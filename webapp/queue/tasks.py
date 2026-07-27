"""RQ task functions for background-job smoke testing."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import os
import pickle
from pathlib import Path

import docuscospacy as ds
import pandas as pd
import polars as pl
from polars import DataFrame
from rq import get_current_job

from webapp.persistence import (
    ArtifactIdentity,
    SharedArtifactWorkflow,
    build_shared_collocation_identity,
    build_shared_frequency_identity,
    build_shared_keyness_identity,
    build_shared_keyness_parts_identity,
    build_shared_ngram_identity,
    registry_service,
)
from webapp.utilities.ai.plotbot import run_plotbot_serialized_service
from webapp.utilities.configuration.logging_config import get_logger


logger = get_logger()


def _get_shared_artifact_workflow() -> SharedArtifactWorkflow:
    """Build a shared-artifact workflow against the current registry service."""

    return SharedArtifactWorkflow(registry_service, logger)


def _identity_from_job_row(job_row) -> ArtifactIdentity:
    """Reconstruct a normalized artifact identity from a control-plane job row."""

    return ArtifactIdentity(
        artifact_type=job_row.artifact_type,
        scope=job_row.scope,
        owner_principal_id=job_row.requester_principal_id,
        selector_hash=job_row.selector_hash,
        selector_payload=job_row.selector_payload,
        parameter_hash=job_row.parameter_hash,
        parameter_payload=job_row.parameter_payload,
        pipeline_version=job_row.pipeline_version,
        model_version=job_row.model_version,
    )


def _load_internal_target_tokens(corpus_path: str) -> DataFrame:
    """Load built-in target tokens from the file-backed internal corpus path."""

    ds_tokens_path = Path(corpus_path) / "ds_tokens.gz"
    with gzip.open(ds_tokens_path, "rb") as handle:
        return pickle.load(handle)


def _ensure_shared_frequency_artifact(corpus_path: str):
    """Ensure a shared frequency artifact exists for a built-in target corpus."""

    shared_artifact_workflow = _get_shared_artifact_workflow()
    identity = build_shared_frequency_identity(target_source=corpus_path)
    ready_artifact = registry_service.find_ready_artifact(identity)
    if ready_artifact is not None:
        return ready_artifact

    decision = shared_artifact_workflow.reserve(
        identity,
        cache_name="frequency",
        ready_loader=lambda: registry_service.find_ready_artifact(identity),
        poll_attempts=20,
        poll_interval_seconds=0.5,
    )
    if decision.state == "ready":
        return decision.payload
    if decision.state == "pending":
        raise RuntimeError("Shared frequency artifact is still pending.")
    if decision.state != "reserved":
        raise RuntimeError(f"Unexpected frequency reservation state: {decision.state}")

    ds_tokens = _load_internal_target_tokens(corpus_path)
    ft_pos, ft_ds = ds.frequency_table(ds_tokens, count_by="both")
    artifact = shared_artifact_workflow.store(
        identity,
        decision.job_id,
        cache_name="frequency",
        store_func=lambda artifact_identity: registry_service.store_frequency_bundle(
            artifact_identity,
            {"ft_pos": ft_pos, "ft_ds": ft_ds},
        ),
    )
    if artifact is None:
        raise RuntimeError("Failed to store shared frequency artifact.")
    return artifact


def _load_builtin_tokens(corpus_path: str) -> DataFrame:
    """Load built-in corpus tokens from a file-backed corpus directory."""

    return _load_internal_target_tokens(corpus_path)


def _subset_tokens_by_categories(tokens: DataFrame, categories: list[str]) -> DataFrame:
    """Return tokens whose doc_id prefix belongs to the selected categories."""

    return (
        tokens
        .with_columns(
            pl.col("doc_id").str.split_exact("_", 0)
            .struct.rename_fields(["cat_id"])
            .alias("id")
        )
        .unnest("id")
        .filter(pl.col("cat_id").is_in(categories))
        .drop("cat_id")
    )


def _build_keyness_parts_metadata(
    target_categories: list[str],
    reference_categories: list[str],
    target_tokens: DataFrame,
    reference_tokens: DataFrame,
) -> dict[str, list[str]]:
    """Build metadata expected by the corpus-parts results page."""

    tar_tokens_pos = target_tokens.group_by(
        ["doc_id", "pos_id", "pos_tag"]
    ).agg(pl.col("token").str.concat("")).filter(pl.col("pos_tag") != "Y").height
    ref_tokens_pos = reference_tokens.group_by(
        ["doc_id", "pos_id", "pos_tag"]
    ).agg(pl.col("token").str.concat("")).filter(pl.col("pos_tag") != "Y").height
    tar_tokens_ds = target_tokens.group_by(
        ["doc_id", "ds_id", "ds_tag"]
    ).agg(pl.col("token").str.concat("")).filter(
        ~(pl.col("token").str.contains("^[[[:punct:]] ]+$") & pl.col("ds_tag").str.contains("Untagged"))
    ).height
    ref_tokens_ds = reference_tokens.group_by(
        ["doc_id", "ds_id", "ds_tag"]
    ).agg(pl.col("token").str.concat("")).filter(
        ~(pl.col("token").str.contains("^[[[:punct:]] ]+$") & pl.col("ds_tag").str.contains("Untagged"))
    ).height
    tar_ndocs = target_tokens.get_column("doc_id").unique().len()
    ref_ndocs = reference_tokens.get_column("doc_id").unique().len()
    return {
        "keyness_parts": [
            list(target_categories),
            list(reference_categories),
            str(tar_tokens_pos),
            str(ref_tokens_pos),
            str(tar_tokens_ds),
            str(ref_tokens_ds),
            str(tar_ndocs),
            str(ref_ndocs),
        ]
    }


def run_registry_smoke_test(control_plane_job_id: int) -> int:
    """Run a tiny registry-backed background job and persist a JSON artifact."""

    job_row = registry_service.get_job_by_id(control_plane_job_id)
    if job_row is None:
        raise ValueError(f"Unknown control-plane job id: {control_plane_job_id}")

    identity = _identity_from_job_row(job_row)
    current_rq_job = get_current_job()
    worker_id = current_rq_job.id if current_rq_job is not None else f"pid-{os.getpid()}"
    registry_service.mark_job_running(control_plane_job_id, worker_id=worker_id)

    try:
        payload = {
            "status": "ok",
            "control_plane_job_id": control_plane_job_id,
            "selector_hash": identity.selector_hash,
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "worker_id": worker_id,
            "worker_pid": os.getpid(),
        }
        artifact = registry_service.store_json_artifact(identity, payload)
        registry_service.mark_job_completed(control_plane_job_id, artifact.artifact_id)
        return artifact.artifact_id
    except Exception as exc:
        registry_service.mark_job_failed(control_plane_job_id, str(exc))
        raise


def run_internal_target_preparation(control_plane_job_id: int, corpus_path: str) -> int:
    """Prepare a built-in target corpus off the Streamlit rerun path."""

    job_row = registry_service.get_job_by_id(control_plane_job_id)
    if job_row is None:
        raise ValueError(f"Unknown control-plane job id: {control_plane_job_id}")

    identity = _identity_from_job_row(job_row)
    current_rq_job = get_current_job()
    worker_id = current_rq_job.id if current_rq_job is not None else f"pid-{os.getpid()}"
    registry_service.mark_job_running(control_plane_job_id, worker_id=worker_id)

    try:
        frequency_artifact = _ensure_shared_frequency_artifact(corpus_path)
        payload = {
            "status": "ok",
            "control_plane_job_id": control_plane_job_id,
            "corp_path": corpus_path,
            "frequency_artifact_id": frequency_artifact.artifact_id,
            "selector_hash": identity.selector_hash,
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "worker_id": worker_id,
            "worker_pid": os.getpid(),
        }
        artifact = registry_service.store_json_artifact(identity, payload)
        registry_service.mark_job_completed(control_plane_job_id, artifact.artifact_id)
        return artifact.artifact_id
    except Exception as exc:
        registry_service.mark_job_failed(control_plane_job_id, str(exc))
        raise


def run_keyness_preparation(
    control_plane_job_id: int,
    target_source: str,
    reference_source: str,
    threshold: float,
    swap_target: bool,
) -> int:
    """Generate shared keyness tables for a built-in target/reference pair."""

    job_row = registry_service.get_job_by_id(control_plane_job_id)
    if job_row is None:
        raise ValueError(f"Unknown control-plane job id: {control_plane_job_id}")

    identity = build_shared_keyness_identity(
        target_source=target_source,
        reference_source=reference_source,
        threshold=threshold,
        swap_target=swap_target,
    )
    current_rq_job = get_current_job()
    worker_id = current_rq_job.id if current_rq_job is not None else f"pid-{os.getpid()}"
    registry_service.mark_job_running(control_plane_job_id, worker_id=worker_id)

    try:
        target_tokens = _load_builtin_tokens(target_source)
        reference_tokens = _load_builtin_tokens(reference_source)
        wc_tar_pos, wc_tar_ds = ds.frequency_table(target_tokens, count_by="both")
        tc_tar_pos, tc_tar_ds = ds.tags_table(target_tokens, count_by="both")
        wc_ref_pos, wc_ref_ds = ds.frequency_table(reference_tokens, count_by="both")
        tc_ref_pos, tc_ref_ds = ds.tags_table(reference_tokens, count_by="both")

        keyness_frames = {
            "kw_pos": ds.keyness_table(
                wc_tar_pos,
                wc_ref_pos,
                threshold=threshold,
                swap_target=swap_target,
            ),
            "kw_ds": ds.keyness_table(
                wc_tar_ds,
                wc_ref_ds,
                threshold=threshold,
                swap_target=swap_target,
            ),
            "kt_pos": ds.keyness_table(
                tc_tar_pos,
                tc_ref_pos,
                tags_only=True,
                threshold=threshold,
                swap_target=swap_target,
            ),
            "kt_ds": ds.keyness_table(
                tc_tar_ds,
                tc_ref_ds,
                tags_only=True,
                threshold=threshold,
                swap_target=swap_target,
            ),
        }
        if any(frame is None or getattr(frame, "height", 0) == 0 for frame in keyness_frames.values()):
            raise RuntimeError("Keyness computation returned no results.")

        artifact = registry_service.store_keyness_bundle(identity, keyness_frames)
        registry_service.mark_job_completed(control_plane_job_id, artifact.artifact_id)
        return artifact.artifact_id
    except Exception as exc:
        registry_service.mark_job_failed(control_plane_job_id, str(exc))
        raise


def run_collocation_preparation(
    control_plane_job_id: int,
    target_source: str,
    node_word: str,
    node_tag: str | None,
    to_left: int,
    to_right: int,
    stat_mode: str,
    count_by: str,
) -> int:
    """Generate a shared collocation table for a built-in target corpus."""

    job_row = registry_service.get_job_by_id(control_plane_job_id)
    if job_row is None:
        raise ValueError(f"Unknown control-plane job id: {control_plane_job_id}")

    identity = build_shared_collocation_identity(
        target_source=target_source,
        node_word=node_word,
        node_tag=node_tag,
        to_left=to_left,
        to_right=to_right,
        stat_mode=stat_mode,
        count_by=count_by,
    )
    current_rq_job = get_current_job()
    worker_id = current_rq_job.id if current_rq_job is not None else f"pid-{os.getpid()}"
    registry_service.mark_job_running(control_plane_job_id, worker_id=worker_id)

    try:
        target_tokens = _load_builtin_tokens(target_source)
        collocations = ds.coll_table(
            target_tokens,
            node_word=node_word,
            node_tag=node_tag,
            preceding=to_left,
            following=to_right,
            statistic=stat_mode,
            count_by=count_by,
        )
        if collocations is None or getattr(collocations, "height", 0) == 0:
            raise RuntimeError("Collocation computation returned no results.")

        artifact = registry_service.store_collocation_bundle(identity, collocations)
        registry_service.mark_job_completed(control_plane_job_id, artifact.artifact_id)
        return artifact.artifact_id
    except Exception as exc:
        registry_service.mark_job_failed(control_plane_job_id, str(exc))
        raise


def run_keyness_parts_preparation(
    control_plane_job_id: int,
    target_source: str,
    target_categories: list[str],
    reference_categories: list[str],
    threshold: float,
    swap_target: bool,
) -> int:
    """Generate shared keyness tables for selected parts of a built-in target corpus."""

    job_row = registry_service.get_job_by_id(control_plane_job_id)
    if job_row is None:
        raise ValueError(f"Unknown control-plane job id: {control_plane_job_id}")

    identity = build_shared_keyness_parts_identity(
        target_source=target_source,
        target_categories=target_categories,
        reference_categories=reference_categories,
        threshold=threshold,
        swap_target=swap_target,
    )
    current_rq_job = get_current_job()
    worker_id = current_rq_job.id if current_rq_job is not None else f"pid-{os.getpid()}"
    registry_service.mark_job_running(control_plane_job_id, worker_id=worker_id)

    try:
        tokens = _load_builtin_tokens(target_source)
        target_tokens = _subset_tokens_by_categories(tokens, target_categories)
        reference_tokens = _subset_tokens_by_categories(tokens, reference_categories)
        wc_tar_pos, wc_tar_ds = ds.frequency_table(target_tokens, count_by="both")
        tc_tar_pos, tc_tar_ds = ds.tags_table(target_tokens, count_by="both")
        wc_ref_pos, wc_ref_ds = ds.frequency_table(reference_tokens, count_by="both")
        tc_ref_pos, tc_ref_ds = ds.tags_table(reference_tokens, count_by="both")

        keyness_frames = {
            "kw_pos_cp": ds.keyness_table(
                wc_tar_pos,
                wc_ref_pos,
                threshold=threshold,
                swap_target=swap_target,
            ),
            "kw_ds_cp": ds.keyness_table(
                wc_tar_ds,
                wc_ref_ds,
                threshold=threshold,
                swap_target=swap_target,
            ),
            "kt_pos_cp": ds.keyness_table(
                tc_tar_pos,
                tc_ref_pos,
                tags_only=True,
                threshold=threshold,
                swap_target=swap_target,
            ),
            "kt_ds_cp": ds.keyness_table(
                tc_tar_ds,
                tc_ref_ds,
                tags_only=True,
                threshold=threshold,
                swap_target=swap_target,
            ),
        }
        if any(frame is None or getattr(frame, "height", 0) == 0 for frame in keyness_frames.values()):
            raise RuntimeError("Corpus-parts keyness computation returned no results.")

        metadata = _build_keyness_parts_metadata(
            target_categories,
            reference_categories,
            target_tokens,
            reference_tokens,
        )
        artifact = registry_service.store_keyness_parts_bundle(
            identity,
            keyness_frames,
            metadata,
        )
        registry_service.mark_job_completed(control_plane_job_id, artifact.artifact_id)
        return artifact.artifact_id
    except Exception as exc:
        registry_service.mark_job_failed(control_plane_job_id, str(exc))
        raise


def run_ngram_preparation(
    control_plane_job_id: int,
    target_source: str,
    analysis_type: str,
    ngram_span: int,
    count_by: str,
    from_anchor: str | None = None,
    node_word: str | None = None,
    tag: str | None = None,
    position: int | None = None,
    search_type: str | None = None,
) -> int:
    """Generate a shared n-gram/cluster table for a built-in target corpus."""

    job_row = registry_service.get_job_by_id(control_plane_job_id)
    if job_row is None:
        raise ValueError(f"Unknown control-plane job id: {control_plane_job_id}")

    identity = build_shared_ngram_identity(
        target_source=target_source,
        analysis_type=analysis_type,
        ngram_span=ngram_span,
        count_by=count_by,
        from_anchor=from_anchor,
        node_word=node_word,
        tag=tag,
        position=position,
        search_type=search_type,
    )
    current_rq_job = get_current_job()
    worker_id = current_rq_job.id if current_rq_job is not None else f"pid-{os.getpid()}"
    registry_service.mark_job_running(control_plane_job_id, worker_id=worker_id)

    try:
        tokens = _load_builtin_tokens(target_source)
        if analysis_type == "ngrams":
            ngrams = ds.ngrams(
                tokens_table=tokens,
                span=ngram_span,
                count_by=count_by,
            )
        elif analysis_type == "clusters" and from_anchor == "Token":
            ngrams = ds.clusters_by_token(
                tokens_table=tokens,
                node_word=node_word,
                node_position=position,
                span=ngram_span,
                search_type=search_type,
                count_by=count_by,
            )
        elif analysis_type == "clusters" and from_anchor == "Tag":
            ngrams = ds.clusters_by_tag(
                tokens_table=tokens,
                tag=tag,
                tag_position=position,
                span=ngram_span,
                count_by=count_by,
            )
        else:
            raise ValueError(f"Unsupported n-gram analysis type: {analysis_type}")

        if ngrams is None or getattr(ngrams, "height", 0) == 0:
            raise RuntimeError("N-gram/cluster computation returned no results.")
        if analysis_type == "clusters" and getattr(ngrams, "height", 0) > 100000:
            raise RuntimeError("Cluster computation returned too many results.")

        artifact = registry_service.store_ngram_bundle(identity, ngrams)
        registry_service.mark_job_completed(control_plane_job_id, artifact.artifact_id)
        return artifact.artifact_id
    except Exception as exc:
        registry_service.mark_job_failed(control_plane_job_id, str(exc))
        raise


def run_plotbot_generation(
    control_plane_job_id: int,
    dataframe_records: list[dict[str, object]],
    plot_lib: str,
    user_input: str,
    llm_params: dict[str, object],
    schema: str | None = None,
    code_chunk: str | None = None,
    cached_code: str | None = None,
    api_key: str = "",
) -> int:
    """Generate Plotbot code and store a compact serialized result artifact."""

    job_row = registry_service.get_job_by_id(control_plane_job_id)
    if job_row is None:
        raise ValueError(f"Unknown control-plane job id: {control_plane_job_id}")

    identity = _identity_from_job_row(job_row)
    current_rq_job = get_current_job()
    worker_id = current_rq_job.id if current_rq_job is not None else f"pid-{os.getpid()}"
    registry_service.mark_job_running(control_plane_job_id, worker_id=worker_id)

    try:
        df = pd.DataFrame.from_records(dataframe_records)
        result = run_plotbot_serialized_service(
            df=df,
            plot_lib=plot_lib,
            user_input=user_input,
            api_key=api_key,
            llm_params=llm_params,
            schema=schema,
            code_chunk=code_chunk,
            cached_code=cached_code,
            track_quota=False,
            include_svg=True,
        )
        payload = {
            "status": "ok" if result.success else "error",
            "control_plane_job_id": control_plane_job_id,
            "selector_hash": identity.selector_hash,
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "worker_id": worker_id,
            "worker_pid": os.getpid(),
            "plot_lib": plot_lib,
            "user_input": user_input,
            "result": asdict(result),
        }
        artifact = registry_service.store_json_artifact(identity, payload)
        registry_service.mark_job_completed(control_plane_job_id, artifact.artifact_id)
        return artifact.artifact_id
    except Exception as exc:
        registry_service.mark_job_failed(control_plane_job_id, str(exc))
        raise