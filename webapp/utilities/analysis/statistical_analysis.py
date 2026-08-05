"""
Corpus generation and statistical analysis utilities.

This module provides functions for generating frequency tables, tag tables,
keyness analysis, and statistical comparisons between corpora.
"""

import polars as pl
import streamlit as st
import docuscospacy as ds
from time import perf_counter
from webapp.utilities.core import app_core
from webapp.utilities.configuration.logging_config import get_logger
from webapp.utilities.analysis.correlation import pearson_correlation
from webapp.persistence import (
    SharedArtifactWorkflow,
    build_shared_keyness_identity,
    build_shared_keyness_parts_identity,
    registry_service,
)
from webapp.utilities.state import (
    CorpusKeys, TargetKeys, ReferenceKeys, WarningKeys, SessionKeys
)
from webapp.utilities.corpus import (
    get_corpus_data,
    get_corpus_data_manager,
    set_corpus_data,
)
from webapp.utilities.session import safe_session_get, get_or_init_user_session


logger = get_logger()
shared_artifact_workflow = SharedArtifactWorkflow(registry_service, logger)
SLOW_GENERATION_TRIGGER_MS = 25
FREQUENCY_RENDER_TRACE_KEY = "_frequency_render_trace"


def _log_slow_generation_trigger(
    operation: str,
    start_time: float,
    session_id: str,
) -> None:
    """Log slow pre-rerun generation trigger steps."""

    elapsed_ms = (perf_counter() - start_time) * 1000
    if elapsed_ms >= SLOW_GENERATION_TRIGGER_MS:
        logger.warning(
            "Slow generation trigger op={} session={} duration_ms={:.2f}",
            operation,
            session_id,
            elapsed_ms,
        )


def _get_shared_keyness_identity(user_session_id: str, threshold: float, swap_target: bool):
    """Return a shared keyness identity for built-in corpus comparisons."""

    _, session = get_or_init_user_session()
    target_db = safe_session_get(session, SessionKeys.TARGET_DB, "")
    reference_db = safe_session_get(session, SessionKeys.REFERENCE_DB, "")
    if not target_db or not reference_db:
        return None

    try:
        return build_shared_keyness_identity(
            target_source=target_db,
            reference_source=reference_db,
            threshold=threshold,
            swap_target=swap_target,
        )
    except ValueError:
        return None


def _normalize_category_selection(categories: list) -> list[str]:
    """Return a stable category selection for artifact identity matching."""

    return sorted(str(category) for category in categories)


def _get_keyness_parts_metadata(
    tar_list: list[str],
    ref_list: list[str],
    tar_tokens_pos: int,
    ref_tokens_pos: int,
    tar_tokens_ds: int,
    ref_tokens_ds: int,
    tar_ndocs: int,
    ref_ndocs: int,
) -> dict[str, list[str]]:
    """Build the metadata payload expected by the corpus-parts results page."""

    return {
        "keyness_parts": [
            list(tar_list),
            list(ref_list),
            str(tar_tokens_pos),
            str(ref_tokens_pos),
            str(tar_tokens_ds),
            str(ref_tokens_ds),
            str(tar_ndocs),
            str(ref_ndocs),
        ]
    }


def _get_shared_keyness_parts_identity(
    user_session_id: str,
    threshold: float,
    swap_target: bool,
):
    """Return a shared keyness-parts identity for built-in target corpora."""

    _, session = get_or_init_user_session()
    target_db = safe_session_get(session, SessionKeys.TARGET_DB, "")
    if not target_db:
        return None

    tar_list = _normalize_category_selection(
        list(st.session_state[user_session_id].get('tar', []))
    )
    ref_list = _normalize_category_selection(
        list(st.session_state[user_session_id].get('ref', []))
    )
    if not tar_list or not ref_list:
        return None

    try:
        return build_shared_keyness_parts_identity(
            target_source=target_db,
            target_categories=tar_list,
            reference_categories=ref_list,
            threshold=threshold,
            swap_target=swap_target,
        )
    except ValueError:
        return None


def _load_cached_keyness_tables(
    user_session_id: str,
    threshold: float,
    swap_target: bool,
) -> bool:
    """Load built-in keyness tables from the shared artifact registry if available."""

    identity = _get_shared_keyness_identity(user_session_id, threshold, swap_target)
    loaded = shared_artifact_workflow.load_ready(
        identity,
        registry_service.load_keyness_bundle,
        cache_name="keyness",
    )
    if loaded is None:
        return False

    artifact, _ = loaded
    try:
        attach_keyness_artifact(
            user_session_id,
            artifact.artifact_id,
            artifact_type=artifact.artifact_type,
        )
        st.success('Keywords loaded from shared cache!')
        st.rerun()
    except Exception as exc:
        logger.warning(f"Shared keyness cache load failed: {exc}")
        return False

    return True


def attach_keyness_parts_artifact(
    user_session_id: str,
    artifact_id: int,
    artifact_type: str | None = None,
) -> bool:
    """Attach a ready corpus-parts keyness artifact to the target corpus session."""

    artifact = registry_service.get_public_artifact_by_id(artifact_id)
    if artifact is None or artifact.status != "ready":
        return False
    if artifact_type is None:
        artifact_type = artifact.artifact_type

    payload = registry_service.load_keyness_parts_bundle(artifact)
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict) or "keyness_parts" not in metadata:
        return False

    manager = get_corpus_data_manager(user_session_id, CorpusKeys.TARGET)
    manager.set_artifact_refs(
        artifact_type,
        artifact_id,
        [
            TargetKeys.KW_POS_CP,
            TargetKeys.KW_DS_CP,
            TargetKeys.KT_POS_CP,
            TargetKeys.KT_DS_CP,
        ],
    )
    app_core.session_manager.update_session_state(
        user_session_id,
        SessionKeys.KEYNESS_PARTS,
        True,
    )
    app_core.session_manager.update_metadata(
        user_session_id,
        CorpusKeys.TARGET,
        metadata,
    )
    st.session_state[user_session_id][WarningKeys.KEYNESS_PARTS] = None
    return True


def attach_keyness_artifact(
    user_session_id: str,
    artifact_id: int,
    artifact_type: str | None = None,
) -> bool:
    """Attach a ready keyness artifact to the current target corpus session."""

    if artifact_type is None:
        artifact = registry_service.get_public_artifact_by_id(artifact_id)
        if artifact is None or artifact.status != "ready":
            return False
        artifact_type = artifact.artifact_type

    manager = get_corpus_data_manager(user_session_id, CorpusKeys.TARGET)
    manager.set_artifact_refs(
        artifact_type,
        artifact_id,
        [TargetKeys.KW_POS, TargetKeys.KW_DS, TargetKeys.KT_POS, TargetKeys.KT_DS],
    )
    app_core.session_manager.update_session_state(
        user_session_id,
        SessionKeys.KEYNESS_TABLE,
        True,
    )
    st.session_state[user_session_id][WarningKeys.KEYNESS] = None
    return True


def _load_cached_keyness_parts(
    user_session_id: str,
    threshold: float,
    swap_target: bool,
) -> bool:
    """Load built-in corpus-parts keyness from the shared artifact registry."""

    identity = _get_shared_keyness_parts_identity(user_session_id, threshold, swap_target)
    loaded = shared_artifact_workflow.load_ready(
        identity,
        registry_service.load_keyness_parts_bundle,
        cache_name="keyness-parts",
    )
    if loaded is None:
        return False

    artifact, _ = loaded
    try:
        attach_keyness_parts_artifact(
            user_session_id,
            artifact.artifact_id,
            artifact_type=artifact.artifact_type,
        )
        st.success('Keywords loaded from shared cache!')
        st.rerun()
    except Exception as exc:
        logger.warning(f"Shared keyness-parts cache load failed: {exc}")
        return False

    return True


def _store_cached_keyness_tables(
    user_session_id: str,
    threshold: float,
    swap_target: bool,
    job_id: int | None,
    kw_pos: pl.DataFrame,
    kw_ds: pl.DataFrame,
    kt_pos: pl.DataFrame,
    kt_ds: pl.DataFrame,
) -> None:
    """Store built-in keyness tables in the shared artifact registry."""

    identity = _get_shared_keyness_identity(user_session_id, threshold, swap_target)
    artifact = shared_artifact_workflow.store(
        identity,
        job_id,
        cache_name="keyness",
        store_func=lambda artifact_identity: registry_service.store_keyness_bundle(
            artifact_identity,
            {
                "kw_pos": kw_pos,
                "kw_ds": kw_ds,
                "kt_pos": kt_pos,
                "kt_ds": kt_ds,
            },
        ),
    )
    if artifact is not None:
        manager = get_corpus_data_manager(user_session_id, CorpusKeys.TARGET)
        manager.set_artifact_refs(
            artifact.artifact_type,
            artifact.artifact_id,
            [TargetKeys.KW_POS, TargetKeys.KW_DS, TargetKeys.KT_POS, TargetKeys.KT_DS],
        )


def _store_cached_keyness_parts(
    user_session_id: str,
    threshold: float,
    swap_target: bool,
    job_id: int | None,
    keyness_frames: dict[str, pl.DataFrame],
    metadata: dict[str, list[str]],
) -> None:
    """Store built-in corpus-parts keyness in the shared artifact registry."""

    identity = _get_shared_keyness_parts_identity(user_session_id, threshold, swap_target)
    artifact = shared_artifact_workflow.store(
        identity,
        job_id,
        cache_name="keyness-parts",
        store_func=lambda artifact_identity: registry_service.store_keyness_parts_bundle(
            artifact_identity,
            keyness_frames,
            metadata,
        ),
    )
    if artifact is not None:
        manager = get_corpus_data_manager(user_session_id, CorpusKeys.TARGET)
        manager.set_artifact_refs(
            artifact.artifact_type,
            artifact.artifact_id,
            [
                TargetKeys.KW_POS_CP,
                TargetKeys.KW_DS_CP,
                TargetKeys.KT_POS_CP,
                TargetKeys.KT_DS_CP,
            ],
        )


def _reserve_shared_keyness_artifact(
    user_session_id: str,
    threshold: float,
    swap_target: bool,
) -> int | None:
    """Reserve a built-in keyness artifact or defer to an in-flight computation."""

    identity = _get_shared_keyness_identity(user_session_id, threshold, swap_target)
    decision = shared_artifact_workflow.reserve(
        identity,
        cache_name="keyness",
        ready_loader=lambda: (
            registry_service.load_keyness_bundle(artifact)
            if (artifact := registry_service.find_ready_artifact(identity)) is not None
            else None
        ),
    )

    if decision.state == "reserved":
        return decision.job_id

    if decision.state == "ready":
        return _load_cached_keyness_tables(user_session_id, threshold, swap_target)

    if decision.state == "pending":
        st.session_state[user_session_id][WarningKeys.KEYNESS] = (
            "This built-in keyness comparison is already being generated. "
            "Please try again shortly.",
            ":material/schedule:",
        )
        return None

    return None


def _reserve_shared_keyness_parts_artifact(
    user_session_id: str,
    threshold: float,
    swap_target: bool,
) -> int | None:
    """Reserve a built-in corpus-parts keyness artifact or defer if in flight."""

    identity = _get_shared_keyness_parts_identity(user_session_id, threshold, swap_target)
    decision = shared_artifact_workflow.reserve(
        identity,
        cache_name="keyness-parts",
        ready_loader=lambda: (
            registry_service.load_keyness_parts_bundle(artifact)
            if (artifact := registry_service.find_ready_artifact(identity)) is not None
            else None
        ),
    )

    if decision.state == "reserved":
        return decision.job_id

    if decision.state == "ready":
        return _load_cached_keyness_parts(user_session_id, threshold, swap_target)

    if decision.state == "pending":
        st.session_state[user_session_id][WarningKeys.KEYNESS_PARTS] = (
            "This built-in corpus-parts comparison is already being generated. "
            "Please try again shortly.",
            ":material/schedule:",
        )
        return None

    return None


def generate_frequency_table(user_session_id: str) -> None:
    """
    Load frequency tables for the target corpus.

    Parameters
    ----------
    user_session_id : str
        The session ID for the user.

    Returns
    -------
    None
    """
    manager = get_corpus_data_manager(user_session_id, CorpusKeys.TARGET)
    if not manager.is_ready():
        st.session_state[user_session_id][WarningKeys.FREQUENCY] = (
            "Frequency table cannot be generated: no tokens found in the target corpus.",
            ":material/sentiment_stressed:"
        )
        return

    generation_start = perf_counter()
    st.session_state[user_session_id][FREQUENCY_RENDER_TRACE_KEY] = {
        "started_at": generation_start,
        "logged": False,
    }

    update_start = perf_counter()
    st.session_state[user_session_id][SessionKeys.FREQ_TABLE_TRANSIENT] = True
    update_elapsed_ms = (perf_counter() - update_start) * 1000
    st.session_state[user_session_id][FREQUENCY_RENDER_TRACE_KEY][
        "update_session_state_ms"
    ] = update_elapsed_ms
    _log_slow_generation_trigger(
        "generate_frequency_table.update_session_state",
        update_start,
        user_session_id,
    )

    st.session_state[user_session_id][WarningKeys.FREQUENCY] = None
    st.session_state[user_session_id][FREQUENCY_RENDER_TRACE_KEY][
        "pre_rerun_ms"
    ] = (perf_counter() - generation_start) * 1000
    _log_slow_generation_trigger(
        "generate_frequency_table.pre_rerun_total",
        generation_start,
        user_session_id,
    )
    st.rerun()


def generate_tags_table(
        user_session_id: str
) -> None:
    """
    Load tags tables for the target corpus.

    Parameters
    ----------
    user_session_id : str
        The session ID for the user.

    Returns
    -------
    None
    """
    manager = get_corpus_data_manager(user_session_id, CorpusKeys.TARGET)
    if not manager.is_ready():
        st.session_state[user_session_id][WarningKeys.TAGS] = (
            "Tags table cannot be generated: no tokens found in the target corpus.",
            ":material/info:"
        )
        return

    app_core.session_manager.update_session_state(user_session_id, 'tags_table', True)
    st.session_state[user_session_id][WarningKeys.TAGS] = None
    st.rerun()


def generate_keyness_tables(
        user_session_id: str,
        threshold: float = 0.01,
        swap_target: bool = False
        ) -> None:
    """
    Generate keyness tables comparing target and reference corpora.

    Parameters
    ----------
    user_session_id : str
        The session ID for the user.
    threshold : float, optional
        Statistical significance threshold for keyness analysis.
    swap_target : bool, optional
        Whether to swap target and reference for comparison.

    Returns
    -------
    None
    """
    if _load_cached_keyness_tables(user_session_id, threshold, swap_target):
        return

    job_id = _reserve_shared_keyness_artifact(user_session_id, threshold, swap_target)
    if (
        job_id is None and
        _get_shared_keyness_identity(user_session_id, threshold, swap_target)
    ):
        return

    # --- Try to get all required frequency/tag tables ---
    try:
        wc_tar_pos = get_corpus_data(user_session_id, CorpusKeys.TARGET, TargetKeys.FT_POS)
        wc_tar_ds = get_corpus_data(user_session_id, CorpusKeys.TARGET, TargetKeys.FT_DS)
        tc_tar_pos = get_corpus_data(user_session_id, CorpusKeys.TARGET, TargetKeys.TT_POS)
        tc_tar_ds = get_corpus_data(user_session_id, CorpusKeys.TARGET, TargetKeys.TT_DS)
        wc_ref_pos = get_corpus_data(
            user_session_id, CorpusKeys.REFERENCE, ReferenceKeys.FT_POS
        )
        wc_ref_ds = get_corpus_data(
            user_session_id, CorpusKeys.REFERENCE, ReferenceKeys.FT_DS
        )
        tc_ref_pos = get_corpus_data(
            user_session_id, CorpusKeys.REFERENCE, ReferenceKeys.TT_POS
        )
        tc_ref_ds = get_corpus_data(
            user_session_id, CorpusKeys.REFERENCE, ReferenceKeys.TT_DS
        )
    except (KeyError, ValueError):
        st.session_state[user_session_id][WarningKeys.KEYNESS] = (
            """
            Keyness cannot be computed: missing frequency or tag tables.
            Please generate frequency and tag tables for both corpora.
            """,
            ":material/sentiment_stressed:"
        )
        return

    freq_tables = [
        wc_tar_pos, wc_tar_ds, tc_tar_pos, tc_tar_ds,
        wc_ref_pos, wc_ref_ds, tc_ref_pos, tc_ref_ds
    ]
    if any(df is None or getattr(df, "height", 0) == 0 for df in freq_tables):
        st.session_state[user_session_id][WarningKeys.KEYNESS] = (
            "Keyness cannot be computed: one or more required tables are empty.",
            ":material/sentiment_stressed:"
        )
        return

    try:
        kw_pos = ds.keyness_table(
            wc_tar_pos,
            wc_ref_pos,
            threshold=threshold,
            swap_target=swap_target,
        )
        kw_ds = ds.keyness_table(
            wc_tar_ds,
            wc_ref_ds,
            threshold=threshold,
            swap_target=swap_target,
        )
        kt_pos = ds.keyness_table(
            tc_tar_pos,
            tc_ref_pos,
            tags_only=True,
            threshold=threshold,
            swap_target=swap_target,
        )
        kt_ds = ds.keyness_table(
            tc_tar_ds,
            tc_ref_ds,
            tags_only=True,
            threshold=threshold,
            swap_target=swap_target,
        )
    except Exception as exc:
        if job_id is not None:
            registry_service.mark_job_failed(job_id, str(exc))
        raise

    keyness_tables = [kw_pos, kw_ds, kt_pos, kt_ds]
    if any(df is None or getattr(df, "height", 0) == 0 for df in keyness_tables):
        st.session_state[user_session_id][WarningKeys.KEYNESS] = (
            "Keyness computation returned no results. Try different data.",
            ":material/counter_0:"
        )
        return

    # Store results using the corpus data manager
    if _get_shared_keyness_identity(user_session_id, threshold, swap_target) is None:
        set_corpus_data(user_session_id, CorpusKeys.TARGET, TargetKeys.KW_POS, kw_pos)
        set_corpus_data(user_session_id, CorpusKeys.TARGET, TargetKeys.KW_DS, kw_ds)
        set_corpus_data(user_session_id, CorpusKeys.TARGET, TargetKeys.KT_POS, kt_pos)
        set_corpus_data(user_session_id, CorpusKeys.TARGET, TargetKeys.KT_DS, kt_ds)
    _store_cached_keyness_tables(
        user_session_id,
        threshold,
        swap_target,
        job_id,
        kw_pos,
        kw_ds,
        kt_pos,
        kt_ds,
    )

    app_core.session_manager.update_session_state(user_session_id, 'keyness_table', True)
    st.session_state[user_session_id][WarningKeys.KEYNESS] = None
    st.success('Keywords generated!')
    st.rerun()


def generate_keyness_parts(
        user_session_id: str,
        threshold: float = 0.01,
        swap_target: bool = False
        ) -> None:
    # --- Check for metadata ---
    _, session = get_or_init_user_session()
    if safe_session_get(session, 'has_meta', False) is False:
        st.session_state[user_session_id]["keyness_parts_warning"] = (
            """
            No metadata found for the target corpus.
            Please load or generate metadata first
            from **Manage Corpus Data**.
            """,
            ":material/sentiment_stressed:"
        )
        return

    tar_list = list(st.session_state[user_session_id].get('tar', []))
    ref_list = list(st.session_state[user_session_id].get('ref', []))

    # --- Check for empty categories ---
    if len(tar_list) == 0 or len(ref_list) == 0:
        st.session_state[user_session_id]["keyness_parts_warning"] = (
            "You must select at least one category for both target and reference parts.",
            ":material/info:"
        )
        return

    if _load_cached_keyness_parts(user_session_id, threshold, swap_target):
        return

    job_id = _reserve_shared_keyness_parts_artifact(
        user_session_id,
        threshold,
        swap_target,
    )
    if (
        job_id is None and
        _get_shared_keyness_parts_identity(user_session_id, threshold, swap_target)
    ):
        return

    # --- Main logic ---
    tok_pl = get_corpus_data(user_session_id, CorpusKeys.TARGET, TargetKeys.DS_TOKENS)

    tar_pl = subset_pl(tok_pl, tar_list)
    ref_pl = subset_pl(tok_pl, ref_list)

    wc_tar_pos, wc_tar_ds = ds.frequency_table(tar_pl, count_by="both")
    tc_tar_pos, tc_tar_ds = ds.tags_table(tar_pl, count_by="both")
    wc_ref_pos, wc_ref_ds = ds.frequency_table(ref_pl, count_by="both")
    tc_ref_pos, tc_ref_ds = ds.tags_table(ref_pl, count_by="both")

    kw_pos_cp = ds.keyness_table(wc_tar_pos, wc_ref_pos, threshold=threshold, swap_target=swap_target)  # noqa: E501
    kw_ds_cp = ds.keyness_table(wc_tar_ds, wc_ref_ds, threshold=threshold, swap_target=swap_target)  # noqa: E501
    kt_pos_cp = ds.keyness_table(tc_tar_pos, tc_ref_pos, tags_only=True, threshold=threshold, swap_target=swap_target)  # noqa: E501
    kt_ds_cp = ds.keyness_table(tc_tar_ds, tc_ref_ds, tags_only=True, threshold=threshold, swap_target=swap_target)  # noqa: E501

    # --- Check for empty results ---
    keyness_tables = [kw_pos_cp, kw_ds_cp, kt_pos_cp, kt_ds_cp]
    if any(df is None or getattr(df, "height", 0) == 0 for df in keyness_tables):
        st.session_state[user_session_id]["keyness_parts_warning"] = (
            """
            Keyness computation for corpus parts returned no results.
            Try different categories.
            """,
            ":material/info:"
        )
        return

    tar_tokens_pos = tar_pl.group_by(
        ["doc_id", "pos_id", "pos_tag"]
    ).agg(pl.col("token").str.concat("")).filter(pl.col("pos_tag") != "Y").height
    ref_tokens_pos = ref_pl.group_by(
        ["doc_id", "pos_id", "pos_tag"]
    ).agg(pl.col("token").str.concat("")).filter(pl.col("pos_tag") != "Y").height

    tar_tokens_ds = tar_pl.group_by(
        ["doc_id", "ds_id", "ds_tag"]
    ).agg(pl.col("token").str.concat("")).filter(
        ~(pl.col("token").str.contains("^[[[:punct:]] ]+$") & pl.col("ds_tag").str.contains("Untagged"))  # noqa: E501
    ).height
    ref_tokens_ds = ref_pl.group_by(
        ["doc_id", "ds_id", "ds_tag"]
    ).agg(pl.col("token").str.concat("")).filter(
        ~(pl.col("token").str.contains("^[[[:punct:]] ]+$") & pl.col("ds_tag").str.contains("Untagged"))  # noqa: E501
    ).height

    tar_ndocs = tar_pl.get_column("doc_id").unique().len()
    ref_ndocs = ref_pl.get_column("doc_id").unique().len()

    metadata = _get_keyness_parts_metadata(
        tar_list,
        ref_list,
        tar_tokens_pos,
        ref_tokens_pos,
        tar_tokens_ds,
        ref_tokens_ds,
        tar_ndocs,
        ref_ndocs,
    )

    # --- Save results and clear warning ---
    if _get_shared_keyness_parts_identity(user_session_id, threshold, swap_target) is None:
        set_corpus_data(user_session_id, CorpusKeys.TARGET, "kw_pos_cp", kw_pos_cp)
        set_corpus_data(user_session_id, CorpusKeys.TARGET, "kw_ds_cp", kw_ds_cp)
        set_corpus_data(user_session_id, CorpusKeys.TARGET, "kt_pos_cp", kt_pos_cp)
        set_corpus_data(user_session_id, CorpusKeys.TARGET, "kt_ds_cp", kt_ds_cp)
    _store_cached_keyness_parts(
        user_session_id,
        threshold,
        swap_target,
        job_id,
        {
            "kw_pos_cp": kw_pos_cp,
            "kw_ds_cp": kw_ds_cp,
            "kt_pos_cp": kt_pos_cp,
            "kt_ds_cp": kt_ds_cp,
        },
        metadata,
    )

    app_core.session_manager.update_session_state(user_session_id, 'keyness_parts', True)

    app_core.session_manager.update_metadata(
        user_session_id,
        'target',
        metadata
    )

    st.session_state[user_session_id]["keyness_parts_warning"] = None
    st.success('Keywords generated!')
    st.rerun()


def subset_pl(
        tok_pl,
        select_ids: list
        ) -> pl.DataFrame:
    token_subset = (
        tok_pl
        .with_columns(
            pl.col("doc_id").str.split_exact("_", 0)
            .struct.rename_fields(["cat_id"])
            .alias("id")
        )
        .unnest("id")
        .filter(pl.col("cat_id").is_in(select_ids))
        .drop("cat_id")
        )
    return token_subset


def freq_simplify_pl(frequency_table) -> pl.DataFrame:
    """
    Simplifies a frequency table DataFrame by replacing part-of-speech tags
    with simplified tags and ensuring the DataFrame has the required structure.

    Parameters
    ----------
    frequency_table : pl.DataFrame
        A Polars DataFrame with columns: Token, Tag, AF, RF, Range.

    Returns
    -------
    pl.DataFrame
        A Polars DataFrame with simplified part-of-speech tags.

    Raises
    ------
    ValueError
        If the DataFrame does not have the required columns or if the tags
        do not match the expected part-of-speech patterns.
    """
    # import polars as pl  # Moved to module level

    required_columns = {'Token', 'Tag', 'AF', 'RF', 'Range'}
    table_columns = set(frequency_table.columns)
    if not required_columns.issubset(table_columns):
        raise ValueError("""
                         Invalid DataFrame.
                         Expected a DataFrame produced by frequency_table
                         with columns: Token, Tag, AF, RF, Range.
                         """)
    tag_prefix = ["NN", "VV", "II"]
    if (not any(
        x.startswith(tuple(tag_prefix)) for x in
        frequency_table.get_column("Tag").to_list()
                )):
        raise ValueError("""
                         Invalid DataFrame.
                         Expected a frequency table with part-of-speech tags.
                         """)

    simple_df = (
        frequency_table
        .with_columns(
            pl.selectors.starts_with("Tag")
            .str.replace('^NN\\S*$', '#NounCommon')
            .str.replace('^VV\\S*$', '#VerbLex')
            .str.replace('^J\\S*$', '#Adjective')
            .str.replace('^R\\S*$', '#Adverb')
            .str.replace('^P\\S*$', '#Pronoun')
            .str.replace('^I\\S*$', '#Preposition')
            .str.replace('^C\\S*$', '#Conjunction')
            .str.replace('^N\\S*$', '#NounOther')
            .str.replace('^VB\\S*$', '#VerbBe')
            .str.replace('^V\\S*$', '#VerbOther')
        )
        .with_columns(
            pl.when(pl.selectors.starts_with("Tag").str.starts_with("#"))
            .then(pl.selectors.starts_with("Tag"))
            .otherwise(
                pl.selectors.starts_with("Tag").str.replace('^\\S+$', '#Other')
                ))
        .with_columns(
            pl.selectors.starts_with("Tag").str.replace("#", "")
        ))

    return simple_df


def correlation_update(
    cc_dict: dict,
    df,
    x: str,
    y: str,
    group_col: str,
    highlight_groups: list
) -> dict:
    """
    Updates cc_dict with highlight and non-highlight group correlations.

    Parameters
    ----------
    cc_dict : dict
        Dictionary to store correlation results
    df : DataFrame
        DataFrame containing the data
    x : str
        Column name for x variable
    y : str
        Column name for y variable
    group_col : str
        Column name for grouping variable
    highlight_groups : list
        List of groups to highlight

    Returns
    -------
    dict
        Updated correlation dictionary
    """
    # Convert to pandas if polars
    if isinstance(df, pl.DataFrame):
        df = df.to_pandas()

    # Highlight group
    df_high = df[df[group_col].isin(highlight_groups)]
    if len(df_high) > 2:
        cc_high = pearson_correlation(df_high[x], df_high[y])
        cc_dict['highlight'] = {
            'df': len(df_high.index) - 2,
            'r': round(cc_high.statistic, 3),
            'p': round(cc_high.pvalue, 5)
        }
    else:
        cc_dict['highlight'] = None

    # Non-highlight group
    df_non = df[~df[group_col].isin(highlight_groups)]
    if len(df_non) > 2:
        cc_non = pearson_correlation(df_non[x], df_non[y])
        cc_dict['non_highlight'] = {
            'df': len(df_non.index) - 2,
            'r': round(cc_non.statistic, 3),
            'p': round(cc_non.pvalue, 5)
        }
    else:
        cc_dict['non_highlight'] = None

    return cc_dict


def update_pca_plot(coord_data, contrib_data, variance, pca_idx) -> tuple:
    """
    Update PCA plot data based on the selected PCA index.

    Parameters
    ----------
    coord_data : DataFrame
        PCA coordinate data
    contrib_data : DataFrame
        PCA contribution data
    variance : array-like
        Variance explained by each component
    pca_idx : int
        PCA index (1-based)

    Returns
    -------
    tuple
        Tuple containing PCA x, y, contribution x, y, variance explained,
        and plot data
    """
    pca_x = coord_data.columns[pca_idx - 1]
    pca_y = coord_data.columns[pca_idx]

    mean_x = contrib_data[pca_x].abs().mean()
    mean_y = contrib_data[pca_y].abs().mean()

    # Always use .copy() after filtering
    contrib_x = contrib_data[contrib_data[pca_x].abs() > mean_x].copy()
    contrib_x.sort_values(by=pca_x, ascending=False, inplace=True)
    contrib_x_values = contrib_x.loc[:, pca_x].tolist()
    contrib_x_values = ['%.2f' % x for x in contrib_x_values]
    contrib_x_values = [x + "%" for x in contrib_x_values]
    contrib_x_tags = contrib_x.loc[:, "Tag"].tolist()
    contrib_x = list(zip(contrib_x_tags, contrib_x_values))
    contrib_x = list(map(', '.join, contrib_x))
    contrib_x = '; '.join(contrib_x)

    contrib_y = contrib_data[contrib_data[pca_y].abs() > mean_y].copy()
    contrib_y.sort_values(by=pca_y, ascending=False, inplace=True)
    contrib_y_values = contrib_y.loc[:, pca_y].tolist()
    contrib_y_values = ['%.2f' % y for y in contrib_y_values]
    contrib_y_values = [y + "%" for y in contrib_y_values]
    contrib_y_tags = contrib_y.loc[:, "Tag"].tolist()
    contrib_y = list(zip(contrib_y_tags, contrib_y_values))
    contrib_y = list(map(', '.join, contrib_y))
    contrib_y = '; '.join(contrib_y)

    contrib_1 = contrib_data[contrib_data[pca_x].abs() > 0].copy()
    contrib_1[pca_x] = contrib_1[pca_x].div(100)
    contrib_1.sort_values(by=pca_x, ascending=True, inplace=True)

    contrib_2 = contrib_data[contrib_data[pca_y].abs() > 0].copy()
    contrib_2[pca_y] = contrib_2[pca_y].div(100)
    contrib_2.sort_values(by=pca_y, ascending=True, inplace=True)

    # Handle variance data - could be array-like or dict
    if isinstance(variance, dict):
        # Try to get variance for the PC components
        pc1_key = f"PC{pca_idx}"
        pc2_key = f"PC{pca_idx + 1}"
        ve_1 = "{:.2%}".format(variance.get(pc1_key, 0))
        ve_2 = "{:.2%}".format(variance.get(pc2_key, 0))
    else:
        # Assume array-like structure
        try:
            var1 = variance[pca_idx - 1] if len(variance) > pca_idx - 1 else 0
            var2 = variance[pca_idx] if len(variance) > pca_idx else 0
            ve_1 = "{:.2%}".format(var1)
            ve_2 = "{:.2%}".format(var2)
        except (IndexError, TypeError):
            ve_1 = "0.00%"
            ve_2 = "0.00%"

    # For plotting: keep the filtered and sorted DataFrames
    contrib_1_plot = contrib_data[contrib_data[pca_x].abs() > 0][["Tag", pca_x]].copy()
    contrib_1_plot[pca_x] = contrib_1_plot[pca_x] / 100
    contrib_1_plot.sort_values(by=pca_x, ascending=True, inplace=True)

    contrib_2_plot = contrib_data[contrib_data[pca_y].abs() > 0][["Tag", pca_y]].copy()
    contrib_2_plot[pca_y] = contrib_2_plot[pca_y] / 100
    contrib_2_plot.sort_values(by=pca_y, ascending=True, inplace=True)

    return (
        pca_x, pca_y, contrib_x, contrib_y, ve_1, ve_2,
        contrib_1_plot, contrib_2_plot
    )
