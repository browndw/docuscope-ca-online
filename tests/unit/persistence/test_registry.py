"""Unit tests for `ArtifactRegistryService` — the shared artifact/dedup engine.

Covers reservation state transitions (reserved/pending/ready), self-healing
when storage goes missing, job lifecycle transitions, and store/load round
trips for every artifact bundle type (exercising the atomic staged-write
path added for review item #5).
"""

from __future__ import annotations

import shutil

import polars as pl
import pytest

from webapp.persistence import Base
from webapp.persistence.database import build_engine, create_session_factory
from webapp.persistence.registry import (
    COLLOCATION_ARTIFACT_TYPE,
    FREQUENCY_ARTIFACT_TYPE,
    KEYNESS_ARTIFACT_TYPE,
    KEYNESS_PARTS_ARTIFACT_TYPE,
    NGRAM_ARTIFACT_TYPE,
    ArtifactIdentity,
    ArtifactRegistryService,
    PUBLIC_OWNER_PRINCIPAL_ID,
    build_shared_frequency_identity,
    build_shared_keyness_identity,
)


@pytest.fixture
def registry(tmp_path, monkeypatch):
    db_path = tmp_path / "control_plane.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("DOCUSCOPE_ARTIFACT_STORE_ROOT", str(tmp_path / "_artifacts"))
    build_engine.cache_clear()
    create_session_factory.cache_clear()
    engine = build_engine()
    Base.metadata.create_all(engine)
    yield ArtifactRegistryService()
    build_engine.cache_clear()
    create_session_factory.cache_clear()


def _identity(**overrides) -> ArtifactIdentity:
    defaults: dict = dict(
        artifact_type="json_artifact",
        scope="public",
        selector_hash="selector-1",
        selector_payload={"source": "demo"},
        parameter_hash="params-1",
        parameter_payload={"threshold": 0.01},
        pipeline_version="1.0.0",
        model_version="preprocessed",
        owner_principal_id=None,
    )
    defaults.update(overrides)
    return ArtifactIdentity(**defaults)


class TestSharedArtifactPolicy:
    def test_public_identity_uses_non_null_owner_sentinel(self):
        identity = build_shared_frequency_identity(
            target_source="builtin:ld/B_MICUSP",
        )

        assert identity.owner_principal_id == PUBLIC_OWNER_PRINCIPAL_ID

    def test_private_identity_requires_owner(self):
        with pytest.raises(ValueError, match="require an owner"):
            _identity(scope="private", owner_principal_id=None)

    def test_builtin_identity_uses_bundled_model_metadata(self):
        identity = build_shared_frequency_identity(
            target_source="builtin:ld/B_MICUSP",
        )

        assert identity.model_version == "docusco_spacy@1.5+911539e"

    def test_equivalent_builtin_sources_share_identity(self):
        first = build_shared_frequency_identity(
            target_source="builtin:ld/B_MICUSP",
        )
        second = build_shared_frequency_identity(
            target_source="builtin:ld/B_MICUSP",
        )

        assert first == second

    def test_uploaded_source_is_not_eligible(self, tmp_path):
        with pytest.raises(ValueError, match="built-in corpus source"):
            build_shared_frequency_identity(target_source=str(tmp_path))

    def test_mixed_keyness_sources_are_not_eligible(self, tmp_path):
        with pytest.raises(ValueError, match="built-in corpus source"):
            build_shared_keyness_identity(
                target_source="builtin:ld/B_MICUSP",
                reference_source=str(tmp_path),
                threshold=0.01,
                swap_target=False,
            )

    def test_model_families_have_distinct_fingerprints(self):
        large_dictionary = build_shared_frequency_identity(
            target_source="builtin:ld/B_MICUSP",
        )
        common_dictionary = build_shared_frequency_identity(
            target_source="builtin:cd/B_MICUSP",
        )

        assert large_dictionary.model_version == "docusco_spacy@1.5+911539e"
        assert common_dictionary.model_version == "docusco_spacy_cd@1.5+911539e"

    def test_keyness_rejects_mixed_model_families(self):
        with pytest.raises(ValueError, match="same model"):
            build_shared_keyness_identity(
                target_source="builtin:ld/B_MICUSP",
                reference_source="builtin:cd/B_MICUSP",
                threshold=0.01,
                swap_target=False,
            )

    def test_model_version_override_cannot_bypass_metadata(self):
        with pytest.raises(ValueError, match="bundled model metadata"):
            build_shared_frequency_identity(
                target_source="builtin:ld/B_MICUSP",
                model_version="preprocessed",
            )


class TestRegistryReadAuthorization:
    def test_principal_reads_allow_public_rows(self, registry):
        reservation = registry.reserve_artifact(_identity())

        assert registry.get_artifact_by_id_for_principal(
            reservation.artifact.artifact_id,
            "any-principal",
        ) is not None
        assert registry.get_job_by_id_for_principal(
            reservation.job.job_id,
            "any-principal",
        ) is not None

    def test_public_reads_reject_private_rows(self, registry):
        private_identity = _identity(
            scope="private",
            owner_principal_id="owner-1",
        )
        reservation = registry.reserve_artifact(private_identity)

        assert (
            registry.get_public_artifact_by_id(reservation.artifact.artifact_id)
            is None
        )
        assert registry.get_public_job_by_id(reservation.job.job_id) is None

    def test_owner_can_read_private_artifact_and_job(self, registry):
        identity = _identity(scope="private", owner_principal_id="owner-1")
        reservation = registry.reserve_artifact(identity)

        artifact = registry.get_artifact_by_id_for_principal(
            reservation.artifact.artifact_id,
            "owner-1",
        )
        job = registry.get_job_by_id_for_principal(
            reservation.job.job_id,
            "owner-1",
        )

        assert artifact is not None
        assert job is not None

    def test_other_principal_cannot_read_private_artifact_or_job(self, registry):
        identity = _identity(scope="private", owner_principal_id="owner-1")
        reservation = registry.reserve_artifact(identity)

        assert (
            registry.get_artifact_by_id_for_principal(
                reservation.artifact.artifact_id,
                "owner-2",
            )
            is None
        )
        assert (
            registry.get_job_by_id_for_principal(
                reservation.job.job_id,
                "owner-2",
            )
            is None
        )

    def test_empty_principal_cannot_read_private_rows(self, registry):
        identity = _identity(scope="private", owner_principal_id="owner-1")
        reservation = registry.reserve_artifact(identity)

        assert registry.get_artifact_by_id_for_principal(
            reservation.artifact.artifact_id,
            "",
        ) is None
        assert registry.get_job_by_id_for_principal(
            reservation.job.job_id,
            "",
        ) is None

    def test_sharing_principal_can_read_private_artifact_and_job(self, registry):
        identity = _identity(scope="private", owner_principal_id="owner-1")
        reservation = registry.reserve_artifact(identity)
        with registry._session_factory() as session:
            artifact = session.get(
                type(reservation.artifact),
                reservation.artifact.artifact_id,
            )
            artifact.sharing_principal_id = "shared-with-1"
            session.commit()

        assert registry.get_artifact_by_id_for_principal(
            reservation.artifact.artifact_id,
            "shared-with-1",
        ) is not None
        assert registry.get_job_by_id_for_principal(
            reservation.job.job_id,
            "shared-with-1",
        ) is not None


class TestReserveArtifact:
    def test_first_reservation_creates_pending_artifact_and_job(self, registry):
        result = registry.reserve_artifact(_identity())

        assert result.state == "reserved"
        assert result.artifact is not None
        assert result.artifact.status == "pending"
        assert result.job is not None
        assert result.job.status == "pending"

    def test_concurrent_reservation_returns_pending_with_same_job(self, registry):
        identity = _identity()
        first = registry.reserve_artifact(identity)
        second = registry.reserve_artifact(identity)

        assert second.state == "pending"
        assert second.job is not None
        assert second.job.job_id == first.job.job_id

    def test_reservation_after_ready_returns_ready_and_increments_access(self, registry):
        identity = _identity()
        stored = registry.store_json_artifact(identity, {"value": 1})
        assert stored.access_count == 1

        result = registry.reserve_artifact(identity)

        assert result.state == "ready"
        assert result.artifact.artifact_id == stored.artifact_id
        assert result.artifact.access_count == 2

    def test_reservation_self_heals_when_ready_storage_missing(self, registry):
        identity = _identity()
        stored = registry.store_json_artifact(identity, {"value": 1})
        shutil.rmtree(stored.storage_uri)

        result = registry.reserve_artifact(identity)

        assert result.state == "reserved"
        assert result.artifact.status == "pending"
        assert result.job is not None

    def test_orphaned_pending_artifact_recovers_with_new_job(self, registry):
        identity = _identity()
        first = registry.reserve_artifact(identity)

        # Simulate an interrupted enqueue: the artifact is still "pending" but
        # its job is no longer active (missing/failed), so the next
        # reservation attempt should recover by creating a fresh job.
        with registry._session_factory() as session:
            job = session.get(type(first.job), first.job.job_id)
            job.status = "failed"
            session.commit()

        second = registry.reserve_artifact(identity)

        assert second.state == "reserved"
        assert second.job is not None
        assert second.job.job_id != first.job.job_id


class TestFindReadyArtifact:
    def test_returns_none_when_no_artifact_exists(self, registry):
        assert registry.find_ready_artifact(_identity()) is None

    def test_returns_ready_artifact_and_increments_access(self, registry):
        identity = _identity()
        stored = registry.store_json_artifact(identity, {"value": 1})

        found = registry.find_ready_artifact(identity)

        assert found is not None
        assert found.artifact_id == stored.artifact_id
        assert found.access_count == 2

    def test_marks_failed_and_returns_none_when_storage_missing(self, registry):
        identity = _identity()
        stored = registry.store_json_artifact(identity, {"value": 1})
        shutil.rmtree(stored.storage_uri)

        assert registry.find_ready_artifact(identity) is None

        refreshed = registry.get_artifact_by_id_internal(stored.artifact_id)
        assert refreshed.status == "failed"

    def test_get_artifact_by_id_returns_none_when_ready_storage_missing(self, registry):
        identity = _identity()
        stored = registry.store_json_artifact(identity, {"value": 1})
        shutil.rmtree(stored.storage_uri)

        assert registry.get_artifact_by_id_internal(stored.artifact_id) is None

        with registry._session_factory() as session:
            refreshed = session.get(type(stored), stored.artifact_id)
            assert refreshed.status == "failed"


class TestJobLifecycle:
    def test_mark_job_running_then_completed_links_artifact(self, registry):
        identity = _identity()
        reservation = registry.reserve_artifact(identity)

        registry.mark_job_running(reservation.job.job_id, worker_id="worker-1")
        job = registry.get_job_by_id_internal(reservation.job.job_id)
        assert job.status == "running"
        assert job.worker_id == "worker-1"

        artifact = registry.store_json_artifact(identity, {"value": 42})
        registry.mark_job_completed(reservation.job.job_id, artifact.artifact_id)

        job = registry.get_job_by_id_internal(reservation.job.job_id)
        assert job.status == "completed"
        assert job.artifact_id == artifact.artifact_id

    def test_private_reservation_expires_but_public_does_not(
        self,
        registry,
        monkeypatch,
    ):
        monkeypatch.setenv("DOCUSCOPE_PRIVATE_ARTIFACT_TTL_HOURS", "6")

        private = registry.reserve_artifact(
            _identity(scope="private", owner_principal_id="user-1")
        )
        public = registry.reserve_artifact(_identity(selector_hash="public"))

        assert private.artifact.expires_at is not None
        assert private.artifact.expires_at > private.artifact.created_at
        assert public.artifact.expires_at is None

    def test_mark_job_failed_releases_pending_artifact(self, registry):
        identity = _identity()
        reservation = registry.reserve_artifact(identity)

        registry.mark_job_failed(reservation.job.job_id, "boom")

        job = registry.get_job_by_id_internal(reservation.job.job_id)
        assert job.status == "failed"
        assert job.failure_reason == "boom"

        artifact = registry.get_artifact_by_id_internal(
            reservation.artifact.artifact_id
        )
        assert artifact.status == "failed"

    def test_record_job_retry_keeps_reservation_active(self, registry):
        identity = _identity()
        reservation = registry.reserve_artifact(identity)
        registry.mark_job_running(reservation.job.job_id, worker_id="worker-1")

        registry.record_job_retry(reservation.job.job_id, "temporary failure")

        job = registry.get_job_by_id_internal(reservation.job.job_id)
        assert job.status == "running"
        assert job.retry_count == 1
        assert job.failure_reason == "temporary failure"
        assert job.finished_at is None
        artifact = registry.get_artifact_by_id_internal(
            reservation.artifact.artifact_id
        )
        assert artifact.status == "pending"

    def test_mark_job_failed_on_unknown_job_is_a_no_op(self, registry):
        # Should not raise even though no job exists with this id.
        registry.mark_job_failed(999_999, "irrelevant")


class TestArtifactBundleRoundTrips:
    def test_json_artifact_round_trip(self, registry):
        identity = _identity(artifact_type="json_artifact")

        artifact = registry.store_json_artifact(identity, {"hello": "world"})
        assert artifact.status == "ready"

        payload = registry.load_json_artifact(artifact)
        assert payload == {"hello": "world"}

    def test_model_and_pipeline_versions_use_distinct_storage(self, registry):
        original = registry.store_json_artifact(
            _identity(model_version="model@1", pipeline_version="pipeline@1"),
            {"version": "original"},
        )
        new_model = registry.store_json_artifact(
            _identity(model_version="model@2", pipeline_version="pipeline@1"),
            {"version": "new-model"},
        )
        new_pipeline = registry.store_json_artifact(
            _identity(model_version="model@1", pipeline_version="pipeline@2"),
            {"version": "new-pipeline"},
        )

        storage_uris = {
            original.storage_uri,
            new_model.storage_uri,
            new_pipeline.storage_uri,
        }
        assert len(storage_uris) == 3
        assert registry.load_json_artifact(original) == {"version": "original"}

    def test_keyness_bundle_store_load_and_dispatch(self, registry):
        identity = _identity(artifact_type=KEYNESS_ARTIFACT_TYPE)
        frames = {
            "kw_pos": pl.DataFrame({"Tag": ["NN1"], "LL": [1.0]}),
            "kw_ds": pl.DataFrame({"Tag": ["NN1"], "LL": [2.0]}),
            "kt_pos": pl.DataFrame({"Tag": ["NN1"], "LL": [3.0]}),
            "kt_ds": pl.DataFrame({"Tag": ["NN1"], "LL": [4.0]}),
        }

        artifact = registry.store_keyness_bundle(identity, frames)
        assert artifact.status == "ready"

        loaded = registry.load_artifact_payload(artifact)
        assert loaded["kw_pos"]["LL"][0] == 1.0
        assert loaded["kt_ds"]["LL"][0] == 4.0

    def test_keyness_parts_bundle_store_load_and_dispatch(self, registry):
        identity = _identity(artifact_type=KEYNESS_PARTS_ARTIFACT_TYPE)
        frames = {
            "kw_pos_cp": pl.DataFrame({"Tag": ["NN1"], "LL": [1.0]}),
            "kw_ds_cp": pl.DataFrame({"Tag": ["NN1"], "LL": [2.0]}),
            "kt_pos_cp": pl.DataFrame({"Tag": ["NN1"], "LL": [3.0]}),
            "kt_ds_cp": pl.DataFrame({"Tag": ["NN1"], "LL": [4.0]}),
        }

        artifact = registry.store_keyness_parts_bundle(
            identity, frames, metadata={"threshold": 0.05}
        )
        loaded = registry.load_artifact_payload(artifact)

        assert loaded["kw_pos_cp"]["LL"][0] == 1.0
        assert loaded["metadata"] == {"threshold": 0.05}

    def test_frequency_bundle_store_load_and_dispatch(self, registry):
        identity = _identity(artifact_type=FREQUENCY_ARTIFACT_TYPE)
        frames = {
            "ft_pos": pl.DataFrame({"Tag": ["NN1"], "AF": [10]}),
            "ft_ds": pl.DataFrame({"Tag": ["NN1"], "AF": [20]}),
        }

        artifact = registry.store_frequency_bundle(identity, frames)
        loaded = registry.load_artifact_payload(artifact)

        assert loaded["ft_pos"]["AF"][0] == 10
        assert loaded["ft_ds"]["AF"][0] == 20

    def test_collocation_bundle_store_load_and_dispatch(self, registry):
        identity = _identity(artifact_type=COLLOCATION_ARTIFACT_TYPE)
        collocations = pl.DataFrame({"Token": ["cat"], "MI": [1.5]})

        artifact = registry.store_collocation_bundle(identity, collocations)
        loaded = registry.load_artifact_payload(artifact)

        assert loaded["collocations"]["MI"][0] == 1.5

    def test_ngram_bundle_store_load_and_dispatch(self, registry):
        identity = _identity(artifact_type=NGRAM_ARTIFACT_TYPE)
        ngrams = pl.DataFrame({"Ngram": ["a b"], "Frequency": [3]})

        artifact = registry.store_ngram_bundle(identity, ngrams)
        loaded = registry.load_artifact_payload(artifact)

        assert loaded["ngrams"]["Frequency"][0] == 3

    def test_restoring_an_artifact_overwrites_cleanly_with_no_stray_dirs(self, registry):
        identity = _identity(artifact_type=COLLOCATION_ARTIFACT_TYPE)
        first = registry.store_collocation_bundle(
            identity, pl.DataFrame({"Token": ["cat"], "MI": [1.0]})
        )
        second = registry.store_collocation_bundle(
            identity, pl.DataFrame({"Token": ["dog"], "MI": [2.0]})
        )

        assert first.artifact_id == second.artifact_id

        from webapp.persistence.registry import _load_cached_collocation_bundle
        _load_cached_collocation_bundle.cache_clear()
        loaded = registry.load_artifact_payload(second)
        assert loaded["collocations"]["Token"][0] == "dog"

        artifact_dir_parent = registry.get_artifact_by_id_internal(
            second.artifact_id
        )
        from pathlib import Path
        siblings = {p.name for p in Path(artifact_dir_parent.storage_uri).parent.iterdir()}
        assert siblings == {Path(artifact_dir_parent.storage_uri).name}

    def test_load_artifact_payload_raises_for_unsupported_type(self, registry):
        identity = _identity(artifact_type="json_artifact")
        artifact = registry.store_json_artifact(identity, {"a": 1})
        artifact.artifact_type = "not_a_real_type"

        with pytest.raises(ValueError):
            registry.load_artifact_payload(artifact)
