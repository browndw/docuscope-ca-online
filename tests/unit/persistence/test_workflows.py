"""Unit tests for `SharedArtifactWorkflow` (reserve/load/store control flow).

Uses a mocked registry/logger so these tests stay fast and independent of the
database — the real `ArtifactRegistryService` behavior is covered separately
in test_registry.py.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from webapp.persistence.workflows import SharedArtifactWorkflow


@pytest.fixture
def registry():
    return MagicMock()


@pytest.fixture
def logger():
    return MagicMock()


@pytest.fixture
def workflow(registry, logger):
    return SharedArtifactWorkflow(registry, logger)


class TestLoadReady:
    def test_bypasses_when_identity_is_none(self, workflow, registry):
        result = workflow.load_ready(None, loader=MagicMock(), cache_name="keyness")

        assert result is None
        registry.find_ready_artifact.assert_not_called()

    def test_returns_none_on_cache_miss(self, workflow, registry):
        identity = MagicMock(selector_hash="sel", parameter_hash="params")
        registry.find_ready_artifact.return_value = None

        result = workflow.load_ready(identity, loader=MagicMock(), cache_name="keyness")

        assert result is None

    def test_returns_artifact_and_payload_on_hit(self, workflow, registry):
        identity = MagicMock(selector_hash="sel", parameter_hash="params")
        artifact = MagicMock(artifact_id=1)
        registry.find_ready_artifact.return_value = artifact
        loader = MagicMock(return_value={"data": 1})

        result = workflow.load_ready(identity, loader=loader, cache_name="keyness")

        assert result == (artifact, {"data": 1})
        loader.assert_called_once_with(artifact)

    def test_returns_none_when_loader_raises(self, workflow, registry, logger):
        identity = MagicMock(selector_hash="sel", parameter_hash="params")
        registry.find_ready_artifact.return_value = MagicMock(artifact_id=1)
        loader = MagicMock(side_effect=RuntimeError("corrupt"))

        result = workflow.load_ready(identity, loader=loader, cache_name="keyness")

        assert result is None
        logger.warning.assert_called_once()


class TestReserve:
    def test_bypasses_when_identity_is_none(self, workflow, registry):
        decision = workflow.reserve(None, cache_name="keyness")

        assert decision.state == "bypass"
        registry.reserve_artifact.assert_not_called()

    def test_bypasses_when_reservation_raises(self, workflow, registry, logger):
        identity = MagicMock(selector_hash="sel")
        registry.reserve_artifact.side_effect = RuntimeError("db down")

        decision = workflow.reserve(identity, cache_name="keyness")

        assert decision.state == "bypass"
        logger.warning.assert_called_once()

    def test_reserved_marks_job_running(self, workflow, registry):
        identity = MagicMock(selector_hash="sel")
        job = MagicMock(job_id=42)
        registry.reserve_artifact.return_value = MagicMock(state="reserved", job=job)

        decision = workflow.reserve(identity, cache_name="keyness")

        assert decision.state == "reserved"
        assert decision.job_id == 42
        registry.mark_job_running.assert_called_once_with(42)

    def test_ready_uses_ready_loader_payload(self, workflow, registry):
        identity = MagicMock(selector_hash="sel")
        registry.reserve_artifact.return_value = MagicMock(state="ready", job=None)
        ready_loader = MagicMock(return_value={"data": 1})

        decision = workflow.reserve(
            identity, cache_name="keyness", ready_loader=ready_loader
        )

        assert decision.state == "ready"
        assert decision.payload == {"data": 1}

    def test_ready_without_payload_falls_through_to_bypass(self, workflow, registry):
        identity = MagicMock(selector_hash="sel")
        registry.reserve_artifact.return_value = MagicMock(state="ready", job=None)
        ready_loader = MagicMock(return_value=None)

        decision = workflow.reserve(
            identity, cache_name="keyness", ready_loader=ready_loader
        )

        assert decision.state == "bypass"

    def test_pending_without_polling_returns_pending(self, workflow, registry):
        identity = MagicMock(selector_hash="sel")
        registry.reserve_artifact.return_value = MagicMock(state="pending", job=None)

        decision = workflow.reserve(identity, cache_name="keyness")

        assert decision.state == "pending"

    def test_pending_polls_until_ready_loader_returns_payload(
        self, workflow, registry, monkeypatch
    ):
        identity = MagicMock(selector_hash="sel")
        registry.reserve_artifact.return_value = MagicMock(state="pending", job=None)
        ready_loader = MagicMock(side_effect=[None, None, {"data": 1}])
        monkeypatch.setattr("webapp.persistence.workflows.time.sleep", MagicMock())

        decision = workflow.reserve(
            identity,
            cache_name="keyness",
            ready_loader=ready_loader,
            poll_attempts=3,
            poll_interval_seconds=0.01,
        )

        assert decision.state == "ready"
        assert decision.payload == {"data": 1}
        assert ready_loader.call_count == 3

    def test_pending_polling_exhausted_returns_pending(
        self, workflow, registry, monkeypatch
    ):
        identity = MagicMock(selector_hash="sel")
        registry.reserve_artifact.return_value = MagicMock(state="pending", job=None)
        ready_loader = MagicMock(return_value=None)
        monkeypatch.setattr("webapp.persistence.workflows.time.sleep", MagicMock())

        decision = workflow.reserve(
            identity,
            cache_name="keyness",
            ready_loader=ready_loader,
            poll_attempts=2,
            poll_interval_seconds=0.01,
        )

        assert decision.state == "pending"
        assert ready_loader.call_count == 2

    def test_unrecognized_state_falls_through_to_bypass(self, workflow, registry):
        identity = MagicMock(selector_hash="sel")
        registry.reserve_artifact.return_value = MagicMock(state="something_else", job=None)

        decision = workflow.reserve(identity, cache_name="keyness")

        assert decision.state == "bypass"


class TestStore:
    def test_skips_when_identity_is_none(self, workflow, registry):
        result = workflow.store(
            None, job_id=1, cache_name="keyness", store_func=MagicMock()
        )

        assert result is None
        registry.mark_job_completed.assert_not_called()

    def test_stores_and_completes_job(self, workflow, registry):
        identity = MagicMock(selector_hash="sel")
        artifact = MagicMock(artifact_id=7)
        store_func = MagicMock(return_value=artifact)

        result = workflow.store(
            identity, job_id=42, cache_name="keyness", store_func=store_func
        )

        assert result is artifact
        store_func.assert_called_once_with(identity)
        registry.mark_job_completed.assert_called_once_with(42, 7)

    def test_store_without_job_id_skips_completion(self, workflow, registry):
        identity = MagicMock(selector_hash="sel")
        artifact = MagicMock(artifact_id=7)
        store_func = MagicMock(return_value=artifact)

        result = workflow.store(
            identity, job_id=None, cache_name="keyness", store_func=store_func
        )

        assert result is artifact
        registry.mark_job_completed.assert_not_called()

    def test_store_failure_marks_job_failed_and_returns_none(
        self, workflow, registry, logger
    ):
        identity = MagicMock(selector_hash="sel")
        store_func = MagicMock(side_effect=RuntimeError("disk full"))

        result = workflow.store(
            identity, job_id=42, cache_name="keyness", store_func=store_func
        )

        assert result is None
        registry.mark_job_failed.assert_called_once_with(42, "disk full")
        logger.warning.assert_called_once()

    def test_store_failure_without_job_id_skips_failure_marking(
        self, workflow, registry
    ):
        identity = MagicMock(selector_hash="sel")
        store_func = MagicMock(side_effect=RuntimeError("disk full"))

        result = workflow.store(
            identity, job_id=None, cache_name="keyness", store_func=store_func
        )

        assert result is None
        registry.mark_job_failed.assert_not_called()
