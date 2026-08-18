from __future__ import annotations

import gzip
import pickle
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
from rq.job import JobStatus

from webapp.persistence import Base
from webapp.persistence.database import build_engine, create_session_factory
from webapp.persistence.models import ArtifactRecord
from webapp.persistence.registry import ArtifactIdentity, ArtifactRegistryService
from webapp.queue import client as client_module
from webapp.queue import tasks as tasks_module


@pytest.fixture
def control_plane_env(tmp_path, monkeypatch):
    db_path = tmp_path / "control_plane.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("DOCUSCOPE_RQ_ENABLED", "1")
    build_engine.cache_clear()
    create_session_factory.cache_clear()
    engine = build_engine()
    Base.metadata.create_all(engine)
    yield tmp_path
    build_engine.cache_clear()
    create_session_factory.cache_clear()


def test_enqueue_registry_smoke_test_deduplicates_pending_jobs(
    control_plane_env,
    monkeypatch,
):
    registry = ArtifactRegistryService()
    monkeypatch.setattr(client_module, "registry_service", registry)

    class FakeQueue:
        def __init__(self):
            self.enqueued = []
            self.jobs = {}

        def fetch_job(self, job_id):
            return self.jobs.get(job_id)

        def enqueue(self, func, *args, **kwargs):
            self.enqueued.append((func, args, kwargs))
            job = SimpleNamespace(id="rq-job-1", get_status=lambda: JobStatus.QUEUED)
            self.jobs[kwargs["job_id"]] = job
            return job

    fake_queue = FakeQueue()
    monkeypatch.setattr(client_module, "get_queue", lambda: fake_queue)

    first = client_module.enqueue_registry_smoke_test(request_key="demo")
    second = client_module.enqueue_registry_smoke_test(request_key="demo")

    assert first.state == "queued"
    assert first.control_plane_job_id is not None
    assert first.rq_job_id == "rq-job-1"
    assert second.state == "pending"
    assert second.control_plane_job_id == first.control_plane_job_id
    assert len(fake_queue.enqueued) == 1

    job_row = registry.get_job_by_id_internal(first.control_plane_job_id)
    assert job_row is not None
    assert job_row.status == "pending"


def test_run_registry_smoke_test_stores_json_artifact(control_plane_env, monkeypatch):
    registry = ArtifactRegistryService()
    monkeypatch.setattr(client_module, "registry_service", registry)
    monkeypatch.setattr(tasks_module, "registry_service", registry)

    from webapp.persistence import registry as registry_module

    artifact_root = Path(control_plane_env) / "_artifacts"
    monkeypatch.setattr(registry_module, "ARTIFACT_STORE_ROOT", artifact_root)
    monkeypatch.setattr(
        tasks_module,
        "get_current_job",
        lambda: SimpleNamespace(id="rq-job-42"),
    )

    class FakeQueue:
        def fetch_job(self, _job_id):
            return None

        def enqueue(self, func, *args, **kwargs):
            return SimpleNamespace(id="rq-job-42")

    monkeypatch.setattr(client_module, "get_queue", lambda: FakeQueue())

    enqueued = client_module.enqueue_registry_smoke_test(request_key="artifact-demo")
    artifact_id = tasks_module.run_registry_smoke_test(enqueued.control_plane_job_id)

    job_row = registry.get_job_by_id_internal(enqueued.control_plane_job_id)
    assert job_row is not None
    assert job_row.status == "completed"
    assert job_row.artifact_id == artifact_id

    artifact = registry.get_artifact_by_id_internal(artifact_id)
    assert artifact is not None
    payload = registry.load_json_artifact(artifact)
    assert payload["status"] == "ok"
    assert payload["control_plane_job_id"] == enqueued.control_plane_job_id
    assert payload["worker_id"] == "rq-job-42"


def test_registry_smoke_retry_keeps_one_reservation_then_completes(
    control_plane_env,
    monkeypatch,
):
    registry = ArtifactRegistryService()
    monkeypatch.setattr(client_module, "registry_service", registry)
    monkeypatch.setattr(tasks_module, "registry_service", registry)

    from webapp.persistence import registry as registry_module

    artifact_root = Path(control_plane_env) / "_artifacts"
    monkeypatch.setattr(registry_module, "ARTIFACT_STORE_ROOT", artifact_root)
    rq_job = SimpleNamespace(
        id="rq-job-retry",
        retries_left=1,
        get_status=lambda: JobStatus.QUEUED,
    )

    class FakeQueue:
        def __init__(self):
            self.job = None
            self.enqueue_kwargs = None

        def fetch_job(self, _job_id):
            return self.job

        def enqueue(self, _func, *_args, **kwargs):
            self.enqueue_kwargs = kwargs
            self.job = rq_job
            return rq_job

    fake_queue = FakeQueue()
    monkeypatch.setattr(client_module, "get_queue", lambda: fake_queue)
    monkeypatch.setattr(tasks_module, "get_current_job", lambda: rq_job)

    first = client_module.enqueue_registry_smoke_test(request_key="retry-demo")
    original_store = registry.store_json_artifact
    monkeypatch.setattr(
        registry,
        "store_json_artifact",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("temporary failure")
        ),
    )

    with pytest.raises(RuntimeError, match="temporary failure"):
        tasks_module.run_registry_smoke_test(first.control_plane_job_id)
    tasks_module.mark_control_plane_job_failed(
        SimpleNamespace(args=(first.control_plane_job_id,), retries_left=1),
        None,
        RuntimeError,
        RuntimeError("temporary failure"),
        None,
    )

    pending = client_module.enqueue_registry_smoke_test(request_key="retry-demo")
    job_row = registry.get_job_by_id_internal(first.control_plane_job_id)
    assert pending.state == "pending"
    assert pending.control_plane_job_id == first.control_plane_job_id
    assert job_row.status == "running"
    assert job_row.retry_count == 1
    assert (
        fake_queue.enqueue_kwargs["on_failure"]
        is client_module.CONTROL_PLANE_FAILURE_CALLBACK
    )

    rq_job.retries_left = 0
    monkeypatch.setattr(registry, "store_json_artifact", original_store)
    artifact_id = tasks_module.run_registry_smoke_test(first.control_plane_job_id)

    completed = registry.get_job_by_id_internal(first.control_plane_job_id)
    assert completed.status == "completed"
    assert completed.artifact_id == artifact_id
    assert completed.retry_count == 1


def test_terminal_rq_failure_releases_control_plane_reservation(
    control_plane_env,
    monkeypatch,
):
    registry = ArtifactRegistryService()
    monkeypatch.setattr(tasks_module, "registry_service", registry)
    identity = client_module.build_queue_smoke_identity("terminal-failure")
    reservation = registry.reserve_artifact(identity)
    rq_job = SimpleNamespace(args=(reservation.job.job_id,))

    tasks_module.mark_control_plane_job_failed(
        rq_job,
        None,
        RuntimeError,
        RuntimeError("retries exhausted"),
        None,
    )

    failed = registry.get_job_by_id_internal(reservation.job.job_id)
    artifact = registry.get_artifact_by_id_internal(reservation.artifact.artifact_id)
    assert failed.status == "failed"
    assert failed.failure_reason == "retries exhausted"
    assert artifact.status == "failed"


def test_registry_json_artifact_uses_configured_store_root(
    control_plane_env,
    monkeypatch,
):
    artifact_root = Path(control_plane_env) / "configured_artifacts"
    monkeypatch.setenv("DOCUSCOPE_ARTIFACT_STORE_ROOT", str(artifact_root))

    registry = ArtifactRegistryService()
    identity = ArtifactIdentity(
        artifact_type="configured_json",
        scope="public",
        selector_hash="selector",
        selector_payload={"selector": "demo"},
        parameter_hash="params",
        parameter_payload={"params": "demo"},
        pipeline_version="test",
        model_version="test",
    )

    artifact = registry.store_json_artifact(identity, {"ok": True})

    assert Path(artifact.storage_uri).is_relative_to(artifact_root)
    assert registry.load_json_artifact(artifact) == {"ok": True}


def test_enqueue_internal_target_preparation_deduplicates_pending_jobs(
    control_plane_env,
    monkeypatch,
):
    registry = ArtifactRegistryService()
    monkeypatch.setattr(client_module, "registry_service", registry)

    class FakeQueue:
        def __init__(self):
            self.enqueued = []
            self.jobs = {}

        def fetch_job(self, job_id):
            return self.jobs.get(job_id)

        def enqueue(self, func, *args, **kwargs):
            self.enqueued.append((func, args, kwargs))
            job = SimpleNamespace(id="rq-job-target-1", get_status=lambda: JobStatus.QUEUED)
            self.jobs[kwargs["job_id"]] = job
            return job

    fake_queue = FakeQueue()
    monkeypatch.setattr(client_module, "get_queue", lambda: fake_queue)

    first = client_module.enqueue_internal_target_preparation("/tmp/demo-corpus")
    second = client_module.enqueue_internal_target_preparation("/tmp/demo-corpus")

    assert first.state == "queued"
    assert first.control_plane_job_id is not None
    assert second.state == "pending"
    assert second.control_plane_job_id == first.control_plane_job_id
    assert len(fake_queue.enqueued) == 1


def test_run_internal_target_preparation_stores_result_and_frequency_artifact(
    control_plane_env,
    monkeypatch,
):
    registry = ArtifactRegistryService()
    monkeypatch.setattr(client_module, "registry_service", registry)
    monkeypatch.setattr(tasks_module, "registry_service", registry)

    from webapp.persistence import registry as registry_module
    from webapp import corpus_paths

    artifact_root = Path(control_plane_env) / "_artifacts"
    monkeypatch.setattr(registry_module, "ARTIFACT_STORE_ROOT", artifact_root)
    monkeypatch.setattr(
        corpus_paths,
        "_builtin_corpus_root",
        lambda: Path(control_plane_env),
    )
    monkeypatch.setattr(
        tasks_module,
        "get_current_job",
        lambda: SimpleNamespace(id="rq-job-target-42"),
    )

    corpus_dir = Path(control_plane_env) / "ld" / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    ds_tokens = pl.DataFrame(
        {
            "doc_id": ["doc1", "doc1"],
            "token": ["alpha", "beta"],
            "pos_tag": ["NN1", "NN1"],
            "ds_tag": ["Actors", "Actors"],
            "pos_id": [1, 2],
            "ds_id": [1, 2],
        },
        schema={
            "doc_id": pl.String,
            "token": pl.String,
            "pos_tag": pl.String,
            "ds_tag": pl.String,
            "pos_id": pl.UInt32,
            "ds_id": pl.UInt32,
        },
    )
    with gzip.open(corpus_dir / "ds_tokens.gz", "wb") as handle:
        pickle.dump(ds_tokens, handle, protocol=pickle.HIGHEST_PROTOCOL)

    class FakeQueue:
        def fetch_job(self, _job_id):
            return None

        def enqueue(self, func, *args, **kwargs):
            return SimpleNamespace(id="rq-job-target-42")

    monkeypatch.setattr(client_module, "get_queue", lambda: FakeQueue())

    enqueued = client_module.enqueue_internal_target_preparation(str(corpus_dir))
    artifact_id = tasks_module.run_internal_target_preparation(
        enqueued.control_plane_job_id,
        str(corpus_dir),
    )

    job_row = registry.get_job_by_id_internal(enqueued.control_plane_job_id)
    assert job_row is not None
    assert job_row.status == "completed"
    assert job_row.artifact_id == artifact_id

    artifact = registry.get_artifact_by_id_internal(artifact_id)
    assert artifact is not None
    payload = registry.load_json_artifact(artifact)
    assert payload["status"] == "ok"
    assert payload["control_plane_job_id"] == enqueued.control_plane_job_id
    assert payload["frequency_artifact_id"] is not None


def test_enqueue_internal_target_preparation_recovers_orphaned_pending_artifact(
    control_plane_env,
    monkeypatch,
):
    registry = ArtifactRegistryService()
    monkeypatch.setattr(client_module, "registry_service", registry)

    identity = client_module.build_internal_target_identity("/tmp/orphaned-corpus")
    session_factory = create_session_factory()
    with session_factory() as db_session:
        db_session.add(
            ArtifactRecord(
                artifact_type=identity.artifact_type,
                scope=identity.scope,
                owner_principal_id=identity.owner_principal_id,
                selector_hash=identity.selector_hash,
                selector_payload=identity.selector_payload,
                pipeline_version=identity.pipeline_version,
                model_version=identity.model_version,
                parameter_hash=identity.parameter_hash,
                parameter_payload=identity.parameter_payload,
                storage_uri="",
                status="pending",
            )
        )
        db_session.commit()

    class FakeQueue:
        def __init__(self):
            self.enqueued = []

        def fetch_job(self, _job_id):
            return None

        def enqueue(self, func, *args, **kwargs):
            self.enqueued.append((func, args, kwargs))
            return SimpleNamespace(id="rq-job-target-orphaned")

    fake_queue = FakeQueue()
    monkeypatch.setattr(client_module, "get_queue", lambda: fake_queue)

    result = client_module.enqueue_internal_target_preparation("/tmp/orphaned-corpus")

    assert result.state == "queued"
    assert result.control_plane_job_id is not None
    assert len(fake_queue.enqueued) == 1


def test_enqueue_plotbot_generation_deduplicates_pending_jobs(
    control_plane_env,
    monkeypatch,
):
    monkeypatch.setattr(
        client_module.registry_service,
        "reserve_artifact",
        lambda _identity: pytest.fail("Plotbot must not reserve an artifact"),
    )

    class FakeQueue:
        def __init__(self):
            self.enqueued = []
            self.jobs = {}

        def fetch_job(self, job_id):
            return self.jobs.get(job_id)

        def enqueue(self, func, *args, **kwargs):
            self.enqueued.append((func, args, kwargs))
            job = SimpleNamespace(
                id="rq-job-plotbot-1",
                get_status=lambda **_kwargs: JobStatus.QUEUED,
            )
            self.jobs[kwargs["job_id"]] = job
            return job

    fake_queue = FakeQueue()
    monkeypatch.setattr(client_module, "get_plotbot_queue", lambda: fake_queue)

    dataframe_records = [{"label": "a", "value": 1}, {"label": "b", "value": 2}]
    llm_params = {
        "temperature": 0.1,
        "max_tokens": 500,
        "top_p": 0.7,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }
    first = client_module.enqueue_plotbot_generation(
        dataframe_records=dataframe_records,
        source_refs=["builtin:ld/B_MICUSP"],
        plot_lib="plotly.express",
        user_input="Make a bar chart of value by label.",
        llm_params=llm_params,
        requester_principal_id="student@example.edu",
    )
    second = client_module.enqueue_plotbot_generation(
        dataframe_records=dataframe_records,
        source_refs=["builtin:ld/B_MICUSP"],
        plot_lib="plotly.express",
        user_input="Make a bar chart of value by label.",
        llm_params=llm_params,
        requester_principal_id="student@example.edu",
    )

    assert first.state == "queued"
    assert first.rq_job_id == "rq-job-plotbot-1"
    assert second.state == "pending"
    assert second.rq_job_id == first.rq_job_id
    assert len(fake_queue.enqueued) == 1
    _, enqueued_args, enqueued_kwargs = fake_queue.enqueued[0]
    assert type(enqueued_args[3]) is dict
    assert enqueued_kwargs["description"] == (
        f"Plotbot generation ({enqueued_kwargs['job_id']})"
    )


def test_enqueue_plotbot_generation_reuses_retained_rq_result(
    control_plane_env,
    monkeypatch,
):
    monkeypatch.setattr(
        client_module.registry_service,
        "reserve_artifact",
        lambda _identity: pytest.fail("Plotbot must not reserve an artifact"),
    )
    finished_job = SimpleNamespace(
        id="plotbot-finished",
        result={"status": "ok", "result": {}},
        get_status=lambda **_kwargs: JobStatus.FINISHED,
    )
    fake_queue = SimpleNamespace(
        fetch_job=lambda _job_id: finished_job,
        enqueue=lambda *_args, **_kwargs: pytest.fail("job must not be enqueued"),
    )
    monkeypatch.setattr(client_module, "get_plotbot_queue", lambda: fake_queue)

    result = client_module.enqueue_plotbot_generation(
        dataframe_records=[{"label": "a", "value": 1}],
        source_refs=["builtin:ld/B_MICUSP"],
        plot_lib="plotly.express",
        user_input="Make a bar chart.",
        llm_params={},
        requester_principal_id="student:session-1",
    )

    assert result.state == "ready"
    assert result.rq_job_id == "plotbot-finished"


def test_enqueue_plotbot_generation_replaces_failed_rq_job(
    control_plane_env,
    monkeypatch,
):
    deleted = []
    failed_job = SimpleNamespace(
        id="plotbot-failed",
        result=None,
        get_status=lambda **_kwargs: JobStatus.FAILED,
        delete=lambda: deleted.append(True),
    )

    class FakeQueue:
        def fetch_job(self, _job_id):
            return failed_job

        def enqueue(self, _func, *_args, **kwargs):
            assert deleted == [True]
            return SimpleNamespace(id=kwargs["job_id"])

    monkeypatch.setattr(client_module, "get_plotbot_queue", lambda: FakeQueue())

    result = client_module.enqueue_plotbot_generation(
        dataframe_records=[{"label": "a", "value": 1}],
        source_refs=["builtin:ld/B_MICUSP"],
        plot_lib="plotly.express",
        user_input="Make a bar chart.",
        llm_params={},
        requester_principal_id="student:session-1",
    )

    assert result.state == "queued"
    assert result.rq_job_id.startswith("plotbot-")


@pytest.mark.parametrize(
    "source_refs",
    [
        ["/tmp/uploaded-corpus"],
        ["builtin:ld/B_MICUSP", "/tmp/uploaded-reference"],
    ],
)
def test_enqueue_plotbot_generation_rejects_private_sources_before_reservation(
    monkeypatch,
    source_refs,
):
    monkeypatch.setattr(
        client_module,
        "get_redis_queue_config",
        lambda: pytest.fail("queue configuration must not be accessed"),
    )
    monkeypatch.setattr(
        client_module.registry_service,
        "reserve_artifact",
        lambda _identity: pytest.fail("registry must not be accessed"),
    )

    with pytest.raises(ValueError, match="built-in-only"):
        client_module.enqueue_plotbot_generation(
            dataframe_records=[{"label": "private", "value": 1}],
            source_refs=source_refs,
            plot_lib="plotly.express",
            user_input="Make a bar chart.",
            llm_params={},
        )


def test_run_plotbot_generation_returns_ephemeral_serialized_result(
    control_plane_env,
    monkeypatch,
):
    artifact_root = Path(control_plane_env) / "_artifacts"
    monkeypatch.setattr(
        client_module.registry_service,
        "reserve_artifact",
        lambda _identity: pytest.fail("Plotbot must not reserve an artifact"),
    )

    class FailingRegistry:
        def __getattr__(self, _name):
            pytest.fail("Plotbot worker must not access the registry")

    monkeypatch.setattr(tasks_module, "registry_service", FailingRegistry())

    class FakeQueue:
        def fetch_job(self, _job_id):
            return None

        def enqueue(self, func, *args, **kwargs):
            return SimpleNamespace(id="rq-job-plotbot-42")

    monkeypatch.setattr(client_module, "get_plotbot_queue", lambda: FakeQueue())

    dataframe_records = [{"label": "a", "value": 1}, {"label": "b", "value": 2}]
    llm_params = {
        "temperature": 0.1,
        "max_tokens": 500,
        "top_p": 0.7,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }
    cached_code = "fig = px.bar(df, x='label', y='value')"

    enqueued = client_module.enqueue_plotbot_generation(
        dataframe_records=dataframe_records,
        source_refs=["builtin:ld/B_MICUSP"],
        plot_lib="plotly.express",
        user_input="Make a bar chart of value by label.",
        llm_params=llm_params,
        cached_code=cached_code,
        requester_principal_id="student@example.edu",
    )
    payload = tasks_module.run_plotbot_generation(
        dataframe_records,
        "plotly.express",
        "Make a bar chart of value by label.",
        llm_params,
        None,
        None,
        cached_code,
    )

    assert enqueued.rq_job_id == "rq-job-plotbot-42"
    assert payload["status"] == "ok"
    assert payload["result"]["result_type"] == "plot"
    assert "<svg" in payload["result"]["plot_svg"]
    assert "dataframe_records" not in payload
    assert "df" not in payload
    assert (
        payload["result"]["code"] == "fig = px.bar(df, x='label', y='value')"
    )
    assert not artifact_root.exists()
