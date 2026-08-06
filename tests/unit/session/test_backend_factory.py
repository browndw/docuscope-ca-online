"""Tests for automatic session backend selection."""

from importlib import import_module
from unittest.mock import patch

import pytest

from webapp.config.unified import config


backend_module = import_module("webapp.utilities.storage.backend_factory")


@pytest.fixture(autouse=True)
def reset_backend_factory():
    backend_module.backend_factory.clear_cache()
    config.clear_desktop_fallback()
    yield
    backend_module.backend_factory.clear_cache()
    config.clear_desktop_fallback()


def test_local_postgres_failure_falls_back_to_desktop_memory(monkeypatch):
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql+psycopg://user:password@localhost:5432/docuscope",
    )
    memory_backend = object()

    with (
        patch.object(
            backend_module,
            "PostgresSessionBackend",
            side_effect=RuntimeError("connection refused"),
        ),
        patch.object(backend_module, "_create_memory_backend", return_value=memory_backend),
    ):
        backend = backend_module.backend_factory.get_backend()

    assert backend is memory_backend
    assert config.is_desktop_mode() is True


def test_remote_postgres_failure_remains_fatal(monkeypatch):
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql+psycopg://user:password@postgres:5432/docuscope",
    )

    with patch.object(
        backend_module,
        "PostgresSessionBackend",
        side_effect=RuntimeError("connection refused"),
    ):
        with pytest.raises(RuntimeError, match="connection refused"):
            backend_module.backend_factory.get_backend()

    assert config.is_desktop_mode() is False


def test_explicit_postgres_backend_does_not_fall_back(monkeypatch):
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql+psycopg://user:password@localhost:5432/docuscope",
    )

    with patch.object(
        backend_module,
        "PostgresSessionBackend",
        side_effect=RuntimeError("connection refused"),
    ):
        with pytest.raises(RuntimeError, match="connection refused"):
            backend_module.backend_factory.get_backend("postgres")

    assert config.is_desktop_mode() is False


def test_local_fallback_can_be_disabled(monkeypatch):
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql+psycopg://user:password@localhost:5432/docuscope",
    )
    monkeypatch.setenv("DOCUSCOPE_DISABLE_DESKTOP_FALLBACK", "1")

    with patch.object(
        backend_module,
        "PostgresSessionBackend",
        side_effect=RuntimeError("connection refused"),
    ):
        with pytest.raises(RuntimeError, match="connection refused"):
            backend_module.backend_factory.get_backend()

    assert config.is_desktop_mode() is False
