"""Tests for control-plane SQLAlchemy engine construction."""

from unittest.mock import patch

from webapp.persistence.config import DatabaseConfig
from webapp.persistence.database import build_engine


def test_psycopg_engine_disables_automatic_prepared_statements():
    """Transaction-pooling proxies must not receive connection-bound prepares."""

    config = DatabaseConfig(
        url="postgresql+psycopg://user:password@pooler:5432/docuscope",
    )
    engine = object()
    build_engine.cache_clear()

    with (
        patch("webapp.persistence.database.get_database_config", return_value=config),
        patch("webapp.persistence.database.create_engine", return_value=engine) as create,
    ):
        assert build_engine() is engine

    create.assert_called_once_with(
        config.url,
        echo=False,
        pool_pre_ping=True,
        connect_args={"prepare_threshold": None},
        future=True,
    )
    build_engine.cache_clear()


def test_non_psycopg_engine_has_no_driver_connect_args():
    """Non-psycopg database URLs should retain their normal driver behavior."""

    config = DatabaseConfig(url="sqlite:///control-plane.db")
    engine = object()
    build_engine.cache_clear()

    with (
        patch("webapp.persistence.database.get_database_config", return_value=config),
        patch("webapp.persistence.database.create_engine", return_value=engine) as create,
    ):
        assert build_engine() is engine

    create.assert_called_once_with(
        config.url,
        echo=False,
        pool_pre_ping=True,
        connect_args={},
        future=True,
    )
    build_engine.cache_clear()
