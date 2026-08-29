"""SQLAlchemy engine and session helpers for the control plane."""

from __future__ import annotations

from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from webapp.persistence.config import DatabaseConfig, get_database_config


class Base(DeclarativeBase):
    """Declarative base for control-plane models."""


def _connect_args(database_url: str) -> dict[str, object]:
    """Return driver options compatible with transaction-pooling proxies."""

    if database_url.startswith("postgresql+psycopg"):
        return {"prepare_threshold": None}
    return {}


@lru_cache(maxsize=1)
def build_engine() -> object:
    """Build and cache the primary SQLAlchemy engine."""

    config: DatabaseConfig = get_database_config()
    return create_engine(
        config.url,
        echo=config.echo_sql,
        pool_pre_ping=config.pool_pre_ping,
        connect_args=_connect_args(config.url),
        future=True,
    )


@lru_cache(maxsize=1)
def create_session_factory() -> sessionmaker:
    """Build and cache the primary session factory."""

    return sessionmaker(
        bind=build_engine(),
        autoflush=False,
        expire_on_commit=False,
    )


def initialize_database_schema() -> None:
    """Create control-plane tables when they do not yet exist."""

    Base.metadata.create_all(build_engine())
