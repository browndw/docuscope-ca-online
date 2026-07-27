"""Database configuration helpers for the Postgres control plane."""

from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class DatabaseConfig:
    """Resolved database settings for the control-plane database."""

    url: str
    echo_sql: bool = False
    pool_pre_ping: bool = True


def _build_database_url() -> str:
    database_url = os.environ.get("DATABASE_URL")
    if database_url:
        return database_url

    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    database = os.environ.get("POSTGRES_DB", "docuscope_ca")
    user = os.environ.get("POSTGRES_USER", "docuscope")
    password = os.environ.get("POSTGRES_PASSWORD", "docuscope")
    return f"postgresql+psycopg://{user}:{password}@{host}:{port}/{database}"


def get_database_config() -> DatabaseConfig:
    """Return database settings derived from environment variables."""

    return DatabaseConfig(
        url=_build_database_url(),
        echo_sql=os.environ.get("SQL_ECHO", "0") == "1",
        pool_pre_ping=True,
    )
