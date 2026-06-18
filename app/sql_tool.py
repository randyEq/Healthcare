"""Small SQL helper utilities for application database queries."""

from __future__ import annotations

from typing import Any
from urllib.parse import quote_plus

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from loguru import logger

from app.config import settings

_engine: Engine | None = None


def _sqlalchemy_database_url() -> str:
    """Return a SQLAlchemy-compatible MySQL URL."""
    if settings.database_url:
        if settings.database_url.startswith("mysql://"):
            return settings.database_url.replace("mysql://", "mysql+pymysql://", 1)
        return settings.database_url

    required_settings = {
        "DB_HOST": settings.db_host,
        "DB_PORT": settings.db_port,
        "DB_NAME": settings.db_name,
        "DB_USER": settings.db_user,
    }
    missing = [
        name for name, value in required_settings.items() if value in (None, "")
    ]
    if settings.db_password is None:
        missing.append("DB_PASSWORD")
    if missing:
        raise ValueError(
            "Database settings are incomplete. Set DATABASE_URL or configure "
            f"{', '.join(missing)} in .env."
        )

    user = quote_plus(settings.db_user or "")
    password = quote_plus(settings.db_password or "")
    host = settings.db_host
    port = settings.db_port
    database = quote_plus(settings.db_name or "")
    return f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"


def _create_engine() -> Engine:
    global _engine
    if _engine is None:
        logger.debug("[SQL] Creating SQLAlchemy engine")
        _engine = create_engine(_sqlalchemy_database_url(), pool_pre_ping=True)
    return _engine


def execute_sql(
    query: str, params: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Execute a SQL query and return rows as dictionaries."""
    engine = _create_engine()
    with engine.connect() as connection:
        result = connection.execute(text(query), params or {})
        rows = [dict(row._mapping) for row in result]
        logger.debug(f"[SQL] Query returned {len(rows)} rows")
        return rows


def execute_write(query: str, params: dict[str, Any] | None = None) -> int:
    """Execute a write query and return the affected row count."""
    engine = _create_engine()
    with engine.begin() as connection:
        result = connection.execute(text(query), params or {})
        rowcount = int(result.rowcount or 0)
        logger.debug(f"[SQL] Write affected {rowcount} rows")
        return rowcount


def test_database_connection() -> bool:
    """Return True when the configured database accepts a simple query."""
    rows = execute_sql("SELECT 1 AS ok")
    return bool(rows and rows[0].get("ok") == 1)


def get_disease_severity() -> list[dict[str, Any]]:
    """Fetch disease severity data from the patientcare database."""
    query = """
    SELECT
        disease_name,
        severity_group,
        severity_level
    FROM patientcare.disease;
    """
    return execute_sql(query)
