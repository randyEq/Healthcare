"""Small SQL helper utilities for application database queries."""

from __future__ import annotations

from typing import Any
from urllib.parse import quote_plus

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from loguru import logger

from app.config import settings

_engine: Engine | None = None


def database_config_summary() -> dict[str, Any]:
    """Return non-secret database connection details for logs."""
    if settings.database_url:
        return {
            "source": "DATABASE_URL",
            "host": "<from_url>",
            "port": "<from_url>",
            "database": "<from_url>",
            "user": "<from_url>",
        }

    return {
        "source": "DB_*",
        "host": settings.db_host,
        "port": settings.db_port,
        "database": settings.db_name,
        "user": settings.db_user,
    }


def is_database_configured() -> bool:
    """Return True when enough DB settings exist to attempt a connection."""
    if settings.database_url:
        return True
    return all(
        value not in (None, "")
        for value in (
            settings.db_host,
            settings.db_port,
            settings.db_name,
            settings.db_user,
            settings.db_password,
        )
    )


def _safe_error_message(exc: Exception) -> str:
    """Return an error string with known secrets removed."""
    message = str(getattr(exc, "orig", exc))
    if settings.db_password:
        message = message.replace(settings.db_password, "***redacted***")
    if settings.database_url:
        message = message.replace(settings.database_url, "***redacted_database_url***")
    return message


def _statement_kind(query: str) -> str:
    """Return the first SQL keyword for compact logging."""
    stripped = query.strip()
    return stripped.split(None, 1)[0].upper() if stripped else "UNKNOWN"


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
        logger.info(
            "[SQL] Creating SQLAlchemy engine target={}",
            database_config_summary(),
        )
        _engine = create_engine(_sqlalchemy_database_url(), pool_pre_ping=True)
    return _engine


def execute_sql(
    query: str, params: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Execute a SQL query and return rows as dictionaries."""
    engine = _create_engine()
    statement_kind = _statement_kind(query)
    try:
        with engine.connect() as connection:
            logger.debug(
                "[SQL] Read start statement={} target={}",
                statement_kind,
                database_config_summary(),
            )
            result = connection.execute(text(query), params or {})
            rows = [dict(row._mapping) for row in result]
            logger.info(
                "[SQL] Read success statement={} rows={} target={}",
                statement_kind,
                len(rows),
                database_config_summary(),
            )
            return rows
    except Exception as exc:
        logger.error(
            "[SQL] Read failure statement={} target={} error_type={} error={}",
            statement_kind,
            database_config_summary(),
            exc.__class__.__name__,
            _safe_error_message(exc),
        )
        raise


def execute_write(query: str, params: dict[str, Any] | None = None) -> int:
    """Execute a write query and return the affected row count."""
    engine = _create_engine()
    statement_kind = _statement_kind(query)
    try:
        with engine.begin() as connection:
            logger.debug(
                "[SQL] Write start statement={} target={}",
                statement_kind,
                database_config_summary(),
            )
            result = connection.execute(text(query), params or {})
            rowcount = int(result.rowcount or 0)
            logger.info(
                "[SQL] Write success statement={} affected_rows={} target={}",
                statement_kind,
                rowcount,
                database_config_summary(),
            )
            return rowcount
    except Exception as exc:
        logger.error(
            "[SQL] Write failure statement={} target={} error_type={} error={}",
            statement_kind,
            database_config_summary(),
            exc.__class__.__name__,
            _safe_error_message(exc),
        )
        raise


def test_database_connection() -> bool:
    """Return True when the configured database accepts a simple query."""
    logger.info("[SQL] Connection test start target={}", database_config_summary())
    try:
        rows = execute_sql("SELECT 1 AS ok")
        ok = bool(rows and rows[0].get("ok") == 1)
        if ok:
            logger.info(
                "[SQL] Connection test success target={}",
                database_config_summary(),
            )
        else:
            logger.error(
                "[SQL] Connection test failed target={} reason=unexpected_result",
                database_config_summary(),
            )
        return ok
    except Exception as exc:
        logger.error(
            "[SQL] Connection test failed target={} error_type={} error={}",
            database_config_summary(),
            exc.__class__.__name__,
            _safe_error_message(exc),
        )
        return False


def get_disease_severity() -> list[dict[str, Any]]:
    """Fetch disease severity data from the patientcare database."""
    query = """
    SELECT
        disease_name,
        severity_group,
        severity_level
    FROM disease;
    """
    return execute_sql(query)
