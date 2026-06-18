"""Configuration for Healthcare CDSS."""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import AliasChoices, Field
from loguru import logger


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # LLM Configuration
    openai_api_key: str = Field(default="...", alias="OPENAI_API_KEY")
    openai_base_url: Optional[str] = Field(default=None, alias="OPENAI_BASE_URL")
    llm_model_name: str = Field(
        default="gpt-4.1-nano",
        validation_alias=AliasChoices("OPENAI_MODEL", "LLM_MODEL_NAME"),
    )
    embedding_model_name: str = Field(
        default="text-embedding-3-small", alias="EMBEDDING_MODEL_NAME"
    )

    # LangSmith monitoring / tracing
    langsmith_tracing: bool = Field(
        default=False,
        validation_alias=AliasChoices("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2"),
    )
    langsmith_api_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY"),
    )
    langsmith_project: str = Field(
        default="healthcare-cdss",
        validation_alias=AliasChoices("LANGSMITH_PROJECT", "LANGCHAIN_PROJECT"),
    )
    langsmith_endpoint: str = Field(
        default="https://api.smith.langchain.com",
        validation_alias=AliasChoices("LANGSMITH_ENDPOINT", "LANGCHAIN_ENDPOINT"),
    )

    # FAISS
    faiss_index_path: str = Field(
        default="/workspace/vector_db", alias="FAISS_INDEX_PATH"
    )

    # App
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")
    app_env: str = Field(default="development", alias="APP_ENV")
    app_debug: bool = Field(default=True, alias="APP_DEBUG")
    session_secret: Optional[str] = Field(default=None, alias="SESSION_SECRET")

    # Database (disabled by default for local development)
    enable_db: bool = Field(default=False, alias="ENABLE_DB")
    db_host: Optional[str] = Field(default=None, alias="DB_HOST")
    db_port: Optional[int] = Field(default=None, alias="DB_PORT")
    db_name: Optional[str] = Field(default=None, alias="DB_NAME")
    db_user: Optional[str] = Field(default=None, alias="DB_USER")
    db_password: Optional[str] = Field(default=None, alias="DB_PASSWORD")
    database_url: Optional[str] = Field(
        default=None,
        alias="DATABASE_URL",
    )

    # Remote biomedical MCP server
    medical_apis_mcp_url: str = Field(
        default="https://mcp.cloud.curiloo.com/tools/unified/mcp",
        alias="MEDICAL_APIS_MCP_URL",
    )

    # Load environment from a project-relative .env by default
    _BASE_DIR = Path(__file__).resolve().parent.parent
    model_config = {
        "env_file": str(_BASE_DIR / ".env"),
        "env_file_encoding": "utf-8",
        "populate_by_name": True,
        "extra": "ignore",
    }


settings = Settings()


def configure_langsmith() -> None:
    """Configure LangSmith tracing environment variables for LangChain."""
    tracing_enabled = bool(settings.langsmith_tracing)
    os.environ["LANGSMITH_TRACING"] = "true" if tracing_enabled else "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "true" if tracing_enabled else "false"

    if not tracing_enabled:
        logger.info("[LangSmith] Monitoring disabled")
        return

    if settings.langsmith_api_key:
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
    else:
        logger.warning(
            "[LangSmith] Monitoring enabled but LANGSMITH_API_KEY is not set"
        )

    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
    os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
    os.environ["LANGCHAIN_ENDPOINT"] = settings.langsmith_endpoint

    logger.info(
        "[LangSmith] Monitoring enabled project={} endpoint={} api_key_set={}",
        settings.langsmith_project,
        settings.langsmith_endpoint,
        bool(settings.langsmith_api_key),
    )


configure_langsmith()
