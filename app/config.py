"""Configuration for Healthcare CDSS."""
import os
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # LLM Configuration (OpenAI-compatible via Drytis gateway)
    openai_api_key: str = Field(default="...", alias="OPENAI_API_KEY")
    openai_base_url: str = Field(default="https://llm.drytis.ai/v1", alias="OPENAI_BASE_URL")
    llm_model_name: str = Field(default="gpt-4o", alias="LLM_MODEL_NAME")
    embedding_model_name: str = Field(default="text-embedding-3-small", alias="EMBEDDING_MODEL_NAME")

    # FAISS
    faiss_index_path: str = Field(default="/workspace/vector_db", alias="FAISS_INDEX_PATH")

    # App
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")
    app_env: str = Field(default="development", alias="APP_ENV")
    app_debug: bool = Field(default=True, alias="APP_DEBUG")

    # Database
    db_host: str = Field(default="localhost", alias="DB_HOST")
    db_port: int = Field(default=3306, alias="DB_PORT")
    db_name: str = Field(default="cdss", alias="DB_NAME")
    db_user: str = Field(default="root", alias="DB_USER")
    db_password: str = Field(default="", alias="DB_PASSWORD")

    model_config = {
        "env_file": "/workspace/.env",
        "env_file_encoding": "utf-8",
        "populate_by_name": True,
        "extra": "ignore",
    }


settings = Settings()
