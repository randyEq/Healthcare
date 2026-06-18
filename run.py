#!/usr/bin/env python3
"""Entry point for the Healthcare CDSS application."""
import uvicorn
from loguru import logger

from app.config import settings


def main() -> None:
    logger.info("=" * 60)
    logger.info("Starting Healthcare CDSS")
    logger.info(f"  Host:  {settings.app_host}")
    logger.info(f"  Port:  {settings.app_port}")
    logger.info(f"  Env:   {settings.app_env}")
    llm_base = settings.openai_base_url or "OpenAI default endpoint"
    logger.info(f"  LLM:   {settings.llm_model_name} @ {llm_base}")
    logger.info(f"  FAISS: {settings.faiss_index_path}")
    logger.info(
        "  LangSmith: {} project={} api_key_set={}",
        "enabled" if settings.langsmith_tracing else "disabled",
        settings.langsmith_project,
        bool(settings.langsmith_api_key),
    )
    logger.info("=" * 60)

    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
