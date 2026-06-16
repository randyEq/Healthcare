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
    logger.info(f"  LLM:   {settings.llm_model_name} @ {settings.openai_base_url}")
    logger.info(f"  FAISS: {settings.faiss_index_path}")
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
