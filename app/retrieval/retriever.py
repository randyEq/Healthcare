"""FAISS retriever for medical knowledge base.

Uses HuggingFace sentence-transformers for local embeddings (no API cost)
and FAISS for fast vector similarity search.
"""

from __future__ import annotations

from pathlib import Path
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from loguru import logger

from app.config import settings


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Create the local HuggingFace embedding model.

    Uses sentence-transformers/all-MiniLM-L6-v2 by default — fast, lightweight,
    and runs locally (no API calls).
    """
    model_name = settings.embedding_model_name
    logger.info(f"[Retriever] Loading embedding model: {model_name}")
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def load_vector_store() -> FAISS | None:
    """Load the FAISS index from disk (cached at module level).

    Returns None if no index exists yet — caller should handle gracefully.
    """
    index_path = Path(settings.faiss_index_path)
    if not index_path.exists() or not (index_path / "index.faiss").exists():
        logger.warning(f"[Retriever] No FAISS index found at {index_path}")
        return None

    try:
        embeddings = get_embeddings()
        store = FAISS.load_local(
            folder_path=str(index_path),
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info(f"[Retriever] FAISS index loaded from {index_path}")
        return store
    except Exception as e:
        logger.error(f"[Retriever] Failed to load FAISS index: {e}")
        return None


def retrieve(query: str, k: int = 5) -> str:
    """Retrieve relevant medical knowledge for a query.

    Returns a formatted string of retrieved context, or a fallback message
    if the index is not available.
    """
    store = load_vector_store()
    if store is None:
        logger.warning("[Retriever] Vector store not available — using empty context")
        return "No medical knowledge base available. Proceeding with general reasoning."

    try:
        docs = store.similarity_search(query, k=k)
        if not docs:
            return "No relevant medical knowledge retrieved."

        formatted: list[str] = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown")
            condition = doc.metadata.get("condition", "")
            header = f"[{i}] {condition}" if condition else f"[{i}] (Source: {source})"
            content = doc.page_content.strip()
            formatted.append(f"{header}\n{content}")

        return "\n\n---\n\n".join(formatted)
    except Exception as e:
        logger.error(f"[Retriever] Retrieval failed: {e}")
        return f"Retrieval error: {e}. Proceeding with general reasoning."
