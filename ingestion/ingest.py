#!/usr/bin/env python3
"""
FAISS Ingestion Pipeline — separate script for building and storing the vector database.

Usage:
    cd /workspace
    python3 -m ingestion.ingest                 # Ingest sample data
    python3 -m ingestion.ingest --dir <path>     # Ingest documents from a directory
    python3 -m ingestion.ingest --verify         # Verify the index was created

This script is SEPARATE from the main application. It:
1. Loads medical knowledge documents (JSON, TXT, PDF supported)
2. Chunks them using LangChain's RecursiveCharacterTextSplitter
3. Embeds using HuggingFace sentence-transformers (local, no API cost)
4. Stores FAISS index to disk at FAISS_INDEX_PATH
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from loguru import logger

from app.config import settings


def get_embeddings() -> HuggingFaceEmbeddings:
    """Create the local HuggingFace embedding model."""
    logger.info(f"[Ingest] Embedding model: {settings.embedding_model_name}")
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_json_documents(json_path: Path) -> list[Document]:
    """Load medical knowledge from a JSON file.

    Expected format: list of objects with at least a 'content' field.
    """
    logger.info(f"[Ingest] Loading JSON data from {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents: list[Document] = []
    for item in data:
        content = item.get("content", "")
        if not content:
            continue

        metadata = {k: v for k, v in item.items() if k != "content"}
        metadata.setdefault("source", "medical_knowledge.json")
        documents.append(Document(page_content=content, metadata=metadata))

    logger.info(f"[Ingest] Loaded {len(documents)} documents from JSON")
    return documents


def load_pdf_documents(pdf_path: Path) -> list[Document]:
    """Load a PDF file using LangChain's PyPDFLoader.

    Handles multi-page PDFs and extracts text page by page.
    """
    logger.info(f"[Ingest] Loading PDF: {pdf_path.name}")
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()
    logger.info(f"[Ingest]   {pdf_path.name} → {len(pages)} pages extracted")
    return pages


def load_text_document(txt_path: Path) -> list[Document]:
    """Load a plain text file."""
    logger.info(f"[Ingest] Loading TXT: {txt_path.name}")
    loader = TextLoader(str(txt_path), encoding="utf-8")
    return loader.load()


def load_directory_documents(dir_path: Path) -> list[Document]:
    """Load all supported documents from a directory.

    Supported formats: .pdf, .txt, .json
    """
    documents: list[Document] = []
    for file_path in sorted(dir_path.iterdir()):
        if file_path.name.startswith("."):
            continue
        suffix = file_path.suffix.lower()
        if suffix == ".json":
            documents.extend(load_json_documents(file_path))
        elif suffix == ".txt":
            documents.extend(load_text_document(file_path))
        elif suffix == ".pdf":
            try:
                documents.extend(load_pdf_documents(file_path))
            except Exception as e:
                logger.error(f"[Ingest] Failed to load PDF {file_path.name}: {e}")
        else:
            logger.debug(f"[Ingest] Skipping unsupported file: {file_path.name}")
    logger.info(f"[Ingest] Loaded {len(documents)} documents from {dir_path}")
    return documents


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split documents into chunks for better retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    logger.info(
        f"[Ingest] Chunked {len(documents)} documents into {len(chunks)} chunks"
    )
    return chunks


def build_and_save_index(documents: list[Document]) -> None:
    """Create FAISS index from documents and save to disk."""
    if not documents:
        logger.error("[Ingest] No documents to index!")
        return

    index_path = Path(settings.faiss_index_path)
    index_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"[Ingest] Building FAISS index with {len(documents)} chunks...")
    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(documents, embeddings)

    vector_store.save_local(str(index_path))
    logger.info(f"[Ingest] ✅ FAISS index saved to {index_path}")

    # Verify files exist
    faiss_file = index_path / "index.faiss"
    pkl_file = index_path / "index.pkl"
    if faiss_file.exists() and pkl_file.exists():
        logger.info(f"[Ingest]   index.faiss: {faiss_file.stat().st_size:,} bytes")
        logger.info(f"[Ingest]   index.pkl:   {pkl_file.stat().st_size:,} bytes")
    else:
        logger.error("[Ingest] ❌ Index files missing after save!")


def verify_index() -> bool:
    """Verify the FAISS index exists and can be loaded."""
    index_path = Path(settings.faiss_index_path)
    faiss_file = index_path / "index.faiss"
    pkl_file = index_path / "index.pkl"

    if not faiss_file.exists() or not pkl_file.exists():
        logger.error(f"[Verify] ❌ Index not found at {index_path}")
        return False

    try:
        embeddings = get_embeddings()
        store = FAISS.load_local(
            folder_path=str(index_path),
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )
        # Test a query
        results = store.similarity_search("headache fever", k=3)
        logger.info(
            f"[Verify] ✅ Index loaded successfully — {len(results)} test results returned"
        )
        for i, doc in enumerate(results, 1):
            condition = doc.metadata.get("condition", "unknown")
            logger.info(f"[Verify]   [{i}] {condition}")
        return True
    except Exception as e:
        logger.error(f"[Verify] ❌ Failed to load index: {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FAISS Ingestion Pipeline for Healthcare CDSS"
    )
    parser.add_argument("--dir", type=str, help="Directory of documents to ingest")
    parser.add_argument(
        "--verify", action="store_true", help="Verify the existing index"
    )
    parser.add_argument(
        "--data", type=str, default=None, help="Specific JSON data file"
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Healthcare CDSS — FAISS Ingestion Pipeline")
    logger.info("=" * 60)
    logger.info(f"[Ingest] FAISS path:  {settings.faiss_index_path}")
    logger.info(f"[Ingest] Embedding:   {settings.embedding_model_name}")
    logger.info("=" * 60)

    if args.verify:
        success = verify_index()
        sys.exit(0 if success else 1)

    # ── Load Documents ──
    if args.dir:
        documents = load_directory_documents(Path(args.dir))
    elif args.data:
        documents = load_json_documents(Path(args.data))
    else:
        # Default: load sample medical knowledge + any user-uploaded PDFs/TXTs
        documents = []
        default_data = Path(__file__).parent / "data" / "medical_knowledge.json"
        if default_data.exists():
            documents.extend(load_json_documents(default_data))
        # Also load any user-uploaded documents
        docs_dir = Path(__file__).parent / "documents"
        if docs_dir.exists():
            dir_docs = load_directory_documents(docs_dir)
            if dir_docs:
                documents.extend(dir_docs)

    if not documents:
        logger.error("[Ingest] No documents found!")
        sys.exit(1)

    # ── Chunk ──
    chunks = chunk_documents(documents)

    # ── Build & Save ──
    build_and_save_index(chunks)

    # ── Verify ──
    logger.info("[Ingest] Running verification...")
    verify_index()

    logger.info("[Ingest] Done!")


if __name__ == "__main__":
    main()
