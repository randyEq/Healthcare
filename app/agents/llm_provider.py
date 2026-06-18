"""Simple factory to provide a chat LLM compatible with LangChain chains."""

from __future__ import annotations

from typing import Any

from app.config import settings


def get_chat_llm(**kwargs: Any):
    """Return a chat model instance compatible with `prompt | llm` chains.

    kwargs are passed through to the underlying model constructor (e.g.
    `temperature`).
    """
    # Prefer OpenAI-compatible client when key starts with sk-
    api_key = (settings.openai_api_key or "").strip()
    if api_key.startswith("sk-"):
        from langchain_openai import ChatOpenAI

        model_kwargs = {
            "model": settings.llm_model_name,
            "api_key": api_key,
            **kwargs,
        }
        if settings.openai_base_url:
            model_kwargs["base_url"] = settings.openai_base_url

        return ChatOpenAI(
            **model_kwargs,
        )

    # Fallback: use HuggingFaceHub if we have an hf_ token
    if api_key.startswith("hf_"):
        # Try community import first (newer langchain split), then fall back
        tried = []
        for module_path in ("langchain_community.chat_models", "langchain.chat_models"):
            try:
                mod = __import__(module_path, fromlist=["HuggingFaceHub"])  # type: ignore
                HuggingFaceHub = getattr(mod, "HuggingFaceHub")
                return HuggingFaceHub(
                    repo_id=settings.llm_model_name,
                    huggingfacehub_api_token=api_key,
                    **kwargs,
                )
            except Exception as exc:
                tried.append((module_path, str(exc)))
        # If we get here, none of the HuggingFace imports worked — fall back
        # to the OpenAI-compatible client to avoid crashing at startup. This
        # will likely produce an authentication error at runtime if the key
        # is not OpenAI-style, but lets the app run and handle the error
        # gracefully at request time.
        try:
            from langchain_openai import ChatOpenAI

            model_kwargs = {
                "model": settings.llm_model_name,
                "api_key": api_key,
                **kwargs,
            }
            if settings.openai_base_url:
                model_kwargs["base_url"] = settings.openai_base_url

            return ChatOpenAI(**model_kwargs)
        except Exception:
            raise ImportError(
                f"Could not import HuggingFaceHub and failed to fall back to ChatOpenAI. Tried: {tried}"
            )

    # No valid key provided — raise so the application can surface a clear error
    raise RuntimeError(
        "OPENAI_API_KEY is missing or invalid. Set OPENAI_API_KEY to an OpenAI `sk-` key."
    )
