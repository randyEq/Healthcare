# Infrastructure

## Environment Variables (app/.env)
- AZURE_OPENAI_API_KEY — LLM API key
- AZURE_OPENAI_ENDPOINT — LLM endpoint URL
- AZURE_OPENAI_API_VERSION — API version
- AZURE_OPENAI_DEPLOYMENT_LLM — LLM deployment name
- AZURE_OPENAI_DEPLOYMENT_EMBEDDING — Embedding deployment name
- FAISS_INDEX_PATH — Path to FAISS index directory
- DATABASE_URL — MySQL connection

## Background Services
- cdss-api: uvicorn app.main:app --host 0.0.0.0 --port 8000

## Caddy Proxy
- Reverse proxy at / → :8000

## Ports
- 8000 (Uvicorn)
