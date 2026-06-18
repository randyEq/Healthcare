# 🏥 Healthcare Clinical Decision Support System (CDSS)

An AI-powered multi-agent system built with **LangGraph** and **LangChain** that assists users with symptom analysis, potential medical conditions, treatment options, medication safety, and recommended next steps.

## ✨ Features

- **🧠 Multi-Agent Architecture** — 4 specialized agents working in sequence via LangGraph
- **🔍 RAG with FAISS** — Retrieves relevant medical knowledge from a vector database
- **💬 Human-in-the-Loop** — Asks follow-up questions when patient info is insufficient
- **💊 Cost-Effective Analysis** — Evaluates treatment options with cost-effectiveness ratings
- **📊 Structured Output** — Patient-friendly responses with diagnosis, severity, urgency, and next steps
- **🧾 Conversation Memory** — Maintains context across multiple turns
- **📄 PDF Ingestion** — Separate pipeline to ingest medical PDFs, TXTs, and JSON into FAISS

## 🏗️ Architecture

```
User Input → Planner Agent → [Insufficient Info?]
                                ├── YES → Follow-up Questions (Human-in-the-Loop)
                                └── NO  → RAG Agent → Cost-Effective Agent → Summary Agent → Response
```

### Agents

| Agent                             | Role                                                                                                            |
| --------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| **Planner Agent**                 | Analyzes user input, checks info completeness (0-100 score), generates follow-up questions if needed            |
| **RAG Agent (Patient/Diagnosis)** | Retrieves medical knowledge from FAISS vector DB, performs clinical reasoning, generates differential diagnoses |
| **Cost-Effective Analysis Agent** | Evaluates treatment cost-effectiveness, medication safety, drug interactions, risk assessment                   |
| **Summary Agent**                 | Consolidates all findings into structured patient-friendly markdown response                                    |

## 🛠️ Tech Stack

| Component      | Technology                                           |
| -------------- | ---------------------------------------------------- |
| **Framework**  | LangGraph, LangChain                                 |
| **LLM**        | Llama-3.1-8B-Instruct via HuggingFace Router         |
| **Embeddings** | sentence-transformers/all-MiniLM-L6-v2 (local, free) |
| **Vector DB**  | FAISS (local persistent storage)                     |
| **Web Server** | FastAPI + Uvicorn                                    |
| **Real-time**  | WebSocket                                            |
| **Frontend**   | HTML, CSS, Vanilla JS                                |

## 📂 Project Structure

```
healthcare-cdss/
├── app/
│   ├── main.py                  # FastAPI app + WebSocket server
│   ├── config.py                # Environment configuration
│   ├── prompts.py               # Agent prompt templates
│   ├── graph/
│   │   ├── workflow.py          # LangGraph StateGraph definition
│   │   ├── nodes.py             # Node wrapper functions
│   │   └── state.py             # TypedDict state schema
│   ├── agents/
│   │   ├── planner.py           # Planner Agent
│   │   ├── rag.py               # RAG Agent (diagnosis)
│   │   ├── analyst.py           # Cost-Effective Analysis Agent
│   │   └── summary.py           # Summary Agent
│   ├── memory/
│   │   └── conversation.py      # Conversation memory
│   └── retrieval/
│       └── retriever.py         # FAISS retriever
├── ingestion/
│   ├── ingest.py                # FAISS ingestion pipeline (PDF/TXT/JSON)
│   ├── data/
│   │   └── medical_knowledge.json  # 20 built-in medical conditions
│   └── documents/               # ← Upload your PDFs here
├── templates/
│   └── index.html               # Chat UI
├── static/
│   ├── style.css                # Styling
│   └── app.js                   # WebSocket chat client + markdown renderer
├── tests/
│   └── test_agents.py           # 11 unit tests
├── .env.example                 # Environment variable template
├── requirements.txt             # Python dependencies
└── run.py                       # Entry point
```

## 🚀 Setup & Installation

### 1. Clone the repo

```bash
git clone https://github.com/gajulasaikumar/healthcare.git
cd healthcare
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env with your actual API keys
```

### 4. Build the FAISS index (ingestion pipeline)

```bash
# Ingest the built-in sample medical knowledge
python -m ingestion.ingest

# OR ingest your own PDFs/TXTs
python -m ingestion.ingest --dir /path/to/your/documents

# Verify the index
python -m ingestion.ingest --verify
```

### 5. Run the server

```bash
python run.py
```

The app will be available at `http://localhost:8000`

## 📄 Adding Your Own Medical PDFs

1. Place PDF files in the `ingestion/documents/` folder
2. Run the ingestion pipeline:
   ```bash
   python -m ingestion.ingest
   ```
3. The FAISS index rebuilds automatically — the RAG agent picks up new knowledge on the next query

**Supported formats:** PDF (`.pdf`), Text (`.txt`), JSON (`.json`)

## 🧪 Running Tests

```bash
python -m pytest tests/test_agents.py -v
```

## 🔧 Configuration

All configuration is via environment variables (`.env` file):

| Variable               | Description         | Default                                   |
| ---------------------- | ------------------- | ----------------------------------------- |
| `OPENAI_API_KEY`       | HuggingFace API key | —                                         |
| `OPENAI_BASE_URL`      | LLM endpoint URL    | `https://router.huggingface.co/v1`        |
| `LLM_MODEL_NAME`       | LLM model name      | `meta-llama/Llama-3.1-8B-Instruct:novita` |
| `EMBEDDING_MODEL_NAME` | Embedding model     | `sentence-transformers/all-MiniLM-L6-v2`  |
| `FAISS_INDEX_PATH`     | Path to FAISS index | `/workspace/vector_db`                    |
| `APP_HOST`             | Server host         | `0.0.0.0`                                 |
| `APP_PORT`             | Server port         | `8000`                                    |
| `LANGSMITH_TRACING`    | Enable LangSmith monitoring | `false`                         |
| `LANGSMITH_API_KEY`    | LangSmith API key   | -                                         |
| `LANGSMITH_PROJECT`    | LangSmith project name | `healthcare-cdss`                      |
| `LANGSMITH_ENDPOINT`   | LangSmith API endpoint | `https://api.smith.langchain.com`      |

### LangSmith Monitoring

To trace LangGraph/LangChain runs in LangSmith, add these values to `.env`:

```env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=healthcare-cdss
```

Restart the server after changing `.env`. Startup logs will show whether
LangSmith is enabled and which project is being used.

## ⚕️ Disclaimer

This AI-generated information is for **educational purposes only** and does **NOT** replace professional medical advice. Please consult a qualified healthcare provider for medical concerns.
