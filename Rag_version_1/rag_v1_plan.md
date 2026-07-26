# RAG PDF Q&A System — Version 1 Plan

## 1. Goal

Get an end-to-end working pipeline: upload a PDF → ask a question → get an answer.
No memory, no caching, no evaluation, no Kubernetes, no AWS yet. Just prove the core RAG loop works, locally, in Docker.

Since the underlying goal of this project is to **understand how RAG systems are built at a production level** (not just a demo script), V1 also bakes in a few production habits early — not full production hardening (that's V7), but the *coding discipline* that makes later versions easier to layer on:
- Typed config (pydantic-settings) instead of loose `os.getenv()` calls
- Typed request/response schemas on every FastAPI endpoint (not just `dict`)
- Structured logging instead of `print()`
- Basic input validation (file type/size) and explicit error handling with proper HTTP status codes
- Clear separation of concerns (ingestion / retrieval / vectorstore / embeddings as separate modules, as already planned) — this is what makes it possible to swap pieces later (e.g. Pinecone → Qdrant, or add caching) without a rewrite

**Definition of Done for V1:**
- User can upload a PDF via the React UI
- Backend chunks it, embeds it, stores vectors in Pinecone
- User can ask a question via the UI
- Backend retrieves relevant chunks and returns an OpenAI-generated answer
- Whole thing runs via `docker-compose up` locally
- Each pipeline stage (ingestion, retrieval) has been validated independently in the trial notebook before being wired into the FastAPI app

---

## 2. Architecture (V1)

```
┌─────────────┐      HTTP       ┌──────────────┐
│  React UI   │ ───────────────▶│   FastAPI    │
│ (Vite app)  │◀─────────────── │   Backend    │
└─────────────┘                 └──────┬───────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
            ┌───────────────┐  ┌───────────────┐  ┌──────────────┐
            │  PDF Loader    │  │  Embedding    │  │   OpenAI     │
            │  + Chunker     │  │  Model (HF)   │  │   (gen)      │
            │ (LangChain)    │  │ bge-base-en   │  │  gpt-4o-mini │
            └───────┬────────┘  └───────┬───────┘  └──────────────┘
                    │                   │
                    ▼                   ▼
                  ┌───────────────────────┐
                  │   Pinecone Vector DB   │
                  └───────────────────────┘
```

No Redis, no LangGraph, no LangSmith, no memory in this version — pure linear pipeline.

---

## 3. Tech stack (V1 scope only)

| Layer | Tool |
|---|---|
| API | FastAPI |
| PDF parsing | LangChain `PyPDFLoader` |
| Chunking | LangChain `RecursiveCharacterTextSplitter` |
| Embeddings | `BAAI/bge-base-en-v1.5` via `sentence-transformers` |
| Vector DB | Pinecone |
| LLM | OpenAI `gpt-4o-mini` |
| Orchestration | LangChain `RetrievalQA` chain (no LangGraph yet) |
| Frontend | React (Vite) |
| Containerization | Docker + docker-compose |

---

## 4. Project structure

```
rag-app/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app, routes
│   │   ├── config.py            # env vars, settings
│   │   ├── ingestion.py         # PDF load + chunk + embed + upsert
│   │   ├── retrieval.py         # query -> retrieve -> generate
│   │   ├── embeddings.py        # embedding model wrapper
│   │   └── vectorstore.py       # Pinecone client setup
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/
│   │   │   ├── UploadPanel.jsx
│   │   │   └── QueryPanel.jsx
│   │   └── api.js               # axios calls to backend
│   ├── package.json
│   └── Dockerfile
├── notebooks/
│   └── rag_v1_trial.ipynb       # scratchpad — validate the pipeline before it becomes app code
├── docker-compose.yml
├── .env.example
├── .gitignore
└── README.md
```

Since this is going up on GitHub, two things worth having from day one:
- **`.gitignore`** — exclude `.env` (never commit real API keys), `__pycache__/`, `node_modules/`, `*.pyc`, and any local sample PDFs you test with if they're large or not meant to be public
- **`README.md`** — a short "what this is / how to run it" doc; even a stub now saves you from writing it retroactively at V7. Worth noting in the README that this is a learning project tracking staged versions (V1 → V7) so anyone browsing the repo/commit history understands the structure

---

## 5. Backend build steps

### Step 1 — Project setup
- Create virtualenv, install: `fastapi`, `uvicorn`, `langchain`, `langchain-community`, `langchain-openai`, `pinecone-client`, `sentence-transformers`, `pypdf`, `python-multipart`, `python-dotenv`
- Set up `.env` with: `OPENAI_API_KEY`, `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`, `PINECONE_ENVIRONMENT`

### Step 2 — Pinecone setup (`vectorstore.py`)
- Create Pinecone index (dimension must match embedding model output — `bge-base-en-v1.5` = 768 dims)
- One index for now, use `namespace` per uploaded document to keep documents separable

### Step 3 — Embedding wrapper (`embeddings.py`)
- Wrap `sentence-transformers` `bge-base-en-v1.5` behind a simple `embed(texts: list[str]) -> list[list[float]]` function
- Load model once at startup (not per-request) — keep it in memory

### Step 4 — Ingestion pipeline (`ingestion.py`)
1. Accept uploaded PDF file (bytes)
2. Save temporarily / read in-memory
3. Load with `PyPDFLoader`
4. Split with `RecursiveCharacterTextSplitter` (chunk_size=900, chunk_overlap=150)
5. Embed all chunks
6. Upsert to Pinecone with metadata: `{ text: chunk_text, source: filename, page: page_num }`
7. Return doc_id / namespace to caller

### Step 5 — Retrieval + generation (`retrieval.py`)
1. Accept a question + namespace
2. Embed the question
3. Query Pinecone top-k (start with k=4)
4. Build a prompt: system instruction + retrieved chunks + question
5. Call OpenAI `gpt-4o-mini` via `langchain-openai` `ChatOpenAI`
6. Return answer + (optionally) source chunks used

### Step 6 — FastAPI routes (`main.py`)
- `POST /upload` → multipart file upload → calls ingestion.py → returns `{ namespace, chunks_indexed }`
- `POST /query` → `{ namespace, question }` → calls retrieval.py → returns `{ answer, sources }`
- `GET /health` → simple health check

### Step 7 — Dockerize backend
- `Dockerfile`: python:3.11-slim base, install requirements, copy app, run via `uvicorn app.main:app --host 0.0.0.0 --port 8000`

---

## 6. Frontend build steps

### Step 1 — Scaffold
- `npm create vite@latest frontend -- --template react`
- Install `axios`

### Step 2 — `UploadPanel.jsx`
- File input + "Upload" button
- On submit → `POST /upload` (multipart/form-data)
- Show status: "Uploading...", "Indexed ✅ (namespace: xyz)"
- Store returned `namespace` in local state (needed for querying)

### Step 3 — `QueryPanel.jsx`
- Text input + "Ask" button (disabled until a doc is uploaded)
- On submit → `POST /query` with `{ namespace, question }`
- Display returned answer in a simple text block

### Step 4 — `api.js`
- Centralize axios calls to backend base URL (from env var, e.g. `VITE_API_URL`)

### Step 5 — Dockerize frontend
- Multi-stage Dockerfile: build with `npm run build`, serve via `nginx` or simple static server

---

## 7. docker-compose.yml (local run)

- `backend` service (port 8000) — env vars from `.env`
- `frontend` service (port 5173 or 3000) — depends_on backend

No Redis service yet — that's V3.

---

## 8. Testing plan for V1

- Manual: upload a sample PDF, ask 3-5 questions of varying specificity, sanity-check answers
- Edge cases to check: very short PDF, PDF with no extractable text (scanned image) — should fail gracefully with a clear error, not crash
- Check Pinecone namespace isolation — upload two different PDFs, confirm queries don't leak across documents

---

## 9. Explicitly out of scope for V1 (deferred)

- Memory / conversation history → **V2**
- Redis caching → **V3**
- LangSmith evaluation/tracing → **V4**
- Prompt engineering / guardrails / citations → **V5**
- Hybrid retrieval, reranking, smarter chunking → **V6**
- Kubernetes, AWS deployment, autoscaling → **V7**

---

## 10. Trial notebook (before writing app code)

Before arranging anything into the `backend/app/` module structure, validate each piece of the pipeline in isolation in `notebooks/rag_v1_trial.ipynb`. This is standard practice even in production teams — prototype the logic in a notebook first, confirm it works and produces sane output, *then* refactor into clean modules/functions/classes for the actual service.

The trial notebook covers, as separate testable cells:
1. Install + import check
2. Load a sample PDF with `PyPDFLoader`, inspect raw extracted text
3. Chunk it with `RecursiveCharacterTextSplitter`, inspect chunk count/sizes
4. Load the `bge-base-en-v1.5` embedding model, embed a sample chunk, check vector shape
5. Connect to Pinecone, create/connect to an index, upsert the embedded chunks
6. Take a test question, embed it, query Pinecone, inspect retrieved chunks
7. Build the final prompt and call OpenAI, inspect the generated answer

Once every cell runs cleanly end-to-end, that logic gets lifted into `ingestion.py` and `retrieval.py` as proper functions — the notebook is a scratchpad, not the final code.

## 11. Suggested build order (within V1)

1. Backend ingestion pipeline (test via `/docs` Swagger UI, no frontend needed yet)
2. Backend retrieval pipeline (test via Swagger UI)
3. Dockerize backend, confirm it works in a container
4. Build React frontend, wire to backend
5. Dockerize frontend
6. docker-compose end-to-end test
