import logging
import uuid

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.vectorstore import ensure_index_exists, VectorStoreError
from app.ingestion import ingest_pdf, IngestionError
from app.retrieval import RetrievalError
from app.graph import ask_with_memory, get_graph
from app.embeddings import EmbeddingModelError
from app.redis_client import ping_redis, RedisConnectionError
from app.schemas import UploadResponse, QueryRequest, QueryResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB — generous for now, revisit if needed

app = FastAPI(title="RAG PDF Q&A API", version="2.0.0")

# Allow the React dev server to call this API. Tighten this list before any real deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    """
    Fail loudly at boot rather than on the first request that needs a
    downstream service. V3 adds two checks here:
      - ping_redis(): confirms Redis is reachable
      - get_graph(): builds the LangGraph graph AND, as a side effect,
        calls RedisSaver.setup() to create Redis's search indices. Doing
        this at startup (not lazily on the first /query) means a broken
        Redis setup surfaces immediately, not on some user's first question.
    """
    logger.info("Running startup checks")
    try:
        ensure_index_exists()
    except VectorStoreError as exc:
        logger.error("Startup failed: %s", exc)
        raise

    try:
        ping_redis()
    except RedisConnectionError as exc:
        logger.error("Startup failed: %s", exc)
        raise

    get_graph()
    logger.info("Startup checks passed")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_bytes = await file.read()

    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Max size is {MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB.",
        )

    try:
        result = ingest_pdf(file_bytes, filename=file.filename)
    except IngestionError as exc:
        logger.error("Ingestion failed for '%s': %s", file.filename, exc)
        raise HTTPException(status_code=422, detail=str(exc))
    except (VectorStoreError, EmbeddingModelError) as exc:
        logger.error("Upstream service failure during ingestion: %s", exc)
        raise HTTPException(status_code=502, detail="A downstream service failed during ingestion.")

    return UploadResponse(
        namespace=result["namespace"],
        chunks_indexed=result["chunks_indexed"],
        filename=file.filename,
    )



@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    # V2: every conversation is identified by a session_id. If the client
    # didn't send one (first question of a new conversation), generate one
    # here and return it — the client is expected to send it back on every
    # follow-up question in the same conversation.
    session_id = request.session_id or str(uuid.uuid4())

    try:
        result = ask_with_memory(request.question, request.namespace, session_id)
    except RetrievalError as exc:
        logger.error("Retrieval failed for namespace '%s': %s", request.namespace, exc)
        raise HTTPException(status_code=422, detail=str(exc))
    except (VectorStoreError, EmbeddingModelError) as exc:
        logger.error("Upstream service failure during query: %s", exc)
        raise HTTPException(status_code=502, detail="A downstream service failed during retrieval.")

    return QueryResponse(answer=result["answer"], sources=result["sources"], session_id=session_id)
