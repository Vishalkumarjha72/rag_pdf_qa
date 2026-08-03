# V4: load .env into the REAL process environment (os.environ), not just into
# our own pydantic-settings Settings object. This matters specifically for
# LangSmith tracing, which reads LANGSMITH_* vars directly from os.environ
# rather than through any of our own code — pydantic-settings' env_file="..."
# parses .env into ITS OWN model only, it does NOT set them globally. Must
# run before any LangChain/LangGraph imports so tracing is active from the
# very first traced call. override=False (the default) means this never
# clobbers a real env var already set by the shell or Docker's env_file: —
# safe to call unconditionally in both local and containerized runs.
from dotenv import load_dotenv
load_dotenv()

import json
import logging
import uuid

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.vectorstore import ensure_index_exists, VectorStoreError
from app.ingestion import ingest_pdf, IngestionError
from app.retrieval import RetrievalError
from app.graph import astream_answer, get_final_state, get_graph
from app.embeddings import EmbeddingModelError
from app.redis_client import ping_redis, RedisConnectionError
from app.schemas import UploadResponse, QueryRequest

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
async def on_startup():
    """
    Fail loudly at boot rather than on the first request that needs a
    downstream service. V3 adds two checks here:
      - ping_redis(): confirms Redis is reachable
      - get_graph(): builds the LangGraph graph AND, as a side effect,
        calls the checkpointer's asetup() to create Redis's search indices
        AND capture the running event loop (needed by AsyncRedisSaver).
        Doing this at startup (not lazily on the first /query) means a
        broken Redis setup surfaces immediately, not on some user's first
        question — and running it here, inside FastAPI's own startup
        event loop, is what makes the loop-capture work correctly.
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

    await get_graph()
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



@app.post("/query")
async def query(request: QueryRequest):
    """
    V3: streams the answer back as Server-Sent Events instead of one JSON
    blob. Two event shapes, one per line, each prefixed "data: ":
      {"type": "token", "text": "..."}   -- zero or more, as the answer generates
      {"type": "done", "session_id": ..., "sources": [...], "metadata": {...}}
      {"type": "error", "detail": "..."} -- instead of "done", if something failed

    Can't use response_model=QueryResponse anymore, or raise HTTPException
    partway through — once streaming has started, the HTTP response is
    already committed, so errors have to become a "type": "error" event
    inside the stream rather than a normal FastAPI error response.
    """
    session_id = request.session_id or str(uuid.uuid4())

    async def event_generator():
        try:
            got_any_token = False
            async for token in astream_answer(request.question, request.namespace, session_id):
                got_any_token = True
                yield f"data: {json.dumps({'type': 'token', 'text': token})}\n\n"

            final = await get_final_state(session_id)

            if not got_any_token:
                # Cache hit: check_cache_node produced the answer directly,
                # no LLM call to stream from. Send the whole thing as one
                # chunk so the frontend still has SOMETHING to render before
                # the "done" event, instead of an answer that just silently
                # never streamed anything.
                yield f"data: {json.dumps({'type': 'token', 'text': final['answer']})}\n\n"

            yield f"data: {json.dumps({'type': 'done', 'session_id': session_id, 'sources': final['sources'], 'metadata': final['metadata']})}\n\n"

        except RetrievalError as exc:
            logger.error("Retrieval failed for namespace '%s': %s", request.namespace, exc)
            yield f"data: {json.dumps({'type': 'error', 'detail': str(exc)})}\n\n"
        except (VectorStoreError, EmbeddingModelError) as exc:
            logger.error("Upstream service failure during query: %s", exc)
            yield f"data: {json.dumps({'type': 'error', 'detail': 'A downstream service failed during retrieval.'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
