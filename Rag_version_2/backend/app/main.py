import logging

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.vectorstore import ensure_index_exists, VectorStoreError
from app.ingestion import ingest_pdf, IngestionError
from app.retrieval import answer_question, RetrievalError
from app.embeddings import EmbeddingModelError
from app.schemas import UploadResponse, QueryRequest, QueryResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB — generous for now, revisit if needed

app = FastAPI(title="RAG PDF Q&A API", version="1.0.0")

# Allow the React dev server to call this API. Tighten this list before any real deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    """Make sure the Pinecone index exists before the app starts serving requests."""
    logger.info("Running startup checks")
    try:
        ensure_index_exists()
    except VectorStoreError as exc:
        logger.error("Startup failed: %s", exc)
        raise


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
    try:
        result = answer_question(request.question, request.namespace)
    except RetrievalError as exc:
        logger.error("Retrieval failed for namespace '%s': %s", request.namespace, exc)
        raise HTTPException(status_code=422, detail=str(exc))
    except (VectorStoreError, EmbeddingModelError) as exc:
        logger.error("Upstream service failure during query: %s", exc)
        raise HTTPException(status_code=502, detail="A downstream service failed during retrieval.")

    return QueryResponse(answer=result["answer"], sources=result["sources"])
