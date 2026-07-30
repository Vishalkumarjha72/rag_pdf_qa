import logging
import os
import tempfile
import uuid

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.embeddings import embed_texts, EmbeddingModelError
from app.vectorstore import get_index, VectorStoreError

logger = logging.getLogger(__name__)

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
UPSERT_BATCH_SIZE = 100  # Pinecone recommends batching upserts rather than one massive call


class IngestionError(Exception):
    """Raised when a PDF fails to load, chunk, embed, or index."""


def _make_namespace(filename: str) -> str:
    """
    Builds a unique namespace per uploaded document, so vectors from different
    PDFs never mix during retrieval. Combines a slugified filename with a short
    random suffix in case the same file is uploaded twice.
    """
    base = os.path.splitext(filename)[0].lower().replace(" ", "-")
    suffix = uuid.uuid4().hex[:8]
    return f"{base}-{suffix}"


def _load_and_split(file_bytes: bytes, filename: str) -> list:
    """
    Writes the uploaded PDF bytes to a temp file (PyPDFLoader needs a real path),
    loads it, and splits it into chunks. Temp file is always cleaned up.
    """
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
        tmp_file.write(file_bytes)
        tmp_path = tmp_file.name

    try:
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
    except Exception as exc:
        logger.error("Failed to load PDF '%s': %s", filename, exc)
        raise IngestionError(f"Could not read PDF '{filename}'. Is it a valid, non-corrupted PDF?") from exc
    finally:
        os.remove(tmp_path)

    if not documents:
        raise IngestionError(f"No extractable text found in '{filename}'. It may be a scanned/image-only PDF.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(documents)

    if not chunks:
        raise IngestionError(f"'{filename}' produced no chunks after splitting.")

    return chunks


def ingest_pdf(file_bytes: bytes, filename: str) -> dict:
    """
    Full ingestion pipeline for one PDF:
      load -> split -> embed -> upsert into Pinecone under a new namespace.

    Returns: {"namespace": str, "chunks_indexed": int}
    """
    logger.info("Starting ingestion for '%s'", filename)

    chunks = _load_and_split(file_bytes, filename)
    namespace = _make_namespace(filename)

    chunk_texts = [chunk.page_content for chunk in chunks]
    try:
        vectors = embed_texts(chunk_texts)
    except EmbeddingModelError as exc:
        raise IngestionError(f"Failed to embed chunks for '{filename}'") from exc

    records = []
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        records.append({
            "id": f"{namespace}-chunk-{i}",
            "values": vector,
            "metadata": {
                "text": chunk.page_content,
                "source": filename,
                "page": chunk.metadata.get("page", -1),
            },
        })

    try:
        index = get_index()
        for batch_start in range(0, len(records), UPSERT_BATCH_SIZE):
            batch = records[batch_start:batch_start + UPSERT_BATCH_SIZE]
            index.upsert(vectors=batch, namespace=namespace)
    except VectorStoreError:
        raise
    except Exception as exc:
        logger.error("Failed to upsert vectors for '%s': %s", filename, exc)
        raise IngestionError(f"Failed to index '{filename}' into the vector store") from exc

    logger.info("Ingestion complete for '%s': %d chunks indexed under namespace '%s'",
                filename, len(records), namespace)

    return {"namespace": namespace, "chunks_indexed": len(records)}
