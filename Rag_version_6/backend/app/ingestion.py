import logging
import os
import tempfile
import uuid

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


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


def _infer_section_title(text: str) -> str | None:
    """Infer a short heading-like title from the first lines of a chunk."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None

    for line in lines[:4]:
        if len(line.split()) > 12:
            continue
        if line.endswith((".", "!", "?", ":")):
            continue
        if line.lower().startswith(("this", "the", "a", "an")) and len(line.split()) <= 4:
            continue
        return line

    return None


def _build_chunk_metadata(chunks: list, filename: str) -> list:
    """Enrich chunks with retrieval-friendly metadata: section title, chunk index, and doc title."""
    document_title = os.path.splitext(filename)[0].replace("-", " ").replace("_", " ").strip()
    enriched_chunks = []

    for index, chunk in enumerate(chunks):
        metadata = dict(chunk.metadata or {})
        metadata.update({
            "chunk_index": index,
            "chunk_length": len(chunk.page_content.split()),
            "document_title": document_title or filename,
        })

        section_title = _infer_section_title(chunk.page_content)
        if section_title:
            metadata["section_title"] = section_title
        else:
            metadata["section_title"] = f"Section {index + 1}"

        enriched_chunks.append(type(chunk)(page_content=chunk.page_content, metadata=metadata))

    return enriched_chunks


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

    return _build_chunk_metadata(chunks, filename)


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
        from app.embeddings import embed_texts, EmbeddingModelError
    except ImportError as exc:
        raise IngestionError(f"Embedding dependencies are not available for '{filename}'") from exc

    try:
        vectors = embed_texts(chunk_texts)
    except EmbeddingModelError as exc:
        raise IngestionError(f"Failed to embed chunks for '{filename}'") from exc

    records = []
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        metadata = dict(chunk.metadata or {})
        records.append({
            "id": f"{namespace}-chunk-{i}",
            "values": vector,
            "metadata": {
                "text": chunk.page_content,
                "source": filename,
                "page": metadata.get("page", -1),
                "chunk_index": metadata.get("chunk_index", i),
                "section_title": metadata.get("section_title", f"Section {i + 1}"),
                "document_title": metadata.get("document_title", os.path.splitext(filename)[0]),
                "chunk_length": metadata.get("chunk_length", len(chunk.page_content.split())),
            },
        })

    try:
        from app.vectorstore import get_index, VectorStoreError
    except ImportError as exc:
        raise IngestionError(f"Vector store dependencies are not available for '{filename}'") from exc

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

    # V6: also persist the raw chunk corpus for BM25 keyword search (see
    # app/bm25_store.py). Pinecone only gives us dense vector search, so the
    # hybrid retrieval pipeline in retrieval.py needs this second, separate
    # index to combine with. Deliberately not fatal if it fails — falling
    # back to dense-only retrieval is better than failing the whole upload.
    try:
        from app.bm25_store import store_corpus
        store_corpus(namespace, chunks)
    except Exception as exc:
        logger.error("Failed to store BM25 corpus for '%s': %s", filename, exc)

    logger.info("Ingestion complete for '%s': %d chunks indexed under namespace '%s'",
                filename, len(records), namespace)

    return {"namespace": namespace, "chunks_indexed": len(records)}
