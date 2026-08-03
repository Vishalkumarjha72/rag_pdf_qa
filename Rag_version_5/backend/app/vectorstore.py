import logging

from pinecone import Pinecone, ServerlessSpec
from pinecone.exceptions import PineconeApiException

from app.config import settings

logger = logging.getLogger(__name__)


class VectorStoreError(Exception):
    """Raised when the Pinecone vector store can't be reached or set up."""


_pinecone_client: Pinecone | None = None


def get_pinecone_client() -> Pinecone:
    """
    Returns a singleton Pinecone client. Created once and reused across
    requests instead of reconnecting every time.
    """
    global _pinecone_client
    if _pinecone_client is None:
        logger.info("Initializing Pinecone client")
        _pinecone_client = Pinecone(api_key=settings.pinecone_api_key)
    return _pinecone_client


def ensure_index_exists() -> None:
    """
    Creates the Pinecone index if it doesn't already exist.
    Safe to call on every app startup — it's a no-op if the index is already there.
    """
    pc = get_pinecone_client()

    try:
        existing_indexes = [index["name"] for index in pc.list_indexes()]
    except PineconeApiException as exc:
        logger.error("Failed to list Pinecone indexes: %s", exc)
        raise VectorStoreError("Could not reach Pinecone to check existing indexes") from exc

    if settings.pinecone_index_name in existing_indexes:
        logger.info("Pinecone index '%s' already exists", settings.pinecone_index_name)
        return

    logger.info("Creating Pinecone index '%s'", settings.pinecone_index_name)
    try:
        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=settings.embedding_dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=settings.pinecone_cloud,
                region=settings.pinecone_environment,
            ),
        )
    except PineconeApiException as exc:
        logger.error("Failed to create Pinecone index: %s", exc)
        raise VectorStoreError(f"Could not create index '{settings.pinecone_index_name}'") from exc


def get_index():
    """
    Returns a handle to the configured Pinecone index.
    Call ensure_index_exists() at app startup before using this.
    """
    pc = get_pinecone_client()
    return pc.Index(settings.pinecone_index_name)
