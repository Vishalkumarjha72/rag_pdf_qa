import logging

from sentence_transformers import SentenceTransformer

from app.config import settings

logger = logging.getLogger(__name__)


class EmbeddingModelError(Exception):
    """Raised when the embedding model fails to load or encode text."""


_embedding_model: SentenceTransformer | None = None


#same singleton pattern as vectorstore.py's Pinecone client. Loading bge-base-en-v1.5 is relatively expensive (downloads/loads weights into memory), so it happens once and is reused — never per-request.
def get_embedding_model() -> SentenceTransformer:
    """
    Returns a singleton embedding model instance.
    Loaded once at first use (or app startup) and reused across requests —
    loading this model is expensive, so we never want to do it per-request.
    """
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading embedding model '%s'", settings.embedding_model_name)
        try:
            _embedding_model = SentenceTransformer(settings.embedding_model_name)
        except Exception as exc:
            # Broad except here is deliberate: SentenceTransformer/huggingface_hub can raise
            # several different exception types under the hood (network errors, OSError for
            # missing/corrupt files, HFValidationError for a bad model name, etc.). We don't
            # need to handle each differently — any failure here means the app can't start,
            # so we wrap it in one clear, actionable error instead of leaking a random traceback.
            logger.error("Failed to load embedding model '%s': %s", settings.embedding_model_name, exc)
            raise EmbeddingModelError(
                f"Could not load embedding model '{settings.embedding_model_name}'. "
                "Check your internet connection and that the model name is correct."
            ) from exc
        logger.info("Embedding model loaded")
    return _embedding_model



#batch embedding, used during ingestion when you have many chunks at once. Returns plain Python lists (not numpy arrays) since that's what Pinecone's upsert expects.
def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embeds a batch of texts (e.g. document chunks during ingestion).
    Returns a list of vectors, one per input text, in the same order.
    """
    if not texts:
        return []

    model = get_embedding_model()
    try:
        vectors = model.encode(texts, show_progress_bar=False)
    except Exception as exc:
        logger.error("Failed to embed %d texts: %s", len(texts), exc)
        raise EmbeddingModelError("Failed to embed input texts") from exc
    return vectors.tolist()



#single-text embedding, used at query time to embed the user's question.
def embed_text(text: str) -> list[float]:
    """
    Embeds a single text (e.g. a user's question at query time).
    """
    model = get_embedding_model()
    try:
        vector = model.encode(text)
    except Exception as exc:
        logger.error("Failed to embed text: %s", exc)
        raise EmbeddingModelError("Failed to embed input text") from exc
    return vector.tolist()
