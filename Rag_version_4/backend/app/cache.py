"""
Query result cache — separate from RedisSaver's conversation checkpointing
(see graph.py). This caches the final {answer, sources} for a specific
question against a specific document, so identical questions don't re-hit
OpenAI/Pinecone.

Keyed by (namespace, standalone_question), NOT (session_id, question) —
deliberate: two different conversations asking the same underlying question
about the same document should share a cache entry, and two differently-
worded versions of the same question (different raw `question`, same
condensed `standalone_question`) should too. This is why caching happens
after the condense step, not before it.
"""

import hashlib
import json
import logging

from app.redis_client import get_redis_client

logger = logging.getLogger(__name__)

CACHE_KEY_PREFIX = "query_cache"
DEFAULT_TTL_SECONDS = 60 * 60  # 1 hour


def _build_cache_key(namespace: str, standalone_question: str) -> str:
    """
    Builds a stable cache key from namespace + standalone_question.
    Normalizes case/whitespace first so trivial differences (extra spaces,
    capitalization) don't create separate cache entries for what's really
    the same question.

    Hashed (not stored raw) because standalone_question could theoretically
    contain characters that aren't safe/clean in a Redis key, and a fixed-
    length key is simpler to reason about regardless of question length.
    """
    normalized = f"{namespace}:{standalone_question.strip().lower()}"
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"{CACHE_KEY_PREFIX}:{digest}"


def get_cached_answer(namespace: str, standalone_question: str) -> dict | None:
    """
    Returns the cached {"answer": str, "sources": list[dict]} for this
    namespace + standalone_question, or None on a cache miss.
    """
    client = get_redis_client()
    key = _build_cache_key(namespace, standalone_question)

    cached_value = client.get(key)
    if cached_value is None:
        logger.info("Cache MISS for key %s", key)
        return None

    logger.info("Cache HIT for key %s", key)
    return json.loads(cached_value)


def set_cached_answer(
    namespace: str,
    standalone_question: str,
    result: dict,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
) -> None:
    """
    Stores {"answer": str, "sources": list[dict]} under a TTL, so stale
    entries expire automatically instead of accumulating forever.
    """
    client = get_redis_client()
    key = _build_cache_key(namespace, standalone_question)
    client.setex(key, ttl_seconds, json.dumps(result))
    logger.info("Cached answer under key %s (TTL %ds)", key, ttl_seconds)
