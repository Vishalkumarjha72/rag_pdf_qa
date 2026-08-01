import logging

import redis

from app.config import settings

logger = logging.getLogger(__name__)


class RedisConnectionError(Exception):
    """Raised when Redis can't be reached."""


_redis_client: redis.Redis | None = None


def get_redis_client() -> redis.Redis:
    """
    Returns a singleton Redis client. Created once and reused across
    requests instead of reconnecting every time — same pattern as
    get_pinecone_client() in vectorstore.py and get_llm() in retrieval.py.

    decode_responses=True means values come back as str, not bytes —
    simpler to work with since everything we store here (cached JSON,
    checkpointer state) is text-based.
    """
    global _redis_client
    if _redis_client is None:
        logger.info("Initializing Redis client (%s:%s)", settings.redis_host, settings.redis_port)
        _redis_client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            decode_responses=True,
        )
    return _redis_client


def ping_redis() -> None:
    """
    Verifies Redis is actually reachable. Call at app startup, same as
    ensure_index_exists() does for Pinecone — fail loudly at startup
    rather than on the first request that needs it.
    """
    client = get_redis_client()
    try:
        client.ping()
    except redis.exceptions.RedisError as exc:
        logger.error("Could not reach Redis at %s:%s: %s", settings.redis_host, settings.redis_port, exc)
        raise RedisConnectionError(
            f"Could not reach Redis at {settings.redis_host}:{settings.redis_port}"
        ) from exc
