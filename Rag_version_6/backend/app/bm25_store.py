"""
Keyword-search corpus store, backing the BM25 side of V6's hybrid retrieval
(see rag_v6_plan.md section 3.1). Pinecone only supports dense vector
search, so keyword/BM25 scoring needs the raw chunk texts available
somewhere outside the vector index — this module persists each document's
full chunk corpus to Redis at ingestion time and rebuilds an in-memory
BM25Okapi index from it on demand at query time.

Keyed by namespace so multiple uploaded PDFs don't share a corpus.
"""

import json
import logging

from rank_bm25 import BM25Okapi

from app.redis_client import get_redis_client

logger = logging.getLogger(__name__)

CORPUS_KEY_PREFIX = "bm25_corpus"

# In-process cache of built BM25Okapi indexes, keyed by namespace. Rebuilding
# an index from the stored corpus on every single query would re-tokenize
# every chunk in the document each time — cheap for small PDFs, but there's
# no reason to redo it when the corpus for a namespace never changes after
# ingestion (namespaces are effectively immutable once created — see
# ingestion.py's _make_namespace). Cleared only by process restart.
_bm25_index_cache: dict[str, tuple[BM25Okapi, list[dict]]] = {}


def _tokenize(text: str) -> list[str]:
    """Simple lowercase whitespace tokenizer — BM25 doesn't need anything fancier."""
    return text.lower().split()


def store_corpus(namespace: str, chunks: list) -> None:
    """
    Persists every chunk's text + metadata for this namespace to Redis, so
    keyword_search() can rebuild a BM25 index later without re-reading the
    original PDF. Called once at the end of ingest_pdf(), right alongside
    the Pinecone upsert — same source chunks, two indexes.
    """
    client = get_redis_client()
    corpus = [
        {
            "chunk_index": chunk.metadata.get("chunk_index"),
            "text": chunk.page_content,
            "source": namespace,
            "page": chunk.metadata.get("page", -1),
            "section_title": chunk.metadata.get("section_title", "Unknown section"),
            "document_title": chunk.metadata.get("document_title", namespace),
            "chunk_length": chunk.metadata.get("chunk_length", len(chunk.page_content.split())),
        }
        for chunk in chunks
    ]
    key = f"{CORPUS_KEY_PREFIX}:{namespace}"
    client.set(key, json.dumps(corpus))
    logger.info("Stored BM25 corpus for namespace '%s' (%d chunks)", namespace, len(corpus))


def _get_index(namespace: str):
    """Returns (BM25Okapi, corpus) for a namespace, building + caching it on first use."""
    if namespace in _bm25_index_cache:
        return _bm25_index_cache[namespace]

    client = get_redis_client()
    key = f"{CORPUS_KEY_PREFIX}:{namespace}"
    raw = client.get(key)
    if raw is None:
        logger.warning("No BM25 corpus found for namespace '%s' (keyword search skipped)", namespace)
        return None

    corpus = json.loads(raw)
    tokenized = [_tokenize(entry["text"]) for entry in corpus]
    index = BM25Okapi(tokenized)

    _bm25_index_cache[namespace] = (index, corpus)
    return index, corpus


def keyword_search(namespace: str, query: str, top_k: int) -> list[dict]:
    """
    Returns the top_k chunks by BM25 keyword-overlap score for this
    namespace, shaped identically to retrieval.py's dense results
    (text/source/page/score/metadata) so the two candidate lists can be
    merged interchangeably by reciprocal_rank_fusion() in retrieval.py.

    Returns [] if this namespace has no stored BM25 corpus yet (e.g. a
    document ingested before V6 added this) — callers should treat that
    the same as "keyword search found nothing" and fall back to dense-only.
    """
    cached = _get_index(namespace)
    if cached is None:
        return []

    index, corpus = cached
    scores = index.get_scores(_tokenize(query))

    ranked = sorted(range(len(corpus)), key=lambda i: scores[i], reverse=True)[:top_k]

    return [
        {
            "text": corpus[i]["text"],
            "source": corpus[i]["source"],
            "page": corpus[i]["page"],
            "score": float(scores[i]),
            "metadata": {
                "chunk_index": corpus[i]["chunk_index"],
                "section_title": corpus[i]["section_title"],
                "document_title": corpus[i]["document_title"],
                "chunk_length": corpus[i]["chunk_length"],
            },
        }
        for i in ranked
        if scores[i] > 0  # a score of 0 means zero keyword overlap — not a real match, don't fake a rank for it
    ]
