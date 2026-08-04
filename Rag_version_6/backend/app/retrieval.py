import logging

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from app.config import settings
from app.embeddings import embed_text, EmbeddingModelError
from app.vectorstore import get_index, VectorStoreError
from app.schemas import AnswerMetadata
from app.prompts import SYSTEM_PROMPT, ANSWER_PROMPT_TEMPLATE
from app.bm25_store import keyword_search

logger = logging.getLogger(__name__)

TOP_K = 4  # final chunk count handed to generation — unchanged since V1
RRF_K = 60  # smoothing constant from the original Reciprocal Rank Fusion paper

_reranker = None  # sentence_transformers.CrossEncoder singleton, lazy-loaded (see _get_reranker)

METADATA_SYSTEM_PROMPT = (
    "You will be shown a question, the numbered context chunks that were retrieved for it, "
    "and the answer that was generated. Assess your own answer: how confident are you that "
    "it's well-supported by the given chunks, and which chunk numbers did you actually rely "
    "on to produce it? Only cite chunks that genuinely contributed — do not cite a chunk just "
    "because it was retrieved if you didn't actually use it."
)

_llm: ChatOpenAI | None = None
_metadata_llm = None  # ChatOpenAI wrapped with .with_structured_output(AnswerMetadata)


class RetrievalError(Exception):
    """Raised when a query fails to retrieve context or generate an answer."""


def get_llm() -> ChatOpenAI:
    """
    Returns a singleton ChatOpenAI client, same reuse pattern as the
    embedding model and Pinecone client — created once, not per-request.

    Public (no leading underscore) because graph.py (V2) reuses this
    same singleton instead of creating its own OpenAI client.
    """
    global _llm
    if _llm is None:
        logger.info("Initializing OpenAI client")
        _llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=settings.openai_api_key)
    return _llm


def _get_metadata_llm():
    """
    Returns a singleton ChatOpenAI client wrapped with
    .with_structured_output(AnswerMetadata) — LangChain's mechanism for
    forcing the model's response to conform to a Pydantic schema instead
    of returning free-form text. Calling .invoke() on this returns an
    AnswerMetadata INSTANCE directly (already validated), not a string
    you'd have to parse yourself.

    Kept as a separate singleton from get_llm() because .with_structured_output()
    wraps the base model into a differently-shaped runnable — it's a distinct
    tool with a distinct job (constrained JSON, not free text), not just a
    different prompt on the same client.
    """
    global _metadata_llm
    if _metadata_llm is None:
        _metadata_llm = get_llm().with_structured_output(AnswerMetadata)
    return _metadata_llm


def generate_answer_metadata(question: str, answer: str, chunks: list[dict]) -> AnswerMetadata:
    """
    V3: a SECOND LLM call, separate from the one that generated `answer`,
    asking the model to self-assess its own answer against the chunks it
    was given. Deliberately kept separate from generate_node's main call —
    see rag_v3_plan.md section 3.3 for why structured output and streaming
    don't mix well; this runs as a plain (non-streamed) call after the
    streamed answer text is already complete.

    Short-circuits with a fixed low-confidence/no-citations result when
    there were no chunks at all (the "I couldn't find relevant content"
    case) — no point spending an API call asking the model to assess
    citations for context it was never given.
    """
    if not chunks:
        return AnswerMetadata(confidence="low", cited_chunk_indices=[])

    numbered_chunks = "\n\n".join(
        f"[{i}] {chunk['text']}" for i, chunk in enumerate(chunks)
    )
    user_prompt = (
        f"Question: {question}\n\n"
        f"Retrieved chunks:\n{numbered_chunks}\n\n"
        f"Answer given: {answer}"
    )

    metadata_llm = _get_metadata_llm()
    try:
        return metadata_llm.invoke([
            SystemMessage(content=METADATA_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])
    except Exception as exc:
        # Metadata is a nice-to-have, not core functionality — if this call
        # fails, log it but don't take down the whole /query request over it.
        logger.error("Failed to generate answer metadata: %s", exc)
        return AnswerMetadata(confidence="low", cited_chunk_indices=[])


def _dense_search(question_vector: list[float], namespace: str, top_k: int) -> list[dict]:
    """Pinecone dense similarity search, normalized to the shared candidate shape."""
    try:
        index = get_index()
        results = index.query(
            vector=question_vector,
            top_k=top_k,
            namespace=namespace,
            include_metadata=True,
        )
    except VectorStoreError:
        raise
    except Exception as exc:
        logger.error("Failed to query Pinecone for namespace '%s': %s", namespace, exc)
        raise RetrievalError("Failed to retrieve relevant context") from exc

    matches = results.get("matches", [])
    if not matches:
        logger.warning("No dense matches found for namespace '%s'", namespace)

    return [
        {
            "text": match["metadata"]["text"],
            "source": match["metadata"].get("source", "unknown"),
            "page": match["metadata"].get("page", -1),
            "score": match["score"],
            "metadata": {
                "chunk_index": match["metadata"].get("chunk_index", -1),
                "section_title": match["metadata"].get("section_title", "Unknown section"),
                "document_title": match["metadata"].get("document_title", "Unknown document"),
                "chunk_length": match["metadata"].get("chunk_length", 0),
            },
        }
        for match in matches
    ]


def _reciprocal_rank_fusion(result_lists: list[list[dict]], k: int = RRF_K) -> list[dict]:
    """
    V6 Step 3: merges multiple ranked candidate lists (dense + keyword) into
    a single ranked list via Reciprocal Rank Fusion. Each chunk's fused
    score is the sum of 1/(k + rank) across every list it appears in (rank
    is 0-indexed position within that list).

    Chunks are matched by (source, chunk_index) rather than raw score,
    because dense cosine similarity and BM25 scores live on completely
    different scales and aren't directly comparable — RRF sidesteps that by
    fusing on RANK instead. A chunk that shows up near the top of both lists
    ends up ranked above one that's #1 in only one list.
    """
    fused_scores: dict[tuple, float] = {}
    chunk_lookup: dict[tuple, dict] = {}

    for result_list in result_lists:
        for rank, chunk in enumerate(result_list):
            key = (chunk["source"], chunk["metadata"].get("chunk_index"))
            fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (k + rank)
            chunk_lookup.setdefault(key, chunk)

    ranked_keys = sorted(fused_scores, key=lambda key: fused_scores[key], reverse=True)
    fused = []
    for key in ranked_keys:
        chunk = dict(chunk_lookup[key])
        chunk["score"] = fused_scores[key]  # replace the source-specific score with the fused RRF score
        fused.append(chunk)
    return fused


def _get_reranker():
    """
    Lazy-loaded singleton cross-encoder, same pattern as get_llm()/get_index().
    Uses cross-encoder/ms-marco-MiniLM-L-6-v2 via sentence-transformers
    (already a dependency for the embedding model) — small and fast enough
    to rerank a handful of candidates per request without adding much
    latency, unlike calling out to a hosted reranking API.
    """
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        logger.info("Loading cross-encoder reranker model")
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


def _rerank(question: str, candidates: list[dict], top_n: int) -> list[dict]:
    """
    V6 Step 4: reranking. RRF fusion combines rank signals from dense +
    keyword search, but neither signal actually reads the question and
    chunk together — a cross-encoder does, scoring each (question, chunk)
    pair jointly. That's slower per-pair than embedding similarity, which
    is why it only runs over the already-small fused candidate set, not the
    whole document.
    """
    if not candidates:
        return candidates

    reranker = _get_reranker()
    pairs = [(question, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)

    reranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
    result = []
    for chunk, score in reranked[:top_n]:
        chunk = dict(chunk)
        chunk["score"] = float(score)
        result.append(chunk)
    return result


def retrieve_chunks(question: str, namespace: str) -> list[dict]:
    """
    V6 hybrid retrieval pipeline: dense (Pinecone) + keyword (BM25)
    candidates -> RRF fusion -> cross-encoder reranking -> top TOP_K chunks.

    settings.enable_hybrid_retrieval / settings.enable_reranking (see
    config.py) each independently fall back toward the simpler V1-V5
    behavior — hybrid off means dense-only candidates go straight into
    fusion's place; reranking off means the fused list is just truncated to
    TOP_K by its existing rank. Useful for an apples-to-apples comparison
    against the V5 baseline (rag_v6_plan.md Step 4/6).

    Public because graph.py (V2+) calls this directly from its retrieve node.
    """
    try:
        question_vector = embed_text(question)
    except EmbeddingModelError as exc:
        raise RetrievalError("Failed to embed the question") from exc

    # Widen the candidate pool when either hybrid fusion or reranking will
    # run afterward — both need more than TOP_K raw candidates to be useful.
    # If both are off, there's no point asking Pinecone for more than TOP_K.
    candidate_k = (
        settings.retrieval_candidate_k
        if (settings.enable_hybrid_retrieval or settings.enable_reranking)
        else TOP_K
    )

    dense_chunks = _dense_search(question_vector, namespace, candidate_k)

    if settings.enable_hybrid_retrieval:
        keyword_chunks = keyword_search(namespace, question, candidate_k)
        candidates = _reciprocal_rank_fusion([dense_chunks, keyword_chunks])
    else:
        candidates = dense_chunks

    if settings.enable_reranking and candidates:
        final_chunks = _rerank(question, candidates, TOP_K)
    else:
        final_chunks = candidates[:TOP_K]

    if final_chunks:
        top_metadata = [
            {
                "score": c["score"],
                "section_title": c["metadata"]["section_title"],
                "chunk_index": c["metadata"]["chunk_index"],
            }
            for c in final_chunks[:3]
        ]
        logger.info(
            "Retrieved %d final chunks for namespace '%s' (hybrid=%s, rerank=%s, top: %s)",
            len(final_chunks), namespace, settings.enable_hybrid_retrieval, settings.enable_reranking, top_metadata,
        )
    else:
        logger.warning("No chunks retrieved at all for namespace '%s'", namespace)

    return final_chunks


def answer_question(question: str, namespace: str) -> dict:
    """
    Full retrieval pipeline: embed question -> retrieve top-k chunks -> generate answer.
    This is the V1 stateless path — still used if you ever want a no-memory query.

    Returns: {"answer": str, "sources": list[dict]}
    """
    logger.info("Answering question for namespace '%s'", namespace)

    chunks = retrieve_chunks(question, namespace)

    if not chunks:
        return {
            "answer": "I couldn't find any relevant content in this document to answer that question.",
            "sources": [],
        }

    context = "\n\n".join(chunk["text"] for chunk in chunks)
    user_prompt = ANSWER_PROMPT_TEMPLATE.format(context=context, question=question)

    llm = get_llm()
    try:
        response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])
    except Exception as exc:
        logger.error("OpenAI generation failed: %s", exc)
        raise RetrievalError("Failed to generate an answer") from exc

    return {
        "answer": response.content,
        "sources": chunks,
    }
