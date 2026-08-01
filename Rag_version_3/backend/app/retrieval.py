import logging

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from app.config import settings
from app.embeddings import embed_text, EmbeddingModelError
from app.vectorstore import get_index, VectorStoreError
from app.schemas import AnswerMetadata

logger = logging.getLogger(__name__)

TOP_K = 4

SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions based only on the provided context "
    "from a document. If the answer isn't in the context, say you don't know — do not "
    "make up information."
)

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


def retrieve_chunks(question: str, namespace: str) -> list[dict]:
    """
    Embeds the question and retrieves the top-k most relevant chunks
    from Pinecone for the given namespace.

    Public because graph.py (V2) calls this directly from its retrieve
    node instead of duplicating the Pinecone query logic.
    """
    try:
        question_vector = embed_text(question)
    except EmbeddingModelError as exc:
        raise RetrievalError("Failed to embed the question") from exc

    try:
        index = get_index()
        results = index.query(
            vector=question_vector,
            top_k=TOP_K,
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
        logger.warning("No matches found for namespace '%s'", namespace)

    return [
        {
            "text": match["metadata"]["text"],
            "source": match["metadata"].get("source", "unknown"),
            "page": match["metadata"].get("page", -1),
            "score": match["score"],
        }
        for match in matches
    ]


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
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

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
