import logging

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from app.config import settings
from app.embeddings import embed_text, EmbeddingModelError
from app.vectorstore import get_index, VectorStoreError

logger = logging.getLogger(__name__)

TOP_K = 4

SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions based only on the provided context "
    "from a document. If the answer isn't in the context, say you don't know — do not "
    "make up information."
)

_llm: ChatOpenAI | None = None


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
