"""
V3 conversation graph: condense -> check_cache -> [retrieve -> generate] -> END.

V2 flow was condense -> retrieve -> generate. V3 adds:
  - RedisSaver checkpointer (persists across restarts, was MemorySaver)
  - a check_cache node + conditional routing: on a cache hit, skip
    retrieve/generate entirely and go straight to END; on a miss, proceed
    as before, and generate_node writes the fresh result into the cache.
"""

import logging
from typing import TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.redis import RedisSaver

from app.retrieval import get_llm, retrieve_chunks, generate_answer_metadata, SYSTEM_PROMPT, RetrievalError
from app.redis_client import get_redis_client
from app.cache import get_cached_answer, set_cached_answer

logger = logging.getLogger(__name__)


class ConversationState(TypedDict):
    # Full chat history for this session so far (HumanMessage/AIMessage objects).
    # This grows by one Human + one AI message every turn.
    messages: list[BaseMessage]

    # The raw question exactly as the user typed it this turn.
    question: str

    # The condense node rewrites `question` (using `messages` for context)
    # into a standalone version that makes sense without prior turns.
    # e.g. "what about the second one?" -> "what is the second ingredient
    # in the bread recipe?"
    # Empty on the very first turn, since there's no history to resolve against.
    standalone_question: str

    # Chunks retrieved from Pinecone using `standalone_question` (or restored
    # from cache on a hit — either way, this is what the frontend shows as
    # "sources" for this turn).
    retrieved_chunks: list[dict]

    # Which document (Pinecone namespace) this conversation is scoped to.
    # Set once per session, doesn't change turn to turn.
    namespace: str

    # The final answer text for THIS turn only. Also embedded inside
    # `messages` as the latest AIMessage, but kept here separately too
    # so the API layer can read it directly without digging through history.
    answer: str

    # Set explicitly by check_cache_node EVERY turn (never left stale from a
    # prior turn) — True routes straight to END, False routes to retrieve.
    cache_hit: bool

    # AnswerMetadata as a plain dict ({"confidence": str, "cited_chunk_indices":
    # list[int]}), not the Pydantic object itself — simpler for the checkpointer
    # to serialize. Converted to a real AnswerMetadata instance at the API layer.
    metadata: dict


CONDENSE_SYSTEM_PROMPT = (
    "Given the conversation so far and a new follow-up question, rewrite the "
    "follow-up question as a standalone question that can be understood without "
    "the conversation history. Do not answer the question — only rewrite it. "
    "If it is already standalone, return it unchanged."
)


def condense_node(state: ConversationState) -> dict:
    """
    Rewrites `question` into `standalone_question` using prior chat history.
    Skipped (passthrough) on the first turn, since there's no history yet
    to resolve pronouns/references against.
    """
    messages_so_far = state.get("messages", [])
    question = state["question"]

    if not messages_so_far:
        return {"standalone_question": question}

    llm = get_llm()
    try:
        response = llm.invoke(
            [SystemMessage(content=CONDENSE_SYSTEM_PROMPT)]
            + messages_so_far
            + [HumanMessage(content=f"Follow-up question: {question}")]
        )
    except Exception as exc:
        logger.error("OpenAI call failed during question condensing: %s", exc)
        raise RetrievalError("Failed to process the follow-up question") from exc

    return {"standalone_question": response.content.strip()}


def check_cache_node(state: ConversationState) -> dict:
    """
    Looks up (namespace, standalone_question) in the query cache.

    On a HIT: builds the same {messages, answer, retrieved_chunks} shape
    generate_node would have produced, using the cached values instead of
    calling the LLM/Pinecone at all. Sets cache_hit=True, which routes
    the graph straight to END (see get_graph()'s conditional edge).

    On a MISS: only sets cache_hit=False. retrieve_node/generate_node run
    normally afterward, and generate_node is responsible for writing the
    fresh result into the cache once it's produced.
    """
    namespace = state["namespace"]
    standalone_question = state["standalone_question"]

    cached = get_cached_answer(namespace, standalone_question)
    if cached is None:
        return {"cache_hit": False}

    messages_so_far = state.get("messages", [])
    question = state["question"]
    updated_messages = messages_so_far + [
        HumanMessage(content=question),
        AIMessage(content=cached["answer"]),
    ]

    return {
        "cache_hit": True,
        "answer": cached["answer"],
        "retrieved_chunks": cached["sources"],
        "metadata": cached.get("metadata", {"confidence": "low", "cited_chunk_indices": []}),
        "messages": updated_messages,
    }


def route_after_cache_check(state: ConversationState) -> str:
    """Conditional edge: skip straight to END on a cache hit, else retrieve normally."""
    return END if state.get("cache_hit") else "retrieve"


def retrieve_node(state: ConversationState) -> dict:
    """
    Retrieves top-k chunks from Pinecone using the CONDENSED question,
    not the raw one — this is the whole point of the condense step.
    Only runs on a cache MISS (see route_after_cache_check).

    retrieve_chunks() already raises RetrievalError internally on failure,
    so nothing to wrap here.
    """
    chunks = retrieve_chunks(state["standalone_question"], state["namespace"])
    return {"retrieved_chunks": chunks}


def generate_node(state: ConversationState) -> dict:
    """
    Generates the answer using retrieved chunks + prior chat history (for
    tone/continuity) + the ORIGINAL raw question (not the standalone rewrite —
    the user should see their own phrasing reflected naturally in the reply).

    Only runs on a cache MISS. Writes the fresh result into the cache before
    returning, so the NEXT time this (namespace, standalone_question) pair
    comes in — from this session or any other — check_cache_node will hit.
    """
    messages_so_far = state.get("messages", [])
    chunks = state.get("retrieved_chunks", [])
    question = state["question"]

    if not chunks:
        answer_text = (
            "I couldn't find any relevant content in this document to answer that question."
        )
    else:
        context = "\n\n".join(chunk["text"] for chunk in chunks)
        user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

        llm = get_llm()
        try:
            response = llm.invoke(
                [SystemMessage(content=SYSTEM_PROMPT)]
                + messages_so_far
                + [HumanMessage(content=user_prompt)]
            )
        except Exception as exc:
            logger.error("OpenAI call failed during answer generation: %s", exc)
            raise RetrievalError("Failed to generate an answer") from exc

        answer_text = response.content

    metadata = generate_answer_metadata(question, answer_text, chunks)
    metadata_dict = metadata.model_dump()

    set_cached_answer(
        state["namespace"],
        state["standalone_question"],
        {"answer": answer_text, "sources": chunks, "metadata": metadata_dict},
    )

    updated_messages = messages_so_far + [
        HumanMessage(content=question),
        AIMessage(content=answer_text),
    ]

    return {"messages": updated_messages, "answer": answer_text, "metadata": metadata_dict}


_compiled_graph = None


def get_graph():
    """
    Builds and compiles the graph once, reusing it across requests —
    same singleton pattern as get_llm()/get_index()/get_embedding_model()
    elsewhere in this codebase.

    Graph shape: START -> condense -> check_cache -> (conditional) ->
                   either END (cache hit) or retrieve -> generate -> END

    RedisSaver persists conversation state in Redis instead of the backend
    process's memory — so a restart no longer loses history (V2's
    limitation). We pass our own Redis client (redis_client.py) via
    redis_client= rather than letting RedisSaver open its own connection,
    so it reuses the app's existing connection pool.

    .setup() creates the Redis search indices RedisSaver needs — safe to
    call every startup (same idea as ensure_index_exists() for Pinecone).
    """
    global _compiled_graph
    if _compiled_graph is None:
        redis_client = get_redis_client()
        checkpointer = RedisSaver(redis_client=redis_client)
        checkpointer.setup()

        builder = StateGraph(ConversationState)
        builder.add_node("condense", condense_node)
        builder.add_node("check_cache", check_cache_node)
        builder.add_node("retrieve", retrieve_node)
        builder.add_node("generate", generate_node)

        builder.add_edge(START, "condense")
        builder.add_edge("condense", "check_cache")
        builder.add_conditional_edges("check_cache", route_after_cache_check, ["retrieve", END])
        builder.add_edge("retrieve", "generate")
        builder.add_edge("generate", END)

        _compiled_graph = builder.compile(checkpointer=checkpointer)

    return _compiled_graph


def ask_with_memory(question: str, namespace: str, session_id: str) -> dict:
    """
    Entry point the FastAPI layer calls. Runs one turn of the conversation
    identified by session_id.

    We only pass `question` and `namespace` as input — NOT `messages`.
    LangGraph's checkpointer automatically restores the prior `messages`
    state for this session_id (thread_id) behind the scenes and merges it
    in, so we never have to manually load/save history ourselves.

    RetrievalError raised by any node propagates up unchanged — the
    FastAPI layer maps it to a 422, same as V1's stateless path.
    """
    graph = get_graph()
    config = {"configurable": {"thread_id": session_id}}

    result = graph.invoke({"question": question, "namespace": namespace}, config=config)

    return {
        "answer": result["answer"],
        "sources": result["retrieved_chunks"],
        "metadata": result["metadata"],
    }
