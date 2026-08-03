"""
V3 conversation graph: condense -> check_cache -> [retrieve -> generate] -> END.

V2 flow was condense -> retrieve -> generate. V3 adds:
  - AsyncRedisSaver checkpointer (persists across restarts, was MemorySaver)
  - a check_cache node + conditional routing: on a cache hit, skip
    retrieve/generate entirely and go straight to END; on a miss, proceed
    as before, and generate_node writes the fresh result into the cache.
"""

import logging
from typing import TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.redis import AsyncRedisSaver

from app.retrieval import (
    get_llm,
    retrieve_chunks,
    generate_answer_metadata,
    SYSTEM_PROMPT,
    ANSWER_PROMPT_TEMPLATE,
    RetrievalError,
)
from app.redis_client import get_async_redis_client
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
    "Given the conversation so far and a new follow-up question, rewrite the follow-up "
    "question as a fully standalone question that can be understood with NO access to "
    "the conversation history. Do not answer the question — only rewrite it.\n\n"
    "Critically: replace every pronoun and vague reference (\"that\", \"it\", \"this\", "
    "\"those\", \"the second one\", etc.) with the SPECIFIC topic, term, or entity it "
    "refers to, pulled from the prior conversation. A rewrite that still contains an "
    "unresolved pronoun has failed at the one job this rewrite exists to do — it will be "
    "used to search a document for relevant content, and a vague question retrieves "
    "vague, useless results.\n\n"
    "Example:\n"
    "  Prior answer mentioned: term frequency-inverse document frequency (TF-IDF)\n"
    "  Follow-up: \"Can you say more about that?\"\n"
    "  Good rewrite: \"Can you explain more about term frequency-inverse document "
    "frequency (TF-IDF) and how it's used to evaluate term importance in documents?\"\n"
    "  Bad rewrite (still vague, do NOT do this): \"Can you provide more details about "
    "that?\"\n\n"
    "If the follow-up question is already fully standalone (no pronouns or vague "
    "references at all), return it unchanged."
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


def _is_bad_answer(answer_text: str, chunks: list[dict]) -> bool:
    """Detect clearly invalid answers so we can fall back safely."""
    if not answer_text or not answer_text.strip():
        return True
    if chunks and len(answer_text.strip()) < 15:
        return True
    return False


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
        user_prompt = ANSWER_PROMPT_TEMPLATE.format(context=context, question=question)

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

        if _is_bad_answer(answer_text, chunks):
            logger.warning(
                "Generated answer was empty or too short; falling back to a safe response."
            )
            answer_text = (
                "I couldn't generate a citation-backed answer from the provided document. "
                "Please try rephrasing the question or ask about another part of the document."
            )

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


async def get_graph():
    """
    Builds and compiles the graph once, reusing it across requests —
    same singleton pattern as get_llm()/get_index()/get_embedding_model()
    elsewhere in this codebase. ASYNC because AsyncRedisSaver.setup() is
    async, and because everything calling this now runs through async
    graph execution (.ainvoke()/.astream()) — see the module docstring's
    note on why RedisSaver (sync) couldn't be used for streaming.

    Graph shape: START -> condense -> check_cache -> (conditional) ->
                   either END (cache hit) or retrieve -> generate -> END

    AsyncRedisSaver persists conversation state in Redis instead of the
    backend process's memory — so a restart no longer loses history (V2's
    limitation). We pass our own async Redis client (redis_client.py) via
    redis_client= rather than letting it open its own connection, so it
    reuses the app's existing connection pool.

    .setup() creates the Redis search indices the checkpointer needs —
    safe to call every startup (same idea as ensure_index_exists() for
    Pinecone).
    """
    global _compiled_graph
    if _compiled_graph is None:
        redis_client = get_async_redis_client()
        checkpointer = AsyncRedisSaver(redis_client=redis_client)
        await checkpointer.asetup()

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


async def ask_with_memory(question: str, namespace: str, session_id: str) -> dict:
    """
    Entry point for a single, non-streamed turn (used by test_graph.py).
    Async now — see get_graph() — so callers need `await` (test_graph.py
    wraps its calls in asyncio.run()).

    We only pass `question` and `namespace` as input — NOT `messages`.
    LangGraph's checkpointer automatically restores the prior `messages`
    state for this session_id (thread_id) behind the scenes and merges it
    in, so we never have to manually load/save history ourselves.

    RetrievalError raised by any node propagates up unchanged — the
    FastAPI layer maps it to a 422, same as V1's stateless path.
    """
    graph = await get_graph()
    config = {"configurable": {"thread_id": session_id}}

    result = await graph.ainvoke({"question": question, "namespace": namespace}, config=config)

    return {
        "answer": result["answer"],
        "sources": result["retrieved_chunks"],
        "metadata": result["metadata"],
    }


async def astream_answer(question: str, namespace: str, session_id: str):
    """
    Async generator yielding answer TEXT CHUNKS (tokens) as they're generated
    — the streaming counterpart to ask_with_memory(). Used by the /query
    endpoint's SSE response.

    Uses graph.astream(..., stream_mode="messages"), which surfaces token-by-
    token output from EVERY chat-model call inside the graph, not just the
    one we care about — condense_node also calls the LLM internally. Each
    yielded (chunk, metadata) pair's metadata includes "langgraph_node", so
    we filter to only forward chunks that came from the "generate" node,
    which is the only one whose output the user should actually see stream.

    `if chunk.content` also filters out empty chunks — this matters because
    generate_node ALSO makes a second, structured-output LLM call
    (generate_answer_metadata) after the main answer. That call uses
    function/tool-calling under the hood, so its streamed chunks have empty
    .content (the data comes through as tool-call arguments instead) — they
    get skipped here automatically rather than leaking partial JSON into
    what the user sees as the answer.

    On a CACHE HIT: check_cache_node produces the whole answer as a plain
    dict return, not an LLM call — there's nothing to stream. This generator
    will yield ZERO chunks in that case. The caller (main.py) is responsible
    for detecting that and falling back to sending the full cached answer
    as a single chunk instead — see get_final_state() below, which is how
    it gets that answer text either way.
    """
    graph = await get_graph()
    config = {"configurable": {"thread_id": session_id}}

    async for chunk, chunk_metadata in graph.astream(
        {"question": question, "namespace": namespace},
        config=config,
        stream_mode="messages",
    ):
        if chunk_metadata.get("langgraph_node") == "generate" and chunk.content:
            yield chunk.content


async def get_final_state(session_id: str) -> dict:
    """
    Reads back the graph's checkpointed state for this session AFTER a turn
    has finished (streamed or not) — this is how the /query endpoint gets
    `sources`, `metadata`, and (on a cache hit) the full `answer` text, none
    of which come through the token stream itself. Async because it goes
    through the same AsyncRedisSaver-backed graph as everything else here.
    """
    graph = await get_graph()
    config = {"configurable": {"thread_id": session_id}}
    state = (await graph.aget_state(config)).values
    return {
        "answer": state["answer"],
        "sources": state["retrieved_chunks"],
        "metadata": state["metadata"],
    }