"""
V2 conversation graph: condense -> retrieve -> generate.

Ties together the conversation state (ConversationState) with three nodes
and a LangGraph checkpointer, so multi-turn conversations resolve follow-up
questions correctly instead of treating every question as standalone (V1's
behavior).
"""

from typing import TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from app.retrieval import get_llm, retrieve_chunks, SYSTEM_PROMPT


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

    # Chunks retrieved from Pinecone using `standalone_question`.
    # Same shape as V1's retrieval.py: {text, source, page, score}
    retrieved_chunks: list[dict]

    # Which document (Pinecone namespace) this conversation is scoped to.
    # Set once per session, doesn't change turn to turn.
    namespace: str

    # The final answer text for THIS turn only. Also embedded inside
    # `messages` as the latest AIMessage, but kept here separately too
    # so the API layer can read it directly without digging through history.
    answer: str


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
    response = llm.invoke(
        [SystemMessage(content=CONDENSE_SYSTEM_PROMPT)]
        + messages_so_far
        + [HumanMessage(content=f"Follow-up question: {question}")]
    )
    return {"standalone_question": response.content.strip()}


def retrieve_node(state: ConversationState) -> dict:
    """
    Retrieves top-k chunks from Pinecone using the CONDENSED question,
    not the raw one — this is the whole point of the condense step.
    """
    chunks = retrieve_chunks(state["standalone_question"], state["namespace"])
    return {"retrieved_chunks": chunks}


def generate_node(state: ConversationState) -> dict:
    """
    Generates the answer using retrieved chunks + prior chat history (for
    tone/continuity) + the ORIGINAL raw question (not the standalone rewrite —
    the user should see their own phrasing reflected naturally in the reply).

    Appends this turn's Human + AI messages onto the history so the next
    turn's condense/generate nodes see it too.
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
        response = llm.invoke(
            [SystemMessage(content=SYSTEM_PROMPT)]
            + messages_so_far
            + [HumanMessage(content=user_prompt)]
        )
        answer_text = response.content

    updated_messages = messages_so_far + [
        HumanMessage(content=question),
        AIMessage(content=answer_text),
    ]

    return {"messages": updated_messages, "answer": answer_text}


_compiled_graph = None


def get_graph():
    """
    Builds and compiles the graph once, reusing it across requests —
    same singleton pattern as get_llm()/get_index()/get_embedding_model()
    elsewhere in this codebase.

    MemorySaver = in-process checkpointer. Keys conversation state by
    thread_id (we pass session_id as thread_id). Lost on backend restart —
    intentional for V2, gets swapped for a Redis-backed checkpointer in V3
    without changing anything below.
    """
    global _compiled_graph
    if _compiled_graph is None:
        builder = StateGraph(ConversationState)
        builder.add_node("condense", condense_node)
        builder.add_node("retrieve", retrieve_node)
        builder.add_node("generate", generate_node)

        builder.add_edge(START, "condense")
        builder.add_edge("condense", "retrieve")
        builder.add_edge("retrieve", "generate")
        builder.add_edge("generate", END)

        # the checkpointer automatically saves/restores the entire ConversationState
        _compiled_graph = builder.compile(checkpointer=MemorySaver())

    return _compiled_graph


def ask_with_memory(question: str, namespace: str, session_id: str) -> dict:
    """
    Entry point the FastAPI layer calls. Runs one turn of the conversation
    identified by session_id.

    We only pass `question` and `namespace` as input — NOT `messages`.
    LangGraph's checkpointer automatically restores the prior `messages`
    state for this session_id (thread_id) behind the scenes and merges it
    in, so we never have to manually load/save history ourselves.
    """
    graph = get_graph()
    config = {"configurable": {"thread_id": session_id}}

    result = graph.invoke({"question": question, "namespace": namespace}, config=config)

    return {
        "answer": result["answer"],
        "sources": result["retrieved_chunks"],
    }
