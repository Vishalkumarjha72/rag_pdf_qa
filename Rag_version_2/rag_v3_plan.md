# RAG PDF Q&A System — Version 3 Plan

## 1. Goal

V3 expands beyond the original "just add Redis caching" scope to cover three related upgrades:

1. **Redis-backed persistence** — swap V2's in-process `MemorySaver` for a Redis-backed checkpointer, so conversation memory survives a backend restart (explicitly deferred from V2).
2. **Redis caching** — cache repeated identical questions against the same document, so they don't re-hit OpenAI/re-retrieve from Pinecone every time.
3. **Streaming responses** — answers appear token-by-token in the frontend instead of waiting for the full response, matching the ChatGPT-style UX you're used to as a user of these tools.
4. **Structured output via Pydantic** — force the LLM to return validated, structured data (not just a raw string) for at least part of the response, using LangChain's structured-output support.

**Definition of Done for V3:**
- Restarting the backend mid-conversation no longer loses history (fixes the V2 limitation)
- Asking the exact same question twice against the same document returns the cached answer near-instantly the second time
- The frontend shows the answer streaming in progressively, not as one blocking wait
- At least one piece of the response (see Section 4) is Pydantic-validated structured output, not just parsed from raw text

---

## 2. A heads-up before we start

LangGraph's Redis checkpointer package and its exact streaming API are areas that move fast, and this plan is being written from general architectural knowledge rather than verified against the current docs (this is genuinely a "check the latest docs when we get there" situation, more so than V1/V2 were — same caution as when we jumped `langgraph` versions and hit several moved imports). Treat the package names and method calls below as **the intended shape**, not gospel — we'll verify each one against the installed version's actual API right before using it, the same way we debugged the `langchain.text_splitter` → `langchain_text_splitters` move last version.

---

## 3. Key design decisions

### 3.1 — Two different Redis use cases, not one

It's tempting to think "add Redis" is a single change, but it's actually two independent responsibilities that happen to use the same database:

| Use case | What it stores | Keyed by |
|---|---|---|
| **Checkpointer** (conversation memory) | Full `ConversationState` (messages, namespace, etc.) | `session_id` (thread_id) |
| **Query cache** | Just the final `{answer, sources}` for a specific question | hash of `(namespace, standalone_question)` |

These solve different problems — one is "remember this conversation," the other is "don't redo work for a question we've already answered." We'll build them as two separate, independently-testable pieces, even though both live in the same Redis instance.

### 3.2 — Cache invalidation scope

The cache key is `(namespace, standalone_question)`, not `(session_id, question)` — this is deliberate. Two different users (different sessions) asking the *same underlying question* about the *same document* should hit the same cache entry; two *rephrased* versions of the same question (different raw `question`, same condensed `standalone_question`) also should. That's why we cache off the standalone question, not the raw one — same principle as retrieval already caring about the condensed form, not the literal user input.

Cache entries should expire (TTL) rather than live forever — if you were building live/frequently-updated documents this would matter a lot; for static uploaded PDFs it matters less, but it's still good practice so stale cache entries don't accumulate forever. A TTL of an hour or so is a reasonable starting point, easy to tune later.

### 3.3 — Streaming: keep it to plain text, not structured output

Streaming works naturally for plain text (tokens arrive one at a time, you just append them to the screen as they come). Streaming a Pydantic-structured object is a genuinely harder problem — you'd need to stream *partial, incomplete* JSON and progressively validate it, which adds real complexity for uncertain UX benefit at this stage.

So the design keeps these separate:
- The **answer text** streams token-by-token (this is the part users actually watch and benefit from seeing stream)
- The **structured output** (Section 4) is generated as a separate, non-streamed step — either alongside or right after the streamed answer, not interleaved with it

### 3.4 — What streaming changes about the architecture

Right now `/query` is a normal request/response endpoint — the frontend calls it, waits, gets one JSON blob back. Streaming means the backend needs to send data incrementally as it's generated. The common approaches:

- **Server-Sent Events (SSE)** — a stream of text chunks over a long-lived HTTP response. Simpler to implement in FastAPI (`StreamingResponse`), simpler to consume in the browser than a full WebSocket, and this is the standard pattern for LLM streaming (it's what ChatGPT's own API-facing endpoint uses).
- **WebSockets** — bidirectional, more setup, generally overkill for one-directional "server sends tokens to client" streaming.

We'll go with **SSE via FastAPI's `StreamingResponse`**, since it's the better fit for this use case and keeps the frontend change simpler (an `EventSource`/`fetch` + `ReadableStream` reader, no new connection-management complexity).

One real complication worth naming now: **`session_id` and `sources` still need to reach the frontend somehow**, and those aren't naturally "streamable" — they're single values that exist all at once. Common pattern: send the token stream first, then a final special "done" event carrying `session_id` and `sources` together, which the frontend recognizes as the end-of-stream signal.

---

## 4. Structured output — what to actually structure

Rather than trying to structure the entire response (the answer itself is genuinely better as free-form prose), a good learning-sized target is adding a Pydantic-validated **citation/confidence** layer alongside the streamed answer:

```python
class AnswerMetadata(BaseModel):
    confidence: Literal["high", "medium", "low"]
    cited_chunk_indices: list[int]  # which of the retrieved chunks were actually used
```

This is generated via LangChain's structured-output support (`llm.with_structured_output(AnswerMetadata)`), as a **second, separate LLM call** after the streamed answer completes — asking the model to self-assess which chunks it actually drew from and how confident it is. This is a genuinely useful production pattern (knowing *which* sources were actually used, not just which were retrieved) and gives you hands-on practice with Pydantic-constrained LLM output without entangling it with the streaming logic.

---

## 5. Tech stack additions (V3 scope)

| Layer | Tool |
|---|---|
| Redis client | `redis` (Python) |
| Redis-backed checkpointer | A LangGraph Redis checkpoint package (exact name to confirm at implementation time — verify against current LangGraph docs, this has likely changed since general knowledge here was current) |
| Query result caching | Same `redis` client, plain key-value with TTL (`SETEX`) |
| Streaming transport | FastAPI `StreamingResponse` (SSE), no new dependency |
| Structured output | `llm.with_structured_output(YourPydanticModel)` — part of `langchain-core`, already installed |
| Local Redis for development | Redis via `docker-compose` (a new `redis` service, official `redis:alpine` image) |

---

## 6. Project structure changes

```
backend/app/
├── main.py              # /query becomes a streaming endpoint
├── graph.py             # checkpointer swapped to Redis; add structured-output node/call
├── cache.py             # NEW — query cache get/set helpers
├── redis_client.py      # NEW — Redis connection singleton (same pattern as get_index()/get_llm())
├── schemas.py            # AnswerMetadata model added

frontend/src/
├── api.js                # askQuestion() becomes a streaming call, not a single POST/await
├── QueryPanel.jsx        # renders tokens as they arrive; shows confidence/citations once available
```

`docker-compose.yml` gains a `redis` service that both `backend` and (indirectly) the conversation graph depend on.

---

## 7. Backend build steps

### Step 1 — Add Redis to the stack
- Add a `redis` service to `docker-compose.yml` (official `redis:alpine` image, exposed on the default `6379`)
- Install the `redis` Python package, pin it in `requirements.txt`
- Build `redis_client.py` — a singleton Redis connection, matching the existing pattern (`get_index()` in `vectorstore.py`, `get_llm()` in `retrieval.py`)
- Test the connection in isolation before touching anything else (mirrors the "prove it works standalone first" habit)

### Step 2 — Swap the checkpointer
- Install the Redis-backed LangGraph checkpointer package (confirm exact name/API against current docs first)
- Replace `MemorySaver()` in `graph.py`'s `get_graph()` with the Redis-backed equivalent, pointed at the same Redis instance from Step 1
- Test: start a conversation, restart the backend process, continue the conversation with the same `session_id` — history should now survive, unlike V2

### Step 3 — Add the query cache (`cache.py`)
- `get_cached_answer(namespace, standalone_question) -> dict | None`
- `set_cached_answer(namespace, standalone_question, result, ttl_seconds=3600)`
- Wire into the graph: right after the condense node produces `standalone_question`, check the cache before running retrieve/generate at all; if hit, skip straight to returning the cached result
- Test: ask the same question twice, confirm the second call is dramatically faster and doesn't hit OpenAI/Pinecone (check your OpenAI usage dashboard or add a log line to confirm)

### Step 4 — Add structured output
- Define `AnswerMetadata` in `schemas.py`
- After the (now-cached-or-generated) answer is produced, make a second `with_structured_output(AnswerMetadata)` call asking the model to self-assess confidence and which chunks it used
- Test in isolation first — hardcoded answer + chunks, confirm the model reliably returns valid `AnswerMetadata` instances

### Step 5 — Convert `/query` to a streaming endpoint
- Switch to FastAPI's `StreamingResponse`, yielding SSE-formatted chunks as the LLM generates tokens (LangGraph/LangChain support streaming token-by-token from a running graph — exact method to confirm against current docs)
- Stream the answer tokens first, then a final event carrying `session_id`, `sources`, and the `AnswerMetadata`
- Test via `curl` or Swagger UI's raw response view before touching the frontend, so you can see the actual event stream format before building a consumer for it

---

## 8. Frontend build steps

### Step 1 — Update `api.js` for streaming
- Replace the single `await apiClient.post(...)` in `askQuestion()` with a `fetch()` call that reads the response body as a stream (`response.body.getReader()`), parsing SSE-formatted chunks as they arrive
- Provide a callback-based API so `QueryPanel` can react to each token as it streams in, rather than getting one final return value

### Step 2 — Update `QueryPanel.jsx` for progressive rendering
- As tokens arrive, append them to the current turn's answer text in state, so the UI updates live
- Once the final event arrives, attach `sources` and the confidence/citation metadata to that turn
- Add a small UI element showing confidence level and which sources were actually cited (using the new `AnswerMetadata`), distinct from the full retrieved-sources list already shown

---

## 9. Testing plan for V3

- **Persistence**: start a conversation, restart the backend, continue — history intact
- **Caching**: ask the same question twice, confirm the second is near-instant and doesn't re-hit OpenAI (check logs/usage)
- **Cache correctness**: ask the same underlying question two *differently worded* ways in the same conversation — confirm the second hits the cache too (since both should condense to a similar standalone question) — and confirm a genuinely different question does NOT wrongly hit the cache
- **Streaming**: confirm tokens visibly appear progressively in the browser, not all at once
- **Structured output**: confirm `AnswerMetadata` is always valid (no malformed confidence values, cited indices actually exist in the retrieved chunks) across several different questions
- **Full stack**: `docker-compose up --build` with the new `redis` service, end-to-end test through the browser

---

## 10. Explicitly out of scope for V3 (deferred)

- Measuring whether any of this (condense step, structured citations, etc.) actually improves answer *quality* — that's evaluation, and belongs in **V4** (LangSmith)
- Any further prompt refinement beyond what's needed to support structured output → **V5**
- Chunking/retrieval quality improvements → **V6**
- Cache warming, cache eviction policies beyond simple TTL, or multi-instance Redis considerations — real production concerns, but more depth than this stage needs

---

## 11. Suggested build order (within V3)

1. Redis connection singleton, tested in isolation
2. Swap checkpointer to Redis-backed, verify persistence across a backend restart
3. Add query caching, verify cache hits/misses behave correctly
4. Add structured output (`AnswerMetadata`), tested in isolation with hardcoded input
5. Convert `/query` to streaming, verify raw event stream via curl/Swagger before touching frontend
6. Update frontend to consume the stream and render progressively
7. Add `redis` service to `docker-compose.yml`, full end-to-end test
