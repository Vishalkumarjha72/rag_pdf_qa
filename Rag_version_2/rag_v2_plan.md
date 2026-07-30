# RAG PDF Q&A System — Version 2 Plan

## 1. Goal

Add conversational memory so follow-up questions work — e.g. "what about the second point?" or "explain that more" should correctly resolve against the prior turn, instead of being treated as a standalone question with no context.

V1's `/query` was fully stateless: every question was embedded and searched in isolation, with no idea what was asked before it. V2 introduces the concept of a **conversation** (a sequence of question/answer turns tied together by a session), using LangGraph to manage that state.

**Definition of Done for V2:**
- A user can ask a question, then ask a natural follow-up that references the previous answer, and get a correct response
- Each conversation is isolated — two different sessions never see each other's history
- Memory is per-process (in-memory), not yet persisted to Redis — that's explicitly deferred to V3
- Frontend has a way to start a fresh conversation (clearing history) without restarting the whole app

---

## 2. Key design decision: history-aware retrieval

The naive approach — just stuffing full chat history into the generation prompt — doesn't fix the retrieval step. If the user asks "what about the second point?", embedding *that exact sentence* and searching Pinecone with it will retrieve garbage, because "the second point" means nothing without context.

The standard fix: before retrieving, use the LLM to **condense** the latest question + chat history into a standalone question first, *then* embed and search with that rewritten version. Generation still uses the raw chat history for tone/continuity, but retrieval uses the condensed question.

So the flow becomes:
```
question + chat_history ──▶ [condense node] ──▶ standalone_question
                                                       │
                                                       ▼
                                              [retrieve node] ──▶ chunks
                                                       │
                                                       ▼
                                      question + chunks + chat_history
                                                       │
                                                       ▼
                                              [generate node] ──▶ answer
```

This is exactly the kind of multi-step, stateful flow LangGraph is built for — V1 used a single straight-line LangChain call, V2 is where LangGraph actually earns its place in the stack.

---

## 3. Tech stack additions (V2 scope only)

| Layer | Tool |
|---|---|
| Conversation orchestration | LangGraph (`StateGraph`) |
| Memory persistence (this version) | LangGraph `MemorySaver` (in-process, lost on restart) |
| Session identity | UUID `session_id` / `thread_id`, generated client-side |

No Redis yet — `MemorySaver` keeps everything in the backend process's memory. V3 swaps this exact checkpointer for a Redis-backed one without touching the graph logic itself, which is the whole point of building it this way now.

---

## 4. Project structure changes

```
backend/app/
├── main.py              # /query gains session_id handling
├── config.py            # unchanged
├── ingestion.py         # unchanged
├── retrieval.py         # becomes a thin wrapper OR gets replaced by graph.py
├── graph.py             # NEW — LangGraph StateGraph: condense → retrieve → generate
├── embeddings.py         # unchanged
├── vectorstore.py       # unchanged
└── schemas.py            # QueryRequest gets optional session_id; QueryResponse returns it

frontend/src/
├── App.jsx               # holds sessionId state, "New Conversation" button
├── QueryPanel.jsx        # sends session_id with each query, renders full turn history
├── api.js                # askQuestion() now passes session_id
```

---

## 5. Backend build steps

### Step 1 — Install LangGraph
- `pip install langgraph`, add to `requirements.txt`

### Step 2 — Define conversation state
- A `TypedDict` (or LangGraph's message-state helper) holding: `messages` (chat history), `question`, `standalone_question`, `retrieved_chunks`, `namespace`

### Step 3 — Build the graph (`graph.py`)
- **Node 1 — condense**: given `messages` + new `question`, call the LLM to produce `standalone_question`. Skip this node entirely if `messages` is empty (first turn of a conversation — no history to condense against).
- **Node 2 — retrieve**: embed `standalone_question`, query Pinecone (same logic as V1's retrieval, just called from inside the graph)
- **Node 3 — generate**: build prompt from `messages` + retrieved chunks + original `question`, call `ChatOpenAI`, append the new Q&A pair to `messages`
- Wire nodes together with `StateGraph`, compile with `MemorySaver` as the checkpointer

### Step 4 — Update `/query` endpoint
- Accept optional `session_id` in the request body — if missing, generate a new UUID server-side and return it
- Pass `session_id` as the `thread_id` to the graph invocation — this is what LangGraph uses to look up the right conversation's history in the checkpointer
- Return `{ answer, sources, session_id }` — frontend needs to hang onto `session_id` for the next turn

### Step 5 — Update schemas
- `QueryRequest.session_id: str | None = None`
- `QueryResponse.session_id: str`

---

## 6. Frontend build steps

### Step 1 — Track session state in `App.jsx`
- `const [sessionId, setSessionId] = useState(null)`
- On first successful query response, store the returned `session_id`; send it on every subsequent call

### Step 2 — Update `QueryPanel.jsx`
- Render the full back-and-forth (not just the latest answer) — a simple list of `{ role: 'user' | 'assistant', text }` turns, so the user can actually see the conversation building up
- Pass `sessionId` down as a prop, include it in the `askQuestion()` call

### Step 3 — "New Conversation" button
- Clears `sessionId` back to `null` and clears the displayed turn list
- Next question after this will get a fresh `session_id` from the backend (since `sessionId` is `null`, it's omitted from the request)

### Step 4 — Update `api.js`
- `askQuestion(question, namespace, sessionId)` — include `session_id` in the POST body when present

---

## 7. Testing plan for V2

- Ask a question, then a follow-up using a pronoun ("it", "that", "the second one") — confirm the answer correctly resolves what's being referred to
- Open two separate browser sessions (or use "New Conversation"), confirm their histories never mix
- Restart the backend mid-conversation — confirm history is lost (expected for V2; this gets fixed in V3) rather than crashing
- Ask a totally unrelated new question after several turns — confirm the condense step doesn't wrongly drag in irrelevant old context

---

## 8. Explicitly out of scope for V2 (deferred)

- Persisting memory across backend restarts → **V3** (Redis-backed checkpointer)
- Caching repeated identical questions → **V3**
- Evaluating whether the condense step actually improves answer quality, measurably → **V4**
- Any change to prompt wording/citations beyond what's needed for history-awareness → **V5**
- Chunking/retrieval quality improvements → **V6**

---

## 9. Suggested build order (within V2)

1. Add LangGraph dependency, define state shape
2. Build the graph in isolation — test condense/retrieve/generate nodes with hardcoded input, no FastAPI involved yet (mirrors V1's "trial notebook first" habit)
3. Wire the graph into `/query` via `session_id`, test multi-turn behavior through Swagger UI
4. Update frontend to track `session_id` and render conversation history
5. Add "New Conversation" button
6. End-to-end test through `docker-compose up`
