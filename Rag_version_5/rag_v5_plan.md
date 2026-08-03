# RAG PDF Q&A System — Version 5 Plan

## 1. Goal

V5 is about making the model behave more reliably and transparently through better prompting: stronger system instructions, guardrails for answer style, and explicit citation behavior. The objective is not to change the retrieval stack yet, but to make the answers more consistent, grounded, and easy to evaluate.

**Definition of Done for V5:**
- The graph uses a clear, explicit system prompt that defines the assistant role, answer format, and citation expectations.
- The prompt architecture includes guardrails to reduce hallucination, avoid speculative answers, and keep responses concise when asked.
- Answers include citations to retrieved chunks or document sections when appropriate, not just free-form text.
- The system can optionally emit structured metadata for answer confidence, cited sources, or answer category.
- Prompt improvements are isolated from the runtime graph so they can be iterated on without changing the retrieval architecture.

---

## 2. Heads-up before we start

Prompt engineering is often the likeliest lever for quality improvement in an existing RAG pipeline, but it can also be deceptive: a small wording change can appear to help on one question and fail on another. V5 is therefore intentionally about building a robust prompting framework, not chasing one-off prompt hacks.

This version should keep the same core stack from V4, with no new external services or major architecture changes. The work is mostly on the prompt templates, graph node wiring, and validation around how answers are generated and annotated.

---

## 3. Key design decisions

### 3.1 — Prompting should be modular, not hardcoded

Keep prompts in a reusable place rather than embedding them directly inside the graph logic. That means:
- `system_prompt` / `assistant_prompt` templates live in a dedicated prompt module or config file
- question-specific prompt stitching happens through a small helper, not a monolithic string constructed in place
- different prompt variants can be swapped for A/B testing or future versions without rewriting graph code

### 3.2 — Guardrails should be explicit and rigorous

The safest and most repeatable prompt improvements are guardrails, not just wishful language. Good guardrails for this version include:
- refuse to answer if the question is outside the PDF's scope
- answer using only information from the retrieved sources
- cite source names/indices when asserting facts
- keep follow-up questions separate from the current answer rather than inventing new context

That means the prompt should explicitly say something like: "If the answer is not supported by the provided sources, say 'I don't know' or explain that the information is not contained in the document." This is the most valuable behaviour boost V5 can deliver.

### 3.3 — Citations should be a normal part of the answer path

Rather than bolting citations on after the fact, V5 should bake them into the response generation step:
- the answer should be generated with an explicit instruction to cite the retrieved chunks or section titles
- the model should be allowed to answer in natural language but still produce a citation list or inline references
- citations should be easy to parse for frontend display and evaluation

A good structure is: answer text + `sources:` metadata, or an answer with bracketed citations like `[source 2]`.

### 3.4 — Structured answer metadata is optional but valuable

If the existing graph already has support for structured output, use it to capture metadata such as:
- `confidence` or `certainty` level
- `cited_chunk_indices`
- `answer_type` (factual, yes/no, summary, unknown)

This is not required to ship V5, but if it is already available from V4's structured-output work, it should be used to make citations and guardrail adherence easier to reason about and evaluate.

---

## 4. Tech stack additions (V5 scope)

| Layer | Tool |
|---|---|
| Prompt orchestration | Dedicated prompt templates / helper module |
| Guardrails | Explicit system/assistant prompt instructions |
| Citation output | Answer template with source citation guidance |
| Structured metadata | Existing LangChain structured-output support, if present |

No new dependencies are required for V5 beyond the existing FastAPI/LangChain/LangGraph stack.

---

## 5. Project structure changes

```
backend/app/
├── prompts.py            # NEW or expanded — centralized prompt templates and helper functions
├── graph.py              # updated to wire improved prompts into the answer generation node
├── schemas.py            # optional: structured metadata models for citations/confidence
├── eval/                 # existing V4 evaluation tooling can be reused to measure prompt changes
```

If you already have a prompt module in V4, extend it instead of adding a new file.

---

## 6. Build steps

### Step 1 — Define the system-level prompt
- Create a clear assistant identity: "You are a PDF research assistant. Use only the provided document excerpts and citations when answering."
- Add explicit refusal behavior for unsupported questions.
- Add answer-style constraints: concise, bullet lists for enumerations, single-sentence summaries for direct questions when asked.
- Keep the system prompt stable and isolated from per-question content.

### Step 2 — Define the answer-generation prompt template
- Build a template that accepts:
  - `system_prompt`
  - `user_question`
  - `retrieved_sources`
  - any additional instructions such as citation style
- Instruct the model to:
  - answer in natural language
  - cite source indices or section titles for every factual claim
  - avoid speculation and say "I don't know" when the document does not contain the answer
- If you are using a structured-output model, include the output schema in the prompt flow.

### Step 3 — Integrate the new prompts into the graph
- Replace the existing raw answer prompt or generation node with the new prompt helper.
- Keep retrieval and condensation unchanged from V4; the improvements are only in how the final answer is asked.
- If the graph has separate “condense question” and “answer with sources” nodes, wire the citation-aware prompt into the answer node.

### Step 4 — Add prompt validation and fallback behavior
- Sanity-check that the graph is receiving both the prompt template and the retrieved sources.
- If the answer node ever returns an empty string or malformed citations, fallback to a safe response like "I couldn't generate a citation-backed answer from the document."
- Optionally add a lightweight test in `test_graph.py` that asserts the prompt contains the new citation rules and refusal guardrails.

### Step 5 — Evaluate prompt quality with V4 tooling
- Reuse the V4 `eval/` suite to compare prompt variants if possible.
- Add at least one evaluation question that specifically checks citation presence and refusal behavior for unsupported queries.
- Run the evaluation and compare results against the V4 baseline to measure whether prompt changes improved groundedness and correctness.

---

## 7. Testing plan for V5

- Prompt sanity: confirm the deployed prompt includes the new citation and refusal instructions.
- Manual checks: ask supported factual questions, unsupported questions, and follow-ups; verify the answer either cites sources or says it cannot answer.
- Citation check: inspect the system output and ensure citations reference retrieved chunks or sections, not generic source text.
- Evaluation: run the existing V4 eval dataset and add at least one prompt-specific case for groundedness/citation behavior.
- Regression check: confirm answers still return under the same latency profile and that retrieval is unchanged.

---

## 8. Explicitly out of scope for V5 (deferred)

- Changing the retrieval architecture, chunking strategy, or embeddings — that belongs in **V6**.
- Full production deployment hardening, autoscaling, or monitoring — that belongs in **V7**.
- Rewriting the UI or graph transport layer for streaming or new connection modes.
- Building new dataset sources or expanding to multiple PDFs beyond the existing V4 dataset.

---

## 9. Suggested build order (within V5)

1. Create or extend `backend/app/prompts.py` with system and answer templates.
2. Wire those prompts into the answer-generation node in `backend/app/graph.py`.
3. Add reuseable citation and refusal guardrails to the prompt templates.
4. Sanity-check output with a few manual queries.
5. Run the V4 evaluation suite and validate citation/refusal behavior on a targeted case.
