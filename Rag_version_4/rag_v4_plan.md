# RAG PDF Q&A System — Version 4 Plan

## 1. Goal

Every version so far has been evaluated by eyeballing a handful of manual test questions. V4 replaces that with **LangSmith** — tracing every graph run automatically, and building a small evaluation dataset + automated scoring so answer/retrieval quality becomes a measured number, not a vibe.

**Definition of Done for V4:**
- Every `/query` call automatically produces a trace visible in the LangSmith UI (no code changes needed beyond config — LangSmith hooks in via env vars and callback instrumentation LangChain/LangGraph already support)
- A small, real evaluation dataset exists (question + reference answer pairs, tied to your actual test PDF(s))
- An automated eval run scores the graph against that dataset on at least two dimensions: **correctness** (does the answer match the reference) and **groundedness** (is the answer actually supported by the retrieved chunks, not hallucinated)
- Results are viewable as a scored "experiment" in the LangSmith UI, giving you a concrete baseline number to compare against later when V5/V6 change prompts or retrieval

---

## 2. Heads-up before we start

Same caveat as V3's Redis section: LangSmith's SDK surface (the `evaluate()` API, dataset upload helpers, evaluator function signatures) is another area I'll want to verify against the actually-installed version before writing code, rather than assume from general knowledge. `langsmith` is already installed transitively (pulled in by `langchain`), but we'll confirm its version and inspect the real API the same way we did for `langgraph-checkpoint-redis` in V3, rather than risk another round of guessed-then-broken imports.

You'll also need a **LangSmith account** (free tier exists) and an API key from smith.langchain.com — this is the one external signup step in this plan, everything else is code.

---

## 3. Key design decisions

### 3.1 — Tracing first, evaluation second

These are two separate, independently useful things:
- **Tracing**: LangSmith automatically records every step of every graph run (which nodes ran, what each LLM call's prompt/response was, how long each step took, cache hits vs misses) as soon as tracing is enabled — zero code changes to `graph.py` itself, just environment configuration. This alone is worth having even before any formal evaluation exists — it's a genuinely better debugging tool than the `[DEBUG]` print statements we've been hand-rolling in `test_graph.py` all this time.
- **Evaluation**: a deliberate, repeatable process — a fixed set of test questions, run against the graph, scored against some criteria. This needs an actual dataset to be built, which tracing does not give you for free.

Step 1 gets you tracing (small, fast, immediately useful). Everything after that builds toward evaluation.

### 3.2 — What "groundedness" means here, and why V3 already set this up nicely

A **correctness** evaluator asks: "does this answer match what the reference answer says?" — useful, but it can't catch a specific and important RAG failure mode: an answer that sounds right but wasn't actually supported by the retrieved context (the model filling gaps from its own general knowledge instead of the document).

A **groundedness** (or "faithfulness") evaluator asks a different question: "is this answer actually backed by the retrieved chunks?" This is where V3's `AnswerMetadata` (confidence + `cited_chunk_indices`) becomes directly useful — not as a replacement for a real evaluator, but as an interesting thing to compare against one. Part of this version's evaluation can specifically check: **does the model's own self-reported confidence actually correlate with whether the answer was truly grounded**, according to an independent LLM-graded evaluator? If "high confidence" answers turn out to be ungrounded just as often as "low confidence" ones, that's a genuinely useful, concrete finding about whether V3's structured self-assessment is trustworthy or just decorative.

### 3.3 — Where the dataset lives

Two reasonable options: build the dataset as a local JSON file checked into the repo (simple, versionable, no dependency on LangSmith's own dataset storage), or upload it as a LangSmith **Dataset** via their SDK (lets you use their hosted `evaluate()` runner and UI comparison features directly). Leaning toward starting with a local JSON file for Step 2 (simplest, fully within your control, easy to hand-edit), then uploading it to LangSmith in Step 3 once the shape is settled — avoids churn from iterating on dataset content directly inside LangSmith's UI while still figuring out what good test questions even look like.

---

## 4. Tech stack additions (V4 scope)

| Layer | Tool |
|---|---|
| Tracing | `langsmith` (already installed transitively) + `LANGCHAIN_TRACING_V2`/`LANGCHAIN_API_KEY`/`LANGCHAIN_PROJECT` env vars |
| Evaluation dataset | Local JSON file initially, uploaded to a LangSmith Dataset once finalized |
| Evaluators | LangSmith's built-in LLM-graded evaluators where they fit (e.g. correctness-vs-reference), custom Python evaluator functions where they don't (e.g. the confidence-vs-groundedness comparison from 3.2) |
| Eval runner | LangSmith SDK's `evaluate()` function (exact import path/signature to confirm against the installed version) |

---

## 5. Project structure changes

```
backend/
├── eval/                    # NEW — dev-time evaluation tooling, not runtime app code
│   ├── dataset.json          # question + reference_answer pairs
│   ├── evaluators.py         # correctness, groundedness, confidence-correlation evaluator functions
│   └── run_eval.py           # script that runs the graph against the dataset and scores it
```

Kept separate from `app/`, same reasoning as `test_*.py` scripts living at the `backend/` root rather than inside `app/` — this is tooling you run deliberately, not part of the deployed application.

---

## 6. Build steps

### Step 1 — Enable tracing
- Add `LANGCHAIN_TRACING_V2=true`, `LANGCHAIN_API_KEY=...`, `LANGCHAIN_PROJECT=rag-pdf-qa` to `.env`/`.env.example`
- Confirm `langsmith` is actually installed and check its resolved version before assuming any API details
- Run a normal `/query` call (same as any prior manual test), then check the LangSmith UI — confirm a trace appears showing the condense/check_cache/retrieve/generate steps

### Step 2 — Build the evaluation dataset
- Write 10-20 question/reference-answer pairs against your existing test PDF(s) in `eval/dataset.json` — a mix of straightforward factual questions and a few multi-turn follow-ups, since that's the whole point of V2/V3's work
- Keep reference answers as natural-language descriptions of what a correct answer should cover, not exact strings to match verbatim (an LLM-graded evaluator compares meaning, not exact text)

### Step 3 — Write the evaluators (`eval/evaluators.py`)
- Verify the actual LangSmith evaluator function signature against the installed SDK version first
- `correctness_evaluator`: compares the graph's answer against the reference answer
- `groundedness_evaluator`: checks whether the answer is actually supported by the retrieved chunks (not the reference answer — the actual retrieved context for that run)
- `confidence_correlation_evaluator`: compares the graph's own `AnswerMetadata.confidence` against the groundedness evaluator's independent judgment — flags cases where they disagree

### Step 4 — Write the eval runner (`eval/run_eval.py`)
- Loads `dataset.json`, runs each question through the graph (a fresh `session_id` per question, since these are independent evaluation questions, not a conversation), collects results
- Uses LangSmith's `evaluate()` (or the closest current equivalent) to score and upload results as a named experiment
- Test: run it once, confirm a scored experiment appears in the LangSmith UI with per-question results you can actually read

### Step 5 — Establish a baseline
- Run the full eval suite once against V3's current code, note the aggregate scores (e.g. "correctness: 85%, groundedness: 78%")
- This becomes the number V5 (better prompts) and V6 (better chunking/retrieval) get compared against later — the whole reason evaluation exists is to prove those future changes actually helped, not just "feel" like they helped

---

## 7. Testing plan for V4

- Tracing: a normal `/query` call produces a readable trace in the LangSmith UI
- Dataset: spot-check a handful of `dataset.json` entries manually — do the reference answers actually reflect what's in the source PDF?
- Evaluators: run each evaluator against one deliberately-good and one deliberately-bad hand-crafted example first (not through the full dataset), confirming they score in the expected direction, before trusting them across the whole dataset
- Full run: `run_eval.py` completes, produces a viewable scored experiment, and the aggregate numbers are in a plausible range (not suspiciously 100% or 0% across the board, which would suggest a broken evaluator rather than a perfect/terrible system)

---

## 8. Explicitly out of scope for V4 (deferred)

- Acting on what the evaluation reveals (improving prompts or retrieval based on low scores) → **V5**/**V6** — V4 is purely about measuring, not fixing
- Continuous/automated evaluation on every code change (CI integration) — a real production concern, more depth than this stage needs
- Expanding the dataset to cover multiple different PDFs — start with one well-understood document, expand later if useful

---

## 9. Suggested build order (within V4)

1. Enable tracing, confirm a trace appears for a normal query
2. Build the dataset (`eval/dataset.json`), hand-reviewed for accuracy against the source PDF
3. Write evaluators, sanity-tested against one good/one bad hand-crafted example each
4. Write and run `run_eval.py` against the full dataset
5. Record the baseline scores somewhere durable (this plan doc, or a short `EVAL_BASELINE.md`) so V5/V6 have something concrete to compare against
