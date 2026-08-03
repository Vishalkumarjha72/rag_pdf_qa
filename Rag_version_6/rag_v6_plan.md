# RAG PDF Q&A System — Version 6 Plan

## 1. Goal

V5 focused on making the model answer more reliably through better prompting. V6 shifts the effort to the retrieval stack itself: improving how chunks are created, how relevant context is found, and how the final answer is grounded in the best evidence.

The goal is not just to retrieve more chunks, but to retrieve the right chunks more consistently.

**Definition of Done for V6:**
- Retrieval quality improves measurably over the V5 baseline on the existing evaluation dataset.
- The system uses a more robust retrieval pipeline than simple dense similarity alone.
- Chunking is more structure-aware and preserves useful context for PDF documents.
- Relevant metadata (section, page, title, document) is available to improve retrieval and citation quality.
- The retrieval changes are isolated enough to compare against V5 without rewriting the whole graph.

---

## 2. Heads-up before we start

V6 is the first version where the main lever should be retrieval quality rather than prompting. That usually has a bigger impact on answer correctness and groundedness, but it is also easier to get wrong if the changes are not measured carefully.

The plan below intentionally keeps the work focused on:
- chunking strategy
- retrieval strategy
- reranking and context selection
- evaluation of retrieval improvements

This version should avoid jumping straight into a full-scale production architecture change. The aim is a focused retrieval upgrade that is easy to compare and reason about.

---

## 3. Key design decisions

### 3.1 — Retrieval should be hybrid, not purely semantic

The current setup is effectively dense vector retrieval. V6 should move toward a hybrid pipeline that combines:
- semantic similarity from embeddings
- keyword overlap for exact match and terminology-heavy questions

This is especially useful for PDF questions that use very specific terms, exact names, dates, or document-specific vocabulary that embedding similarity alone can miss.

A practical implementation is:
- retrieve a larger candidate set using dense search
- retrieve a second candidate set using keyword-based scoring
- merge them with a simple fusion strategy such as Reciprocal Rank Fusion (RRF)

### 3.2 — Chunking should be more structure-aware

The current fixed-size chunking is a decent baseline, but it is likely too blunt for PDFs. V6 should improve chunk quality by making chunk boundaries more aware of:
- paragraph breaks
- section headings
- document structure
- overlap around meaningful transitions

The goal is to reduce chunk fragmentation so each chunk is more self-contained and easier to cite.

### 3.3 — Metadata should become part of the retrieval workflow

Each chunk should carry useful metadata such as:
- document name
- page number
- section title
- chunk index

This makes retrieval more precise and helps with citation quality. It also makes future filtering easier, for example: “only retrieve from section X” or “only retrieve from this PDF namespace.”

### 3.4 — Reranking should reduce noise before generation

Even a good retrieval system can return too many weak matches. V6 should add a reranking step:
- retrieve a broader set of candidates
- rerank them to keep the top 3–5 most relevant chunks
- pass only that refined context to the answer generator

This tends to improve groundedness and reduce the chance that the model is distracted by irrelevant context.

### 3.5 — Retrieval should be measurable, not just “better-looking”

V6 should include explicit retrieval-focused evaluation, not just end-to-end answer scoring. Good metrics for this version include:
- recall@k
- precision@k
- hit rate on known answer-bearing chunks
- end-to-end groundedness and correctness against the existing eval dataset

That gives a clear before/after comparison for the retrieval changes.

---

## 4. Tech stack additions (V6 scope)

| Layer | Tool / Approach |
|---|---|
| Hybrid retrieval | Dense embeddings + keyword-based matching |
| Ranking | Reciprocal Rank Fusion or a lightweight reranker |
| Chunking | Improved recursive/structure-aware splitting with metadata |
| Metadata | Section/page/title-aware chunk records |
| Evaluation | Retrieval-focused metrics plus existing answer evaluators |

A small dependency such as a BM25-style keyword retriever may be added if it improves the retrieval pipeline without overcomplicating the setup.

---

## 5. Project structure changes

```text
backend/app/
├── ingestion.py          # updated chunking strategy and metadata enrichment
├── retrieval.py          # hybrid retrieval + reranking flow
├── schemas.py            # optional chunk metadata model expansion
├── vectorstore.py        # retrieval config and index-query adjustments
└── eval/                 # optional retrieval-specific eval helpers
```

The main implementation work should stay in the backend retrieval path, not in the frontend.

---

## 6. Build steps

### Step 1 — Audit the current retrieval behavior
- Inspect the current retrieval path in the backend.
- Run a small set of representative questions and inspect which chunks are returned.
- Identify failure modes: missed terminology, overly broad chunks, irrelevant context, or weak citations.

### Step 2 — Improve chunking and metadata
- Adjust the ingestion pipeline to create more meaningful chunks.
- Preserve heading and paragraph boundaries where possible.
- Attach metadata such as page number and section title for every chunk.
- Tune chunk size and overlap based on a few sample PDFs.

### Step 3 — Add hybrid retrieval
- Implement a second retrieval signal based on keyword overlap.
- Merge the dense and keyword results into a single ranked candidate list.
- Keep the retrieval interface stable so the rest of the graph can stay mostly unchanged.

### Step 4 — Add reranking
- Retrieve a larger candidate set first.
- Rerank the top candidates down to the best subset before passing them to the answer node.
- Make the reranking step configurable so it can be turned off for debugging.

### Step 5 — Integrate retrieval metrics into eval
- Extend the existing evaluation workflow to include retrieval-focused checks.
- Compare V5 and V6 on the same questions and note whether retrieval quality improved.
- Keep the evaluation simple enough to run repeatedly.

### Step 6 — Measure and document the baseline
- Record the V6 retrieval baseline and compare it to V5.
- Capture the main improvements, regressions, and any latency trade-offs.

---

## 7. Testing plan for V6

- Retrieval sanity checks: confirm the top retrieved chunks actually contain the answer-bearing content for a sample of questions.
- Chunk quality checks: inspect a few chunks manually to ensure they are not too fragmented or too broad.
- End-to-end answer checks: verify that improved retrieval leads to better grounded answers and fewer hallucinated responses.
- Evaluation run: compare V5 vs V6 on the same eval dataset and look for meaningful gains in correctness and groundedness.
- Regression checks: ensure the system still returns answers under reasonable latency and does not fail when retrieval returns fewer candidates than expected.

---

## 8. Explicitly out of scope for V6 (deferred)

- Rewriting the UI or frontend experience.
- New external services beyond what is already needed for retrieval.
- Large prompt rewrites beyond small retrieval-related instruction updates.
- Full multi-document cross-collection search or advanced agentic workflows.

---

## 9. Suggested build order (within V6)

1. Audit current retrieval behavior and collect a small set of failure cases.
2. Improve chunking and metadata enrichment in ingestion.
3. Implement hybrid retrieval.
4. Add reranking and context trimming.
5. Run the eval suite and compare results against V5.
6. Record the new retrieval baseline for future versions.
