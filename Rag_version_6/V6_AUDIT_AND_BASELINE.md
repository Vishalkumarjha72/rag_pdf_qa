# V6 — Retrieval Audit & Baseline (Steps 1 & 6)

## Step 1 — Audit of pre-V6 retrieval behavior

Code-level audit of the retrieval path as it stood before this V6 work (equivalent to
the V5 baseline, since V6 was branched from it):

- **Single signal**: `retrieve_chunks()` ran pure dense cosine similarity against
  Pinecone (`index.query(...)`), `top_k=4`, nothing else. No keyword/exact-match
  signal at all — a question using a specific term, name, or figure verbatim from the
  PDF had no advantage over a semantically-similar paraphrase; both competed purely on
  embedding distance.
- **No reranking**: whatever Pinecone returned in its top-4 was passed straight to
  generation. No second pass re-scored the (question, chunk) pairs jointly.
- **Chunking**: already reasonably good going into V6 — `ingestion.py`'s
  `RecursiveCharacterTextSplitter` (900/150) plus `_build_chunk_metadata()` already
  attaches `chunk_index`, `section_title` (heuristically inferred from the first
  non-trivial line of a chunk), `document_title`, and `page`. This part of the V6 plan
  (Step 2) was already implemented and tested (`test_ingestion.py`) before this
  session — it is NOT heading/structure-aware (still a fixed-size splitter, not a
  layout-aware one), just metadata-enriched.
- **Known failure mode (carried over from V4's eval findings, see
  `[[rag-system]]` memory / V4 section of `rag_v4_plan.md`)**: retrieved chunks were
  concatenated with a naive `"\n\n".join(...)`, which could splice unrelated chunks
  together with no boundary markers, occasionally causing the model to lose track of
  which sentence belonged to which topic. Chunk-level metadata now exists to eventually
  label each chunk in the prompt (e.g. `[source N — Section Title, p.X]`), but the
  join itself in `retrieval.py`/`graph.py` was NOT changed in this V6 session — noted
  here as a candidate for V7, not fixed now, to keep this session's diff focused on the
  plan's actual Step 3/4 scope (hybrid retrieval + reranking).

## What this session implemented (Steps 3–5)

- **Step 3 — Hybrid retrieval**: `app/bm25_store.py` (new) persists each document's
  raw chunk corpus to Redis at ingestion time and builds an in-memory `BM25Okapi`
  index per namespace on first query. `retrieval.py`'s `retrieve_chunks()` now pulls a
  wider dense candidate pool (`RETRIEVAL_CANDIDATE_K`, default 10) AND a BM25 keyword
  candidate pool of the same size, then merges them with `_reciprocal_rank_fusion()`
  (RRF, k=60).
- **Step 4 — Reranking**: `_rerank()` runs the fused candidates through a
  `cross-encoder/ms-marco-MiniLM-L-6-v2` cross-encoder (via `sentence-transformers`,
  already a dependency) and keeps the top `TOP_K=4` by joint (question, chunk) score.
  Both hybrid retrieval and reranking are independently toggleable via
  `ENABLE_HYBRID_RETRIEVAL` / `ENABLE_RERANKING` in `.env`, so either can be switched
  off for a controlled before/after comparison.
- **Step 5 — Retrieval-focused eval**: `eval/retrieval_evaluators.py` (new) adds
  `retrieval_hit_rate_evaluator` (LLM-judged: do the retrieved chunks contain enough
  information to support the *reference* answer, independent of what the model
  generated) and `avg_chunks_retrieved_evaluator` (sanity metric). Both are wired into
  `eval/run_eval.py` alongside the existing V4/V5 answer-level evaluators, under a new
  `v6-hybrid-retrieval-rerank` experiment name.

  **Honest limitation**: `eval/dataset.json` has `{question, reference_answer}` pairs,
  not hand-labeled "these exact chunk indices are answer-bearing" ground truth. True
  `recall@k` / `precision@k` (as literally named in the plan) need that labeling,
  which doesn't exist yet for this dataset. `retrieval_hit_rate_evaluator` is a
  practical LLM-judged proxy for hit rate given what's actually available. If you
  later hand-label answer-bearing chunks per question, swap it for exact recall/precision.

## Step 6 — Baseline: NOT YET RUN

This session could not execute `python -m eval.run_eval` — it requires your local
OpenAI, Pinecone, and LangSmith API keys plus a running local Redis instance, none of
which this session has network access to. **You'll need to run it yourself:**

```bash
cd Rag_version_6/backend
pip install -r requirements.txt   # picks up rank-bm25
python -m eval.upload_dataset     # only if the dataset isn't already uploaded
python -m eval.run_eval
```

Then compare the new `v6-hybrid-retrieval-rerank` experiment against the existing
`v5-prompting` experiment in the LangSmith UI — same dataset, same answer-level
evaluators, so `correctness`/`groundedness`/`citation_presence`/
`confidence_correlation` are directly comparable, plus the two new retrieval-only
metrics. For an isolated look at hybrid retrieval alone vs. hybrid+rerank, set
`ENABLE_RERANKING=false` in `.env` and rerun once more.

Once you've run it, this section should be replaced with the actual before/after
numbers and any regressions or latency trade-offs you observed (per the plan's Step 6).
