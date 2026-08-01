"""
Isolated smoke test for generate_answer_metadata() — no FastAPI, no graph,
no PDF upload. Uses hardcoded question/answer/chunks so this runs fast and
doesn't depend on anything else being set up first.

Run from inside backend/: python test_metadata.py
"""

from app.retrieval import generate_answer_metadata

QUESTION = "What programming language is used in the examples?"

# Chunk 0 is genuinely relevant; chunk 1 is a decoy (present, but irrelevant
# to the question) — a good AnswerMetadata result should cite chunk 0 only.
CHUNKS = [
    {"text": "All code examples in this guide are written in Python 3.11.", "source": "guide.pdf", "page": 1, "score": 0.91},
    {"text": "The recommended editor for this course is VS Code, though any editor works.", "source": "guide.pdf", "page": 2, "score": 0.44},
]

ANSWER = "The examples in this guide are written in Python."


def main():
    print(f"Question: {QUESTION}")
    print(f"Answer:   {ANSWER}")
    print(f"Chunks:   {len(CHUNKS)} provided (chunk 0 relevant, chunk 1 a decoy)\n")

    metadata = generate_answer_metadata(QUESTION, ANSWER, CHUNKS)

    print(f"Result type: {type(metadata).__name__}")
    print(f"Confidence: {metadata.confidence}")
    print(f"Cited chunk indices: {metadata.cited_chunk_indices}")

    print("\nCheck manually:")
    print("  - confidence should be 'high' or 'medium' (the answer IS well-supported)")
    print("  - cited_chunk_indices should be [0] (or include 0), NOT include 1")

    print("\n--- Testing the no-chunks short-circuit (should skip the LLM entirely) ---")
    empty_result = generate_answer_metadata(QUESTION, "I don't know.", [])
    print(f"Confidence: {empty_result.confidence} (expected: low)")
    print(f"Cited chunk indices: {empty_result.cited_chunk_indices} (expected: [])")

    print("\nDone.")


if __name__ == "__main__":
    main()
