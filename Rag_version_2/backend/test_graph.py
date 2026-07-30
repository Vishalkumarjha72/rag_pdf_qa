"""
Isolated smoke test for the V2 conversation graph — no FastAPI involved.

Ingests a PDF (reusing V1's ingestion pipeline), then runs a short
multi-turn conversation against it using the SAME session_id, to verify
that a follow-up question correctly resolves using prior turns instead
of being treated as a standalone query (V1's behavior).

Run from inside backend/: python test_graph.py

Put at least one PDF in backend/books/ before running.
"""

import pathlib
import uuid

from app.ingestion import ingest_pdf, IngestionError
from app.graph import ask_with_memory

BOOKS_DIR = pathlib.Path(__file__).parent / "books"

# Edit these two to match whatever PDF you're testing with —
# the follow-up question should only make sense if the first
# turn's context was actually carried over.
FIRST_QUESTION = "What is this document about?"
FOLLOW_UP_QUESTION = "Can you say more about that?"


def get_first_namespace() -> str | None:
    pdf_paths = sorted(BOOKS_DIR.glob("*.pdf"))
    if not pdf_paths:
        print(f"No PDFs found in '{BOOKS_DIR}'. Add one to test.")
        return None

    pdf_path = pdf_paths[0]
    print(f"Ingesting: {pdf_path.name}")
    try:
        result = ingest_pdf(pdf_path.read_bytes(), filename=pdf_path.name)
        print(f"  Namespace: {result['namespace']}")
        return result["namespace"]
    except IngestionError as e:
        print(f"  Ingestion FAILED: {e}")
        return None


def run_conversation(namespace: str):
    session_id = str(uuid.uuid4())
    print(f"\nSession ID: {session_id}")

    print(f"\nTurn 1 — Question: {FIRST_QUESTION}")
    result_1 = ask_with_memory(FIRST_QUESTION, namespace, session_id)
    print(f"  Answer: {result_1['answer']}")
    print(f"  Sources used: {len(result_1['sources'])}")

    print(f"\nTurn 2 — Question: {FOLLOW_UP_QUESTION}")
    result_2 = ask_with_memory(FOLLOW_UP_QUESTION, namespace, session_id)
    print(f"  Answer: {result_2['answer']}")
    print(f"  Sources used: {len(result_2['sources'])}")

    print(
        "\nCheck manually: does Turn 2's answer make sense as a continuation "
        "of Turn 1, rather than a generic/confused response to 'that'?"
    )


if __name__ == "__main__":
    namespace = get_first_namespace()
    if namespace:
        run_conversation(namespace)
    print("\nDone.")
