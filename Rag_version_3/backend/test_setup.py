"""
Combined smoke test for everything built so far in V1:
  Step 1 - config
  Step 2 - vectorstore (Pinecone)
  Step 3 - embeddings
  Step 4 - ingestion
  Step 5 - retrieval

Run from inside backend/: python test_setup.py

Put any PDF(s) you want to test with inside backend/books/ before running.
"""

import pathlib

from app.config import settings
from app.vectorstore import ensure_index_exists, get_index
from app.embeddings import embed_text
from app.ingestion import ingest_pdf, IngestionError
from app.retrieval import answer_question, RetrievalError

BOOKS_DIR = pathlib.Path(__file__).parent / "books"
TEST_QUESTION = "What is this document about?"  # change this to something specific to your PDF(s)


def test_config():
    print("1. Config")
    print(f"   Pinecone index name: {settings.pinecone_index_name}")
    print(f"   Embedding model: {settings.embedding_model_name}")
    print(f"   Expected dimension: {settings.embedding_dimension}")


def test_vectorstore():
    print("\n2. Vectorstore (Pinecone)")
    ensure_index_exists()
    index = get_index()
    stats = index.describe_index_stats()
    print(f"   Connected. Current vector count: {stats['total_vector_count']}")


def test_embeddings():
    print("\n3. Embeddings")
    vector = embed_text("This is a test sentence.")
    print(f"   Got vector of length {len(vector)}")
    print(f"   Matches expected dimension: {len(vector) == settings.embedding_dimension}")


def test_ingestion() -> list[str]:
    """Ingests every PDF in books/ and returns the list of namespaces created."""
    print("\n4. Ingestion")

    namespaces = []

    if not BOOKS_DIR.exists():
        print(f"   '{BOOKS_DIR}' does not exist — create it and add PDFs to test ingestion.")
        return namespaces

    pdf_paths = sorted(BOOKS_DIR.glob("*.pdf"))

    if not pdf_paths:
        print(f"   No PDFs found in '{BOOKS_DIR}'. Add one or more .pdf files to test ingestion.")
        return namespaces

    for pdf_path in pdf_paths:
        print(f"\n   Ingesting: {pdf_path.name}")
        file_bytes = pdf_path.read_bytes()
        try:
            result = ingest_pdf(file_bytes, filename=pdf_path.name)
            print(f"     Namespace: {result['namespace']}")
            print(f"     Chunks indexed: {result['chunks_indexed']}")
            namespaces.append(result["namespace"])
        except IngestionError as e:
            print(f"     FAILED: {e}")

    return namespaces


def test_retrieval(namespaces: list[str]):
    print("\n5. Retrieval")

    if not namespaces:
        print("   No namespaces to test against (ingestion produced none). Skipping.")
        return

    for namespace in namespaces:
        print(f"\n   Namespace: {namespace}")
        print(f"   Question: {TEST_QUESTION}")
        try:
            result = answer_question(TEST_QUESTION, namespace)
            print(f"   Answer: {result['answer']}")
            print(f"   Sources used: {len(result['sources'])}")
            for src in result["sources"]:
                print(f"     - {src['source']} (page {src['page']}, score {src['score']:.4f})")
        except RetrievalError as e:
            print(f"   FAILED: {e}")


if __name__ == "__main__":
    test_config()
    test_vectorstore()
    test_embeddings()
    namespaces = test_ingestion()
    test_retrieval(namespaces)
    print("\nAll checks complete.")