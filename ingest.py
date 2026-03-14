"""
ingest.py

One-time script to:
1. Load documents from data/dataset.json
2. Generate vector embeddings using Sentence Transformers
3. Create an Endee index
4. Upsert all vectors into Endee

Run this once before starting the chatbot API:
    python ingest.py

Requirements:
- Endee server must be running at http://localhost:8080
  Start with: ./run.sh (inside the cloned endee repo)
"""

from embed_documents import load_documents, embed_documents
from endee_client import get_client, create_index, upsert_vectors


def main():
    print("=" * 50)
    print("  Endee RAG Research Assistant — Data Ingestion")
    print("=" * 50)

    # Step 1: Load documents
    print("\n[1/3] Loading documents...")
    documents = load_documents()
    print(f"      Loaded {len(documents)} documents.")

    # Step 2: Generate embeddings
    print("\n[2/3] Generating vector embeddings...")
    embedded_docs = embed_documents(documents)

    # Step 3: Connect to Endee and upsert vectors
    print("\n[3/3] Connecting to Endee and upserting vectors...")
    client = get_client()
    create_index(client, recreate=False)
    upsert_vectors(client, embedded_docs)

    print("\n✅ Ingestion complete! Knowledge base is ready in Endee.")
    print("   Start the API with: uvicorn app:app --reload")
    print("   Start the UI  with: streamlit run ui.py")


if __name__ == "__main__":
    main()
