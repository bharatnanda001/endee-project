"""
endee_client.py

Manages all interactions with the Endee vector database using the official Python SDK.
Handles index creation, vector upsert, and semantic search queries.

Endee SDK docs: https://docs.endee.io/python-sdk/quickstart
Server runs at: http://localhost:8080
"""

from endee import Endee, Precision

INDEX_NAME = "rag_knowledge_base"
VECTOR_DIMENSION = 384  # Matches all-MiniLM-L6-v2 output


def get_client() -> Endee:
    """Initialize and return the Endee client (connects to localhost:8090)."""
    return Endee(url="http://localhost:8090")


def create_index(client: Endee, recreate: bool = False) -> None:
    """
    Create the vector index in Endee.
    If recreate=True, drops the existing index and rebuilds.
    """
    # Check if index already exists
    existing_indexes = client.list_indexes()
    index_names = [idx.get("name") for idx in existing_indexes] if existing_indexes else []

    if INDEX_NAME in index_names:
        if recreate:
            print(f"Dropping existing index '{INDEX_NAME}'...")
            client.delete_index(name=INDEX_NAME)
        else:
            print(f"Index '{INDEX_NAME}' already exists. Skipping creation.")
            return

    print(f"Creating index '{INDEX_NAME}' (dimension={VECTOR_DIMENSION}, cosine, INT8)...")
    client.create_index(
        name=INDEX_NAME,
        dimension=VECTOR_DIMENSION,
        space_type="cosine",
        precision=Precision.INT8,
    )
    print("Index created successfully.")


def upsert_vectors(client: Endee, embedded_docs: list[dict]) -> None:
    """
    Insert or update vectors into the Endee index.
    Each document includes: id, vector, meta (text, source), filter (category).
    """
    index = client.get_index(name=INDEX_NAME)
    print(f"Upserting {len(embedded_docs)} vectors into Endee...")
    index.upsert(embedded_docs)
    print("Vectors upserted successfully.")


def search_vectors(client: Endee, query_vector: list[float], top_k: int = 3) -> list[dict]:
    """
    Search the Endee index for the most similar vectors to the query.
    Returns a list of results with id and similarity score.
    """
    index = client.get_index(name=INDEX_NAME)
    results = index.query(vector=query_vector, top_k=top_k)
    return results
