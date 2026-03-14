"""
embed_documents.py

Loads documents from dataset.json and generates dense vector embeddings
using the Sentence Transformers model all-MiniLM-L6-v2.
The model produces 384-dimensional vectors, compatible with Endee's default index.
"""

import json
from sentence_transformers import SentenceTransformer

# Load the sentence transformer model (downloads once, cached thereafter)
model = SentenceTransformer("all-MiniLM-L6-v2")

VECTOR_DIMENSION = 384  # all-MiniLM-L6-v2 produces 384-dim vectors


def load_documents(path: str = "data/dataset.json") -> list[dict]:
    """Load documents from the JSON dataset file."""
    with open(path, "r") as f:
        return json.load(f)


def embed_text(text: str) -> list[float]:
    """Convert a single text string into a vector embedding."""
    return model.encode(text).tolist()


def embed_documents(documents: list[dict]) -> list[dict]:
    """
    Generate embeddings for all documents.
    Returns a list of records ready to upsert into Endee.
    """
    print(f"Generating embeddings for {len(documents)} documents...")
    embedded = []
    for doc in documents:
        vector = embed_text(doc["text"])
        embedded.append({
            "id": doc["id"],
            "vector": vector,
            "meta": {
                "text": doc["text"],
                "source": doc.get("source", ""),
            },
            "filter": {
                "category": doc.get("category", "general"),
            }
        })
    print("Embeddings generated successfully.")
    return embedded


if __name__ == "__main__":
    docs = load_documents()
    embedded = embed_documents(docs)
    print(f"Sample embedding length: {len(embedded[0]['vector'])}")
    print(f"Sample document: {embedded[0]['meta']['text'][:60]}...")
