"""
rag_pipeline.py

Core RAG (Retrieval-Augmented Generation) pipeline.

Flow:
  User Question
      ↓
  Embed question → 384-dim vector
      ↓
  Query Endee vector database (semantic similarity search)
      ↓
  Retrieve top-k most relevant documents
      ↓
  Build prompt with retrieved context
      ↓
  Generate answer (currently template-based; replace with LLM for production)
"""

from embed_documents import embed_text
from endee_client import get_client, search_vectors


def retrieve_context(question: str, top_k: int = 3) -> list[dict]:
    """
    Embed the question and retrieve the most similar documents from Endee.
    Returns a list of result dicts with id, similarity, and meta.
    """
    query_vector = embed_text(question)
    client = get_client()
    results = search_vectors(client, query_vector, top_k=top_k)
    return results


def generate_answer(question: str, top_k: int = 3) -> dict:
    """
    Full RAG pipeline:
    1. Retrieve relevant context from Endee
    2. Build a prompt
    3. Return a structured response (replace prompt generation with LLM call)
    """
    results = retrieve_context(question, top_k=top_k)

    # Extract text from retrieved documents' metadata
    context_docs = []
    for r in results:
        meta = r.get("meta") or {}
        text = meta.get("text", "")
        source = meta.get("source", "")
        similarity = round(float(r.get("similarity", 0)), 4)
        context_docs.append({
            "text": text,
            "source": source,
            "similarity": similarity,
        })

    context_text = "\n".join([f"- {d['text']}" for d in context_docs])

    # Build prompt (wire in OpenAI / HuggingFace / Ollama here for production)
    prompt = f"""You are an AI research assistant.
Answer the question using ONLY the context provided below.
If the context does not contain the answer, say "I don't have enough information."

Context:
{context_text}

Question: {question}

Answer:"""

    # Placeholder answer — replace with: openai.ChatCompletion.create(prompt=prompt, ...)
    answer = (
        f"Based on the retrieved documents:\n\n{context_text}\n\n"
        f"[Plug in OpenAI / HuggingFace LLM here for a natural language answer]"
    )

    return {
        "question": question,
        "answer": answer,
        "retrieved_documents": context_docs,
    }
