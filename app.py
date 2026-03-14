"""
app.py

FastAPI backend for the Endee RAG Research Assistant.
Exposes two endpoints:
  GET /chat?q=<question>           — full RAG answer with retrieved context
  GET /search?q=<query>&top_k=<n> — raw semantic search results from Endee

Run:
    uvicorn app:app --reload

Then test:
    http://localhost:8000/chat?q=what+is+RAG
    http://localhost:8000/search?q=vector+database
"""

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from rag_pipeline import generate_answer, retrieve_context

app = FastAPI(
    title="Endee RAG Research Assistant",
    description="A Retrieval-Augmented Generation chatbot powered by Endee Vector Database.",
    version="1.0.0",
)

# Allow Streamlit UI to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "message": "Endee RAG Research Assistant is running.",
        "endpoints": {
            "chat": "/chat?q=your+question",
            "search": "/search?q=your+query&top_k=3",
        }
    }


@app.get("/chat")
def chat(
    q: str = Query(..., description="The question to ask the AI assistant"),
    top_k: int = Query(3, description="Number of results to retrieve"),
):
    """
    Full RAG pipeline:
    1. Embed the question
    2. Search Endee vector database for relevant documents
    3. Return the answer with retrieved context
    """
    result = generate_answer(q, top_k=top_k)
    return {"response": result}


@app.get("/search")
def search(
    q: str = Query(..., description="Search query"),
    top_k: int = Query(3, description="Number of results to return"),
):
    """
    Semantic search endpoint — returns raw Endee search results.
    Useful for exploring what the vector database retrieves.
    """
    results = retrieve_context(q, top_k=top_k)
    return {"query": q, "results": results}
