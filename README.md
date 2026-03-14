# EndeeAI – RAG Research Assistant

> **AI-powered knowledge chatbot** built with the **Endee Vector Database**, Sentence Transformers, FastAPI, and Streamlit. Implements a full RAG (Retrieval-Augmented Generation) pipeline for semantic document retrieval and AI answer generation.

## Problem Statement

Large Language Models (LLMs) are powerful but have a key limitation — **they cannot access private documents, domain-specific knowledge, or real-time information**. Without external retrieval, LLMs hallucinate or return generic answers.

## Solution

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline that:

1. Converts documents into **dense vector embeddings** (384-dimensional)
2. **Stores** embeddings in the **Endee Vector Database** using the official Python SDK
3. **Retrieves** semantically relevant documents using cosine similarity search
4. **Generates** an AI answer grounded in the retrieved context

The result: an AI that can accurately answer questions using your own knowledge base.

---

## System Architecture

```
User Question
     ↓
Sentence Transformer (all-MiniLM-L6-v2)
     ↓    [384-dim dense vector]
Endee Vector Database  ←── upsert (ingest.py)
     ↓    [cosine similarity search, top-k]
Retrieved Documents (with meta + source)
     ↓
RAG Prompt Builder
     ↓
LLM (OpenAI / HuggingFace / Ollama)
     ↓
Answer + Sources → FastAPI + Streamlit UI
```

---

## How Endee is Used

Endee is used as the **core vector database** for this project:

| Operation | Endee SDK Call |
|-----------|---------------|
| Create index | `client.create_index(name, dimension=384, space_type="cosine", precision=Precision.INT8)` |
| Upsert vectors | `index.upsert([{id, vector, meta, filter}])` |
| Search | `index.query(vector=query_vec, top_k=3)` |

Endee's HNSW-based approximate nearest-neighbor search makes this significantly faster than brute-force cosine similarity — crucial for production-scale retrieval.

### Why Endee vs Other Vector DBs?

- 🚀 **High-performance**: Handles millions of vectors with millisecond latency
- 🔍 **Advanced filtering**: Filter by metadata fields (category, source, topic)
- 🔗 **Hybrid search**: Combines dense vector + sparse keyword retrieval
- 🛠️ **Simple SDK**: One-line Python client `Endee()` connects to the local server
- 🏗️ **Production-ready**: CPU-optimized (AVX2/AVX512/NEON), Docker-deployable

---

## Features

- ✅ Vector embeddings via Sentence Transformers (384-dim)
- ✅ Semantic document retrieval using Endee (cosine similarity)
- ✅ Full RAG pipeline with prompt construction
- ✅ REST API via FastAPI (`/chat`, `/search` endpoints)
- ✅ Chat UI via Streamlit (with session history & source inspection)
- ✅ Metadata filtering support (category, source)
- ✅ Easily extensible with OpenAI / HuggingFace LLM

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Vector Database | **Endee** (open-source, localhost:8080) |
| Embeddings | Sentence Transformers `all-MiniLM-L6-v2` |
| Backend API | FastAPI + Uvicorn |
| Chat UI | Streamlit |
| Language | Python 3.10+ |

---

## Project Structure

```
rag-research-assistant/
│
├── app.py               # FastAPI backend (/chat, /search)
├── ui.py                # Streamlit chat interface
├── rag_pipeline.py      # Core RAG logic (retrieve + generate)
├── embed_documents.py   # Embedding model + text → vector
├── endee_client.py      # Endee SDK wrapper (index, upsert, query)
├── ingest.py            # One-time ingestion script
├── data/
│   └── dataset.json     # Knowledge base documents
├── requirements.txt
└── README.md
```

---

## Setup & Run

### 1. Clone the Endee repo and start the server

```bash
git clone https://github.com/endee-io/endee.git
cd endee
chmod +x ./install.sh ./run.sh
./install.sh --release --avx2
./run.sh
```

The Endee server starts at: **http://localhost:8080**

### 2. Clone this project & install dependencies

```bash
# Inside the endee repo folder:
cd rag-research-assistant

pip install -r requirements.txt
```

### 3. Ingest documents into Endee

```bash
python ingest.py
```

This loads `data/dataset.json`, generates embeddings, and upserts vectors into Endee.

### 4. Start the API

```bash
uvicorn app:app --reload
```

API available at: **http://localhost:8000**

### 5. Start the Streamlit UI

```bash
streamlit run ui.py
```

---

## API Reference

### `GET /chat`
Full RAG answer with retrieved documents.

```
GET http://localhost:8000/chat?q=What+is+RAG?
```

Response:
```json
{
  "response": {
    "question": "What is RAG?",
    "answer": "Based on retrieved documents: RAG combines vector search...",
    "retrieved_documents": [
      {
        "text": "Retrieval-Augmented Generation (RAG) combines vector search...",
        "source": "AI Research",
        "similarity": 0.9231
      }
    ]
  }
}
```

### `GET /search`
Raw semantic search results from Endee.

```
GET http://localhost:8000/search?q=vector+database&top_k=3
```

---

## Example Queries

| Question | Retrieved Context |
|----------|-----------------|
| "What is RAG?" | RAG pipelines, LLM generation |
| "How does Endee work?" | Endee HNSW indexing, cosine search |
| "What is semantic search?" | Semantic search, vector similarity |
| "What are foxnuts?" | Foxnut nutrition and health benefits |

---

## Extending with an LLM

Replace the placeholder in `rag_pipeline.py` with:

```python
# OpenAI example
import openai
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": prompt}]
)
answer = response.choices[0].message.content
```

Or use a local model via Ollama / HuggingFace.

---

## License

Apache 2.0 — Built on [Endee](https://github.com/endee-io/endee) (Apache 2.0)
