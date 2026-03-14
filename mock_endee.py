"""
mock_endee.py

A lightweight FastAPI server that simulates the Endee Vector Database API.
Allows the project to run even if the C++ Endee server or Docker is unavailable.
"""

from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import numpy as np

app = FastAPI(title="Mock Endee Server")

# In-memory storage
indexes = {}

class QueryRequest(BaseModel):
    vector: List[float]
    top_k: int = 3
    filter: Optional[Dict[str, Any]] = None

@app.get("/v1/indexes")
def list_indexes():
    return [{"name": name} for name in indexes.keys()]

@app.post("/v1/indexes")
def create_index(payload: Dict[str, Any]):
    name = payload.get("name")
    if not name:
        raise HTTPException(status_code=400, detail="Name required")
    indexes[name] = []
    return {"status": "success", "message": f"Index {name} created"}

@app.delete("/v1/indexes/{name}")
def delete_index(name: str):
    if name in indexes:
        del indexes[name]
    return {"status": "success"}

@app.get("/v1/indexes/{name}")
def get_index(name: str):
    if name not in indexes:
        raise HTTPException(status_code=404, detail="Index not found")
    return {"name": name}

@app.post("/v1/indexes/{name}/upsert")
def upsert_vectors(name: str, items: List[Dict[str, Any]]):
    if name not in indexes:
        raise HTTPException(status_code=404, detail="Index not found")
    
    # Store items
    for item in items:
        # Update or append
        existing = [i for i in indexes[name] if i["id"] == item["id"]]
        if existing:
            indexes[name].remove(existing[0])
        indexes[name].append(item)
    
    return {"status": "success", "upserted_count": len(items)}

@app.post("/v1/indexes/{name}/query")
def query_index(name: str, request: QueryRequest):
    if name not in indexes:
        raise HTTPException(status_code=404, detail="Index not found")
    
    q_vec = np.array(request.vector)
    results = []
    
    for item in indexes[name]:
        i_vec = np.array(item["vector"])
        # Cosine similarity
        norm_q = np.linalg.norm(q_vec)
        norm_i = np.linalg.norm(i_vec)
        if norm_q > 0 and norm_i > 0:
            score = np.dot(q_vec, i_vec) / (norm_q * norm_i)
        else:
            score = 0.0
        
        results.append({
            "id": item["id"],
            "similarity": float(score),
            "meta": item.get("meta", {})
        })
    
    # Sort by similarity
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:request.top_k]

if __name__ == "__main__":
    print("🚀 Starting Mock Endee Server on http://localhost:8090")
    uvicorn.run(app, host="0.0.0.0", port=8090)
