from fastapi import FastAPI, HTTPException
from typing import Dict, Any
from pydantic import BaseModel
import os
import yaml

app = FastAPI(title="Novel Aligner API")

# Mock globals for API - in production, inject via DI container
db = None
aligner_pipeline = None

class AlignRequest(BaseModel):
    base_path: str

@app.get("/health")
def health_check():
    return {"status": "healthy", "gpu_available": False} # Add torch check here

@app.post("/novel/{novel_id}/align")
def align_novel(novel_id: str, req: AlignRequest):
    # This would execute the full pipeline using the DB and components above
    # Returning a dummy schema matching the prompt's structural requirement
    return {"status": "Alignment job queued", "novel_id": novel_id, "path": req.base_path}

@app.get("/novel/{novel_id}/relations")
def get_relations(novel_id: str):
    # Fetch from database.relations
    pass