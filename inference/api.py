from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uuid
import os

from inference.pipeline import run_rag_pipeline
from scripts.build_vectorstore import build_dual_stores
from utils.language_checker import detect_lang

app = FastAPI(title="RAG Pipeline API")


# -------------------------------
# Request/Response Models
# -------------------------------

class QueryRequest(BaseModel):
    query: str
    model_type: str = "groq"  # groq | hf


class QueryResponse(BaseModel):
    answer: str


class BuildRequest(BaseModel):
    strategy: str = "semantic"  # paragraph | recursive | semantic


# -------------------------------
# /query Endpoint
# -------------------------------

@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    chat_id = str(uuid.uuid4())
    lang = detect_lang(request.query)

    answer = run_rag_pipeline(query=request.query, lang=lang, chat_id=chat_id, model_type=request.model_type)
    return {"answer": answer}


# -------------------------------
# /build-store Endpoint
# -------------------------------

@app.post("/build-store")
def build_vector_db(request: BuildRequest):
    # Define directory paths
    en_dir = "data/raw/english"
    bn_dir = "data/raw/bangla"

    # Check if directories exist
    if not os.path.exists(en_dir) or not os.path.exists(bn_dir):
        raise HTTPException(status_code=400, detail="English or Bangla directory not found")

    # Get all files from directories
    en_files = [os.path.join(en_dir, f) for f in os.listdir(en_dir) if os.path.isfile(os.path.join(en_dir, f))]
    bn_files = [os.path.join(bn_dir, f) for f in os.listdir(bn_dir) if os.path.isfile(os.path.join(bn_dir, f))]

    # Check if all files exist
    for f in en_files + bn_files:
        if not os.path.exists(f):
            raise HTTPException(status_code=400, detail=f"File not found: {f}")

    try:
        build_dual_stores(bn_files, en_files, strategy=request.strategy)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"message": f"Vector DBs created using '{request.strategy}' strategy."}
