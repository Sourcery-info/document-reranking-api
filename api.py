from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict
import uvicorn
import argparse
import os
from reranker import get_reranker, rank_documents, unload_reranker
import torch
from __version__ import __version__
import logging
import sys

# Set CUDA device if specified in environment
if torch.cuda.is_available():
    cuda_device = os.environ.get('CUDA_DEVICE')
    if cuda_device is not None:
        try:
            device_id = int(cuda_device)
            if device_id >= 0 and device_id < torch.cuda.device_count():
                torch.cuda.set_device(device_id)
            else:
                print(f"Warning: Invalid CUDA_DEVICE {device_id}. Using default device.")
        except ValueError:
            print(f"Warning: Invalid CUDA_DEVICE value '{cuda_device}'. Using default device.")

app = FastAPI(title="Document Reranking API", version=__version__)

# Add API usage instructions
API_INSTRUCTIONS: Dict = {
    "description": "Document Reranking API - Ranks documents based on their relevance to a question",
    "version": __version__,
    "endpoints": {
        "/": {
            "method": "GET",
            "description": "Returns these API usage instructions"
        },
        "/healthz": {
            "method": "GET",
            "description": "Health check endpoint that returns server status"
        },
        "/test": {
            "method": "GET",
            "description": "Runs a predefined test reranking example to verify functionality"
        },
        "/rank": {
            "method": "POST",
            "description": "Ranks documents based on relevance to a question",
            "request_body": {
                "question": "string - The question to match documents against",
                "documents": "array of strings - The documents to rank",
                "top_k": "integer (optional, default=3) - Number of top matches to return"
            },
            "response": {
                "ranked_documents": "array of objects containing document text and relevance score",
                "execution_time": "float - Time taken to process the request in seconds"
            },
            "example_request": {
                "question": "What is a panda?",
                "documents": [
                    "The giant panda is a bear native to China.",
                    "Python is a programming language.",
                    "Pandas eat bamboo as their main food source."
                ],
                "top_k": 2
            }
        },
        "/unload": {
            "method": "GET",
            "description": "Unloads the reranker model from GPU memory"
        }
    }
}

@app.get("/")
async def root():
    return JSONResponse(content=API_INSTRUCTIONS)

@app.get("/healthz")
async def health_check():
    """Health check endpoint that verifies server and model status"""
    try:
        # Check if model is loaded/can be loaded
        reranker = get_reranker()
        model_loaded = reranker is not None
        
        # Get GPU information
        gpu_info = {
            "gpu_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "current_device": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "current_device_id": torch.cuda.current_device() if torch.cuda.is_available() else None,
            "selected_device": os.environ.get('CUDA_DEVICE', 'default'),
            "memory_allocated": f"{torch.cuda.memory_allocated() / 1024**2:.2f}MB" if torch.cuda.is_available() else None
        }
        
        return {
            "status": "healthy",
            "model_status": {
                "loaded": model_loaded,
                "type": "BAAI/bge-reranker-v2-gemma"
            },
            "gpu_info": gpu_info
        }
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Health check failed: {str(e)}"
        )

@app.get("/test")
async def test_reranking():
    """Test endpoint that runs a predefined reranking example"""
    try:
        test_question = "What is a panda?"
        test_documents = [
            "The giant panda is a bear native to China.",
            "Python is a programming language.",
            "Pandas eat bamboo as their main food source."
        ]
        
        ranked_docs, execution_time = rank_documents(
            question=test_question,
            documents=test_documents,
            top_k=3
        )
        
        return {
            "status": "success",
            "test_results": {
                "question": test_question,
                "ranked_documents": [
                    {"document": doc.document, "score": doc.score}
                    for doc in ranked_docs
                ],
                "execution_time": execution_time
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Test reranking failed: {str(e)}"
        )

class RankingRequest(BaseModel):
    question: str
    documents: List[str]
    top_k: int = 3  # default to top 3

class RankedDocument(BaseModel):
    document: str
    score: float

class RankingResponse(BaseModel):
    ranked_documents: List[RankedDocument]
    execution_time: float

@app.post("/rank", response_model=RankingResponse)
async def rank_documents_endpoint(request: RankingRequest):
    if not request.documents:
        raise HTTPException(status_code=400, detail="No documents provided")
    if request.top_k < 1:
        raise HTTPException(status_code=400, detail="top_k must be >= 1")
    if request.top_k > len(request.documents):
        request.top_k = len(request.documents)
    
    ranked_docs, execution_time = rank_documents(
        question=request.question,
        documents=request.documents,
        top_k=request.top_k
    )
    
    # Convert RankedDocument objects to dictionaries
    ranked_docs_dicts = [
        {"document": doc.document, "score": doc.score}
        for doc in ranked_docs
    ]
    
    return RankingResponse(
        ranked_documents=ranked_docs_dicts,
        execution_time=execution_time
    )

@app.get("/unload")
async def unload_model():
    """Unload the model from GPU memory"""
    unload_reranker()
    return {"status": "success", "message": "Model unloaded from memory"}

def get_args():
    parser = argparse.ArgumentParser(description='Document Reranking API Server')
    parser.add_argument('--host', 
                      default=os.environ.get('RERANK_HOST', '0.0.0.0'),
                      help='Host to run the server on (default: 0.0.0.0)')
    parser.add_argument('--port', 
                      type=int,
                      default=int(os.environ.get('RERANK_PORT', '8000')),
                      help='Port to run the server on (default: 8000)')
    return parser.parse_args()

# Add after imports
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

# Update the startup code at bottom of file
if __name__ == "__main__":
    args = get_args()
    try:
        logging.info(f"Starting server on {args.host}:{args.port}")
        logging.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logging.info(f"CUDA devices: {torch.cuda.device_count()}")
            logging.info(f"Current CUDA device: {torch.cuda.current_device()}")
            logging.info(f"Device name: {torch.cuda.get_device_name()}")
        uvicorn.run(app, host=args.host, port=args.port)
    except Exception as e:
        logging.error(f"Server failed to start: {str(e)}", exc_info=True)
        sys.exit(1) 