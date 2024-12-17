from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict
import uvicorn
import argparse
import os
from reranker import get_reranker, rank_documents

app = FastAPI(title="Document Reranking API")

# Add API usage instructions
API_INSTRUCTIONS: Dict = {
    "description": "Document Reranking API - Ranks documents based on their relevance to a question",
    "endpoints": {
        "/": {
            "method": "GET",
            "description": "Returns these API usage instructions"
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
        }
    }
}

@app.get("/")
async def root():
    return JSONResponse(content=API_INSTRUCTIONS)

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

if __name__ == "__main__":
    args = get_args()
    print(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port) 