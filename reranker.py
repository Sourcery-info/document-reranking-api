from FlagEmbedding import FlagLLMReranker
import time
from typing import List, Tuple
from pydantic import BaseModel
import torch
import gc

# Create a global reranker instance that can be reused
global_reranker = None

class RankedDocument(BaseModel):
    document: str
    score: float

def get_reranker():
    global global_reranker
    if global_reranker is None:
        global_reranker = FlagLLMReranker(
            'BAAI/bge-reranker-v2-gemma',
            use_fp16=True,
            cache_dir='./model_cache'
        )
    return global_reranker

def unload_reranker():
    """Unload the reranker model and clear GPU memory"""
    global global_reranker
    if global_reranker is not None:
        del global_reranker
        global_reranker = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

def rank_documents(
    question: str,
    documents: List[str],
    top_k: int
) -> Tuple[List[RankedDocument], float]:
    """
    Rank documents based on their relevance to the question.
    
    Args:
        question: The query question
        documents: List of documents to rank
        top_k: Number of top documents to return
        
    Returns:
        Tuple containing:
        - List of RankedDocument objects
        - Execution time in seconds
    """
    start_time = time.time()
    
    try:
        # Prepare pairs for ranking
        pairs = [[question, doc] for doc in documents]
        
        # Get reranker and compute scores
        reranker = get_reranker()
        scores = reranker.compute_score(pairs)
        
        # Create document-score pairs and sort
        doc_scores = list(zip(documents, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Take top_k results and convert to RankedDocument objects
        ranked_documents = [
            RankedDocument(document=doc, score=float(score))
            for doc, score in doc_scores[:top_k]
        ]
        
        return ranked_documents, time.time() - start_time
        
    finally:
        # Clear CUDA cache after each ranking operation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()  # Force Python garbage collection