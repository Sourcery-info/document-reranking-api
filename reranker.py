from FlagEmbedding import FlagLLMReranker
import time
from typing import List, Tuple
from pydantic import BaseModel
import torch
import gc
import os
import logging
import sys

# Add at top of file after imports
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

# Create a global reranker instance that can be reused
global_reranker = None

class RankedDocument(BaseModel):
    document: str
    score: float

def get_reranker():
    global global_reranker
    if global_reranker is None:
        try:
            logging.info("Initializing reranker...")
            # Add detailed CUDA diagnostics
            if (os.environ.get('RERANKER_DEBUG') == 'true'):
                logging.info(f"PyTorch version: {torch.__version__}")
                logging.info(f"CUDA available: {torch.cuda.is_available()}")
                logging.info(f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
                logging.info(f"CUDA device count: {torch.cuda.device_count()}")
                logging.info(f"Current CUDA device: {torch.cuda.current_device()}")
                logging.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
            
            # Get selected CUDA device
            device = None
            if torch.cuda.is_available():
                cuda_device = os.environ.get('CUDA_DEVICE')
                if cuda_device is not None:
                    try:
                        device_id = int(cuda_device)
                        if device_id >= 0 and device_id < torch.cuda.device_count():
                            device = f'cuda:{device_id}'
                            logging.info(f"Using specified CUDA device: {device}")
                    except ValueError:
                        logging.warning(f"Invalid CUDA_DEVICE value: {cuda_device}")
                if device is None:
                    device = 'cuda:0'
                    logging.info("Using default CUDA device: cuda:0")
            else:
                device = 'cpu'
                logging.info("CUDA not available, using CPU")

            logging.info(f"Loading model on device: {device}")
            global_reranker = FlagLLMReranker(
                os.environ.get('RERANKER_MODEL', 'BAAI/bge-reranker-v2-gemma'),
                use_fp16=True,
                cache_dir='./model_cache',
                device=device
            )
            logging.info("Reranker initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing reranker: {str(e)}", exc_info=True)
            raise
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
        logging.info(f"Ranking {len(documents)} documents")
        # Prepare pairs for ranking
        pairs = [[question, doc] for doc in documents]
        
        # Get reranker and compute scores
        reranker = get_reranker()
        logging.info("Computing scores...")
        scores = reranker.compute_score(pairs)
        
        # Create document-score pairs and sort
        doc_scores = list(zip(documents, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Take top_k results and convert to RankedDocument objects
        ranked_documents = [
            RankedDocument(document=doc, score=float(score))
            for doc, score in doc_scores[:top_k]
        ]
        
        execution_time = time.time() - start_time
        logging.info(f"Ranking completed in {execution_time:.2f} seconds")
        return ranked_documents, execution_time
        
    except Exception as e:
        logging.error(f"Error during document ranking: {str(e)}", exc_info=True)
        raise
    finally:
        # Clear CUDA cache after each ranking operation
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                gc.collect()
            except Exception as e:
                logging.error(f"Error clearing CUDA cache: {str(e)}")