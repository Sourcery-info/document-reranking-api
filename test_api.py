import pytest
from fastapi.testclient import TestClient
from api import app, API_INSTRUCTIONS
from reranker import rank_documents
from unittest.mock import patch

client = TestClient(app)

# Test data
TEST_QUESTION = "What is a panda?"
TEST_DOCUMENTS = [
    "The giant panda is a bear native to China.",
    "Python is a programming language.",
    "Pandas eat bamboo as their main food source."
]

def test_root_endpoint():
    """Test the root endpoint returns API instructions"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == API_INSTRUCTIONS

def test_healthz_endpoint():
    """Test the health check endpoint"""
    response = client.get("/healthz")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_status" in data
    assert "gpu_info" in data
    
    # Check model status structure
    model_status = data["model_status"]
    assert "loaded" in model_status
    assert "type" in model_status
    assert model_status["type"] == "BAAI/bge-reranker-v2-gemma"
    
    # Check GPU info structure
    gpu_info = data["gpu_info"]
    assert "gpu_available" in gpu_info
    assert "gpu_count" in gpu_info
    assert "current_device" in gpu_info
    assert "current_device_id" in gpu_info
    assert "selected_device" in gpu_info
    assert "memory_allocated" in gpu_info

def test_test_endpoint():
    """Test the test endpoint with predefined example"""
    response = client.get("/test")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "success"
    assert "test_results" in data
    
    results = data["test_results"]
    assert results["question"] == "What is a panda?"
    assert "ranked_documents" in results
    assert "execution_time" in results
    
    # Check ranked documents
    ranked_docs = results["ranked_documents"]
    assert len(ranked_docs) == 3  # Should return all test documents
    assert all("document" in doc and "score" in doc for doc in ranked_docs)
    
    # Verify scores are in descending order
    scores = [doc["score"] for doc in ranked_docs]
    assert scores == sorted(scores, reverse=True)
    
    # Verify panda-related documents rank higher than Python
    python_doc = next(
        doc for doc in ranked_docs 
        if "Python is a programming language" in doc["document"]
    )
    panda_docs = [
        doc for doc in ranked_docs 
        if "panda" in doc["document"].lower()
    ]
    assert all(doc["score"] > python_doc["score"] for doc in panda_docs)

def test_rank_endpoint_success():
    """Test successful document ranking"""
    request_data = {
        "question": TEST_QUESTION,
        "documents": TEST_DOCUMENTS,
        "top_k": 2
    }
    
    response = client.post("/rank", json=request_data)
    assert response.status_code == 200
    
    data = response.json()
    assert "ranked_documents" in data
    assert "execution_time" in data
    assert len(data["ranked_documents"]) == 2
    
    # Check response structure
    first_doc = data["ranked_documents"][0]
    assert "document" in first_doc
    assert "score" in first_doc
    assert isinstance(first_doc["score"], float)

def test_rank_endpoint_validation():
    """Test input validation"""
    # Test empty documents
    response = client.post("/rank", json={
        "question": TEST_QUESTION,
        "documents": [],
        "top_k": 2
    })
    assert response.status_code == 400
    assert "No documents provided" in response.json()["detail"]
    
    # Test invalid top_k
    response = client.post("/rank", json={
        "question": TEST_QUESTION,
        "documents": TEST_DOCUMENTS,
        "top_k": 0
    })
    assert response.status_code == 400
    assert "top_k must be >= 1" in response.json()["detail"]
    
    # Test top_k larger than documents
    response = client.post("/rank", json={
        "question": TEST_QUESTION,
        "documents": TEST_DOCUMENTS,
        "top_k": 5
    })
    assert response.status_code == 200
    assert len(response.json()["ranked_documents"]) == len(TEST_DOCUMENTS)

def test_reranker_function():
    """Test the underlying reranker function"""
    ranked_docs, execution_time = rank_documents(
        question=TEST_QUESTION,
        documents=TEST_DOCUMENTS,
        top_k=2
    )
    
    assert len(ranked_docs) == 2
    assert execution_time > 0
    assert all(hasattr(doc, 'document') and hasattr(doc, 'score') for doc in ranked_docs)
    
    # Check if scores are in descending order
    scores = [doc.score for doc in ranked_docs]
    assert scores == sorted(scores, reverse=True)

def test_rank_endpoint_relevance():
    """Test if the ranking makes semantic sense"""
    request_data = {
        "question": TEST_QUESTION,
        "documents": TEST_DOCUMENTS,
        "top_k": 3
    }
    
    response = client.post("/rank", json=request_data)
    assert response.status_code == 200
    
    ranked_docs = response.json()["ranked_documents"]
    
    # The panda-related documents should rank higher than the Python document
    python_doc = "Python is a programming language."
    panda_docs_ranks = [
        i for i, doc in enumerate(ranked_docs) 
        if doc["document"] != python_doc
    ]
    python_doc_rank = next(
        i for i, doc in enumerate(ranked_docs) 
        if doc["document"] == python_doc
    )
    
    assert all(panda_rank < python_doc_rank for panda_rank in panda_docs_ranks) 

def test_unload_endpoint():
    """Test the model unload endpoint"""
    with patch('api.unload_reranker') as mock_unload:
        response = client.get("/unload")
        
        # Check response
        assert response.status_code == 200
        assert response.json() == {
            "status": "success",
            "message": "Model unloaded from memory"
        }
        
        # Verify unload_reranker was called
        mock_unload.assert_called_once()