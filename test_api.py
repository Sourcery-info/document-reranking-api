import pytest
from fastapi.testclient import TestClient
from api import app, API_INSTRUCTIONS
from reranker import rank_documents

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