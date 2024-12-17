import os

# Test configuration
TEST_HOST = os.getenv('TEST_RERANK_HOST', 'localhost')
TEST_PORT = int(os.getenv('TEST_RERANK_PORT', '8000'))

# Test data
TEST_CASES = [
    {
        "question": "What is a panda?",
        "documents": [
            "The giant panda is a bear native to China.",
            "Python is a programming language.",
            "Pandas eat bamboo as their main food source."
        ],
        "expected_order": [0, 2, 1]  # Expected indices in order of relevance
    }
    # Add more test cases as needed
] 