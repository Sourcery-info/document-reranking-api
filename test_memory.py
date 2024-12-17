import requests
import time
import numpy as np
import os
import argparse
import subprocess
import re

def get_gpu_memory():
    """Returns GPU memory usage in MB using nvidia-smi"""
    try:
        output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
            encoding='utf-8'
        )
        return float(output.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 0

def get_gpu_memory_details():
    """Returns detailed GPU memory information"""
    try:
        output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total,memory.free', '--format=csv,nounits,noheader'],
            encoding='utf-8'
        )
        used, total, free = map(float, output.strip().split(','))
        return used, total, free
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 0, 0, 0

def get_args():
    parser = argparse.ArgumentParser(description='Test memory usage of Document Reranking API')
    parser.add_argument('--host', 
                      default=os.environ.get('RERANK_HOST', 'localhost'),
                      help='API host (default: localhost)')
    parser.add_argument('--port', 
                      type=int,
                      default=int(os.environ.get('RERANK_PORT', '8000')),
                      help='API port (default: 8000)')
    parser.add_argument('--iterations',
                      type=int,
                      default=10,
                      help='Number of test iterations (default: 10)')
    return parser.parse_args()

def test_ranking_endpoint(host, port, num_iterations=10):
    url = f"http://{host}:{port}/rank"
    
    # Test data
    payload = {
        "question": "What is machine learning?",
        "documents": [
            "Machine learning is a branch of artificial intelligence.",
            "Deep learning is a subset of machine learning.",
            "Natural language processing uses machine learning.",
            "Python is a popular programming language for ML.",
            "Data science involves statistical analysis."
        ] * 5,  # Multiply documents to make it more memory-intensive
        "top_k": 3
    }

    print(f"Testing API at {url}")
    print("Starting memory test...")
    print(f"{'Iteration':^10} | {'GPU Used (MB)':^15} | {'GPU Free (MB)':^15} | {'Response Time (s)':^15}")
    print("-" * 65)

    for i in range(num_iterations):
        used, total, free = get_gpu_memory_details()
        
        start_time = time.time()
        response = requests.post(url, json=payload)
        request_time = time.time() - start_time
        
        used_after, _, free_after = get_gpu_memory_details()
        
        print(f"{i:^10} | {used_after:^15.0f} | {free_after:^15.0f} | {request_time:^15.2f}")
        
        # Small delay between requests
        time.sleep(1)

if __name__ == "__main__":
    args = get_args()
    test_ranking_endpoint(args.host, args.port, args.iterations) 