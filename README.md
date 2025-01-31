# Document Reranking API

This is a simple API for reranking documents based on their relevance to a question.

## Setup

You can run locally using either Conda or Pip. Alternatively, you can use Docker.

## Requirements

An NVIDIA GPU with CUDA drivers installed is suggested. CUDA Toolkit and Nvidia GDS are required to use the GPU.

### Conda

```bash
conda env create -f environment.yml
```

Or, if you need to update the environment:

```bash
conda env update -f environment.yml
```

### Pip

```bash
pip install -r requirements.txt
```

### Docker

```bash
docker build -t document-reranking-api .
docker run -p 8000:8000 document-reranking-api
```

## Setup

### Conda

You can install the dependencies using conda:

```bash
conda env create -f environment.yml
conda activate document-reranking-api
```

## Usage

To run the API, use the following command:

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

This will start the server on `http://0.0.0.0:8000`.

## Testing

To run the tests, first install the test dependencies:

```bash
pip install pytest pytest-asyncio
```

Then run the tests using one of these commands:

```bash
# Run all tests
pytest

# Run tests with output
pytest -v

# Run tests with output and print statements
pytest -v -s

# Run a specific test file
pytest test_api.py -v

# Run a specific test function
pytest test_api.py::test_healthz_endpoint -v
```

## API Documentation

The API documentation is available at `http://localhost:8000/docs`.

## Example Usage

To use the API, send a POST request to `http://localhost:8000/rank` with the following JSON body:

```json
{
  "question": "What is a panda?",
  "documents": ["The giant panda is a bear native to China.", "Python is a programming language.", "Pandas eat bamboo as their main food source."],
  "top_k": 2
}
```

## Quick Test

There is a `/test` endpoint that will test the API with a sample question and documents.

## GPU Selection

If you have multiple GPUs, you can select which GPU to use by setting the `CUDA_VISIBLE_DEVICES` environment variable:

```bash
# Use GPU 0
CUDA_VISIBLE_DEVICES=0 uvicorn api:app --reload --host 0.0.0.0 --port 8000

# Use GPU 1
CUDA_VISIBLE_DEVICES=1 uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

You can verify the GPU being used by checking the `/healthz` endpoint, which will show:
- Whether GPU is available
- Total number of GPUs
- Currently selected GPU device
- GPU memory usage

## Environment Variables

The reranker component can be configured using the following environment variables:

### RERANKER_MODEL
- **Description**: Specifies which model to use for reranking
- **Default**: `BAAI/bge-reranker-v2-gemma`
- **Example**: 
  ```bash
  export RERANKER_MODEL="BAAI/bge-reranker-large-v1.5"
  ```

### RERANKER_DEBUG
- **Description**: Enables detailed CUDA diagnostics logging when set to 'true'
- **Default**: Not enabled
- **Example**:
  ```bash
  export RERANKER_DEBUG=true
  ```
  When enabled, logs will include:
  - PyTorch version
  - CUDA availability
  - CUDA version
  - Number of CUDA devices
  - Current CUDA device
  - CUDA device name

### CUDA_DEVICE
- **Description**: Specifies which GPU device to use for the reranker
- **Default**: Uses first available GPU (cuda:0)
- **Example**:
  ```bash
  export CUDA_DEVICE=1  # Use second GPU
  ```

  ### CUDA_VISIBLE_DEVICES
- **Description**: Specifies which GPUs to make available to the reranker
- **Default**: All available GPUs
- **Example**:
  ```bash
  export CUDA_VISIBLE_DEVICES=0,1  # Use GPUs 0 and 1
  ```
