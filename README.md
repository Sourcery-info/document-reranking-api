# Document Reranking API

This is a simple API for reranking documents based on their relevance to a question.

## Version

The current version is managed in `__version__.py` using semantic versioning (MAJOR.MINOR.PATCH):
- MAJOR version for incompatible API changes
- MINOR version for added functionality in a backward compatible manner
- PATCH version for backward compatible bug fixes

The version is exposed in:
- API documentation at `/docs`
- Root endpoint `/` in the response
- FastAPI title

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

If you have multiple GPUs, you can select which GPU to use by setting the `CUDA_DEVICE` environment variable:

```bash
# Use GPU 0 (default)
CUDA_DEVICE=0 uvicorn api:app --reload --host 0.0.0.0 --port 8000

# Use GPU 1
CUDA_DEVICE=1 uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

You can verify the GPU being used by checking the `/healthz` endpoint, which will show:
- Whether GPU is available
- Total number of GPUs
- Currently selected GPU device
- GPU memory usage