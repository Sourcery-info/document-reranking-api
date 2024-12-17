# Document Reranking API

This is a simple API for reranking documents based on their relevance to a question.

## Usage

To run the API, use the following command:

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

This will start the server on `http://0.0.0.0:8000`.

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