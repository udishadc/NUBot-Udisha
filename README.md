# NUBotNUBot: Retrieval-Augmented Generation (RAG) Chatbot

NUBot is an intelligent chatbot designed to assist students and visitors with queries related to Northeastern University, such as courses, faculty, co-op opportunities, and more. It utilizes a Retrieval-Augmented Generation (RAG) approach to provide instant, accurate responses.

## Features

Instant responses to academic-related queries.

Scalable and efficient system for handling high query volumes.

Continuous updates via cloud deployment.

# setup

- add dependencies in **pyproject.toml** under _dependencies_ array
  run

```
pip install .
```

## Backend

- to run backend service

go to root dir **NUBot** then run

```python
python -m src.backend.api
```

now backend will run in [localhost:5000](localhost:5000)
