Here's the reformatted README for your **NUBot** project:

````markdown
# NUBot: Retrieval-Augmented Generation (RAG) Chatbot

NUBot is an intelligent chatbot designed to assist students and visitors with queries related to Northeastern University, such as courses, faculty, co-op opportunities, and more. It utilizes a Retrieval-Augmented Generation (RAG) approach to provide instant, accurate responses.

## Prerequisites

Before setting up the project, install the Python debugger extension in VS Code:

- [Python debugger extension](https://marketplace.visualstudio.com/items?itemName=ms-python.debugpy)

## Features

- Instant responses to academic-related queries.
- Scalable and efficient system for handling high query volumes.
- Continuous updates via cloud deployment.

## Setup

1. Add dependencies in the `pyproject.toml` under the `_dependencies_` array.
2. Run the following command to install them:

   ```bash
   pip install .
   ```
````

## Backend

To run the backend service:

### Option 1: Using Python Command

1. Go to the root directory of **NUBot**.
2. Run the following command:

   ```bash
   python -m src.backend.api
   ```

### Option 2: For Linux/macOS (Terminal)

1. Open the terminal in the **NUBot** directory.
2. Set the environment variable and start the Flask server:

   ```bash
   export FLASK_APP=src.backend.api
   flask run
   ```

### Option 3: For Windows (Command Prompt)

1. Open the terminal in the **NUBot** directory.
2. Set the environment variable and start the Flask server:

   ```bash
   set FLASK_APP=src.backend.api
   flask run
   ```

### Option 4: Running via VS Code (Run and Debug)

1. Open **Run and Debug** in VS Code.
2. Click on the **Run** button to start the backend.

![Run and Debug](./assets/image.png)

The backend will now be running at [http://localhost:5000](http://localhost:5000).

```

### Key Updates:
- Organized the content for better readability.
- Consistently formatted code blocks and commands.
- Clearer headings for each section.
```
