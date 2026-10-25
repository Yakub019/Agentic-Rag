# Agentic RAG

A small agentic retrieval-augmented generation (RAG) demo that reads WebVTT
(`.vtt`) transcript files, splits them into text chunks, indexes them with
Chroma, retrieves relevant chunks for a question, and asks a local Ollama model
to answer using the retrieved context.

## How it works

1. `documents_loader.py` reads every `.vtt` file in `data/`, removes the WebVTT
   header and timestamps, and splits the transcript text into chunks.
2. `main.py` embeds and indexes those chunks in a Chroma vector store, then
   retrieves the three most relevant chunks for the sample question.
3. A LangGraph workflow passes the retrieved context to the local `Qwen3:1.7b`
   model through Ollama. The model is instructed to say it does not know when
   the answer is not in the context.
4. The script prints the answer and the retrieved transcript excerpts.

The vector store is created in memory each time the script runs; it is not
persisted between runs.

## Requirements

- Python 3.13 or newer
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/) installed and running locally
- The Ollama model used by the app:

  ```bash
  ollama pull Qwen3:1.7b
  ```

## Setup

Clone the repository and open its directory:

```bash
git clone https://github.com/Yakub019/Agentic-Rag.git
cd Agentic-Rag
```

Install the project dependencies with uv:

```bash
uv sync
```

Make sure Ollama is running and the model is available:

```bash
ollama serve
ollama list
```

If Ollama is already running as a background service, you do not need to run
`ollama serve` again.

## Run

Run the original retrieval-and-answer workflow:

```bash
uv run python main.py
```

The script currently asks the example question
`What is the Regression Testing?`. To ask a different question, update the
`test_query` value in `main.py` and run the command again.

### Conditional retrieval workflow

`1_agentic_rag.py` adds a routing step before retrieval. The local `Qwen3:4b`
model classifies the question: document-specific questions go through Chroma
retrieval, while general questions are answered directly without retrieving
transcripts. Both paths then generate an answer with the local model.

Pull the model used by this script and run it:

```bash
ollama pull Qwen3:4b
uv run python 1_agentic_rag.py
```

The script uses the same `.vtt` files, chunking settings, and embedding model
as `main.py`. It creates the Chroma collection in memory for each run.

## Add or replace transcript data

Place `.vtt` files in the `data/` directory. The loader processes all matching
files when the app starts. Use only transcripts you have permission to store
and share.

## Model notes

- Answers are generated with the local Ollama model `Qwen3:1.7b`.
- Embeddings use `sentence-transformers/all-MiniLM-L6-v2`; the model may be
  downloaded the first time it is used.
- `main.py` also initializes a Hugging Face endpoint (`Qwen2.5-7B-Instruct`),
  but the current answer-generation workflow uses the Ollama model instead.

## Project files

| File or directory | Purpose |
| --- | --- |
| `main.py` | Builds the vector store and LangGraph retrieval/generation workflow |
| `1_agentic_rag.py` | Routes general questions around retrieval and retrieves documents for specialized questions |
| `documents_loader.py` | Loads and splits WebVTT transcript files |
| `data/` | Transcript files used as the knowledge base |
| `pyproject.toml` | Project metadata and dependencies |
| `uv.lock` | Locked dependency versions for reproducible installs |