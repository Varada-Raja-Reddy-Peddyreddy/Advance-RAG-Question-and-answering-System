# RAG Q&A App

## Overview
The **RAG Q&A App** is a Retrieval-Augmented Generation (RAG) question-and-answering
application built with **Streamlit**, **FAISS**, and **LangChain**. Point it at a
data source — a web page, PDF, Word document, text file, or pasted text — and ask
questions about its content. Answers are grounded in the retrieved passages, and the
sources used for each answer are shown alongside it.

## Features
- Supports multiple input types: **Text, PDF, DOCX, TXT, and web links**.
- Uses **FAISS** (Facebook AI Similarity Search) for vector storage and retrieval.
- Generates embeddings locally with a **HuggingFace sentence-transformers** model.
- Answers with **Groq** (`openai/gpt-oss-120b` by default) for fast LLM inference.
- Built on a modern **LCEL** RAG pipeline and shows the **source passages** behind every answer.
- Interactive **Streamlit** UI with progress indicators and graceful error handling.

## Architecture
1. **Load** — the input is read into LangChain `Document` objects (with source metadata).
2. **Split** — documents are chunked with `RecursiveCharacterTextSplitter`.
3. **Embed & index** — chunks are embedded and stored in a FAISS vector store.
4. **Retrieve & generate** — at query time, the top-k chunks are retrieved and passed,
   via an LCEL chain, to the Groq chat model to produce a grounded answer.

## Installation

### Prerequisites
- Python 3.9+
- pip

### Install dependencies
```bash
pip install -r requirements.txt
```

### Configure your API key
This project uses **Groq** for inference. Create an API key at
[console.groq.com/keys](https://console.groq.com/keys), then copy the example
environment file and add your key:

```bash
cp .env.example .env
# then edit .env and set GROQ_API_KEY=...
```

Alternatively, provide the key through Streamlit secrets
(`.streamlit/secrets.toml`):

```toml
GROQ_API_KEY = "your-groq-api-key-here"
```

> The key is read from the environment (or Streamlit secrets) at runtime — it is
> never committed to the repository. `.env` and `secrets.toml` are git-ignored.

## Usage

### Run the app
```bash
streamlit run app.py
```

### Interacting with the app
1. In the sidebar, select an **Input type** (Link, PDF, Text, DOCX, TXT).
2. Provide the input (upload a file, paste text, or enter URLs).
3. Click **Process** to read and index the data.
4. Type a question and click **Submit** to get a grounded answer.
5. Expand **Sources** to see the passages the answer was based on.

## Configuration
| Variable        | Default                 | Description                          |
| --------------- | ----------------------- | ------------------------------------ |
| `GROQ_API_KEY`  | _(required)_            | Your Groq API key.                   |
| `GROQ_MODEL`    | `openai/gpt-oss-120b`   | Groq chat model used for answering.  |

The embedding model (`sentence-transformers/all-mpnet-base-v2`), chunk size, and
number of retrieved passages can be adjusted at the top of `app.py`.

## Project structure
```
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── .env.example        # Template for environment variables
├── .gitignore
└── README.md
```

## Tech stack
- **Streamlit** — interactive UI
- **FAISS** — vector similarity search
- **LangChain (LCEL)** — retrieval + generation pipeline
- **HuggingFace sentence-transformers** — text embeddings
- **Groq** — fast LLM inference
- **pypdf** / **python-docx** — document parsing

## Notes on this update
This project was modernized from its original 2024 version:
- Migrated off the deprecated `RetrievalQA` chain to a version-robust **LCEL** pipeline.
- Updated the default model — the previous `deepseek-r1-distill-llama-70b` and its
  `llama-3.3-70b-versatile` successor were both deprecated by Groq — to the currently
  recommended `openai/gpt-oss-120b`.
- Replaced the unmaintained **PyPDF2** with **pypdf** and split LangChain into its
  current modular packages.
- Moved API keys out of a committed source file and into `.env` / Streamlit secrets.
- Fixed several runtime bugs and added source citations, caching, and error handling.

## Future enhancements
- Conversational memory (multi-turn follow-up questions).
- Persist the FAISS index to disk for reuse across sessions.
- Deploy on Streamlit Community Cloud or Hugging Face Spaces.

## License
This project is licensed under the **MIT License**.

## Author
VARADA RAJA REDDY
