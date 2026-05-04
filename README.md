# Personal Knowledge Agent

A terminal-based AI assistant that uses Retrieval-Augmented Generation (RAG) to answer questions based on your local documents.

## 🌟 Features

- **TUI Interface**: A modern, responsive terminal user interface with color-coded chat logs and interactive input.
- **Mistral AI Integration**: Leverages Mistral for state-of-the-art embeddings, relevance-based reranking, and high-quality response generation.
- **Robust RAG Pipeline**:
  - **Ingestion**: Supports PDF, TXT, Markdown, and various code formats.
  - **Chunking**: Intelligent document splitting with overlap to maintain context.
  - **Vector Storage**: Fast similarity search using FAISS.
  - **Reranking**: Advanced LLM-based reranking to ensure the most relevant context is prioritized.
- **Batch Ingestion**: Dedicated script to process and index entire directories of knowledge.

## 🛠️ Architecture

The project is organized into modular components:

- **`ingestion/`**: Document loaders and metadata parsers.
- **`chunking/`**: Text splitting logic (naive and structure-aware).
- **`embedding/`**: Integration with Mistral's embedding models.
- **`index/`**: FAISS vector store management (build, save, load).
- **`retrieval/`**: Multi-stage retrieval process including similarity search and LLM reranking.
- **`main.py`**: The heart of the application—a Textual-based chat interface.
- **`ingest_data.py`**: A CLI tool for batch processing raw data into the vector store.

## 🚀 Getting Started

### Prerequisites

Ensure you have Python installed and the necessary libraries:
```bash
pip install textual langchain langchain-mistralai faiss-cpu pypdf
```

### Configuration

The project is configured to use Mistral AI. The API key is managed within `main.py` and `ingest_data.py`, or can be set via environment variables.

### 1. Ingest Data
Place your raw documents (PDFs, text files, etc.) in the `raw/` directory, then run:
```bash
python ingest_data.py
```
This will build the FAISS index and save it to `index/vector_store`.

### 2. Run the Agent
Launch the terminal UI to start chatting with your knowledge base:
```bash
python main.py
```

## ⌨️ TUI Shortcuts

- `Ctrl + C`: Quit the application.
- `Ctrl + L`: Clear the chat history.
- `Enter`: Submit your query.

## 📝 License

MIT
