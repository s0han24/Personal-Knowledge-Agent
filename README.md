# Personal Knowledge Agent

A premium AI assistant that uses Retrieval-Augmented Generation (RAG) to answer questions based on your local documents. Powered by **Mistral AI** and built with **Streamlit**.

## 🌟 Features

- **Streamlit Interface**: A modern, web-based chat interface with native support for **LaTeX math**, syntax-highlighted code, and smooth scrolling.
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
- **`app.py`**: The main application—a Streamlit-based web interface.
- **`ingest_data.py`**: A CLI tool for batch processing raw data into the vector store.

## 🚀 Getting Started

### Prerequisites

Ensure you have Python installed and the necessary libraries:
```bash
pip install streamlit langchain langchain-mistralai faiss-cpu pypdf
```

### Configuration

The project requires a **Mistral AI API key**. Set it as an environment variable:

**Mac/Linux:**
```bash
export MISTRAL_API_KEY="your-mistral-api-key"
```

**Windows (Command Prompt):**
```cmd
set MISTRAL_API_KEY=your-mistral-api-key
```

### 1. Ingest Data
Place your raw documents in the `raw/` directory, then run:
```bash
python ingest_data.py
```

### 2. Run the Agent
Launch the web UI to start chatting with your knowledge base:
```bash
streamlit run app.py
```

## 📝 License

MIT

## 📝 License

MIT
