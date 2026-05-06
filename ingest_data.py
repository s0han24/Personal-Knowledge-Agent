import os
import sys

# Add the current directory to sys.path to ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import data_path, vector_store_path
from ingestion.loaders import load_dir
from ingestion.parsers import parse_dir
from chunking.caller import chunk_documents
from index.faiss_index import build_index, save_index, add_docs

def main():
    print(f"Starting ingestion pipeline (Mistral) for data in: {data_path}")

    # 1. Load documents
    print("Loading documents...")
    try:
        documents = load_dir(data_path)
        print(f"Loaded {len(documents)} documents.")
    except Exception as e:
        print(f"Error loading documents: {e}")
        return

    # 2. Parse documents
    print("Parsing documents...")
    try:
        parsed_documents = parse_dir(documents)
        print(f"Parsed {len(parsed_documents)} documents.")
    except Exception as e:
        print(f"Error parsing documents: {e}")
        return

    # 3. Chunk documents
    print("Chunking documents...")
    try:
        chunks = chunk_documents(parsed_documents)
        print(f"Created {len(chunks)} chunks.")
    except Exception as e:
        print(f"Error chunking documents: {e}")
        return

    # 4. Build index
    print("Building FAISS index and adding documents (Mistral Embeddings)...")
    try:
        # build_index() initializes the empty index using MistralEmbeddings
        vector_store = build_index()
        # add_docs adds the actual content
        vector_store = add_docs(vector_store, chunks)
        print("Index built and documents added.")
    except Exception as e:
        print(f"Error building index: {e}")
        return

    # 5. Save index
    print(f"Saving index to {vector_store_path}...")
    try:
        save_index(vector_store, vector_store_path)
        print("Ingestion pipeline completed successfully!")
    except Exception as e:
        print(f"Error saving index: {e}")
        return

if __name__ == "__main__":
    main()
