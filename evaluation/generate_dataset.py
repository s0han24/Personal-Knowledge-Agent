"""
Dataset Generation Pipeline for RAG Evaluation

Generates question-chunk pairs by:
1. Loading all chunks from the FAISS vector store
2. For each chunk, using Mistral to generate a question that the chunk answers
3. Storing pairs as JSON with a unique chunk identifier

Usage:
    python -m evaluation.generate_dataset [--output evaluation/eval_dataset.json] [--sample N] [--delay SECONDS]
"""

import os
import sys
import json
import time
import hashlib
import argparse
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_mistralai import ChatMistralAI
from index.faiss_index import load_index
from config import vector_store_path


def get_chunk_id(chunk):
    """
    Generate a unique, deterministic identifier for a chunk.
    
    Uses a combination of source file, chunk number, and a content hash
    to create a reproducible ID that can be used to look up the chunk later.
    
    Args:
        chunk: A LangChain Document object with metadata and page_content
    
    Returns:
        A string identifier like "source_file.pdf__chunk_3__a1b2c3d4"
    """
    source = chunk.metadata.get("source", "unknown")
    chunk_num = chunk.metadata.get("chunk_number", 0)
    # Short hash of content for uniqueness (first 8 hex chars)
    content_hash = hashlib.sha256(chunk.page_content.encode("utf-8")).hexdigest()[:8]
    
    # Clean source name for readability
    source_basename = os.path.basename(source)
    
    return f"{source_basename}__chunk_{chunk_num}__{content_hash}"


def extract_all_chunks(index_path=None):
    """
    Extract all Document objects stored in the FAISS vector store.
    
    The FAISS wrapper in LangChain stores documents in an InMemoryDocstore,
    accessible via the index_to_docstore_id mapping.
    
    Args:
        index_path: Optional path to the FAISS index. Defaults to config value.
    
    Returns:
        List of LangChain Document objects
    """
    vector_store = load_index(index_path)
    
    chunks = []
    for doc_id in vector_store.index_to_docstore_id.values():
        doc = vector_store.docstore.search(doc_id)
        if doc and hasattr(doc, "page_content"):
            chunks.append(doc)
    
    print(f"Extracted {len(chunks)} chunks from the vector store.")
    return chunks


def generate_question(llm, chunk_text):
    """
    Use Mistral to generate a natural question that the given chunk answers.
    
    The prompt is designed to produce a single, self-contained question
    that a user might realistically ask and that the chunk can answer.
    
    Args:
        llm: ChatMistralAI instance
        chunk_text: The text content of the chunk
    
    Returns:
        A generated question string, or None if generation fails
    """
    prompt = f"""You are an expert at generating evaluation questions for a Retrieval-Augmented Generation (RAG) system.

Given the following text chunk from a knowledge base, generate exactly ONE question that:
1. Can be directly and fully answered using ONLY the information in this chunk
2. Is specific and detailed — avoid overly broad or vague questions
3. Sounds like a natural question a real user would ask
4. Does NOT reference "the chunk", "the text", "the passage", or "the document" — the question should stand alone
5. Is a complete, grammatically correct question

Text Chunk:
\"\"\"
{chunk_text}
\"\"\"

Respond with ONLY the question, nothing else. No preamble, no explanation, no numbering."""

    try:
        response = llm.invoke(prompt)
        question = response.content.strip()
        # Remove any accidental quotes wrapping the question
        question = question.strip('"').strip("'")
        return question
    except Exception as e:
        print(f"  [ERROR] Failed to generate question: {e}")
        return None


def generate_dataset(
    output_path="evaluation/eval_dataset.json",
    index_path=None,
    sample_size=None,
    delay_seconds=1.0,
    model="mistral-medium-latest",
):
    """
    Main pipeline: extract chunks, generate questions, save dataset.
    
    Args:
        output_path: Path to save the output JSON dataset
        index_path: Optional custom path to the FAISS index
        sample_size: If set, only process this many chunks (useful for testing)
        delay_seconds: Delay between API calls to respect rate limits
        model: Mistral model to use for question generation
    
    Returns:
        The generated dataset as a list of dicts
    """
    # --- Setup ---
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY environment variable is not set.")
    
    llm = ChatMistralAI(model=model, mistral_api_key=api_key)
    
    # --- Extract chunks ---
    print("Loading chunks from vector store...")
    chunks = extract_all_chunks(index_path)
    
    if sample_size and sample_size < len(chunks):
        import random
        random.seed(42)  # Reproducible sampling
        chunks = random.sample(chunks, sample_size)
        print(f"Sampled {sample_size} chunks for dataset generation.")
    
    # --- Generate questions ---
    dataset = []
    failed = 0
    
    print(f"\nGenerating questions for {len(chunks)} chunks...")
    print(f"  Model: {model}")
    print(f"  Rate limit delay: {delay_seconds}s between calls\n")
    
    for i, chunk in enumerate(chunks):
        chunk_id = get_chunk_id(chunk)
        print(f"[{i+1}/{len(chunks)}] Processing {chunk_id}...")
        
        question = generate_question(llm, chunk.page_content)
        
        if question:
            entry = {
                "chunk_id": chunk_id,
                "question": question,
                "chunk_text": chunk.page_content,
                "metadata": {
                    "source": chunk.metadata.get("source", "unknown"),
                    "chunk_number": chunk.metadata.get("chunk_number", 0),
                    "start_index": chunk.metadata.get("start_index"),
                    "end_index": chunk.metadata.get("end_index"),
                    "type": chunk.metadata.get("type", "unknown"),
                    "file_extension": chunk.metadata.get("file_extension", "unknown"),
                },
            }
            dataset.append(entry)
            print(f"  ✓ Q: {question[:80]}...")
        else:
            failed += 1
            print(f"  ✗ Skipped (generation failed)")
        
        # Rate limiting
        if i < len(chunks) - 1:
            time.sleep(delay_seconds)
    
    # --- Save ---
    output = {
        "generated_at": datetime.now().isoformat(),
        "model": model,
        "total_chunks": len(chunks),
        "total_pairs": len(dataset),
        "failed": failed,
        "pairs": dataset,
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"Dataset generation complete!")
    print(f"  Total chunks processed: {len(chunks)}")
    print(f"  Successful pairs:       {len(dataset)}")
    print(f"  Failed:                 {failed}")
    print(f"  Output saved to:        {output_path}")
    print(f"{'='*60}")
    
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate question-chunk pairs for RAG evaluation"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation/eval_dataset.json",
        help="Path to save the output JSON dataset (default: evaluation/eval_dataset.json)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Number of chunks to sample (default: all chunks)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay in seconds between API calls for rate limiting (default: 1.0)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mistral-medium-latest",
        help="Mistral model to use for question generation (default: mistral-medium-latest)",
    )
    parser.add_argument(
        "--index-path",
        type=str,
        default=None,
        help="Custom path to the FAISS index (default: uses config.py value)",
    )
    
    args = parser.parse_args()
    
    generate_dataset(
        output_path=args.output,
        index_path=args.index_path,
        sample_size=args.sample,
        delay_seconds=args.delay,
        model=args.model,
    )
