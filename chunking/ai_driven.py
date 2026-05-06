from chunking.semantic import perform_semantic_chunking
from langchain_mistralai import ChatMistralAI
from langchain_core.documents import Document

# Initialize a small Mistral model for AI-driven chunking and summarization
mistral_ai = ChatMistralAI(model="mistral-large-latest")


# Hits rate limit for Mistral, so just a proof of concept for now. I need to find a workaround for this, viable retrieval strategies include:
# 1. Using windowed retrieval: Retrieve a chunk before and after the current chunk to provide additional context without needing to summarize the entire document.
# 2. Parent-Document Retrieval: Make small child chunks and larger parent chunks; search the Child chunks to find the exact location, but retrieve the Parent chunk to send to the LLM.
def perform_ai_driven_chunking(document, chunk_size=500, chunk_overlap=50):
    """
    Chunks the document into smaller chunks using an AI-driven approach

    Args:
        document: Document object
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks

    Returns:
        List of chunked documents
    """
    # TODO: Implement a chunking strategy that uses AI to determine the optimal way to chunk the document based on its content, structure, and the agent's goals. This could involve using machine learning models to analyze the document and identify natural breakpoints for chunking.
    return perform_semantic_chunking(document, chunk_size, chunk_overlap)

def summarize_content(content):
    """
    Summarizes the content of a document

    Args:
        content: The content to be summarized

    Returns:
        A summary of the content
    """
    summary = mistral_ai.invoke(f"Summarize the following content and return only the summary with no preamble:\n\n{content} \n\nSummary:")
    
    # Return summary text
    return summary.content

def perform_context_enriched_chunking(document, chunk_size=500, chunk_overlap=50):
    """
    Chunks the document into smaller chunks and enriches them with additional context

    Args:
        document: Document object
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks

    Returns:
        List of chunked documents with enriched context
    """
    content = document.page_content
    
    summary = summarize_content(content)
    
    chunks = perform_semantic_chunking(document, chunk_size, chunk_overlap)
    
    enriched_chunks = []
    for chunk in chunks:
        enriched_content = f"{summary}\n\n{chunk.page_content}"
        metadata = chunk.metadata.copy()
        metadata['enriched'] = True
        enriched_chunk = Document(page_content=enriched_content, metadata=metadata)
        enriched_chunks.append(enriched_chunk)
    
    return enriched_chunks
