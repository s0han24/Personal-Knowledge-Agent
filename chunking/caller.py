from .naive import perform_fixed_size_chunking

def chunk_documents(documents, chunking_function=perform_fixed_size_chunking, chunk_size=500, chunk_overlap=50):
    """
    Chunks all the documents into smaller chunks
    
    Args:
        documents: List of Document objects
        chunking_function: The function to use for chunking
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
    
    Returns:
        List of chunked documents
    """
    chunked_documents = []
    for document in documents:
        chunked_document = chunking_function(document, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunked_documents.extend(chunked_document)
    return chunked_documents
