from .naive import perform_fixed_size_chunking

def chunk_documents(documents, chunking_function=perform_fixed_size_chunking):
    """
    Chunks all the documents into smaller chunks
    
    Args:
        documents: List of Document objects
    
    Returns:
        List of chunked documents
    """
    chunked_documents = []
    for document in documents:
        chunked_document = chunking_function(document)
        chunked_documents.extend(chunked_document)
    return chunked_documents
