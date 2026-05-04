from langchain_core.documents import Document

def chunk_document(document):
    """
    Chunks the document into smaller fixed-size chunks with overlap
    
    Args:
        document: Document object
    
    Returns:
        List of chunked documents
    """

    content = document.page_content
    content = content.strip()

    # Create chunks of 500 characters each with overlap of 50 characters
    chunks = []
    for i in range(0, len(content), 450):
        chunks.append(content[i:i+500])

    # Create a list of metadata for each chunk
    metadata = []
    for i in range(len(chunks)):
        metadata.append(document.metadata)
        metadata[i]['chunk_number'] = i + 1
        metadata[i]['start_index'] = i * 450
        metadata[i]['end_index'] = i * 450 + 500
    
    # Create a list of Document objects
    chunked_documents = []
    for i in range(len(chunks)):
        chunked_documents.append(Document(page_content=chunks[i], metadata=metadata[i]))
    
    return chunked_documents

def chunk_docs(documents):
    """
    Chunks all the documents into smaller chunks
    
    Args:
        documents: List of Document objects
    
    Returns:
        List of chunked documents
    """
    chunked_documents = []
    for document in documents:
        chunked_document = chunk_document(document)
        chunked_documents.extend(chunked_document)
    return chunked_documents
