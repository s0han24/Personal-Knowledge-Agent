from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter

def perform_fixed_size_chunking(document, chunk_size=500, chunk_overlap=50):
    """
    Chunks the document into smaller fixed-size chunks with overlap
    
    Args:
        document: Document object
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
    
    Returns:
        List of chunked documents
    """

    content = document.page_content
    content = content.strip()

    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len)
    
    chunks = text_splitter.split_text(content)

    # Create a list of metadata for each chunk
    metadata = []
    for i in range(len(chunks)):
        metadata.append(document.metadata)
        metadata[i]['chunk_number'] = i + 1
        metadata[i]['start_index'] = i * (chunk_size - chunk_overlap)
        metadata[i]['end_index'] = metadata[i]['start_index'] + chunk_size

    # Create a list of Document objects
    chunked_documents = []
    for i in range(len(chunks)):
        chunked_documents.append(Document(page_content=chunks[i], metadata=metadata[i]))
    
    return chunked_documents
