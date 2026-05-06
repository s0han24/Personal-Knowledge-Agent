from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter, RecursiveJsonSplitter, Language

def perform_semantic_chunking(document, chunk_size=500, chunk_overlap=50):
    """_summary_

    Args:
        document (Document): The document to be chunked
        chunk_size (int, optional): The size of each chunk. Defaults to 500.
        chunk_overlap (int, optional): The overlap between chunks. Defaults to 50.
    """
    content = document.page_content
    content = content.strip()
    
    doctype = document.metadata.get('type', 'unknown')
    file_extension = document.metadata.get('file_extension', 'unknown')

    if doctype == 'pdf':
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len)
        chunks = text_splitter.split_text(content)
    elif doctype == 'text' and file_extension == 'md':
        text_splitter = MarkdownHeaderTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len, strip_header=False)
        chunks = text_splitter.split_text(content)
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len)
        chunks = text_splitter.split_text(chunks)
    elif doctype == 'text' and file_extension in ['json']:
        text_splitter = RecursiveJsonSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len)
        chunks = text_splitter.split_text(content)
    else:
        # Assume the document is a code snippet
        extension_to_language = {
            'py': Language.PYTHON,
            'js': Language.JS,
            'ts': Language.TS,
            'java': Language.JAVA,
            'cpp': Language.CPP,
            'go': Language.GO,
            'rb': Language.RUBY,
            'php': Language.PHP,
            'rs': Language.RUST,
            'html': Language.HTML,
            'cs': Language.CSHARP,
            'c': Language.C,
            'scala': Language.SCALA,
        }
        
        language = extension_to_language.get(file_extension)
        
        if language:
            text_splitter = RecursiveCharacterTextSplitter.from_language(
                language=language, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
        else:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            
        chunks = text_splitter.split_text(content)
    
    # Wrap raw text chunks into Document objects with metadata
    chunked_documents = []
    for i, chunk_text in enumerate(chunks):
        chunk_metadata = dict(document.metadata)  # copy to avoid shared-reference bugs
        chunk_metadata['chunk_number'] = i + 1
        chunk_metadata['start_index'] = i * (chunk_size - chunk_overlap)
        chunk_metadata['end_index'] = chunk_metadata['start_index'] + chunk_size
        chunked_documents.append(Document(page_content=chunk_text, metadata=chunk_metadata))
    
    return chunked_documents

def perform_adaptive_chunking(document, chunk_size=500, chunk_overlap=50):
    """
    Chunks the document into smaller adaptive chunks based on its content and structure
    
    Args:
        document: Document object
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
    
    Returns:
        List of chunked documents
    """
    # TODO: Implement a chunking strategy that adapts to the content and creates chunks of varying sizes based on the complexity of the content. For example, we could create larger chunks for simple text and smaller chunks for complex code snippets.
    return perform_semantic_chunking(document, chunk_size, chunk_overlap)


