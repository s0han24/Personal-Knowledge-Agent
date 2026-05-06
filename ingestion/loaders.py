from langchain_community.document_loaders import PyPDFLoader, TextLoader
import os

def load_file(path):
    """
    Loads a single file based on its extension
    
    Args:
        path: Path to the file
    
    Returns:
        List of Document objects
    """
    file_extension = path.split('/')[-1].split('.')[-1]
    
    if file_extension == 'pdf':
        docs = load_pdf(path, file_extension)
        return docs
    else:
        # For simplicity, we treat all non-pdf files as text files. In a real implementation, we would want to handle different file types more robustly.
        docs = load_text(path, file_extension)
        return docs

def load_text(path, file_extension):
    """
    Loads a text file
    
    Args:
        path: Path to the text file
    
    Returns:
        List of Document objects
    """
    loader = TextLoader(path)
    docs = loader.load()
    doctype = 'text' if file_extension == 'txt' else 'code'
    for doc in docs:
        doc.metadata['type'] = doctype
        doc.metadata['file_extension'] = file_extension
    return docs

def load_pdf(path, file_extension):
    """
    Loads a PDF file (digital ones only)
    
    Args:
        path: Path to the PDF file
    
    Returns:
        List of Document objects
    """
    loader = PyPDFLoader(path)
    docs = loader.load()
    for doc in docs:
        doc.metadata['type'] = 'pdf'
        doc.metadata['file_extension'] = file_extension
    return docs

def load_dir(path):
    """
    Loads all the files in a directory
    
    Args:
        path: Path to the directory
    
    Returns:
        List of Document objects
    """
    files = os.listdir(path)
    documents = []
    for file in files:
        full_path = os.path.join(path, file)
        if os.path.isfile(full_path):
            docs = load_file(full_path)
            documents.extend(docs)
    return documents
