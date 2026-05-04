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
    
    if file_extension == 'txt':
        return load_text(path)
    elif file_extension == 'pdf':
        return load_pdf(path)
    elif file_extension == 'md':
        docs = load_text(path)
        for doc in docs:
            doc.metadata['type'] = 'markdown'
        return docs
    else:
        # Assume the file is a code snippet and load it as a text file
        docs = load_text(path)
        for doc in docs:
            doc.metadata['type'] = 'code'
        return docs

def load_text(path):
    """
    Loads a text file
    
    Args:
        path: Path to the text file
    
    Returns:
        List of Document objects
    """
    loader = TextLoader(path)
    return loader.load()

def load_pdf(path):
    """
    Loads a PDF file (digital ones only)
    
    Args:
        path: Path to the PDF file
    
    Returns:
        List of Document objects
    """
    loader = PyPDFLoader(path)
    return loader.load()

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
