import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from embedding.embedder import get_embeddings_obj

from config import vector_store_path

def build_index():
    """
    Builds the FAISS index from the documents
    
    Args:
        docs: List of Document objects
    
    Returns:
        FAISS index
    """
    embeddings = get_embeddings_obj()
    dimension = len(embeddings.embed_query("hello"))
    index = faiss.IndexFlatL2(dimension)
    docstore = InMemoryDocstore()
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=docstore,
        index_to_docstore_id={},
    )
    return vector_store

def add_docs(vector_store, docs):
    """
    Adds documents to the FAISS index
    
    Args:
        vector_store: FAISS index
        docs: List of Document objects
    
    Returns:
        FAISS index
    """
    vector_store.add_documents(docs)
    return vector_store

def save_index(vector_store, path):
    """
    Saves the FAISS index to a file
    
    Args:
        vector_store: FAISS index
        path: Path to save the index
    """
    vector_store.save_local(path)

def load_index(path=None):
    """
    Loads the FAISS index from the default path
    
    Args:
        path: Path to the index
    
    Returns:
        FAISS index
    """
    if path is None:
        path = vector_store_path
    return FAISS.load_local(path, get_embeddings_obj(), allow_dangerous_deserialization=True)
