from langchain_mistralai import MistralAIEmbeddings
from langchain_core.embeddings import Embeddings
import os
from sentence_transformers import SentenceTransformer, util

__EMBEDDING_MODEL__ = None

class CustomEmbeddings(Embeddings):
    def __init__(self):
        global __EMBEDDING_MODEL__
        if __EMBEDDING_MODEL__ is None:
            __EMBEDDING_MODEL__ = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')
    
    def embed_query(self, query):
        return __EMBEDDING_MODEL__.encode(query)
    
    def embed_documents(self, docs):
        return __EMBEDDING_MODEL__.encode(docs)

def get_embeddings_obj():
    """
    Returns the embedding object, initializing it if necessary.
    """
    return CustomEmbeddings()

def get_embeddings(docs):
    """
    Embeds the documents using the MistralAIEmbeddings model
    
    Args:
        docs: List of Document objects
    
    Returns:
        List of embeddings
    """
    embeddings = get_embeddings_obj()
    return embeddings.embed_documents([doc.page_content for doc in docs])
