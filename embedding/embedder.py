from langchain_mistralai import MistralAIEmbeddings
import os

def get_embeddings_obj():
    """
    Returns the embedding object, initializing it if necessary.
    """
    api_key = os.environ.get("MISTRAL_API_KEY") or "uFyz6AUCYQ6F9GMWNaHLRoXGP8TLbqhJ"
    return MistralAIEmbeddings(model="mistral-embed", mistral_api_key=api_key)

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
