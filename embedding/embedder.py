from langchain_mistralai import MistralAIEmbeddings
import os

def get_embeddings_obj():
    """
    Returns the embedding object, initializing it if necessary.
    """
    MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
    if not MISTRAL_API_KEY:
        raise ValueError("MISTRAL_API_KEY environment variable is not set.")
    return MistralAIEmbeddings(model="mistral-embed", mistral_api_key=MISTRAL_API_KEY)

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
