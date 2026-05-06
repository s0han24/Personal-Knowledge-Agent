from langchain_mistralai import ChatMistralAI
from sentence_transformers import CrossEncoder
import re
import os

_MISTRAL_LLM = None
_CROSS_ENCODER_MODEL = None

def get_mistral_llm():
    global _MISTRAL_LLM
    if _MISTRAL_LLM is None:
        MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
        if not MISTRAL_API_KEY:
            raise ValueError("MISTRAL_API_KEY environment variable is not set.")
        _MISTRAL_LLM = ChatMistralAI(model="mistral-small-latest", mistral_api_key=MISTRAL_API_KEY)
    return _MISTRAL_LLM

def get_cross_encoder():
    global _CROSS_ENCODER_MODEL
    if _CROSS_ENCODER_MODEL is None:
        _CROSS_ENCODER_MODEL = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return _CROSS_ENCODER_MODEL

def llm_relevance_score(query, item):
    """
    Compute the relevance score of an item to the query using a language model.

    Args:
        query (str): The input query for which to compute the relevance score.
        item: The item for which to compute the relevance score.
    Returns:
        Relevance score of the item to the query.
    """
    try:
        llm = get_mistral_llm()
        
        prompt = f"Assign a relevance score to item based on its relevance to the query both the query and the item are delimited by triple backticks.\nQuery: ```{query}```\nItem: ```{item.page_content}``` \nRelevance Score (0-1):"
        
        response = llm.invoke(prompt)
        content = response.content.strip()
        
        # Flexible parsing of the response to extract the relevance score
        relevance_score = 0.0
        try:
            relevance_score = float(content)
        except ValueError:
            # If the response is not a valid float, try to extract the score from the text
            match = re.search(r'(\d+(\.\d+)?)', content)
            if match:
                relevance_score = float(match.group(1))
        return relevance_score
    except Exception:
        return 0.0

def cross_encoder_score(query, item):
    """
    Compute the relevance score of an item to the query using a cross encoder.

    Args:
        query (str): The input query for which to compute the relevance score.
        item: The item for which to compute the relevance score.
    Returns:
        Relevance score of the item to the query.
    """
    cross_encoder = get_cross_encoder()
    return cross_encoder.predict([(query, item.page_content)])[0]

def keyword_relevance_score(query, item):
    """
    Compute the relevance score of an item to the query based on keyword matching.

    Args:
        query (str): The input query for which to compute the relevance score.
        item: The item for which to compute the relevance score.
    Returns:
        Relevance score of the item to the query based on keyword matching.
    """
    query_keywords = set(query.lower().split())
    item_keywords = set(item.page_content.lower().split())
    common_keywords = query_keywords.intersection(item_keywords)
    relevance_score = len(common_keywords) / len(query_keywords) if query_keywords else 0.0
    return relevance_score

def rerank(query, retrieved_items, compute_relevance_score=cross_encoder_score):
    """
    Rerank the retrieved items based on their relevance to the query using a language model.

    Args:
        query (str): The input query for which to rerank the retrieved items.
        retrieved_items (list): The list of items retrieved from the initial retrieval step.
    Returns:
        List of the reranked items based on their relevance to the query.
    """
    if retrieved_items is None:
        print('retrieved_items is None')
        return []
    print("reranking...")
    scored_items = [(item, compute_relevance_score(query, item)) for item in retrieved_items]
    reranked_items = sorted(scored_items, key=lambda x: x[1], reverse=True)
    return reranked_items
