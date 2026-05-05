from index.faiss_index import load_index

def retrieve_by_similarity(query, path=None, top_k=5):
    """
    Retrieve the top_k most similar items from the data based on the query.

    Args:
        query (str): The input query for which to find similar items.
        path (str): The path to the FAISS index file.
        top_k (int): The number of top similar items to retrieve.
    Returns:
        List of the top_k most similar items in the vector store.
    """
    vector_store = load_index(path)
    results = vector_store.similarity_search_with_score(query, k=top_k)
    return results

