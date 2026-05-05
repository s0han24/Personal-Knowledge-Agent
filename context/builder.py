def build_context(chunks):
    """
    Builds the context from the documents
    
    Args:
        query: The user's query
        chunks: List of text chunks
    Returns:
        A string containing the context for the query
    """
    # Naive implementation: Concatenate the chunks to form the context
    context = "\n".join(chunks)
    return context
