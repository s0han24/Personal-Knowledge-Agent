import os
from langchain_mistralai import ChatMistralAI
import re

_MISTRAL_LLM = None

def get_mistral_llm(model="mistral-medium-latest"):
    global _MISTRAL_LLM
    # Note: If we need different models, we might need a dict cache, 
    # but for now we'll just cache the one used.
    if _MISTRAL_LLM is None:
        _MISTRAL_LLM = ChatMistralAI(model=model, mistral_api_key=os.environ.get("MISTRAL_API_KEY"))
    return _MISTRAL_LLM

def polish_query(query, context=""):
    """
    Rewrite the query to be optimized for semantic similarity search.
    """
    instructions = f"""
You are an AI expert in knowledge management and semantic search.

Your task is to analyze the user's query given the context and rewrite it to be optimized for semantic similarity search.

CRITICAL INSTRUCTIONS:
1. **Expansion**: If the query is short (1-3 words), expand it into a full, detailed question. Add context, entities, and relationships.
   - *Example*: "Python" -> "What is Python and how is it used in data science?"
2. **Normalization**: Convert the query into a natural language question if it's in keyword format.
   - *Example*: "AI benefits ML" -> "What are the benefits of using AI in machine learning?"
3. **Specificity**: Add details that would help retrieve more relevant documents.
   - *Example*: "Cats" -> "What are the characteristics and care requirements of domestic cats?"
4. **Preserve Meaning**: Do NOT add new factual information or change the intent of the query.
5. **Context Enrichment**: If the query seems incomplete, use your general knowledge to infer the most likely context and add it to the query.
   - *Example*: "Python" -> "What is Python and how is it used in data science?"
6. **Tone**: Use a neutral and objective tone in the rewritten query.
7. **Preserve Intent**: Think step by step and understand the user's intent and rewrite the query to preserve the original intent.
8. **Output Format**: Return your response in the following format(use triple backticks to delimit the thinking process):

```"Thinking Process: <your thinking process>"```
"Rewritten Query: <your rewritten query>"

Context: {context}

User Query: {query}

Response:
"""
    llm = get_mistral_llm(model="mistral-medium-latest")
    response = llm.invoke(instructions.format(query=query))
    content = response.content.strip()  
    # first remove the thinking process
    content = content.split("Rewritten Query:")[1].strip()

    match = re.search(r'Rewritten Query:\s*(.*?)\s*$', content, re.DOTALL)
    if match:
        polished_query = match.group(1).strip()
    else:
        # Fallback to the whole content if no match
        polished_query = content

    content = f"""
    You are given a **rewritten query**, which has been optimized for semantic similarity search. Your task is to answer this query using the retrieved context and your general knowledge.

    **Rewritten Query:** {polished_query}

    **CRITICAL INSTRUCTIONS:**
1. Use the provided context if it is relevant to the query.

2. If the context contains sufficient information:
   - Answer using the context
   - Use citations in the format [Source: <source>, Page: <page>]
   - Prefer accurate referencing over excessive quoting

3. If the context is insufficient or not relevant:
   - Explicitly state: "The retrieved context does not contain sufficient information."
   - Then answer using general knowledge
   - Do NOT fabricate or force citations

4. Answer format:
   - Be direct and concise
   - Include explanations, mathematical formulations, or code if appropriate

5. Think step by step
   """
    return content


def rewrite_query_only(query, context=""):
    """
    Rewrite the query for semantic search, returning ONLY the rewritten query text.
    
    Unlike polish_query() which returns a full LLM prompt template,
    this returns just the rewritten query string — suitable for passing
    directly to FAISS similarity search.
    
    Args:
        query: The original user query
    
    Returns:
        The rewritten query string
    """
    instructions = f"""You are an AI expert in knowledge management and semantic search.

Your task is to analyze the user's query and rewrite it to be optimized for semantic similarity search.

CRITICAL INSTRUCTIONS:
1. If the query is short, expand it into a full, detailed question.
2. Convert keyword-format queries into natural language questions.
3. Add details that would help retrieve more relevant documents.
4. Do NOT add new factual information or change the intent of the query.
5. Return ONLY the rewritten query, nothing else. No preamble, no explanation.

User Query: {query}

Rewritten Query:"""
    llm = get_mistral_llm(model="mistral-medium-latest")
    response = llm.invoke(instructions)
    rewritten = response.content.strip()
    # Remove any accidental quotes
    rewritten = rewritten.strip('"').strip("'")
    return rewritten
