import streamlit as st
import os
import sys

# Add current directory to path for local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from retrieval.retrieve import retrieve_by_similarity
from retrieval.rerank import rerank
from retrieval.rewrite import polish_query
from context.builder import build_context
from index.faiss_index import load_index
from config import vector_store_path
import re

try:
    from langchain_mistralai import ChatMistralAI
except ImportError:
    ChatMistralAI = None

# --- Configuration ---
st.set_page_config(
    page_title="Personal Knowledge Agent",
    layout="wide",
    initial_sidebar_state="expanded"
)

def normalize_latex(text):
    # Inline: \( ... \) → $...$
    text = re.sub(r'\\\((.*?)\\\)', r'$\1$', text)

    # Display: \[ ... \] → $$...$$  (IMPORTANT: DOTALL)
    text = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', text, flags=re.DOTALL)

    # Fix spacing inside inline math
    text = re.sub(r'\$\s*(.*?)\s*\$', r'$\1$', text)

    # Fix spacing inside display math
    text = re.sub(r'\$\$\s*(.*?)\s*\$\$', r'$$\1$$', text, flags=re.DOTALL)

    return text

# --- Styling ---
st.markdown("""
<style>
    .stChatMessage {
        border-radius: 15px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .main {
        background-color: #0e1117;
    }
    h1 {
        color: #00d4ff;
    }
</style>
""", unsafe_allow_html=True)

# --- Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "index" not in st.session_state:
    with st.spinner("Loading knowledge base..."):
        try:
            st.session_state.index = load_index()
            st.toast("Knowledge base loaded!")
        except Exception as e:
            st.error(f"Failed to load index: {e}")
            st.session_state.index = None

if "client" not in st.session_state:
    mistral_api_key = os.environ.get("MISTRAL_API_KEY")
    if not mistral_api_key:
        st.error("MISTRAL_API_KEY environment variable is not set. Please set it to use LLM features.")
        raise ValueError("MISTRAL_API_KEY environment variable is not set.")
        
    if ChatMistralAI and mistral_api_key:
        st.session_state.client = ChatMistralAI(
            model="mistral-large-latest", 
            mistral_api_key=mistral_api_key
        )
    else:
        st.warning("Mistral API key not found or library missing. LLM features disabled.")
        st.session_state.client = None

# --- UI Layout ---
st.title("Personal Knowledge Agent")
st.subheader("Intelligent Retrieval-Augmented Generation")

# Sidebar for controls
with st.sidebar:
    st.header("Settings")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    st.markdown("### System Status")
    if st.session_state.index:
        st.success("FAISS Index: Ready")
    else:
        st.error("FAISS Index: Missing")
    
    if st.session_state.client:
        st.success("Mistral LLM: Connected")
    else:
        st.error("Mistral LLM: Offline")

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "logs" in message:
            with st.expander("🔍 RAG Visibility Logs", expanded=False):
                if "rewritten_query" in message["logs"]:
                    with st.expander("🔍 Rewritten Query"):
                        st.markdown(message["logs"]["rewritten_query"])
                    st.markdown("---")

                with st.expander('🔍 Retrieved Chunks (Vector Search)', expanded=False):
                    for i, chunk in enumerate(message["logs"]["retrieved"]):
                        st.info(f"**Chunk {i+1}** (Score: {chunk['score']:.4f})\n\n{chunk['content']}")
                
                with st.expander('🔍 Reranked Chunks (Relevance)', expanded=False):
                    for i, chunk in enumerate(message["logs"]["reranked"]):
                        st.success(f"**Chunk {i+1}** (Score: {chunk['score']:.4f})\n\n{chunk['content']}")
                
                with st.expander('🔍 Final Context', expanded=False):
                    st.code(message["logs"]["context"], language="text")

# Chat Input
if prompt := st.chat_input("Ask me anything about your knowledge base..."):
    # Display user message and add to history
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate response
    with st.chat_message("assistant"):
        if not st.session_state.index:
            response = "Error: Knowledge base is not loaded. Please ensure the index exists."
            st.error(response)
        elif not st.session_state.client:
            response = "Error: LLM client is not available. Please check your API key."
            st.error(response)
        else:
            with st.status("Searching and thinking...", expanded=True) as status:
                try:
                    # 1. Rewrite Query
                    st.write("Optimizing query for search...")
                    polished_query = polish_query(prompt)

                    # 2. Retrieve
                    st.write("Retrieving relevant documents...")
                    # retrieved_results is now List[Tuple[Document, float]]
                    retrieved_results = retrieve_by_similarity(polished_query, top_k=5)
                    
                    if not retrieved_results:
                        response = "I couldn't find any relevant information in your knowledge base."
                        status.update(label="No results found", state="complete")
                    else:
                        # 3. Rerank
                        st.write("Reranking for precision...")
                        # Pass only Document objects to rerank
                        docs_only = [doc for doc, score in retrieved_results]
                        reranked_results = rerank(polished_query, docs_only)
                        
                        # 3. Build Context
                        st.write("Building context...")
                        # reranked_results is now List[Tuple[Document, float]]
                        context = build_context([doc.page_content for doc, score in reranked_results])
                        
                        # 4. Generate Answer
                        st.write("Generating answer...")
                        llm_prompt = f"Context:\n{context}\n\nQuestion: {prompt}\n\nAnswer the question based strictly on the context provided. If the answer is not in the context, say you don't know."
                        ai_response = st.session_state.client.invoke(llm_prompt)
                        response = ai_response.content
                        
                        # Prepare logs for session state
                        logs = {
                            "rewritten_query": polished_query,
                            "retrieved": [{"content": doc.page_content, "score": score} for doc, score in retrieved_results],
                            "reranked": [{"content": doc.page_content, "score": score} for doc, score in reranked_results],
                            "context": context
                        }
                        
                        status.update(label="Thinking complete!", state="complete", expanded=False)
                except Exception as e:
                    response = f"An error occurred: {str(e)}"
                    logs = None
                    status.update(label="Error occurred", state="error")
        
        response = normalize_latex(response)

        st.markdown(response, unsafe_allow_html=True)

        message_data = {"role": "assistant", "content": response}
        if logs:
            message_data["logs"] = logs
        st.session_state.messages.append(message_data)
    
    # Force a rerun to clear the "dimmed" generation state and let the history loop take over
    st.rerun()
