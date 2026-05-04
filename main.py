import sys
import os

# Set Mistral API Key globally BEFORE importing any helpers
# This ensures that module-level initializations in helpers pick up the key.
MISTRAL_API_KEY = "uFyz6AUCYQ6F9GMWNaHLRoXGP8TLbqhJ"
os.environ["MISTRAL_API_KEY"] = MISTRAL_API_KEY

from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Input, RichLog, Static, Label
from textual.containers import Vertical, Horizontal
from textual import on, work
from textual.binding import Binding

# Add the current directory to sys.path to ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import helpers
from retrieval.retrieve import retrieve_by_similarity
from retrieval.rerank import rerank
from context.builder import build_context
from index.faiss_index import load_index
from config import vector_store_path

try:
    from langchain_mistralai import ChatMistralAI
except ImportError:
    ChatMistralAI = None

class KnowledgeAgentApp(App):
    """A Textual app for the Personal Knowledge Agent."""

    TITLE = "Personal Knowledge Agent (Mistral Edition)"
    SUB_TITLE = "Your intelligent companion, now powered by Mistral"

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=True),
        Binding("ctrl+l", "clear_chat", "Clear Chat", show=True),
    ]

    CSS = """
    Screen {
        background: #1e1e1e;
    }

    #chat-container {
        height: 1fr;
        border: tall $accent;
        background: #252525;
        padding: 1;
        margin: 1;
    }

    #chat-log {
        height: 1fr;
        scrollbar-gutter: stable;
    }

    #input-container {
        height: auto;
        dock: bottom;
        padding: 0 1;
        margin-bottom: 1;
    }

    Input {
        border: double $accent;
        background: #333333;
        color: #ffffff;
    }

    Input:focus {
        border: double #00ff00;
    }

    .user-msg {
        color: #00ff00;
        text-style: bold;
    }

    .agent-msg {
        color: #00bfff;
        text-style: bold;
    }

    .system-msg {
        color: #888888;
        text-style: italic;
    }

    .error-msg {
        color: #ff4500;
        text-style: bold italic;
    }
    """

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id="chat-container"):
            yield RichLog(id="chat-log", highlight=True, markup=True)
        with Horizontal(id="input-container"):
            yield Input(placeholder="Ask a question about your knowledge base...", id="query-input")
        yield Footer()

    def on_mount(self) -> None:
        self.chat_log = self.query_one("#chat-log")
        self.chat_log.write("[system-msg]Agent initialized. Welcome![/system-msg]")
        
        self.index = None
        self.client = None
        
        self.initialize_agent()

    @work(exclusive=True)
    async def initialize_agent(self) -> None:
        """Initialize the FAISS index and Mistral client."""
        self.chat_log.write("[system-msg]Loading knowledge base...[/system-msg]")
        try:
            # Load index
            self.index = load_index()
            self.chat_log.write("[system-msg]Knowledge base loaded successfully.[/system-msg]")
        except Exception as e:
            self.chat_log.write(f"[error-msg]Failed to load knowledge base: {e}[/error-msg]")
            self.chat_log.write("[system-msg]Please ensure the index exists at 'index/vector_store'.[/system-msg]")

        try:
            if ChatMistralAI:
                self.client = ChatMistralAI(model="mistral-large-latest", mistral_api_key=MISTRAL_API_KEY)
                self.chat_log.write("[system-msg]Mistral client ready.[/system-msg]")
            else:
                self.chat_log.write("[error-msg]langchain-mistralai library not found. Please install it.[/error-msg]")
        except Exception as e:
            self.chat_log.write(f"[error-msg]Failed to initialize Mistral client: {e}[/error-msg]")

    @on(Input.Submitted, "#query-input")
    @work(exclusive=True)
    async def handle_query(self, event: Input.Submitted) -> None:
        query = event.value.strip()
        if not query:
            return

        input_widget = self.query_one("#query-input")
        input_widget.value = ""
        
        self.chat_log.write(f"\n[user-msg]You:[/user-msg] {query}")

        if not self.index:
            self.chat_log.write("[agent-msg]Agent:[/agent-msg] [error-msg]Knowledge base not available.[/error-msg]")
            return

        if not self.client:
            self.chat_log.write("[agent-msg]Agent:[/agent-msg] [error-msg]LLM client not available. Check your API key and installation.[/error-msg]")
            return

        self.chat_log.write("[system-msg]Searching and thinking (Mistral)...[/system-msg]")

        try:
            # 1. Retrieve
            retrieved_items = retrieve_by_similarity(query, top_k=5)
            
            if not retrieved_items:
                self.chat_log.write("[agent-msg]Agent:[/agent-msg] I couldn't find any relevant information in your knowledge base.")
                return

            # 2. Rerank
            reranked_items = rerank(query, retrieved_items)
            
            # 3. Build Context
            context = build_context(query, [item.page_content for item in reranked_items])
            
            # 4. Generate Answer
            prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer the question based strictly on the context provided. If the answer is not in the context, say you don't know."
            response = self.client.invoke(prompt)
            
            answer = response.content
            self.chat_log.write(f"\n[agent-msg]Agent:[/agent-msg] {answer}")
            
        except Exception as e:
            self.chat_log.write(f"\n[error-msg]An error occurred: {str(e)}[/error-msg]")

    def action_clear_chat(self) -> None:
        """Action to clear the chat log."""
        self.chat_log.clear()
        self.chat_log.write("[system-msg]Chat cleared.[/system-msg]")

if __name__ == "__main__":
    app = KnowledgeAgentApp()
    app.run()
