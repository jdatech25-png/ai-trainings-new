"""
Example 4: Guided Project - Contextual Chatbot with RAG, Memory & Tools

This guided project demonstrates a production-ready chatbot that combines:
1. RAG (Retrieval Augmented Generation) - Answer questions from documents
2. Conversational Memory - Remember conversation context
3. Tools - Perform calculations, get weather, search information
4. Intelligent routing - Decide when to use RAG vs tools vs general knowledge

This is the culmination of Day 2 training, showcasing how to build
a sophisticated conversational AI system with LangChain.

Architecture:
┌──────────────┐
│ User Question│
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│ Agent with Memory│
└──────┬───────────┘
       │
       ├──────▶ RAG (Document Q&A)
       │
       ├──────▶ Tools (Calculator, Weather, Search)
       │
       └──────▶ General Knowledge (LLM)

Prerequisites:
- Set OPENAI_API_KEY in .env file
- Run build_faiss_store.py to create vector index
- (Optional) Run build_milvus_store.py for Milvus

Usage:
    python examples/example_4_guided_project.py
"""

import os
import sys
from dotenv import load_dotenv
from typing import List, Dict, Any
import warnings

# Suppress all LangChain deprecation warnings for training purposes
warnings.filterwarnings("ignore", message=".*LangChain.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Import centralized managers
from core.memory_manager import MemoryManager, MemoryType

# Load environment variables
load_dotenv()


# ============================================================================
# Tool Definitions
# ============================================================================

@tool
def calculator(expression: str) -> str:
    """
    Performs mathematical calculations.
    Input should be a valid mathematical expression like '5 + 3' or '10 * 2'.
    """
    try:
        allowed_chars = set("0123456789+-*/()%. ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression"
        
        result = eval(expression)
        return f"Calculation result: {expression} = {result}"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def weather_info(city: str) -> str:
    """
    Get current weather information for a city.
    Input should be a city name like 'San Francisco' or 'London'.
    """
    # Mock weather data
    mock_weather = {
        "san francisco": {"temp": "18°C", "condition": "Partly Cloudy", "humidity": "65%"},
        "london": {"temp": "12°C", "condition": "Rainy", "humidity": "80%"},
        "tokyo": {"temp": "20°C", "condition": "Sunny", "humidity": "55%"},
        "new york": {"temp": "15°C", "condition": "Clear", "humidity": "60%"},
        "paris": {"temp": "14°C", "condition": "Cloudy", "humidity": "70%"},
        "seattle": {"temp": "16°C", "condition": "Rainy", "humidity": "75%"},
        "mumbai": {"temp": "30°C", "condition": "Hot and Humid", "humidity": "85%"}
    }
    
    city_lower = city.lower()
    if city_lower in mock_weather:
        w = mock_weather[city_lower]
        return f"Weather in {city}: {w['condition']}, {w['temp']}, Humidity: {w['humidity']}"
    else:
        return f"Weather data not available for {city}. Try: San Francisco, London, Tokyo, New York, Paris, Seattle, or Mumbai"


@tool
def search_documents(query: str) -> str:
    """
    Search company documents for information.
    Use this tool to answer questions about company policies, products, or technical documentation.
    Input should be a search query related to company information.
    """
    try:
        # Load FAISS vector store
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            dimensions=3072
        )
        
        faiss_index_path = "faiss_index"
        if not os.path.exists(faiss_index_path):
            return "Error: Document index not found. Please run build_faiss_store.py first."
        
        vector_store = FAISS.load_local(
            faiss_index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        # Retrieve relevant documents
        docs = vector_store.similarity_search(query, k=3)
        
        if not docs:
            return "No relevant documents found."
        
        # Format context
        context = "\n\n".join([
            f"From {doc.metadata.get('source', 'Unknown')} (page {doc.metadata.get('page', '?')}):\n{doc.page_content}"
            for doc in docs
        ])
        
        return f"Retrieved information from company documents:\n\n{context}"
    
    except Exception as e:
        return f"Error searching documents: {str(e)}"


# ============================================================================
# Contextual Chatbot Class
# ============================================================================

class ContextualChatbot:
    """
    A sophisticated chatbot that combines RAG, memory, and tools.
    
    Features:
    - Remembers conversation history
    - Can answer from documents (RAG)
    - Can use tools (calculator, weather, etc.)
    - Handles general knowledge questions
    """
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        verbose: bool = False
    ):
        """
        Initialize the contextual chatbot.
        
        Args:
            model: OpenAI model to use
            temperature: LLM temperature (0-2)
            verbose: Whether to show agent reasoning
        """
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Define tools
        self.tools = [calculator, weather_info, search_documents]
        
        # Create memory using MemoryManager
        self.memory = MemoryManager.create_memory(
            memory_type=MemoryType.BUFFER,
            memory_key="chat_history",
            return_messages=False
        )
        
        # Create agent prompt
        self.prompt = PromptTemplate.from_template("""
You are a helpful AI assistant with access to tools and company documents.

Chat History:
{chat_history}

Available Tools:
{tools}

Tool Names: {tool_names}

Guidelines:
- For questions about company policies, benefits, products, or documentation, use search_documents tool
- For mathematical calculations, use calculator tool
- For weather information, use weather_info tool
- For general questions, use your knowledge
- Always remember the chat history for context

Use this format:
Question: {input}
Thought: Let me think about what to do
Action: [tool name or "none" if no tool needed]
Action Input: [input for the tool]
Observation: [result of the action]
Thought: I now know how to respond
Final Answer: [your response to the user]

Begin!

Question: {input}
Thought: {agent_scratchpad}
""")
        
        # Create agent
        agent = create_react_agent(self.llm, self.tools, self.prompt)
        
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=verbose,
            handle_parsing_errors=True,
            max_iterations=5
        )
    
    def chat(self, message: str) -> str:
        """
        Send a message to the chatbot and get a response.
        
        Args:
            message: User's message
            
        Returns:
            Chatbot's response
        """
        try:
            result = self.agent_executor.invoke({"input": message})
            return result['output']
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"
    
    def get_memory(self) -> str:
        """Get the conversation memory contents."""
        return self.memory.buffer
    
    def clear_memory(self):
        """Clear the conversation memory."""
        self.memory.clear()
    
    def toggle_verbose(self):
        """Toggle verbose mode on/off."""
        self.agent_executor.verbose = not self.agent_executor.verbose
        return self.agent_executor.verbose


# ============================================================================
# Interactive Chat Interface
# ============================================================================

def interactive_chat():
    """
    Run an interactive chat session with the contextual chatbot.
    """
    print("=" * 70)
    print("Contextual Chatbot - RAG + Memory + Tools")
    print("=" * 70)
    print()
    
    print("🚀 Initializing chatbot...")
    print()
    
    # Check prerequisites
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found")
        print("   Please add it to your .env file")
        return
    
    faiss_path = "faiss_index"
    if not os.path.exists(faiss_path):
        print("⚠️  Warning: FAISS index not found at 'faiss_index/'")
        print("   Document search will not work. Run: python scripts/build_faiss_store.py")
        print()
    else:
        print("✅ FAISS index found - document search enabled")
        print()
    
    # Initialize chatbot
    chatbot = ContextualChatbot(verbose=False)
    
    print("=" * 70)
    print("Chat Session Started")
    print("=" * 70)
    print()
    
    print("🤖 Chatbot Capabilities:")
    print("   • 📚 Search company documents (policies, products, technical docs)")
    print("   • 🧮 Perform calculations")
    print("   • 🌤️  Get weather information")
    print("   • 💭 Remember conversation context")
    print("   • 🧠 Answer general knowledge questions")
    print()
    
    print("💬 Try asking:")
    print("   • 'What are the company benefits?'  (uses RAG)")
    print("   • 'What's 15% of 80,000?'  (uses calculator)")
    print("   • 'What's the weather in Tokyo?'  (uses weather tool)")
    print("   • 'What did we discuss earlier?'  (uses memory)")
    print()
    
    print("Commands:")
    print("   • 'memory' - View conversation history")
    print("   • 'clear' - Clear conversation memory")
    print("   • 'verbose' - Toggle verbose mode (see agent reasoning)")
    print("   • 'tools' - List available tools")
    print("   • 'quit' - Exit")
    print()
    
    while True:
        print("─" * 70)
        user_input = input("You: ").strip()
        
        if not user_input:
            continue
        
        # Handle commands
        if user_input.lower() == 'quit':
            print("\n👋 Thanks for chatting! Goodbye!")
            break
        
        elif user_input.lower() == 'memory':
            print("\n📝 Conversation History:")
            memory = chatbot.get_memory()
            if memory:
                print(memory)
            else:
                print("(empty)")
            print()
            continue
        
        elif user_input.lower() == 'clear':
            chatbot.clear_memory()
            print("\n🗑️  Conversation memory cleared!\n")
            continue
        
        elif user_input.lower() == 'verbose':
            verbose_state = chatbot.toggle_verbose()
            print(f"\n{'✅' if verbose_state else '❌'} Verbose mode: {verbose_state}\n")
            continue
        
        elif user_input.lower() == 'tools':
            print("\n🔧 Available Tools:")
            for tool in chatbot.tools:
                print(f"   • {tool.name}: {tool.description.split('.')[0]}")
            print()
            continue
        
        # Get chatbot response
        print()
        print("🤖 Assistant: ", end="", flush=True)
        response = chatbot.chat(user_input)
        print(response)
        print()


# ============================================================================
# Demo Mode
# ============================================================================

def run_demo():
    """
    Run a guided demo showing all capabilities.
    """
    print("=" * 70)
    print("Guided Demo: Contextual Chatbot")
    print("=" * 70)
    print()
    
    print("🚀 Initializing chatbot...")
    chatbot = ContextualChatbot(verbose=True)
    
    print("\n✅ Chatbot initialized with:")
    print("   • Document search (RAG)")
    print("   • Conversational memory")
    print("   • Calculator tool")
    print("   • Weather tool")
    print()
    
    # Demo conversation
    demo_questions = [
        ("What are the company benefits?", "Document Search (RAG)"),
        ("Calculate 15% of that salary if it's $80,000", "Calculator Tool + Memory"),
        ("What's the weather like in San Francisco?", "Weather Tool")
    ]
    
    for i, (question, capability) in enumerate(demo_questions, 1):
        print("=" * 70)
        print(f"Demo {i}/{len(demo_questions)}: {capability}")
        print("=" * 70)
        print()
        print(f"👤 Question: {question}")
        print()
        print(f"🤖 Processing...")
        print()
        
        response = chatbot.chat(question)
        
        print(f"✅ Response: {response}")
        print()
        
        if i < len(demo_questions):
            input("Press Enter to continue to next demo...\n")
    
    print("=" * 70)
    print("✅ Demo Completed!")
    print()
    print("💡 Key Takeaways:")
    print("   • Chatbot intelligently routes questions to appropriate tools")
    print("   • Memory maintains context across conversation")
    print("   • RAG provides accurate answers from documents")
    print("   • Tools extend capabilities beyond LLM knowledge")
    print("   • All features work together seamlessly")
    print()
    print("📚 This is a production-ready pattern for building contextual AI assistants!")
    print("=" * 70)


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Contextual Chatbot with RAG, Memory & Tools")
    parser.add_argument("--demo", action="store_true", help="Run guided demo")
    args = parser.parse_args()
    
    print("\n🚀 Example 4: Guided Project - Contextual Chatbot\n")
    
    if args.demo:
        run_demo()
    else:
        interactive_chat()


if __name__ == "__main__":
    main()
