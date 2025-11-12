"""
Example 3: Chains with Tools - Extending LLM Capabilities

This example demonstrates:
1. Creating custom tools with @tool decorator
2. Integrating tools with chains
3. Using agents for automatic tool selection
4. Combining multiple tools in one workflow
5. Error handling with tools
6. RAG tool integration with Milvus vector database
7. Real API integrations (weather, web search)

What are Tools?
- Tools extend LLM capabilities with external functions
- Can access real-time data, perform calculations, call APIs
- Tools make LLMs more useful and accurate

Tool Integration Patterns:
┌──────────┐     ┌─────────────┐     ┌──────────┐     ┌────────┐
│ User     │────▶│ LLM decides │────▶│ Execute  │────▶│ Return │
│ Question │     │ which tool  │     │ Tool     │     │ Answer │
└──────────┘     └─────────────┘     └──────────┘     └────────┘

Available Tools:
- calculator: Math operations
- weather_info: Real-time weather using wttr.in API
- search_web: Web search using DuckDuckGo
- rag_tool: Search internal documents from Milvus vector DB
- get_word_length: Count characters
- reverse_string: Reverse text

Prerequisites:
- Set OPENAI_API_KEY in .env file
- (Optional) Configure MILVUS_URI for RAG tool
- Install: pip install duckduckgo-search

Usage:
    python examples/example_3_tools.py          # Interactive mode
    python examples/example_3_tools.py --demo   # Demo mode
"""

import os
import sys
from dotenv import load_dotenv
from typing import Optional, List
import requests
import json
import warnings

# Suppress all LangChain deprecation warnings for training purposes
warnings.filterwarnings("ignore", message=".*LangChain.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate

# Import for DuckDuckGo search
try:
    from duckduckgo_search import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False
    print("⚠️  Warning: duckduckgo-search not installed. Web search will be limited.")

# Load environment variables
load_dotenv()

# ============================================================================
# Initialize Milvus Service (for RAG tool)
# ============================================================================
_milvus_retriever = None

def _get_milvus_retriever():
    """Get or initialize Milvus retriever (lazy loading)"""
    global _milvus_retriever
    
    if _milvus_retriever is not None:
        return _milvus_retriever
    
    try:
        from core.milvus_service import MilvusService
        
        # Check if Milvus is configured
        milvus_uri = os.getenv("MILVUS_URI")
        if not milvus_uri:
            return None
        
        # Initialize MilvusService
        service = MilvusService()
        service.connect(collection_name="training_demo")
        
        # Get retriever
        _milvus_retriever = service.get_retriever(k=3)
        return _milvus_retriever
        
    except Exception as e:
        print(f"⚠️  Milvus not available: {e}")
        return None


# ============================================================================
# Custom Tool Definitions
# ============================================================================

@tool
def calculator(expression: str) -> str:
    """
    Useful for mathematical calculations and arithmetic operations.
    Input should be a valid Python mathematical expression.
    Examples: "5 + 3", "10 * 2", "100 / 4", "(5 + 3) * 2"
    """
    try:
        # Safety: only allow numbers, operators, and parentheses
        allowed_chars = set("0123456789+-*/()%. ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression"
        
        result = eval(expression)
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error calculating: {str(e)}"


@tool
def get_word_length(word: str) -> str:
    """
    Returns the length (number of characters) of a word or phrase.
    Useful when you need to count characters.
    """
    length = len(word)
    return f"The word/phrase '{word}' has {length} characters"


@tool
def reverse_string(text: str) -> str:
    """
    Reverses a string of text.
    Useful for word games or text manipulation tasks.
    """
    reversed_text = text[::-1]
    return f"'{text}' reversed is '{reversed_text}'"


@tool
def weather_info(city: str) -> str:
    """
    Get current weather information for a city using wttr.in service.
    Input should be a city name (e.g., "San Francisco", "London").
    Returns current weather conditions, temperature, and humidity.
    """
    try:
        # Use wttr.in API - free, no API key required
        url = f"https://wttr.in/{city}?format=%C+%t+%h"
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            weather_data = response.text.strip()
            return f"Weather in {city}: {weather_data}"
        else:
            return f"Unable to fetch weather for {city}. Please check the city name."
            
    except requests.exceptions.Timeout:
        return f"Weather service timeout for {city}. Please try again."
    except Exception as e:
        return f"Error fetching weather: {str(e)}"


@tool
def search_web(query: str) -> str:
    """
    Search the web for information using DuckDuckGo search.
    Input should be a search query string.
    Returns relevant search results from the web.
    """
    if not DDGS_AVAILABLE:
        return "Web search is not available. Please install duckduckgo-search: pip install duckduckgo-search"
    
    try:
        # Use DuckDuckGo search
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            
            if not results:
                return f"No search results found for: {query}"
            
            # Format results
            formatted_results = []
            for i, result in enumerate(results, 1):
                title = result.get('title', 'No title')
                snippet = result.get('body', 'No description')
                formatted_results.append(f"{i}. {title}\n   {snippet}")
            
            return f"Search results for '{query}':\n\n" + "\n\n".join(formatted_results)
            
    except Exception as e:
        return f"Error performing web search: {str(e)}"


@tool
def rag_tool(query: str) -> str:
    """
    Search internal documents from the vector database using RAG (Retrieval Augmented Generation).
    Input should be a query about company policies, procedures, or documentation.
    Returns relevant information from the company's knowledge base.
    
    This tool demonstrates integration with Milvus vector database for document retrieval.
    Requires Milvus to be configured and running with documents loaded.
    """
    retriever = _get_milvus_retriever()
    
    if retriever is None:
        return ("RAG tool is not available. Make sure:\n"
                "1. Milvus is running\n"
                "2. MILVUS_URI is set in .env\n"
                "3. Documents are loaded (run: python scripts/build_milvus_store.py)")
    
    try:
        # Retrieve relevant documents
        docs = retriever.retrieve(query)
        
        if not docs:
            return f"No relevant documents found for: {query}"
        
        # Format results
        formatted_results = []
        for i, doc in enumerate(docs, 1):
            content = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            source = doc.metadata.get('source', 'Unknown')
            formatted_results.append(f"{i}. [Source: {source}]\n   {content}")
        
        return f"Documents found for '{query}':\n\n" + "\n\n".join(formatted_results)
        
    except Exception as e:
        return f"Error retrieving documents: {str(e)}"


# ============================================================================
# Demo Functions
# ============================================================================

def demo_basic_tool():
    """
    Demonstrate using a single tool directly.
    """
    print("=" * 70)
    print("Demo 1: Basic Tool Usage")
    print("=" * 70)
    print()
    
    print("🔧 Available Tool: Calculator")
    print()
    
    # Test the calculator tool directly
    expressions = ["5 + 3", "10 * 7", "100 / 4", "(5 + 3) * 2"]
    
    for expr in expressions:
        print(f"Expression: {expr}")
        result = calculator.invoke(expr)
        print(f"Result: {result}")
        print()
    
    print("💡 Key Concept:")
    print("   Tools are functions decorated with @tool that LLMs can call.")
    print()


def demo_agent_with_tools():
    """
    Demonstrate using an agent that can select and use multiple tools.
    """
    print("=" * 70)
    print("Demo 2: Agent with Multiple Tools")
    print("=" * 70)
    print()
    
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Define available tools
    tools = [calculator, get_word_length, reverse_string, weather_info]
    
    print("🔧 Available Tools:")
    for tool in tools:
        print(f"   • {tool.name}: {tool.description.split('.')[0]}")
    print()
    
    # Create ReAct agent prompt
    react_prompt = PromptTemplate.from_template("""
Answer the following question as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}
""")
    
    # Create agent
    agent = create_react_agent(llm, tools, react_prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5
    )
    
    # Test questions
    questions = [
        "What is 523 multiplied by 47?",
        "What's the weather like in Pune?",
        "How many letters are in the word 'hippopotamus'?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"─" * 70)
        print(f"Question {i}: {question}")
        print(f"─" * 70)
        
        try:
            result = agent_executor.invoke({"input": question})
            print(f"\n✅ Answer: {result['output']}\n")
        except Exception as e:
            print(f"❌ Error: {e}\n")
    
    print("💡 Key Concept:")
    print("   Agents automatically choose which tool to use based on the question.")
    print()


def demo_chain_with_tool():
    """
    Demonstrate integrating a tool into a specific chain.
    """
    print("=" * 70)
    print("Demo 3: Chain with Explicit Tool Call")
    print("=" * 70)
    print()
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Chain that uses calculator for any math question
    prompt = ChatPromptTemplate.from_template("""
You are a helpful math assistant. 

When asked a math question:
1. Extract the mathematical expression
2. Use the calculator tool to compute it
3. Explain the result

Question: {question}

Let me calculate that for you.
""")
    
    chain = LLMChain(llm=llm, prompt=prompt, verbose=False)
    
    question = "If I have 15 apples and buy 23 more, how many do I have?"
    
    print(f"Question: {question}")
    print()
    
    # Get LLM to formulate the expression
    response = chain.predict(question=question)
    print(f"LLM Response: {response}")
    print()
    
    # Extract expression and use calculator
    expression = "15 + 23"
    calc_result = calculator.invoke(expression)
    print(f"Calculator Tool: {calc_result}")
    print()
    
    print("💡 Key Concept:")
    print("   Chains can explicitly call tools at specific steps.")
    print()


def demo_sequential_tool_calls():
    """
    Demonstrate an agent making multiple tool calls in sequence to answer one question.
    This clearly shows how agents can chain tool usage.
    """
    print("=" * 70)
    print("Demo 3.5: Sequential Multi-Tool Usage")
    print("=" * 70)
    print()
    print("This demo shows an agent using MULTIPLE tools to answer a single question.")
    print()
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [calculator, get_word_length, weather_info, search_web]
    
    print("🔧 Available Tools:")
    for tool in tools:
        print(f"   • {tool.name}: {tool.description.split('.')[0]}")
    print()
    
    # Create ReAct agent with explicit instructions
    react_prompt = PromptTemplate.from_template("""
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}
""")
    
    agent = create_react_agent(llm, tools, react_prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10
    )
    
    # Questions that naturally require multiple tools
    questions = [
        "What is the weather in Mumbai? Then multiply the temperature number by 2.",
        "How many letters are in the word 'hippopotamus' and what is that number times 5?",
        "Calculate 15 + 7, then tell me how many characters are in the result when written as text."
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"{'═' * 70}")
        print(f"Example {i}: {question}")
        print(f"{'─' * 70}")
        print("Watch how the agent:")
        print("  1️⃣  Identifies it needs multiple tools")
        print("  2️⃣  Calls the first tool and gets a result")
        print("  3️⃣  Uses that result with the second tool")
        print("  4️⃣  Provides a final answer")
        print(f"{'─' * 70}")
        print()
        
        try:
            result = agent_executor.invoke({"input": question})
            print(f"\n✅ Final Answer: {result['output']}\n")
        except Exception as e:
            print(f"❌ Error: {e}\n")
        
        if i < len(questions):
            input("Press Enter to see next example...\n")
    
    print("💡 Key Concept:")
    print("   Agents can chain multiple tool calls together automatically.")
    print("   Each tool's output can be used as input for the next tool.")
    print("   The agent decides the sequence based on the question logic.")
    print()


def demo_multi_tool_workflow():
    """
    Demonstrate a workflow using multiple tools in sequence.
    """
    print("=" * 70)
    print("Demo 4: Multi-Tool Workflow")
    print("=" * 70)
    print()
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [calculator, weather_info, search_web]
    
    print("🔧 Available Tools: calculator, weather_info, search_web")
    print()
    
    # Create agent with better ReAct prompt
    react_prompt = PromptTemplate.from_template("""
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

IMPORTANT: 
- Use weather_info to get weather data (returns temperature as text like "15°C")
- Use calculator to perform mathematical operations on numeric values
- When you need to do math on weather data, first get the weather, then extract the number, then calculate

Begin!

Question: {input}
Thought: {agent_scratchpad}
""")
    
    agent = create_react_agent(llm, tools, react_prompt)    
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=10,  # Increased iterations to allow multiple tool calls
        handle_parsing_errors=True
    )
    
    # Questions demonstrating multi-tool usage
    questions = [
        "What is the weather in London and what is 25 times 4?",
        "Search for 'Python programming' and tell me how many letters are in the word 'programming'"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"{'═' * 70}")
        print(f"Question {i}: {question}")
        print(f"{'─' * 70}")
        print()
        
        try:
            result = agent_executor.invoke({"input": question})
            print(f"\n✅ Final Answer: {result['output']}\n")
        except Exception as e:
            print(f"❌ Error: {e}\n")
        
        if i < len(questions):
            input("Press Enter to continue to next question...\n")
    
    print("💡 Key Concept:")
    print("   Agents can automatically select and chain multiple tool calls.")
    print("   The agent decides which tools to use and in what order based on the question.")
    print()


def interactive_tool_chat():
    """
    Interactive mode: Chat with tool-enabled agent.
    """
    print("=" * 70)
    print("Interactive Tool-Enabled Chat")
    print("=" * 70)
    print()
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Define tools
    tools = [calculator, get_word_length, reverse_string, weather_info, search_web, rag_tool]
    
    print("🔧 Available Tools:")
    for tool in tools:
        print(f"   • {tool.name}")
    print()
    
    # Create agent
    react_prompt = PromptTemplate.from_template("""
Answer questions using available tools when needed.

Tools: {tools}
Tool Names: {tool_names}

Format:
Question: {input}
Thought: {agent_scratchpad}

Question: {input}
""")
    
    agent = create_react_agent(llm, tools, react_prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,  # Set to True to see agent reasoning
        handle_parsing_errors=True,
        max_iterations=5
    )
    
    print("💬 Ask questions! The agent will use tools when needed.")
    print()
    print("Try asking:")
    print("  • 'What is 523 * 847?'  (uses calculator)")
    print("  • 'What's the weather in London?'  (uses weather_info)")
    print("  • 'Search for latest AI news'  (uses search_web)")
    print("  • 'What are the company vacation policies?'  (uses rag_tool)")
    print()
    print("Commands:")
    print("  • 'tools' - List available tools")
    print("  • 'verbose' - Toggle verbose mode")
    print("  • 'quit' - Exit")
    print()
    
    verbose = False
    
    while True:
        print("─" * 70)
        user_input = input("You: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() == 'quit':
            print("👋 Goodbye!")
            break
        
        if user_input.lower() == 'tools':
            print("\n🔧 Available Tools:")
            for tool in tools:
                print(f"   • {tool.name}: {tool.description.split('.')[0]}")
            print()
            continue
        
        if user_input.lower() == 'verbose':
            verbose = not verbose
            agent_executor.verbose = verbose
            print(f"{'✅' if verbose else '❌'} Verbose mode: {verbose}\n")
            continue
        
        # Get response from agent
        print()
        try:
            result = agent_executor.invoke({"input": user_input})
            print(f"Assistant: {result['output']}")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print()


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="LangChain Tools Examples")
    parser.add_argument("--demo", action="store_true", help="Run demo mode")
    args = parser.parse_args()
    
    print("\n🚀 Example 3: Chains with Tools\n")
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found in environment variables")
        print("   Please add it to your .env file")
        sys.exit(1)
    
    if args.demo:
        # Run all demos
        demo_basic_tool()
        input("Press Enter to continue...\n")
        
        demo_agent_with_tools()
        input("Press Enter to continue...\n")
        
        demo_chain_with_tool()
        input("Press Enter to continue...\n")
        
        demo_sequential_tool_calls()
        input("Press Enter to continue...\n")
        
        demo_multi_tool_workflow()
        
        print("=" * 70)
        print("✅ All demos completed!")
        print()
        print("💡 Key Takeaways:")
        print("   • Tools extend LLM capabilities with external functions")
        print("   • @tool decorator creates LangChain-compatible tools")
        print("   • Agents automatically select which tool to use")
        print("   • Agents can chain multiple tools together sequentially")
        print("   • Real API integrations: weather (wttr.in), search (DuckDuckGo)")
        print("   • RAG tool connects to Milvus vector database")
        print("   • Tools enable access to real-time data and computations")
        print()
        print("📚 Next: Run example_4_guided_project.py for complete integration!")
    else:
        # Interactive mode
        interactive_tool_chat()


if __name__ == "__main__":
    main()
