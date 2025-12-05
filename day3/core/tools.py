"""
Tool Definitions - Reusable Tools for Multi-Agent Systems

This module provides a collection of tools that agents can use to:
- Perform calculations
- Get information (weather, time, etc.)
- Search documents
- Access external APIs

These tools can be used with ReAct agents in multi-agent workflows.
"""

import os
from typing import Optional
from langchain.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from datetime import datetime


# ============================================================================
# Basic Utility Tools
# ============================================================================

@tool
def calculator(expression: str) -> str:
    """
    Performs mathematical calculations.
    Input should be a valid mathematical expression like '5 + 3', '10 * 2', or '(50 + 30) / 4'.
    
    Args:
        expression: Mathematical expression to evaluate
        
    Returns:
        Calculation result as a string
    """
    try:
        # Validate only safe characters
        allowed_chars = set("0123456789+-*/()%. ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression. Use only numbers and operators: + - * / ( )"
        
        result = eval(expression)
        return f"Calculation result: {expression} = {result}"
    except Exception as e:
        return f"Error in calculation: {str(e)}"


@tool
def get_current_time(timezone: str = "UTC") -> str:
    """
    Get the current time and date.
    Input should be a timezone like 'UTC', 'America/New_York', 'Asia/Tokyo', etc.
    
    Args:
        timezone: Timezone name (default: UTC)
        
    Returns:
        Current time and date as a string
    """
    try:
        current_time = datetime.now()
        return f"Current time ({timezone}): {current_time.strftime('%Y-%m-%d %H:%M:%S')}"
    except Exception as e:
        return f"Error getting time: {str(e)}"


@tool
def weather_info(city: str) -> str:
    """
    Get current weather information for a city.
    Input should be a city name like 'San Francisco', 'London', 'Tokyo', etc.
    
    Args:
        city: City name
        
    Returns:
        Weather information as a string
    """
    # Mock weather data for demo purposes
    # In production, this would call a real weather API
    mock_weather = {
        "san francisco": {"temp": "18°C", "condition": "Partly Cloudy", "humidity": "65%"},
        "london": {"temp": "12°C", "condition": "Rainy", "humidity": "80%"},
        "tokyo": {"temp": "20°C", "condition": "Sunny", "humidity": "55%"},
        "new york": {"temp": "15°C", "condition": "Clear", "humidity": "60%"},
        "paris": {"temp": "14°C", "condition": "Cloudy", "humidity": "70%"},
        "seattle": {"temp": "16°C", "condition": "Rainy", "humidity": "75%"},
        "mumbai": {"temp": "30°C", "condition": "Hot and Humid", "humidity": "85%"},
        "sydney": {"temp": "22°C", "condition": "Sunny", "humidity": "60%"},
        "berlin": {"temp": "10°C", "condition": "Overcast", "humidity": "72%"},
        "toronto": {"temp": "8°C", "condition": "Cold", "humidity": "55%"}
    }
    
    city_lower = city.lower()
    if city_lower in mock_weather:
        w = mock_weather[city_lower]
        return f"Weather in {city}: {w['condition']}, Temperature: {w['temp']}, Humidity: {w['humidity']}"
    else:
        available = ", ".join([c.title() for c in list(mock_weather.keys())[:5]])
        return f"Weather data not available for {city}. Try: {available}, and others."


# ============================================================================
# Document Search Tool (RAG)
# ============================================================================

@tool
def search_documents(query: str) -> str:
    """
    Search company documents for information using semantic search (RAG).
    Use this tool to answer questions about company policies, products, or technical documentation.
    Input should be a search query related to company information.
    
    Args:
        query: Search query
        
    Returns:
        Retrieved document excerpts
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
            return "No relevant documents found for the query."
        
        # Format context
        context = "\n\n".join([
            f"From {doc.metadata.get('source', 'Unknown')} (page {doc.metadata.get('page', '?')}):\n{doc.page_content}"
            for doc in docs
        ])
        
        return f"Retrieved information from company documents:\n\n{context}"
    
    except Exception as e:
        return f"Error searching documents: {str(e)}"


# ============================================================================
# Information Lookup Tools
# ============================================================================

@tool
def company_info(topic: str) -> str:
    """
    Get information about company topics like benefits, policies, culture, etc.
    Input should be a topic like 'benefits', 'vacation policy', 'remote work', etc.
    
    Args:
        topic: Company topic to look up
        
    Returns:
        Company information
    """
    # Mock company information
    # In production, this might query a database or knowledge base
    info_db = {
        "benefits": """Company Benefits:
- Health Insurance: Comprehensive medical, dental, and vision coverage
- 401(k): Company matches up to 5% of salary
- PTO: 20 days paid time off + 10 holidays
- Professional Development: $2,000 annual learning budget
- Remote Work: Flexible hybrid work arrangement
- Wellness: Free gym membership and mental health support""",
        
        "vacation": """Vacation Policy:
- 20 days PTO per year (increases with tenure)
- 10 company holidays
- Unlimited sick days
- Birthday off
- 4-week sabbatical after 5 years""",
        
        "remote work": """Remote Work Policy:
- Hybrid model: 2-3 days in office per week
- Fully remote available for certain roles
- Home office stipend: $500
- Co-working space reimbursement available
- Flexible hours (core hours: 10am-3pm)""",
        
        "culture": """Company Culture:
- Innovation-driven and collaborative
- Inclusive and diverse workplace
- Focus on work-life balance
- Transparent communication from leadership
- Continuous learning encouraged
- Community involvement and volunteering""",
    }
    
    topic_lower = topic.lower()
    for key in info_db:
        if key in topic_lower:
            return info_db[key]
    
    return f"No specific information found for '{topic}'. Available topics: {', '.join(info_db.keys())}"


@tool
def web_search(query: str) -> str:
    """
    Search the web for current information.
    Use this for questions requiring up-to-date information from the internet.
    Input should be a search query.
    
    Args:
        query: Search query
        
    Returns:
        Search results (mock data for demo)
    """
    # Mock search results
    # In production, integrate with Google Search API, Bing API, or SerpAPI
    return f"""Web search results for '{query}':

1. Recent trends and developments in {query}
2. Expert opinions and analysis on {query}
3. Latest news and updates about {query}

Note: This is mock data. Integrate a real search API for production use."""


# ============================================================================
# Data Processing Tools
# ============================================================================

@tool
def text_analyzer(text: str) -> str:
    """
    Analyze text for word count, sentiment, and key characteristics.
    Input should be the text to analyze.
    
    Args:
        text: Text to analyze
        
    Returns:
        Analysis results
    """
    try:
        words = text.split()
        word_count = len(words)
        char_count = len(text)
        sentence_count = text.count('.') + text.count('!') + text.count('?')
        
        # Simple sentiment analysis based on keywords
        positive_words = ['good', 'great', 'excellent', 'happy', 'success', 'wonderful']
        negative_words = ['bad', 'poor', 'terrible', 'sad', 'failure', 'awful']
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            sentiment = "Positive"
        elif neg_count > pos_count:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
        
        return f"""Text Analysis Results:
- Word Count: {word_count}
- Character Count: {char_count}
- Sentence Count: {sentence_count}
- Sentiment: {sentiment}
- Average Word Length: {char_count / word_count if word_count > 0 else 0:.1f} characters"""
    
    except Exception as e:
        return f"Error analyzing text: {str(e)}"


# ============================================================================
# Tool Collections
# ============================================================================

def get_basic_tools():
    """Get basic utility tools."""
    return [calculator, get_current_time, weather_info]


def get_research_tools():
    """Get tools for research agents."""
    return [search_documents, web_search, company_info]


def get_writer_tools():
    """Get tools for writer agents."""
    return [text_analyzer, calculator]


def get_all_tools():
    """Get all available tools."""
    return [
        calculator,
        get_current_time,
        weather_info,
        search_documents,
        company_info,
        web_search,
        text_analyzer
    ]


# ============================================================================
# Tool Registry
# ============================================================================

TOOL_REGISTRY = {
    "basic": get_basic_tools,
    "research": get_research_tools,
    "writer": get_writer_tools,
    "all": get_all_tools
}

# Individual tool mapping for config-based loading
TOOL_MAP = {
    "calculator": calculator,
    "get_current_time": get_current_time,
    "weather_info": weather_info,
    "search_documents": search_documents,
    "company_info": company_info,
    "web_search": web_search,
    "text_analyzer": text_analyzer
}


def get_tools_by_category(category: str):
    """
    Get tools by category name.
    
    Args:
        category: Tool category ('basic', 'research', 'writer', 'all')
        
    Returns:
        List of tools
    """
    if category in TOOL_REGISTRY:
        return TOOL_REGISTRY[category]()
    else:
        available = ", ".join(TOOL_REGISTRY.keys())
        raise ValueError(f"Unknown tool category: {category}. Available: {available}")


def get_tools_by_name(tool_names: list):
    """
    Get specific tools by their names.
    
    This function is used by the config-based agent creation to load
    tools specified in the YAML configuration.
    
    Args:
        tool_names: List of tool names (strings)
        
    Returns:
        List of tool objects
        
    Example:
        tools = get_tools_by_name(["calculator", "text_analyzer"])
    """
    tools = []
    for tool_name in tool_names:
        if tool_name not in TOOL_MAP:
            available = ", ".join(TOOL_MAP.keys())
            raise ValueError(f"Unknown tool: {tool_name}. Available: {available}")
        tools.append(TOOL_MAP[tool_name])
    return tools

