"""
Streamlit Chat Application - API Client

This is a simplified Streamlit application that calls the FastAPI backend.
All RAG logic is handled by the ChatService via REST API endpoints.

Features:
• Clean chat interface
• Real-time streaming responses from API
• Vector store selection (FAISS/Milvus)
• Configuration options
• Source document display
• Conversation history

Usage:
    1. Start the API server: python scripts/start_api.py
    2. Run this app: streamlit run streamlit_app/app.py

Prerequisites:
- API server running on http://localhost:8000
- Vector store built (FAISS or Milvus)
"""

import streamlit as st
import requests
from typing import Optional

# Page configuration
st.set_page_config(
    page_title="RAG Chat Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .source-doc {
        background-color: #f0f2f6;
        padding: 0.5rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .api-status {
        padding: 0.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .api-healthy {
        background-color: #d4edda;
        color: #155724;
    }
    .api-unhealthy {
        background-color: #f8d7da;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "api_url" not in st.session_state:
    st.session_state.api_url = "http://localhost:8000"


def check_api_health() -> dict:
    """Check if the API server is running and healthy"""
    try:
        response = requests.get(f"{st.session_state.api_url}/health", timeout=2)
        if response.status_code == 200:
            return response.json()
        else:
            return {"status": "unhealthy", "message": f"API returned {response.status_code}"}
    except requests.exceptions.ConnectionError:
        return {"status": "unreachable", "message": "Cannot connect to API server"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def get_api_config() -> Optional[dict]:
    """Get current API configuration"""
    try:
        response = requests.get(f"{st.session_state.api_url}/config", timeout=2)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def update_api_config(temperature: Optional[float] = None, k: Optional[int] = None, llm_model: Optional[str] = None) -> bool:
    """Update API configuration"""
    try:
        payload = {}
        if temperature is not None:
            payload["temperature"] = temperature
        if k is not None:
            payload["k"] = k
        if llm_model is not None:
            payload["llm_model"] = llm_model
        
        response = requests.post(
            f"{st.session_state.api_url}/config",
            json=payload,
            timeout=5
        )
        return response.status_code == 200
    except:
        return False


def chat_with_api(query: str, k: Optional[int] = None) -> Optional[dict]:
    """Send a chat request to the API"""
    try:
        payload = {"query": query}
        if k is not None:
            payload["k"] = k
        
        response = requests.post(
            f"{st.session_state.api_url}/chat",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.Timeout:
        st.error("Request timed out. Please try again.")
        return None
    except Exception as e:
        st.error(f"Error calling API: {str(e)}")
        return None


def chat_stream_with_api(query: str, k: Optional[int] = None):
    """Send a streaming chat request to the API"""
    try:
        payload = {"query": query}
        if k is not None:
            payload["k"] = k
        
        response = requests.post(
            f"{st.session_state.api_url}/chat/stream",
            json=payload,
            stream=True,
            timeout=30
        )
        
        if response.status_code == 200:
            for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                if chunk:
                    yield chunk
        else:
            st.error(f"API Error: {response.status_code}")
    except Exception as e:
        st.error(f"Error streaming from API: {str(e)}")


# Header
st.markdown('<div class="main-header">🤖 RAG Chat Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Powered by FastAPI & Streamlit</div>', unsafe_allow_html=True)

# Sidebar - Simplified
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Health Check
    health = check_api_health()
    
    if health["status"] == "healthy":
        st.markdown(
            f'<div class="api-status api-healthy">✅ API Connected</div>',
            unsafe_allow_html=True
        )
    elif health["status"] == "unreachable":
        st.markdown(
            f'<div class="api-status api-unhealthy">❌ {health["message"]}</div>',
            unsafe_allow_html=True
        )
        st.warning("Start the API server with: `python scripts/start_api.py`")
        st.stop()
    else:
        st.markdown(
            f'<div class="api-status api-unhealthy">⚠️ {health["message"]}</div>',
            unsafe_allow_html=True
        )
    
    st.divider()
    
    # Clear conversation
    if st.button("�️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Default k value (hidden from user)
k_value = 3

# Main chat interface
st.markdown("---")

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Sources hidden per user request
        # if message["role"] == "assistant" and "sources" in message:
        #     with st.expander("📚 View Sources"):
        #         for i, doc in enumerate(message["sources"], 1):
        #             st.markdown(f"""
        #             <div class="source-doc">
        #                 <strong>Source {i}:</strong> {doc['source']} (Page {doc['page']})
        #                 <br><em>{doc['content'][:200]}...</em>
        #             </div>
        #             """, unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ask a question about the documents..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get AI response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # Use streaming API
        full_response = ""
        
        try:
            for chunk in chat_stream_with_api(prompt, k=k_value):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            
            # Get sources (make a separate non-streaming call)
            result = chat_with_api(prompt, k=k_value)
            sources = result.get("sources", []) if result else []
            
            # Sources hidden per user request
            # if sources:
            #     with st.expander("📚 View Sources"):
            #         for i, doc in enumerate(sources, 1):
            #             st.markdown(f"""
            #             <div class="source-doc">
            #                 <strong>Source {i}:</strong> {doc['source']} (Page {doc['page']})
            #                 <br><em>{doc['content'][:200]}...</em>
            #             </div>
            #             """, unsafe_allow_html=True)
            
            # Add assistant message to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "sources": sources
            })
            
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #666; font-size: 0.9rem;">'
    'Powered by <strong>FastAPI</strong>, <strong>LangChain</strong>, and <strong>OpenAI</strong>'
    '</div>',
    unsafe_allow_html=True
)
