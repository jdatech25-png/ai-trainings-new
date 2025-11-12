# Day 2: Chains, Memory & Tools in LangChain - Complete Training Guide

## 🎯 Training Objectives
- Understand and implement LangChain chains for structured AI workflows
- Add conversational memory to maintain context across interactions
- Integrate external tools and APIs with LangChain agents
- Build a contextual chatbot that combines RAG, memory, and tool execution
- Master production-ready patterns for conversational AI systems

---

## 📋 Table of Contents
1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Project Structure](#project-structure)
4. [Detailed Setup Guide](#detailed-setup-guide)
5. [Running the Examples](#running-the-examples)
6. [Core Concepts](#core-concepts)
7. [Training Exercises](#training-exercises)
8. [Troubleshooting](#troubleshooting)
9. [Learning Outcomes](#learning-outcomes)

---

## 📋 Prerequisites

- **Python 3.10+** installed
- **OpenAI API Key** ([Get one here](https://platform.openai.com/api-keys))
- **Internet connection** for real-time API tools (weather, web search)
- **(Optional)** Milvus server for RAG tool in Example 3 and Example 4

---

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Activate virtual environment
source .venv2/bin/activate  # On macOS/Linux
# .venv2\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 2. Generate Sample Documents
```bash
python scripts/generate_sample_pdfs.py
```

### 3. Run Examples in Order
```bash
# Example 1: Simple LLMChain
python examples/example_1_simple_chain.py

# Example 2: Conversational Memory
python examples/example_2_memory.py

# Example 3: Chains with Tools
python examples/example_3_tools.py

# Example 4: Guided Project - Contextual Chatbot with RAG, Memory & Tools
python examples/example_4_guided_project.py
```

---

## 📁 Project Structure

```
day2/
│
├── README.md                          # This comprehensive guide
├── requirements.txt                   # Python dependencies
├── .env.example                       # Environment variables template
├── .gitignore                         # Git ignore rules
│
├── docs/                              # Sample PDF documents
│   ├── company_handbook.pdf          # Sample company policies
│   ├── product_guide.pdf             # Sample product documentation
│   └── technical_manual.pdf          # Sample technical docs
│
├── core/                              # Reusable components
│   ├── __init__.py
│   ├── chains.py                     # LangChain chain utilities
│   ├── memory_manager.py             # Memory management
│   ├── tools.py                      # Custom tool definitions
│   ├── chat_service.py               # Enhanced RAG service with memory
│   ├── document_loader.py            # PDF loading and chunking
│   ├── embeddings.py                 # Embedding model wrapper
│   ├── vector_stores.py              # FAISS and Milvus abstractions
│   ├── retrievers.py                 # Document retrieval logic
│   └── generators.py                 # LLM answer generation
│
├── scripts/                           # Utility scripts
│   ├── generate_sample_pdfs.py       # Generate sample PDFs
│   ├── build_faiss_store.py          # Build FAISS vector index
│   └── build_milvus_store.py         # Build Milvus collection
│
├── examples/                          # Progressive training examples
│   ├── __init__.py
│   ├── example_1_simple_chain.py     # Basic LLMChain patterns
│   ├── example_2_memory.py           # Conversational memory
│   ├── example_3_tools.py            # Tool integration
│   └── example_4_guided_project.py   # Complete chatbot with all features
│
└── streamlit_app/                     # Web application with memory
    ├── __init__.py
    └── app.py                         # Enhanced Streamlit app
```

---

## 🔧 Detailed Setup Guide

### Step 1: Environment Setup

```bash
# Navigate to project directory
cd /path/to/day1

# Activate virtual environment
source .venv2/bin/activate  # On macOS/Linux
# OR
.venv2\Scripts\activate  # On Windows

# Install all dependencies
pip install -r requirements.txt
```

**Expected output**: All packages installed successfully

### Step 2: Configure Environment Variables

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your OpenAI API key
nano .env  # or use your preferred editor
```

**Required in .env**:
```env
OPENAI_API_KEY=sk-your-actual-api-key-here
```

**Optional** (for Example 4 - Milvus):
```env
MILVUS_URI=tcp://localhost:19530
MILVUS_COLLECTION_NAME=training_demo
```

### Step 3: Verify Installation

```bash
# Test core imports
python -c "from core.document_loader import load_local_pdfs; print('✅ Core modules OK')"

# Check if sample PDFs exist
ls docs/*.pdf
```

---

## 📚 Running the Examples

### Example 1: Simple LLMChain

**Purpose**: Learn the fundamentals of LangChain chains for structured AI workflows

```bash
# Interactive mode (default)
python examples/example_1_simple_chain.py

# Demo mode (sample chains)
python examples/example_1_simple_chain.py --demo
```

**What to observe**:
- Basic chain composition (Prompt → LLM → Parser)
- Sequential chain patterns
- Input/output handling
- Structured responses

**Key concepts**:
- LLMChain basics
- PromptTemplate
- OutputParser
- Chain composition (LCEL - LangChain Expression Language)

---

### Example 2: Conversational Memory

**Purpose**: Add memory to maintain context across conversations

```bash
# Interactive mode with different memory types
python examples/example_2_memory.py

# Demo mode comparing memory types
python examples/example_2_memory.py --demo

# Try specific memory type
python examples/example_2_memory.py --memory buffer
python examples/example_2_memory.py --memory entity
```

**What to observe**:
- ConversationBufferMemory (stores all messages)
- ConversationEntityMemory (tracks entities)
- How context is maintained across turns
- Memory retrieval and formatting

**Key concepts**:
- Conversation memory types
- Context window management
- Entity extraction
- Memory integration with chains

**Interactive commands**:
- Type your messages naturally
- Watch how the AI references previous context
- Type `memory` to view memory contents
- Type `clear` to reset memory
- Type `quit` to exit

---

### Example 3: Chains with Tools

**Purpose**: Integrate external tools and APIs with LangChain

```bash
# Interactive mode
python examples/example_3_tools.py

# Demo mode with sample tool calls
python examples/example_3_tools.py --demo
```

**What to observe**:
- Custom tool creation with `@tool` decorator
- Tool selection and execution by agents
- Real API integrations (weather, web search, RAG)
- Multiple tools working together

**Available Tools**:
- **calculator**: Mathematical calculations
- **weather_info**: Real-time weather using wttr.in API
- **search_web**: Web search using DuckDuckGo
- **rag_tool**: Search internal documents from Milvus vector database
- **get_word_length**: Count characters in text
- **reverse_string**: Text manipulation

**Key concepts**:
- Tool definition with `@tool` decorator
- Agent-based tool selection (ReAct pattern)
- Real API integration vs mock data
- RAG integration with vector databases
- Error handling with tools

**Try asking**:
- "What's 523 * 847?"  (uses calculator tool)
- "What's the weather in London?"  (uses weather_info with wttr.in API)
- "Search for latest AI news"  (uses search_web with DuckDuckGo)
- "What are the company vacation policies?"  (uses rag_tool with Milvus)

**Requirements**:
- OpenAI API key (required)
- `pip install duckduckgo-search` for web search
- Milvus configured for RAG tool (optional)

---

### Example 4: Guided Project - Contextual Chatbot

**Purpose**: Build a complete chatbot combining RAG, memory, and tools

```bash
# Make sure FAISS index is built first
python scripts/build_faiss_store.py

# Run the guided project
python examples/example_4_guided_project.py
```

**What to observe**:
- RAG retrieval for document questions
- Memory maintaining conversation context
- Tools for external data/actions
- Seamless integration of all features

**Key concepts**:
- Multi-modal chatbot architecture
- RAG + Memory + Tools pattern
- Production-ready design
- Error handling and graceful degradation

**Sample conversation flow**:
```
You: What are the company benefits?
Bot: [Uses RAG to retrieve from documents]

You: Can you remind me what we just discussed?
Bot: [Uses memory to recall previous context]

You: What's 25% of my salary if it's $80,000?
Bot: [Uses calculator tool]

You: Compare that to the industry average
Bot: [Uses search tool + reasoning]
```

---

## 🧩 Core Concepts

### Overview

Day 2 focuses on three key LangChain concepts that enable powerful conversational AI:

1. **Chains**: Structured workflows for AI tasks
2. **Memory**: Context retention across conversations
3. **Tools**: External capabilities and APIs

### 1. LangChain Chains

**What are Chains?**

Chains are composable building blocks that link prompts, models, and processing steps:

```python
# Simple chain
chain = prompt | llm | output_parser

# Sequential chain
chain = prompt1 | llm1 | parser1 | prompt2 | llm2 | parser2
```

**Why use Chains?**
- ✅ Structured, repeatable workflows
- ✅ Easier debugging and testing
- ✅ Composable and reusable
- ✅ Production-ready patterns

**Chain Types**:
- **LLMChain**: Basic prompt → LLM → output
- **Sequential Chain**: Multi-step processing
- **Router Chain**: Conditional branching
- **Retrieval Chain**: RAG patterns

---

### 2. Conversational Memory

**What is Memory?**

Memory allows chatbots to remember previous interactions:

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

# First turn
chain.run("My name is Alice")  # Stored in memory

# Second turn
chain.run("What's my name?")   # Retrieves from memory
# Output: "Your name is Alice"
```

**Memory Types**:

| Memory Type | Use Case | Pros | Cons |
|------------|----------|------|------|
| **ConversationBufferMemory** | Short conversations | Simple, accurate | Grows with conversation |
| **ConversationSummaryMemory** | Long conversations | Compact | May lose details |
| **ConversationEntityMemory** | Track entities | Extracts key info | More complex |
| **ConversationBufferWindowMemory** | Recent context | Fixed size | Forgets old messages |

**When to use Memory**:
- ✅ Multi-turn conversations
- ✅ Personalized experiences
- ✅ Context-dependent responses
- ✅ Follow-up questions

---

### 3. Tools and Agents

**What are Tools?**

Tools extend LLM capabilities with external functions:

```python
from langchain.tools import tool

@tool
def calculator(expression: str) -> str:
    """Useful for mathematical calculations."""
    return str(eval(expression))

@tool
def weather_info(city: str) -> str:
    """Get current weather for a city using wttr.in API."""
    response = requests.get(f"https://wttr.in/{city}?format=%C+%t+%h")
    return f"Weather in {city}: {response.text}"

@tool
def search_web(query: str) -> str:
    """Search the web using DuckDuckGo."""
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=3))
        return format_results(results)

@tool
def rag_tool(query: str) -> str:
    """Search internal documents from Milvus vector database."""
    retriever = milvus_service.get_retriever(k=3)
    docs = retriever.retrieve(query)
    return format_documents(docs)
```

**Tool Integration Patterns**:

1. **With Chains**: Explicit tool selection
2. **With Agents**: LLM decides which tool to use (ReAct pattern)

**Available Real Tools in Example 3**:
- **calculator**: Math operations (built-in)
- **weather_info**: Real-time weather via wttr.in API
- **search_web**: Web search via DuckDuckGo API
- **rag_tool**: Document search via Milvus vector DB
- **get_word_length**: Character counting
- **reverse_string**: Text manipulation

**Benefits**:
- ✅ Access real-time data from APIs
- ✅ Perform complex calculations
- ✅ Search internal knowledge bases (RAG)
- ✅ Search the web for current information
- ✅ Integrate with existing systems

---

## � Learning Path

### Module 1: Understanding Chains (45 mins)
**Theory**: Why chains? LCEL syntax, composition patterns

**Example**: 1 (Simple LLMChain)

**Concepts**:
- Prompt → LLM → Parser pattern
- Chain composition with `|` operator
- Input/output schemas
- Error handling

**Hands-on**:
- Create a simple translation chain
- Build a multi-step reasoning chain
- Debug chain execution

---

### Module 2: Conversational Memory (45 mins)
**Theory**: Memory types, context management, token limits

**Example**: 2 (Memory)

**Concepts**:
- Buffer vs Summary vs Entity memory
- Memory integration with chains
- Context window management
- Memory persistence

**Hands-on**:
- Compare different memory types
- Build a personal assistant with memory
- Handle long conversations

---

### Module 3: Tools Integration (45 mins)
**Theory**: Tool definition, function calling, agent patterns, real API integration

**Example**: 3 (Tools)

**Concepts**:
- Custom tool creation with `@tool` decorator
- Real API integrations (weather, web search)
- RAG tool with vector database
- Agent-based tool selection
- Error handling with external services

**Hands-on**:
- Create custom tools for your use case
- Integrate real APIs (weather, search)
- Connect tools to Milvus vector database
- Build agents that select appropriate tools
- Handle API failures gracefully

---
- Tool selection and execution
- Agent vs ReAct patterns
- Error handling

**Hands-on**:
- Create custom tools
- Build a tool-using chain
- Handle tool failures gracefully

---

### Module 4: Guided Project (60 mins)
**Example**: 4 (Complete Chatbot)

**Integration**:
- RAG for document Q&A
- Memory for conversation context
- Tools for external actions

**Production Patterns**:
- Error handling
- Fallback strategies
- Performance optimization
- User experience

---

## � Learning Outcomes

By the end of Day 2, participants should be able to:

- [ ] Understand LangChain chain composition patterns (LCEL)
- [ ] Create structured workflows with LLMChain
- [ ] Implement conversational memory (Buffer, Entity)
- [ ] Compare different memory types and their use cases
- [ ] Create custom tools with `@tool` decorator
- [ ] Integrate tools with chains
- [ ] Build agents that decide which tools to use
- [ ] Combine RAG + Memory + Tools in one system
- [ ] Handle errors and edge cases gracefully
- [ ] Design production-ready conversational AI systems

---

## 💡 Key Takeaways

**Day 2 Training Focuses On**:
- **Chains**: Structured AI workflows with LangChain
- **Memory**: Context retention across conversations  
- **Tools**: External capabilities and APIs
- **Integration**: Combining all concepts in production systems

---

## 🎉 Ready to Begin!

You now have a complete Day 2 training setup with:
- ✅ **LangChain chains** - Structured AI workflows
- ✅ **Conversational memory** - Context retention
- ✅ **Tool integration** - External capabilities
- ✅ **Guided project** - Production patterns
- ✅ **Hands-on examples** - Progressive learning

**Start your training**: `python examples/example_1_simple_chain.py`

---

*Last Updated: November 11, 2024*  
*Project Location*: `/Users/amolbhangale/Personal/Amol/JDA Tech/projects/Trainings/day2`
