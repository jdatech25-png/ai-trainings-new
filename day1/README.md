# Day 1: AI RAG Bootcamp - Complete Training Guide

## 🎯 Training Objectives
- Learn RAG (Retrieval Augmented Generation) architecture
- Build progressive examples from simple chat to production RAG systems
- Compare different approaches: OpenAI Assistants, Chat Completion, Vector Stores
- Deploy a complete web application with Streamlit

---

## 📋 Table of Contents
1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Project Structure](#project-structure)
4. [Detailed Setup Guide](#detailed-setup-guide)
5. [Running the Examples](#running-the-examples)
6. [Core Modules](#core-modules)
7. [Training Exercises](#training-exercises)
8. [Troubleshooting](#troubleshooting)
9. [Learning Outcomes](#learning-outcomes)

---

## 📋 Prerequisites

- **Python 3.10+** installed
- **OpenAI API Key** ([Get one here](https://platform.openai.com/api-keys))
- **(Optional)** Milvus server for production RAG example

---

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Activate virtual environment
source .venv1/bin/activate  # On macOS/Linux
# .venv1\Scripts\activate  # On Windows

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
# Example 1: OpenAI Assistant API
python examples/example_1_openai_assistant.py

# Example 2: Chat Completion API
python examples/example_2_chat_completion.py

# Example 3: RAG with FAISS
python scripts/build_faiss_store.py
python examples/example_3_rag_faiss.py

# Example 4: RAG with Milvus (optional - requires Milvus server)
python scripts/build_milvus_store.py
python examples/example_4_rag_milvus.py

# Example 5: Streamlit Application
streamlit run streamlit_app/app.py
```

---

## 📁 Project Structure

```
day1/
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
├── core/                              # Reusable RAG components
│   ├── __init__.py
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
│   ├── example_1_openai_assistant.py # OpenAI Assistants API
│   ├── example_2_chat_completion.py  # Chat Completion API
│   ├── example_3_rag_faiss.py        # RAG with FAISS
│   └── example_4_rag_milvus.py       # RAG with Milvus
│
└── streamlit_app/                     # Web application
    ├── __init__.py
    └── app.py                         # Main Streamlit application
```

---

## 🔧 Detailed Setup Guide

### Step 1: Environment Setup

```bash
# Navigate to project directory
cd /path/to/day1

# Activate virtual environment
source .venv1/bin/activate  # On macOS/Linux
# OR
.venv1\Scripts\activate  # On Windows

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

### Example 1: OpenAI Assistant API

**Purpose**: Learn how to use OpenAI's Assistants API with streaming responses

```bash
# Interactive mode (default)
python examples/example_1_openai_assistant.py

# Demo mode (single question)
python examples/example_1_openai_assistant.py --demo
```

**What to observe**:
- Assistant creation or connection
- Streaming response (text appears gradually)
- Thread-based conversation management
- State management handled by OpenAI

**Key concepts**:
- OpenAI Assistants API
- Streaming responses
- Conversation threads
- Managed state

---

### Example 2: Chat Completion API

**Purpose**: Learn Chat Completion API with system prompts and manual state management

```bash
# Interactive mode (default)
python examples/example_2_chat_completion.py

# Compare different system prompts
python examples/example_2_chat_completion.py --prompts

# Demonstrate conversation memory
python examples/example_2_chat_completion.py --memory
```

**What to observe**:
- System prompt customization
- Manual conversation history management
- Streaming responses
- Lower latency than Assistants API

**Key concepts**:
- Chat Completion API
- System prompts
- Manual state management
- Temperature parameter

**Interactive commands**:
- Type your question to chat
- Type `count` to see message history
- Type `clear` to reset conversation
- Type `quit` to exit

---

### Example 3: RAG with FAISS

**Purpose**: Build your first RAG system with local vector database

#### Step 3a: Build FAISS Index

```bash
python scripts/build_faiss_store.py
```

**Expected output**:
- PDFs loaded and chunked
- Embeddings generated
- FAISS index saved to `faiss_index/`

**Time**: ~30-60 seconds (depends on document size)

#### Step 3b: Query the RAG System

```bash
# Interactive mode (default)
python examples/example_3_rag_faiss.py

# Demo mode with sample questions
python examples/example_3_rag_faiss.py --demo
```

**What to observe**:
- Semantic search finds relevant chunks
- Context is retrieved automatically
- LLM answers based on retrieved context
- Source documents are cited

**Key concepts**:
- RAG (Retrieval Augmented Generation)
- Vector embeddings
- Semantic similarity search
- FAISS vector store
- Context-aware responses

**Interactive commands**:
- Type your question to get RAG answers
- Type `context` to see retrieved documents
- Type `stats` to see system information
- Type `quit` to exit

**Try asking**:
- "What are the company benefits?"
- "How does the API authentication work?"
- "What leave policies does the company have?"

---

### Example 4: RAG with Milvus

**Purpose**: Use production-grade distributed vector database

**Prerequisites**: Milvus server must be running

#### Quick Milvus Setup (Docker):

```bash
# Pull and run Milvus standalone
docker run -d --name milvus-standalone \
  -p 19530:19530 \
  -p 9091:9091 \
  milvusdb/milvus:latest
```

#### Step 4a: Build Milvus Collection

```bash
python scripts/build_milvus_store.py
```

**Expected output**:
- Connected to Milvus
- Documents inserted
- Collection created with HNSW index

#### Step 4b: Query Milvus RAG System

```bash
# Interactive mode (default)
python examples/example_4_rag_milvus.py

# Demo mode
python examples/example_4_rag_milvus.py --demo
```

**What to observe**:
- Similar interface to FAISS
- Production-ready features
- Advanced filtering capabilities
- Distributed architecture support

**Key concepts**:
- Milvus vector database
- Production scalability
- Advanced indexing (HNSW)
- Metadata filtering

---

### Example 5: Streamlit Chat Application

**Purpose**: Complete web-based chat UI with RAG backend

```bash
streamlit run streamlit_app/app.py
```

**Expected output**:
- Browser opens automatically (usually http://localhost:8501)
- Professional chat interface

**Features to explore**:
1. **Sidebar**: Configure vector store, LLM settings
2. **Connect**: Choose FAISS or Milvus backend
3. **Chat**: Ask questions and see streaming responses
4. **Sources**: View retrieved document chunks
5. **Settings**: Adjust k (number of docs), temperature, models

**Sample workflow**:
1. Click "Connect to Vector Store" (FAISS)
2. Ask: "What leave policies does the company have?"
3. Observe streaming response
4. Check source documents
5. Adjust "Number of documents (k)" slider
6. Ask another question to see different retrieval

---

## 🧩 Core Modules

### Overview

The `core/` folder contains reusable components for building RAG applications. These modules provide clean abstractions over LangChain and can be used in your own projects.

### `core/document_loader.py`

**Purpose**: Load and chunk PDF documents

```python
from core.document_loader import load_local_pdfs

# Load all PDFs from a directory
docs = load_local_pdfs("docs/", chunk_size=1000, chunk_overlap=200)

# Returns: List of dicts with 'content', 'source', 'page' keys
print(f"Loaded {len(docs)} chunks")
```

**Features**:
- Recursive PDF loading
- Configurable chunk size and overlap
- Metadata preservation (source file, page number)

---

### `core/embeddings.py`

**Purpose**: Manage OpenAI embeddings

```python
from core.embeddings import EmbeddingManager

# Initialize embedding model
embeddings = EmbeddingManager(model="text-embedding-3-small")

# Embed documents
doc_embeddings = embeddings.embed_documents(["doc1", "doc2"])

# Embed single query
query_embedding = embeddings.embed_query("search query")
```

**Features**:
- Clean wrapper around OpenAI embeddings
- Automatic dimension detection (1536 or 3072)
- Error handling and retries

---

### `core/vector_stores.py`

**Purpose**: Unified interface for FAISS and Milvus

```python
from core.vector_stores import FAISSVectorStore, MilvusVectorStore
from core.embeddings import EmbeddingManager

embeddings = EmbeddingManager()

# FAISS (local, file-based)
faiss_store = FAISSVectorStore(embeddings)
faiss_store.add_documents(docs)
faiss_store.save("faiss_index/")

# Milvus (distributed, production)
milvus_store = MilvusVectorStore(
    embeddings,
    connection_args={"host": "localhost", "port": "19530"},
    collection_name="training_demo"
)
milvus_store.add_documents(docs)
```

**Features**:
- Common interface for both databases
- Easy switching between FAISS and Milvus
- Automatic index creation and optimization

---

### `core/retrievers.py`

**Purpose**: Semantic search and document retrieval

```python
from core.retrievers import DocumentRetriever

# Create retriever
retriever = DocumentRetriever(vector_store, k=3)

# Retrieve relevant documents
docs = retriever.retrieve("What are the benefits?")

# Format for LLM context
context = retriever.format_docs_for_context(docs)

# Get source information
sources = retriever.get_source_info(docs)
```

**Features**:
- Configurable number of results (k)
- Similarity score tracking
- Context formatting for LLMs
- Source extraction for citations

---

### `core/generators.py`

**Purpose**: LLM-based answer generation with RAG

```python
from core.generators import AnswerGenerator

# Initialize generator
generator = AnswerGenerator(model="gpt-4o-mini", temperature=0)

# Generate answer with context
answer = generator.generate(
    query="What are the benefits?",
    context=retrieved_docs
)

# Streaming generation
for chunk in generator.generate_with_stream(query, context):
    print(chunk, end="", flush=True)
```

**Features**:
- RAG prompt template
- Context injection
- Streaming and non-streaming modes
- Customizable system prompts

---

## 🎓 Training Exercises

### Exercise 1: Understanding Chunks

**Objective**: Learn how chunk size affects RAG quality

1. Open `scripts/build_faiss_store.py`
2. Change `chunk_size` to different values (500, 1000, 2000)
3. Rebuild index and query
4. **Question**: How does chunk size affect answer quality?

---

### Exercise 2: System Prompt Experiments

**Objective**: Understand the power of system prompts

1. Run `python examples/example_2_chat_completion.py`
2. Try different system prompts:
   - "You are a technical expert. Use code examples."
   - "You are a teacher for children. Use simple words."
   - "You are a pirate. Talk like a pirate."
3. Ask the same question with each prompt
4. **Question**: How does behavior change?

---

### Exercise 3: Retrieval Tuning

**Objective**: Find optimal number of retrieved documents

1. Open Streamlit app: `streamlit run streamlit_app/app.py`
2. Try different k values (1, 3, 5, 10)
3. Ask: "What are all the features of CloudMaster?"
4. **Question**: What's the optimal k value and why?

---

### Exercise 4: Temperature Effects

**Objective**: Understand temperature in LLM generation

1. In Streamlit app, set temperature to 0
2. Ask a question, note the answer
3. Ask the same question again (should be identical)
4. Set temperature to 1.5
5. Ask the same question multiple times
6. **Question**: When would you use high vs low temperature?

---

## 🐛 Troubleshooting

### Issue: Import Errors

**Symptoms**: `ModuleNotFoundError: No module named 'langchain'`

**Solution**:
```bash
# Ensure virtual environment is activated
source .venv1/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

---

### Issue: "OPENAI_API_KEY not found"

**Symptoms**: API key error when running examples

**Solution**:
```bash
# Verify .env file exists
cat .env

# Should show:
# OPENAI_API_KEY=sk-...

# If empty, edit the file
nano .env
```

---

### Issue: "No PDF files found"

**Symptoms**: Document loader can't find PDFs

**Solution**:
```bash
# Generate sample PDFs
python scripts/generate_sample_pdfs.py

# Verify PDFs exist
ls -l docs/
# Should show:
# company_handbook.pdf
# product_guide.pdf
# technical_manual.pdf
```

---

### Issue: Milvus Connection Failed

**Symptoms**: "Failed to connect to Milvus"

**Solution**:
```bash
# Check if Milvus is running
docker ps | grep milvus

# If not running, start it
docker run -d --name milvus-standalone \
  -p 19530:19530 \
  milvusdb/milvus:latest

# Wait ~30 seconds, then test
python -c "from pymilvus import connections; connections.connect('default', host='localhost', port='19530'); print('✅ Connected')"
```

---

### Issue: FAISS Index Not Found

**Symptoms**: "FAISS index not found"

**Solution**:
```bash
# Build the index first
python scripts/build_faiss_store.py

# Verify it was created
ls -l faiss_index/
```

---

### Issue: Out of Memory

**Symptoms**: Python crashes or slows down significantly

**Solution**:
- Reduce chunk_size in build scripts (e.g., 500 instead of 1000)
- Process fewer documents
- Use smaller embedding model (if available)

---

## 📊 Understanding RAG Results

### Good RAG Response Indicators:
✅ **Cites specific sources** - References actual documents  
✅ **Stays within context** - Only uses retrieved information  
✅ **Admits when uncertain** - Says "not found in documents"  
✅ **Provides relevant details** - Uses specifics from context  

### Poor RAG Response Indicators:
❌ **Hallucinates facts** - Makes up information not in documents  
❌ **Ignores context** - Gives generic answers without using retrieved docs  
❌ **Too generic** - Doesn't leverage specific information  
❌ **Contradicts sources** - Says things opposite to documents  

---

## 🎯 Learning Path

### Module 1: Simple Chat (30 mins)
**Theory**: AI/ML/GenAI evolution, LLMs, Transformer architecture

**Examples**: 1 & 2

**Concepts**:
- Assistants vs Chat Completion APIs
- Streaming responses
- System prompts
- Conversation memory

---

### Module 2: Understanding RAG (45 mins)
**Theory**: Why RAG? Limitations of LLMs, Embeddings, Vector databases

**Script**: `build_faiss_store.py` (with code walkthrough)

**Concepts**:
- Document chunking strategies
- Text embeddings
- Semantic similarity search
- Vector databases (FAISS)

---

### Module 3: RAG Implementation (45 mins)
**Example**: 3 (RAG with FAISS)

**Concepts**:
- Retrieval pipeline
- Context injection
- RAG prompt engineering
- Source attribution

---

### Module 4: Production RAG (30 mins)
**Theory**: FAISS vs Milvus comparison, Scaling considerations

**Script**: `build_milvus_store.py`

**Example**: 4 (RAG with Milvus)

**Concepts**:
- Production scalability
- Distributed vector databases
- Advanced indexing (HNSW)
- Metadata filtering

---

### Module 5: Complete Application (30 mins)
**Demo**: Streamlit app

**Concepts**:
- Web UI/UX design
- Configuration management
- Backend switching
- Deployment considerations

---

## 🎓 Learning Outcomes

By the end of Day 1, participants should be able to:

- [ ] Explain the difference between AI, ML, and GenAI
- [ ] Understand why RAG is needed (LLM limitations)
- [ ] Describe the RAG pipeline: Embed → Search → Retrieve → Generate
- [ ] Use OpenAI API (Assistants and Chat Completion)
- [ ] Load and chunk documents effectively
- [ ] Generate and store text embeddings
- [ ] Perform semantic similarity search
- [ ] Build a complete RAG application
- [ ] Compare FAISS vs Milvus vector databases
- [ ] Deploy a chat interface with Streamlit
- [ ] Tune RAG parameters (chunk size, k, temperature)

---

## 💡 Training Tips

### For Trainers:
1. **Run all examples yourself first** - Understand the complete flow
2. **Prepare answers to common questions** - Reference troubleshooting section
3. **Have fallback plans** - If Milvus fails, focus on FAISS examples
4. **Encourage experimentation** - Let participants modify parameters
5. **Time management** - Allocate ~45 mins per module, with buffer time
6. **Live coding** - Show the code, explain key concepts, then run
7. **Interactive demos** - Let participants suggest questions to ask

### For Participants:
1. **Follow examples in order** - Each builds on previous concepts
2. **Read the code comments** - They explain key decisions
3. **Experiment with parameters** - Change chunk_size, k, temperature
4. **Ask questions** - No question is too basic
5. **Take notes** - Document what works and what doesn't
6. **Try to break it** - Experiment with edge cases
7. **Build something new** - Use core modules to create your own app

---

## 🚀 Next Steps (Day 2+)

After completing Day 1, potential topics for continuation:

- **LangGraph**: Building complex multi-step agent workflows
- **Multi-Agent Systems**: Agents that collaborate and specialize
- **Advanced RAG Techniques**: 
  - Hybrid search (semantic + keyword)
  - Query expansion and rewriting
  - Re-ranking retrieved documents
  - Parent document retrieval
- **Production Deployment**: 
  - Docker containerization
  - REST API creation
  - Monitoring and logging
  - Error handling
- **Evaluation**: 
  - Testing RAG quality
  - Metrics (precision, recall, faithfulness)
  - A/B testing different configurations

---

## 📚 Additional Resources

### Official Documentation
- [LangChain Documentation](https://python.langchain.com/) - Comprehensive framework docs
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference) - API endpoints and parameters
- [Milvus Documentation](https://milvus.io/docs) - Vector database guide
- [Streamlit Documentation](https://docs.streamlit.io/) - Web app framework
- [FAISS GitHub](https://github.com/facebookresearch/faiss) - Vector search library

### Tutorials & Articles
- [RAG from Scratch](https://python.langchain.com/docs/tutorials/rag/) - LangChain RAG tutorial
- [Embeddings Guide](https://platform.openai.com/docs/guides/embeddings) - OpenAI embeddings
- [Vector Database Comparison](https://milvus.io/docs/comparison.md) - When to use what

### Papers
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [RAG: Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401) - RAG research paper

---

## ✅ Pre-Training Checklist

### Before Starting Training:
- [ ] Python 3.9+ installed
- [ ] Virtual environment activated
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] OpenAI API key configured in `.env`
- [ ] Sample PDFs generated
- [ ] FAISS index built
- [ ] (Optional) Milvus server running
- [ ] All examples tested and working
- [ ] Streamlit app loads successfully

### During Training:
- [ ] Share screen clearly
- [ ] Run examples in sequence
- [ ] Encourage questions and discussion
- [ ] Let participants try themselves
- [ ] Show both success and failure cases
- [ ] Explain error messages when they occur

### After Training:
- [ ] Share project repository
- [ ] Provide additional learning resources
- [ ] Collect participant feedback
- [ ] Answer follow-up questions
- [ ] Schedule office hours (if applicable)

---

## 📊 Project Statistics

- **Total Files Created**: 20+
- **Lines of Code**: 2,000+
- **Documentation**: This comprehensive guide
- **Training Duration**: 2-3 hours
- **Skill Level**: Beginner to Intermediate
- **Prerequisites**: Basic Python knowledge

---

## 🎉 Ready to Begin!

You now have a complete, production-ready RAG training project with:
- ✅ **Modular architecture** - Reusable core components
- ✅ **Progressive examples** - Build from simple to complex
- ✅ **Comprehensive documentation** - This complete guide
- ✅ **Production patterns** - Best practices included
- ✅ **Interactive demos** - Hands-on learning
- ✅ **Troubleshooting** - Solutions to common issues

**Start your training**: `python examples/example_1_openai_assistant.py`

---

*Last Updated: November 8, 2024*  
*Project Location*: `/Users/amolbhangale/Personal/Amol/JDA Tech/projects/Trainings/day1`
