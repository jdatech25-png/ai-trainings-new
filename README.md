# AI RAG Training Bootcamp

A comprehensive, hands-on training program covering Retrieval Augmented Generation (RAG) systems and advanced LangChain patterns for production AI applications.

## 📚 Training Structure

### Day 1: RAG Fundamentals & Implementation
Build progressive RAG systems from simple chat to production-ready applications.

**Core Topics:**
- RAG (Retrieval Augmented Generation) architecture
- OpenAI Assistants & Chat Completion APIs
- Document processing, chunking, and embeddings
- Vector databases (FAISS & Milvus)
- Semantic search and retrieval
- Production RAG deployment with Streamlit

**What You'll Build:**
- Simple chat applications with OpenAI APIs
- RAG systems with local (FAISS) and distributed (Milvus) vector stores
- Complete web-based chat interface

👉 [Day 1 Details](./day1/README.md)

---

### Day 2: Chains, Memory & Tools
Master advanced LangChain patterns for intelligent, contextual AI systems.

**Core Topics:**
- LangChain chains and composition patterns (LCEL)
- Conversational memory (Buffer, Entity, Summary)
- Custom tool creation and integration
- Real API integrations (weather, web search)
- Agent-based tool selection (ReAct pattern)
- RAG + Memory + Tools integration

**What You'll Build:**
- Structured AI workflows with chains
- Conversational bots with memory
- Tool-enabled agents with external APIs
- Complete contextual chatbot combining all features

👉 [Day 2 Details](./day2/README.md)

---

## 🎯 Prerequisites

- **Python 3.10+**
- **OpenAI API Key** ([Get one here](https://platform.openai.com/api-keys))
- Basic Python programming knowledge
- (Optional) Docker for Milvus vector database

---

## 🚀 Quick Start

### Setup for Day 1
```bash
cd day1
source .venv1/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Add your OPENAI_API_KEY
python scripts/generate_sample_pdfs.py
```

### Setup for Day 2
```bash
cd day2
source .venv2/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Add your OPENAI_API_KEY
python scripts/generate_sample_pdfs.py
```

---

## 📖 Learning Path

| Module | Key Concepts |
|--------|--------------|
| **Day 1: Simple Chat** | OpenAI APIs, Streaming, System Prompts |
| **Day 1: RAG Basics** | Embeddings, Vector Search, Chunking |
| **Day 1: RAG Implementation** | FAISS, Retrieval, Context Injection |
| **Day 1: Production RAG** | Milvus, Scaling, Deployment |
| **Day 1: Web Application** | Streamlit UI, Configuration |
| **Day 2: Chains** | LangChain Composition, LCEL |
| **Day 2: Memory** | Conversation Context, Memory Types |
| **Day 2: Tools** | Custom Tools, Agents, APIs |
| **Day 2: Integration** | RAG + Memory + Tools |

---

## 🎓 Key Outcomes

### After Day 1, you will:
- ✅ Understand RAG architecture and why it's needed
- ✅ Build RAG systems with FAISS and Milvus
- ✅ Deploy production-ready chat applications
- ✅ Tune RAG parameters for optimal performance

### After Day 2, you will:
- ✅ Create structured AI workflows with chains
- ✅ Implement conversational memory patterns
- ✅ Integrate external tools and APIs
- ✅ Build contextual chatbots combining all techniques

---

## 🛠️ Tech Stack

- **LangChain** - AI application framework
- **OpenAI** - LLM and embeddings
- **FAISS** - Local vector search
- **Milvus** - Production vector database
- **Streamlit** - Web UI framework
- **LangChain Tools** - External integrations

---

## 📂 Repository Structure

```
ai-trainings/
├── README.md              # This file
├── day1/                  # RAG Fundamentals
│   ├── README.md         # Detailed Day 1 guide
│   ├── core/             # Reusable RAG components
│   ├── examples/         # Progressive examples (1-4)
│   ├── scripts/          # Build vector stores
│   └── streamlit_app/    # Web interface
└── day2/                  # Chains, Memory & Tools
    ├── README.md         # Detailed Day 2 guide
    ├── core/             # Enhanced components
    ├── examples/         # Advanced examples (1-4)
    └── streamlit_app/    # Enhanced web interface
```

---

## 💡 Training Approach

- **Progressive Learning**: Build from simple concepts to complex systems
- **Hands-On Practice**: Run code, experiment, and modify
- **Production Patterns**: Learn industry best practices
- **Modular Design**: Reusable components for your own projects

---

## 🔗 Resources

- [Day 1 Full Documentation](./day1/README.md)
- [Day 2 Full Documentation](./day2/README.md)
- [LangChain Documentation](https://python.langchain.com/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
