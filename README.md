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

### Day 3: Multi-Agent Systems with LangGraph
Build sophisticated multi-agent workflows with state management, coordination, and human-in-the-loop capabilities.

**Core Topics:**
- LangGraph architecture (nodes, edges, state, checkpoints)
- Multi-agent communication patterns
- Sequential and parallel agent workflows
- Supervisor pattern for agent coordination
- Conditional routing and dynamic workflows
- Human-in-the-loop with checkpoint persistence
- Swarm intelligence and collaborative agents
- Production-ready multi-agent systems

**What You'll Build:**
- Sequential agent pipelines (researcher → writer → reviewer)
- Supervisor-controlled dynamic workflows
- Quality gates with conditional routing
- Human approval workflows with state persistence
- Parallel swarm intelligence systems
- Production multi-agent applications

👉 [Day 3 Details](./day3/README.md)

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

### Setup for Day 3
```bash
cd day3
source .venv3/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Add your OPENAI_API_KEY
# No additional setup needed - examples are interactive!
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
| **Day 3: LangGraph Basics** | Nodes, Edges, State, Graphs |
| **Day 3: Multi-Agent Patterns** | Sequential, Supervisor, Swarm |
| **Day 3: Advanced Features** | Checkpoints, Human-in-Loop |
| **Day 3: Production Systems** | State Management, Routing |

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

### After Day 3, you will:
- ✅ Build multi-agent systems with LangGraph
- ✅ Implement sequential and parallel agent workflows
- ✅ Create supervisor-controlled dynamic workflows
- ✅ Add human-in-the-loop with checkpoint persistence
- ✅ Design production-ready multi-agent applications
- ✅ Master state management and conditional routing

---

## 🛠️ Tech Stack

- **LangChain** - AI application framework
- **LangGraph** - Multi-agent workflow orchestration
- **OpenAI** - LLM and embeddings
- **FAISS** - Local vector search
- **Milvus** - Production vector database
- **Streamlit** - Web UI framework
- **LangChain Tools** - External integrations
- **SQLite** - Checkpoint persistence

---

## 📂 Repository Structure

```
ai-trainings-new/
├── README.md              # This file
├── day1/                  # RAG Fundamentals
│   ├── README.md         # Detailed Day 1 guide
│   ├── core/             # Reusable RAG components
│   ├── examples/         # Progressive examples (1-5)
│   ├── scripts/          # Build vector stores
│   └── streamlit_app/    # Web interface
├── day2/                  # Chains, Memory & Tools
│   ├── README.md         # Detailed Day 2 guide
│   ├── core/             # Enhanced components
│   ├── examples/         # Advanced examples (1-4)
│   └── streamlit_app/    # Enhanced web interface
└── day3/                  # Multi-Agent Systems
    ├── README.md         # Detailed Day 3 guide
    ├── core/             # LangGraph components
    ├── examples/         # Multi-agent examples (1-6)
    ├── config/           # Agent configurations
    └── checkpoints/      # State persistence
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
- [Day 3 Full Documentation](./day3/README.md)
- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
