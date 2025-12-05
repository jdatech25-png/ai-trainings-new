# AI Training Bootcamp - Coding Agent Instructions

## Project Overview

This is a **progressive 3-day training program** teaching RAG, LangChain patterns, and multi-agent systems. Each day is **self-contained** with isolated dependencies, virtual environments (.venv1, .venv2, .venv3), and distinct architectural patterns.

**Critical**: Day 1, Day 2, and Day 3 are **separate projects** sharing educational concepts but NOT code. Do not cross-reference code between days or suggest refactoring for DRY—repetition is intentional for learning isolation.

## Architecture & Day Structure

### Day 1: RAG Fundamentals (FAISS + Milvus)
- **Pattern**: Service-oriented architecture with `ChatService` as central RAG orchestrator
- **Key Module**: `core/chat_service.py` - All RAG logic (retrieval, generation, streaming)
- **Vector Stores**: FAISS (local, file-based) and Milvus (distributed, server-based)
- **Controllers**: `api/chat_controller.py` - FastAPI endpoints consuming `ChatService`
- **Build Process**: Run `python scripts/build_faiss_store.py` or `build_milvus_store.py` BEFORE examples
- **Example Pattern**: Progressive complexity (OpenAI Assistant → Chat Completion → RAG FAISS → RAG Milvus)

### Day 2: Chains, Memory & Tools
- **Pattern**: LangChain composition with LCEL (LangChain Expression Language)
- **Key Modules**: 
  - `core/chains.py` - Reusable chain patterns (ChainFactory)
  - `core/memory_manager.py` - Conversational memory abstraction
- **Dependencies**: Uses `langchain-classic` for legacy memory classes (intentional for training)
- **Memory Types**: Buffer, Window, Entity, Summary (demos in `examples/example_2_memory.py`)
- **Integration**: Day 2 combines RAG (from Day 1 patterns) + Memory + Tools in `example_4_guided_project.py`

### Day 3: Multi-Agent Systems (LangGraph)
- **Pattern**: Graph-based workflows with stateful agents
- **Key Modules**:
  - `core/graph_state.py` - TypedDict state schemas extending `MessagesState`
  - `core/agents.py` - ReAct agents using `langchain.agents.create_agent`
  - `core/agent_config.py` - YAML-driven config loader (singleton pattern)
- **State Management**: All custom states extend `MessagesState` from `langgraph.graph`
- **Config-Driven**: Agent definitions in `config/agents_config.yaml` (prompts, roles, tools, output schemas)
- **Checkpoint Persistence**: SQLite-based checkpoints in `checkpoints/` for human-in-the-loop workflows
- **Examples Progress**: Simple two agents → Sequential → Supervisor → Conditional edges → Human-in-loop → Swarm

## Critical Developer Workflows

### Environment Setup (ESSENTIAL FIRST STEP)
```bash
# Each day has isolated environment
cd day1  # or day2, day3
source .venv1/bin/activate  # .venv2, .venv3 respectively
pip install -r requirements.txt

# REQUIRED: Create .env with API key
cp .env.example .env
# Edit .env: OPENAI_API_KEY=sk-...
```

### Building Vector Stores (Day 1 & Day 2)
**MUST run before RAG examples or they will fail**:
```bash
# Generate sample PDFs first
python scripts/generate_sample_pdfs.py

# Build FAISS index (local, file-based)
python scripts/build_faiss_store.py  # Creates faiss_index/ directory

# Build Milvus index (requires Milvus server running)
python scripts/build_milvus_store.py  # Creates collection in Milvus
```

### Running Examples (Sequential Order Required)
Examples are numbered for progressive learning—run in order:
```bash
# Day 1
python examples/example_1_openai_assistant.py
python examples/example_2_chat_completion.py
python examples/example_3_rag_faiss.py  # Requires build_faiss_store.py first
python examples/example_4_rag_milvus.py  # Requires Milvus server + build_milvus_store.py

# Day 3 (LangGraph)
python examples/example_1_simple_two_agents.py
python examples/example_5_human_in_loop.py --interactive  # Interactive checkpoint flow
```

### Starting Services
```bash
# FastAPI server (Day 1/2)
python scripts/start_api.py
# Docs at http://localhost:8000/docs

# Streamlit UI (Day 1/2)
streamlit run streamlit_app/app.py
```

## Code Conventions & Patterns

### Import Path Pattern (Universal)
**ALL examples and scripts** use this pattern for importing core modules:
```python
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.chat_service import ChatService  # Now works from any example file
```
This allows running examples directly: `python examples/example_3_rag_faiss.py` without package installation.

### Day 1/2: ChatService Usage Pattern
```python
from core.chat_service import ChatService, VectorStoreType

service = ChatService(
    vector_store_type=VectorStoreType.FAISS,  # or VectorStoreType.MILVUS
    faiss_index_path="faiss_index",
    k=3,  # Number of documents to retrieve
    temperature=0
)
service.initialize()  # Must call before usage
answer = service.get_answer("What are company benefits?")
# Streaming: for chunk in service.get_answer_stream(query): ...
```

### Day 3: LangGraph Construction Pattern
```python
from langgraph.graph import StateGraph, END
from core.graph_state import MultiAgentState
from core import ResearchAgent, WriterAgent  # Config-based agents

# 1. Create graph with state schema
graph = StateGraph(MultiAgentState)

# 2. Add agent nodes
researcher = ResearchAgent()  # Loads from config/agents_config.yaml
graph.add_node("researcher", researcher)
graph.add_node("writer", WriterAgent())

# 3. Define edges (workflow)
graph.add_edge("researcher", "writer")
graph.add_edge("writer", END)

# 4. Set entry point & compile
graph.set_entry_point("researcher")
app = graph.compile()

# 5. Execute
result = app.invoke({"task": "Research AI trends", ...})
```

### Day 3: State Schema Pattern (TypedDict)
All custom states MUST extend `MessagesState`:
```python
from langgraph.graph import MessagesState
from typing import Annotated, Dict, Any
import operator

class MultiAgentState(MessagesState):
    """Extends MessagesState which provides 'messages' field."""
    task: str  # Replaced on update
    agent_outputs: Annotated[Dict[str, Any], lambda x, y: {**x, **y}]  # Merged
```
**Never create states from scratch—always extend `MessagesState`**.

### Day 3: Config-Driven Agents Pattern
Agents load ALL settings from `config/agents_config.yaml`:
```python
# Core agents (in core/__init__.py)
from core.agents import create_research_agent, create_writer_agent

class ResearchAgent(BaseAgent):
    def __init__(self):
        self.name = "researcher"
        self._agent = create_research_agent()  # Loads config, tools, prompts
```
**Do NOT hardcode prompts, models, or tools in agent classes**—use YAML config.

## File & Module Organization

### Core Modules (Shared Pattern Across Days)
- `core/document_loader.py` - PDF loading, chunking (Day 1/2)
- `core/embeddings.py` - OpenAI embeddings wrapper (Day 1/2)
- `core/vector_stores.py` - FAISS/Milvus abstractions (Day 1/2)
- `core/chat_service.py` - Central RAG service (Day 1/2)
- `core/graph_state.py` - TypedDict state schemas (Day 3)
- `core/agents.py` - ReAct agent implementations (Day 3)
- `core/agent_config.py` - YAML config loader singleton (Day 3)

### Scripts (Utility Commands)
- `scripts/generate_sample_pdfs.py` - Creates sample PDFs in `docs/`
- `scripts/build_faiss_store.py` - Builds FAISS index from PDFs
- `scripts/build_milvus_store.py` - Builds Milvus collection from PDFs
- `scripts/start_api.py` - Starts FastAPI server with uvicorn

## Dependency Management

### LangChain Versioning (Critical)
- **Day 1**: `langchain==0.3.7` (stable RAG patterns)
- **Day 2**: `langchain==1.0.5` + `langchain-classic==1.0.0` (for legacy memory classes)
- **Day 3**: Latest LangChain + `langgraph`, `langgraph-checkpoint`, `langgraph_supervisor`

**Never suggest upgrading/downgrading packages across days**—versions are pinned for training stability.

### Vector Store Dependencies
- **FAISS**: `faiss-cpu==1.9.0+` (CPU-only, no CUDA)
- **Milvus**: `pymilvus==2.4.8+` (client library, requires Milvus server)

## Testing & Debugging

### Verifying Vector Store Setup
```bash
# Check FAISS index exists
ls faiss_index/  # Should show index.faiss and index.pkl

# Test FAISS retrieval
python examples/example_3_rag_faiss.py --demo
```

### Common Error Patterns
1. **"FAISS index not found"**: Run `python scripts/build_faiss_store.py`
2. **"OPENAI_API_KEY not set"**: Check `.env` file in current day directory
3. **Import errors from examples**: Ensure you're in correct day directory and venv is activated
4. **"MessagesState not found" (Day 3)**: Import from `langgraph.graph`, not `langgraph.prebuilt`
5. **Day 3 agents not loading config**: Check `config/agents_config.yaml` exists and is valid YAML

### Running Tests
```bash
# Day 2 - Test multi-tool integration
python test_multi_tool.py

# Day 3 - Test checkpoint flow (interactive)
python examples/example_5_human_in_loop.py --interactive
```

## External Dependencies & Integration

### Required Services
- **OpenAI API**: Required for ALL examples (set `OPENAI_API_KEY` in `.env`)
- **Milvus Server** (Optional, Day 1/2): For distributed vector store examples
  ```bash
  # Start Milvus via Docker
  docker-compose up -d milvus-standalone
  ```

### API Integrations (Day 2)
- **Weather API**: Mock implementation in tools (for training)
- **Web Search**: Uses `duckduckgo-search` library (no API key needed)

## Documentation References

- Main README: Project overview and learning path
- Day READMEs: Detailed setup, concepts, and exercises for each day
- `INTERACTIVE_MODE_GUIDE.md` (Day 3): Checkpoint persistence workflow guide
- Docstrings: All modules have comprehensive docstrings explaining patterns

## AI Assistant Guidelines

1. **Respect Day Isolation**: Do not suggest sharing code between days or creating a "common" library
2. **Environment Awareness**: Always verify which day's venv is active before suggesting code
3. **Build Steps First**: For RAG issues, check if vector store was built before debugging code
4. **Config-Driven Day 3**: When editing Day 3 agents, update YAML config, not Python code
5. **State Schema Compliance**: Day 3 states MUST extend `MessagesState`—this is a LangGraph requirement
6. **Sequential Examples**: Recommend running examples in numbered order for optimal learning
7. **Error Context**: When debugging, check if `.env` exists, vector stores are built, and correct venv is active
8. **Do NOT Create Documentation or Test Files**: Unless explicitly requested by the user, do not create:
   - Markdown files (`.md`) - README, guides, documentation
   - Test files (`test_*.py`, `*_test.py`) - unit tests, integration tests
   - Documentation files of any kind
   
   When assisting users, provide instructions in the chat conversation instead of creating documentation files. Only create these files when the user specifically asks for them.
