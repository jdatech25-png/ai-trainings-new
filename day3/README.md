# Day 3: Multi-Agent Systems with LangGraph - Complete Training Guide

## 🎯 Training Objectives
- Understand LangGraph architecture and core concepts
- Master nodes, edges, state management, and checkpoints
- Build multi-agent systems with sequential and dynamic workflows
- Implement supervisor patterns for agent coordination
- Add human-in-the-loop capabilities with checkpoints
- Explore swarm intelligence patterns for collaborative agents
- **NEW**: Use tools with ReAct agents in multi-agent workflows
- Design production-ready multi-agent applications with structured outputs

---

## 📋 Table of Contents
1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Project Structure](#project-structure)
4. [LangGraph Core Concepts](#langgraph-core-concepts)
5. [Tool-Enabled Multi-Agent Systems](#tool-enabled-multi-agent-systems-new)
6. [Running the Examples](#running-the-examples)
7. [Core Modules](#core-modules)
8. [Training Exercises](#training-exercises)
9. [Comparison: Approaches](#comparison-approaches)
10. [Learning Outcomes](#learning-outcomes)

---

## 📋 Prerequisites

- **Completed Day 1 & Day 2** training (RAG, Chains, Memory, Tools)
- **Python 3.10+** installed
- **OpenAI API Key** ([Get one here](https://platform.openai.com/api-keys))
- Basic understanding of:
  - LangChain chains and agents
  - Conversational memory
  - Tool integration

---

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Create and activate virtual environment
python -m venv .venv3
source .venv3/bin/activate  # On macOS/Linux
# .venv3\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 2. Run Examples in Order

All examples run in **interactive mode** by default - simply execute them and follow the prompts:

```bash
# Example 1: Simple Two Agents (Researcher → Writer)
python examples/example_1_simple_two_agents.py

# Example 2: Three Agents Sequential (Researcher → Writer → Reviewer)
python examples/example_2_three_agents_sequential.py

# Example 3: Supervisor Pattern (Supervisor coordinates workers)
python examples/example_3_supervisor_pattern.py

# Example 4: Conditional Edges (Quality gate with routing)
python examples/example_4_conditional_edges.py

# Example 5: Human-in-the-Loop (Checkpoint-based approval workflow)
python examples/example_5_human_in_loop.py

# Example 6: Swarm Pattern (Parallel analysts with convergence)
python examples/example_6_swarm_pattern.py
```

**Note**: All examples use interactive prompts for input. No command-line arguments required!

---

## 📁 Project Structure

```
day3/
│
├── README.md                          # This comprehensive guide
├── requirements.txt                   # Python dependencies (includes langgraph)
├── .env.example                       # Environment variables template
├── .gitignore                         # Git ignore rules
│
├── core/                              # Reusable LangGraph components
│   ├── __init__.py
│   ├── graph_state.py                # TypedDict state schemas
│   ├── agents.py                     # Agent classes (including ReActAgent)
│   ├── tools.py                      # Tool definitions (NEW)
│   └── graph_builder.py              # Graph construction utilities
│
├── examples/                          # Progressive training examples
│   ├── __init__.py
│   ├── example_1_simple_two_agents.py       # Two agents (interactive)
│   ├── example_2_three_agents_sequential.py # Three agent pipeline (interactive)
│   ├── example_3_supervisor_pattern.py      # Supervisor coordination (interactive)
│   ├── example_4_conditional_edges.py       # Dynamic routing (interactive)
│   ├── example_5_human_in_loop.py          # Human approval workflow (checkpoint-based)
│   ├── example_6_swarm_pattern.py          # Swarm collaboration (interactive)
│   └── checkpoints/                        # Example-specific checkpoints
│
└── checkpoints/                       # Main checkpoint storage (auto-created)
    └── human_loop.db                 # SQLite checkpoint database (example 5)
```

---

## 🧠 LangGraph Core Concepts

### What is LangGraph?

**LangGraph** is a framework for building stateful, multi-agent workflows as graphs.

Think of it as:
- **Nodes**: Individual agents or processing steps
- **Edges**: Connections that control flow
- **State**: Shared data that flows through the graph
- **Checkpoints**: Snapshots for persistence and human-in-loop

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Researcher │────▶│   Writer    │────▶│  Reviewer   │
│   (Node)    │     │   (Node)    │     │   (Node)    │
└─────────────┘     └─────────────┘     └─────────────┘
       │                   │                   │
       └───────────────────┴───────────────────┘
                    Shared State
```

### Key Concepts

#### 1. **Nodes** - Processing Units

Nodes are functions that:
- Take the current state as input
- Perform some processing (call LLM, use tools, etc.)
- Return state updates

```python
def research_node(state):
    """Research agent node."""
    task = state["task"]
    # Do research...
    result = {"findings": [...]}
    
    return {
        "agent_outputs": {"researcher": result},
        "next": "writer"
    }
```

**Node Characteristics**:
- ✅ Stateless (pure functions of state)
- ✅ Return only state updates, not full state
- ✅ Can be agents, tools, or custom logic
- ✅ Execute sequentially or in parallel

---

#### 2. **Edges** - Flow Control

Edges define how execution flows between nodes.

**Types of Edges**:

| Edge Type | Purpose | Example |
|-----------|---------|---------|
| **Direct Edge** | Always go to next node | `graph.add_edge("A", "B")` |
| **Conditional Edge** | Route based on state | `graph.add_conditional_edges(...)` |
| **Entry Edge** | Start of graph | `graph.set_entry_point("start")` |
| **End Edge** | Terminate graph | `graph.add_edge("node", END)` |

**Example: Conditional Routing**
```python
def route_after_review(state):
    """Decide next step based on review."""
    if state["agent_outputs"]["reviewer"]["approved"]:
        return "finish"
    else:
        return "writer"  # Revise

graph.add_conditional_edges(
    "reviewer",
    route_after_review,
    {
        "finish": END,
        "writer": "writer"
    }
)
```

---

#### 3. **State** - Shared Data Flow

State is a **TypedDict** that flows through the graph.

**State Design Principles**:
- ✅ Use TypedDict for type safety
- ✅ Annotate lists/dicts for proper merging
- ✅ Keep state minimal (only what's needed)
- ✅ Use structured outputs (JSON) from agents

**Example State**:
```python
from typing import TypedDict, Annotated
import operator

class AgentState(TypedDict):
    # Messages append (don't replace)
    messages: Annotated[list, operator.add]
    
    # Agent outputs merge (don't replace)
    agent_outputs: Annotated[dict, lambda x, y: {**x, **y}]
    
    # Simple fields replace
    task: str
    next: str
```

**State Update Operators**:
```python
# No annotation: Replace
task: str  # New value replaces old

# operator.add: Append to list
messages: Annotated[list, operator.add]  # New messages append

# Custom merger: Merge dicts
agent_outputs: Annotated[dict, lambda x, y: {**x, **y}]
```

---

#### 4. **Checkpoints** - Persistence & Human-in-Loop

Checkpoints enable:
- **Persistence**: Save and resume workflows
- **Human-in-the-Loop**: Pause for approval
- **Time Travel**: Replay from any point
- **Debugging**: Inspect state at each step

**Checkpoint Configuration**:
```python
from langgraph.checkpoint.sqlite import SqliteSaver

# Create checkpointer
checkpointer = SqliteSaver.from_conn_string("checkpoints/checkpoint.db")

# Compile graph with checkpointing
graph = builder.compile(checkpointer=checkpointer)

# Invoke with thread ID (enables resuming)
result = graph.invoke(
    {"task": "..."},
    config={"configurable": {"thread_id": "session_1"}}
)

# Resume from checkpoint
result = graph.invoke(
    None,  # Continue from last state
    config={"configurable": {"thread_id": "session_1"}}
)
```

**Human-in-the-Loop Pattern**:
```python
# 1. Agent executes
# 2. Graph pauses at approval node
# 3. Human reviews output
# 4. Human provides feedback
# 5. Graph resumes with feedback
```

---

#### 5. **Supervisor Pattern** - Coordinated Workflows

A supervisor agent coordinates multiple worker agents.

**Supervisor Responsibilities**:
- Analyze current state
- Decide which worker to execute next
- Determine when task is complete
- Provide instructions to workers

**Flow**:
```
┌────────────┐
│ Supervisor │◀─────┐
└──────┬─────┘      │
       │            │
   ┌───┴────┬───────┴──┬──────────┐
   ▼        ▼          ▼          │
┌─────┐ ┌────────┐ ┌──────────┐  │
│ Res │ │ Writer │ │ Reviewer │  │
└──┬──┘ └───┬────┘ └────┬─────┘  │
   │        │           │         │
   └────────┴───────────┴─────────┘
       (Report back to supervisor)
```

**Benefits**:
- ✅ Centralized decision-making
- ✅ Dynamic workflow adaptation
- ✅ Easy to add new workers
- ✅ Clear separation of concerns

---

#### 6. **Swarm Pattern** - Collaborative Intelligence

Multiple agents work in parallel, sharing knowledge and converging on solutions.

**Swarm Characteristics**:
- Multiple agents process simultaneously
- Shared knowledge base
- Inter-agent communication
- Convergence toward consensus
- Emergent intelligent behavior

**Use Cases**:
- Market analysis (multiple analysts)
- Code review (multiple reviewers)
- Content generation (multiple writers)
- Decision-making (consensus building)

**Flow**:
```
       ┌─────────────────────────┐
       │   Shared Knowledge      │
       └───────────┬─────────────┘
                   │
       ┌───────────┼───────────┐
       ▼           ▼           ▼
   ┌──────┐   ┌──────┐   ┌──────┐
   │Agent1│   │Agent2│   │Agent3│
   └───┬──┘   └───┬──┘   └───┬──┘
       │          │          │
       └──────────┴──────────┘
              Converge
```

---

## � Tool-Enabled Multi-Agent Systems (NEW)

### Overview

Day 3 now includes **tool-enabled agents** using the **ReAct (Reasoning + Acting)** pattern from `langchain_classic.agents`. This allows agents to use real tools (calculator, search, etc.) within multi-agent workflows.

**Enhancement Benefits**:
- ✅ Agents can access real information via tools
- ✅ Perform calculations with accuracy
- ✅ Search documents using RAG
- ✅ Maintain LangGraph compatibility
- ✅ Track tool usage transparently

### New Components

#### 1. **Tools Module** (`core/tools.py`)

Reusable tools organized by category:

**Basic Tools**:
- `calculator` - Mathematical calculations
- `get_current_time` - Current date/time
- `weather_info` - Weather information (mock data for demo)

**Research Tools**:
- `search_documents` - RAG-based document search
- `company_info` - Company policies and benefits lookup
- `web_search` - Web search (mock implementation)

**Writer Tools**:
- `text_analyzer` - Word count, sentiment analysis
- `calculator` - For calculations in content

**Tool Collections**:
```python
from core.tools import get_research_tools, get_writer_tools, get_all_tools

research_tools = get_research_tools()  # For research agents
writer_tools = get_writer_tools()      # For writer agents
all_tools = get_all_tools()            # All available tools
```

#### 2. **ReActAgent** (`core/agents.py`)

New agent class that combines LangGraph with tool usage:

```python
from core.agents import ReActAgent
from core.tools import get_research_tools

# Create a tool-enabled research agent
researcher = ReActAgent(
    name="researcher",
    role="Research topics using available search tools",
    tools=get_research_tools(),
    output_schema={
        "findings": "list of findings",
        "sources": "list of sources",
        "confidence": "float between 0-1"
    },
    verbose=True
)

# Use in a graph
result = researcher(state)
```

**ReActAgent Features**:
- Uses `create_react_agent` from `langchain_classic.agents`
- Returns structured JSON outputs (compatible with LangGraph)
- Tracks which tools were used
- Supports verbose mode for debugging
- Maintains state management compatibility

**ReAct Pattern**:
```
Question/Task → Thought → Action (use tool) → Observation → 
Thought → Action → Observation → ... → Final Answer
```

#### 3. **New Tool-Enabled Examples**

**Example 2B: Sequential Workflow with Tools**
```bash
python examples/tools/example_2b_agents_with_tools.py
```

Architecture:
```
Researcher (+ tools) → Writer (+ tools) → Reviewer → END
```

- **Researcher** uses: search_documents, company_info, web_search
- **Writer** uses: calculator, text_analyzer
- **Reviewer** evaluates without tools

**Example 3B: Supervisor with Tool-Enabled Workers**
```bash
python examples/tools/example_3b_supervisor_with_tools.py
```

Architecture:
```
               Supervisor
                   ↓
    ┌─────────────┼─────────────┐
    ↓             ↓             ↓
Researcher    Writer        Reviewer
(+ tools)     (+ tools)     (no tools)
```

- Supervisor coordinates tool-enabled workers
- Dynamic worker selection based on task
- Each worker has specialized toolset

### Tool Usage Patterns

#### Pattern 1: Create ReAct Agent Directly

```python
from core.agents import ReActAgent
from core.tools import calculator, weather_info

agent = ReActAgent(
    name="analyst",
    role="Analyze data and provide insights",
    tools=[calculator, weather_info],
    output_schema={
        "analysis": "string",
        "metrics": "dict"
    }
)
```

#### Pattern 2: Use in Sequential Graph

```python
from langgraph.graph import StateGraph, END
from core.agents import ReActAgent
from core.tools import get_research_tools, get_writer_tools

# Create tool-enabled agents
researcher = ReActAgent("researcher", "Research topics", get_research_tools())
writer = ReActAgent("writer", "Write content", get_writer_tools())

# Build graph
graph = StateGraph(MultiAgentState)
graph.add_node("researcher", researcher)
graph.add_node("writer", writer)
graph.add_edge("researcher", "writer")
graph.add_edge("writer", END)
graph.set_entry_point("researcher")

# Execute
result = graph.compile().invoke({"task": "Your task"})
```

#### Pattern 3: Mix Tool-Enabled and Regular Agents

```python
from core.agents import ReActAgent, ReviewerAgent
from core.tools import get_research_tools

# Tool-enabled researcher
researcher = ReActAgent("researcher", "Research", get_research_tools())

# Regular reviewer (no tools needed)
reviewer = ReviewerAgent("reviewer")

# Use both in same graph - best of both worlds!
graph.add_node("researcher", researcher)
graph.add_node("reviewer", reviewer)
```

### Tool Categories

| Category | Tools | Use Case |
|----------|-------|----------|
| **basic** | calculator, get_current_time, weather_info | General utilities |
| **research** | search_documents, company_info, web_search | Information gathering |
| **writer** | text_analyzer, calculator | Content creation |
| **all** | All available tools | Maximum flexibility |

### Output Format

ReActAgent maintains LangGraph compatibility with structured outputs:

```json
{
  "agent_outputs": {
    "researcher": {
      "agent_name": "researcher",
      "status": "success",
      "output": {
        "findings": ["finding 1", "finding 2"],
        "sources": ["source 1"],
        "confidence": 0.85
      },
      "reasoning": "Used 3 available tools to complete task",
      "tools_used": ["search_documents", "company_info"],
      "next_action": null
    }
  }
}
```

### When to Use Tool-Enabled vs Regular Agents

**Use Tool-Enabled Agents For**:
- Tasks requiring specific calculations
- Searching documents or databases
- Accessing real-time information
- Tasks requiring structured data lookups
- Example: "Calculate 401k match for $120k salary"

**Use Regular Agents For**:
- Creative writing tasks
- Philosophical or analytical reasoning
- Tasks within LLM knowledge
- General quality review
- Example: "Write a creative marketing story"

**Mix Both**:
- Complex workflows needing both capabilities
- Research (tools) → Writing (LLM) → Review (LLM)
- Use tools where needed, LLM knowledge elsewhere

---

## �📚 Running the Examples

### Example 1: Simple Two Agents

**Purpose**: Learn the basics of LangGraph with two agents communicating sequentially

```bash
python examples/example_1_simple_two_agents.py
```

**Interactive Mode**: 
- Prompts you to enter a research task
- Type 'quit' to exit
- Graph builds once, reuses for multiple tasks

**What to observe**:
- Two agents (researcher, writer) working in sequence
- State flowing between agents
- Structured JSON outputs from each agent
- Simple linear workflow
- Clean, minimal output focused on results

**Concepts covered**:
- Graph creation with `StateGraph`
- Adding nodes with `add_node()`
- Connecting nodes with `add_edge()`
- Setting entry point and end point
- Invoking the graph with interactive input

**Try modifying**:
- Change the task input
- Modify agent roles in `core/agents.py`
- Adjust output schemas
- Add error handling

---

### Example 2: Three Agents Sequential

**Purpose**: Build a complete pipeline with three specialized agents

```bash
python examples/example_2_three_agents_sequential.py
```

**Interactive Mode**:
- Enter a task when prompted
- Graph executes: Researcher → Writer → Reviewer
- Type 'quit' to exit

**What to observe**:
- Research → Write → Review pipeline
- Each agent builds on previous outputs
- Structured JSON from every agent
- Complete workflow from research to final review
- Graph compiled once, reused efficiently

**Concepts covered**:
- Multi-step sequential workflows
- Agent specialization (researcher, writer, reviewer)
- State accumulation across agents
- Final output compilation
- Efficient graph reuse pattern

**Use cases**:
- Content creation pipelines
- Data processing workflows
- Multi-stage analysis

---

### Example 3: Supervisor Pattern

**Purpose**: Implement supervisor-controlled workflow with dynamic agent selection

```bash
python examples/example_3_supervisor_pattern.py
```

**Interactive Mode**:
- Enter a task for the supervisor
- Supervisor dynamically routes to workers
- Type 'quit' to exit

**What to observe**:
- Supervisor decides which agent runs next
- Dynamic workflow based on current state
- Agents report back to supervisor
- Loop continues until supervisor says "FINISH"
- Graph structure allows flexible routing

**Concepts covered**:
- Supervisor agent pattern
- Conditional edges based on supervisor decisions
- Worker agents with specific roles
- Dynamic workflow adaptation
- Centralized decision-making

**Workflow**:
1. Supervisor analyzes task
2. Selects appropriate worker
3. Worker executes and reports back
4. Supervisor decides next step
5. Repeat until complete

---

### Example 4: Conditional Edges

**Purpose**: Master dynamic routing with conditional edges

```bash
python examples/example_4_conditional_edges.py
```

**Interactive Mode**:
- Enter a content creation task
- Reviewer provides quality gate
- Optional verbose mode prompt for detailed output
- Type 'quit' to exit

**What to observe**:
- Routing changes based on review quality score
- Multiple possible paths through graph (approve/reject)
- Conditional logic in edge functions
- Retry loops with max revision limit
- Quality gate pattern in action

**Concepts covered**:
- `add_conditional_edges()` method
- Routing functions
- Edge mapping dictionaries
- State-based decisions
- Max iteration controls

**Patterns**:
- Approval/rejection flows
- Quality gates
- Retry logic
- Dynamic routing based on quality

---

### Example 5: Human-in-the-Loop

**Purpose**: Add human approval checkpoints to workflows

```bash
python examples/example_5_human_in_loop.py
```

**Interactive Mode** (Checkpoint-based):
1. **First run**: Enter thread ID and task → Writer generates content → Pauses → State saved
2. **Second run**: Same thread ID → Shows content → Approve/reject → Resumes workflow
3. **Complete**: Either approved OR max revisions reached

**What to observe**:
- Workflow pauses after writer executes
- Checkpoints save state to SQLite database
- Can resume across program restarts
- Human feedback integration (approval + optional feedback)
- Thread-based session management

**Concepts covered**:
- `SqliteSaver` checkpoint configuration
- `interrupt_after` pattern for pausing
- `update_state()` + `invoke(None, config)` for resuming
- Thread-based state persistence
- Human approval nodes
- State resumption across sessions

**Use cases**:
- Content approval workflows
- Sensitive decision-making
- Quality control gates
- Compliance requirements

**Key Pattern**:
```python
# Compile with interrupt
graph.compile(checkpointer=checkpointer, interrupt_after=["writer"])

# First invoke - runs until interrupt
graph.invoke(initial_state, config)

# Update state with human input
graph.update_state(config, {"approved": True})

# Resume from checkpoint
graph.invoke(None, config)
```

---

### Example 6: Swarm Pattern

**Purpose**: Explore collaborative multi-agent intelligence with parallel execution

```bash
python examples/example_6_swarm_pattern.py
```

**Interactive Mode**:
- Enter an analysis task
- Three specialist analysts execute **simultaneously**
- Aggregator synthesizes all insights
- Type 'quit' to exit

**What to observe**:
- **Parallel execution**: 3 analysts run at same time (not sequential)
- Multiple entry points in graph
- Shared state accumulation from parallel sources
- Convergence pattern (many → one aggregation)
- Final consensus from diverse perspectives

**Concepts covered**:
- Setting multiple entry points for parallel execution
- Parallel node execution (swarm behavior)
- Convergence edges (many analysts → one aggregator)
- Shared state management across parallel paths
- Consensus building from multiple agents
- `recursion_limit` for controlling parallel workflows

**Applications**:
- Market analysis (multiple analyst perspectives)
- Collaborative research
- Multi-perspective decision-making
- Complex problem-solving requiring diverse expertise

**Graph Pattern**:
```
Entry → Analyst 1 ↘
Entry → Analyst 2 → Aggregator → END
Entry → Analyst 3 ↗
```

---

## 🧩 Core Modules

### core/graph_state.py

**State Management**

Defines TypedDict schemas for graph state:

- `AgentState`: Basic agent state
- `MultiAgentState`: Extended multi-agent state
- `SupervisorState`: Supervisor pattern state
- `SwarmState`: Swarm intelligence state
- `HumanLoopState`: Human-in-loop state

**Usage**:
```python
from core.graph_state import MultiAgentState, create_initial_state

state = create_initial_state(
    MultiAgentState,
    task="Write a blog post",
    max_iterations=5
)
```

---

### core/agents.py

**Agent Definitions**

Provides reusable agent classes with JSON outputs:

- `BaseAgent`: Base class for all agents
- `ResearchAgent`: Research and information gathering
- `WriterAgent`: Content creation
- `ReviewerAgent`: Quality review and feedback
- `SupervisorAgent`: Workflow coordination
- **NEW**: `ReActAgent`: Tool-enabled agents using ReAct pattern

**Usage**:
```python
from core.agents import ResearchAgent, WriterAgent, ReActAgent
from core.tools import get_research_tools

# Regular agent
researcher = ResearchAgent()

# Tool-enabled agent
tool_researcher = ReActAgent(
    name="researcher",
    role="Research with tools",
    tools=get_research_tools()
)
```

---

### core/tools.py (NEW)

**Tool Definitions**

Provides reusable tools for agents:

**Individual Tools**:
- `calculator`: Mathematical calculations
- `get_current_time`: Current date/time
- `weather_info`: Weather information
- `search_documents`: RAG-based document search
- `company_info`: Company policies lookup
- `web_search`: Web search
- `text_analyzer`: Text analysis

**Tool Collections**:
```python
from core.tools import (
    get_basic_tools,      # Utility tools
    get_research_tools,   # Research agents
    get_writer_tools,     # Writer agents
    get_all_tools,        # All tools
    get_tools_by_category # Dynamic loading
)
```

**Creating Custom Tools**:
```python
from langchain.tools import tool

@tool
def my_custom_tool(input: str) -> str:
    """Description of what the tool does."""
    # Tool logic here
    return "result"
```

---

### core/graph_builder.py# Use in graph
graph.add_node("researcher", researcher)
graph.add_node("writer", writer)
```

**Custom Agents**:
```python
from core.agents import create_agent_with_tools

analyst = create_agent_with_tools(
    name="sentiment_analyst",
    role="Analyze sentiment in text",
    output_format={
        "sentiment": "positive/negative/neutral",
        "confidence": "float",
        "key_phrases": "list"
    }
)
```

---

### core/graph_builder.py

**Graph Construction Utilities**

Helper class for building LangGraph workflows:

**GraphBuilder Methods**:
- `add_node()`: Add a node to graph
- `add_edge()`: Add direct edge
- `add_conditional_edge()`: Add conditional routing
- `set_entry_point()`: Set start node
- `set_finish_point()`: Add edge to END
- `compile()`: Compile graph for execution

**Convenience Functions**:
- `create_sequential_graph()`: Sequential workflow
- `create_supervisor_graph()`: Supervisor pattern
- `create_parallel_graph()`: Parallel execution
- `create_human_in_loop_graph()`: Human approval

**Usage**:
```python
from core.graph_builder import GraphBuilder
from core.graph_state import AgentState

builder = GraphBuilder(AgentState, checkpoint=True)
builder.add_node("agent1", agent1_func)
builder.add_node("agent2", agent2_func)
builder.add_edge("agent1", "agent2")
builder.set_entry_point("agent1")
builder.set_finish_point("agent2")

graph = builder.compile()
```

---

## 🎓 Training Exercises

### Exercise 1: Add a Fourth Agent

**Objective**: Extend Example 2 with an editor agent

1. Create an `EditorAgent` class
2. Add it to the sequential pipeline
3. Define its JSON output schema
4. Test the complete pipeline

**Expected output**: Research → Write → Review → Edit

---

### Exercise 2: Custom Routing Logic

**Objective**: Implement quality-based routing

1. Add a quality score to reviewer output
2. Route to editor if quality < 7
3. Route to END if quality >= 7
4. Allow max 3 revision loops

---

### Exercise 3: Multi-Supervisor System

**Objective**: Create a hierarchy of supervisors

1. Create a master supervisor
2. Create domain-specific supervisors (content, technical)
3. Master supervisor routes to domain supervisors
4. Domain supervisors manage their workers

---

### Exercise 4: Swarm Consensus

**Objective**: Build a swarm that reaches consensus

1. Create 3-5 analyst agents
2. Each analyzes the same data
3. Agents share findings
4. System converges to consensus
5. Final output is aggregated insights

---

## 💡 Key Design Patterns

### Pattern 1: Sequential Pipeline
```
Input → Agent1 → Agent2 → Agent3 → Output
```
**Use when**: Clear step-by-step process needed

### Pattern 2: Supervisor Coordination
```
          Supervisor
         /    |    \
      W1     W2     W3
```
**Use when**: Dynamic decision-making required

### Pattern 3: Conditional Branching
```
Input → Decision → Branch A / Branch B → Output
```
**Use when**: Different paths based on state

### Pattern 4: Human Approval
```
Agent → Checkpoint → Human → Approve/Reject
```
**Use when**: Human oversight required

### Pattern 5: Swarm Collaboration
```
    Shared Knowledge
   /      |      \
 A1      A2      A3 → Consensus
```
**Use when**: Complex problem needs multiple perspectives

---

## 📊 Comparison: Approaches

### Multi-Agent Patterns

| Approach | Complexity | Use Case | Pros | Cons |
|----------|-----------|----------|------|------|
| Sequential | Low | Linear workflows | Simple, predictable | Inflexible |
| Supervisor | Medium | Dynamic tasks | Adaptive, scalable | More complex |
| Conditional | Medium | Branching logic | Flexible routing | Harder to debug |
| Human-in-Loop | High | Critical decisions | Human oversight | Slower |
| Swarm | High | Complex analysis | Multiple perspectives | Coordination overhead |

### Regular vs Tool-Enabled Agents

| Aspect | Regular Agents | Tool-Enabled Agents (NEW) |
|--------|---------------|---------------------------|
| **Knowledge Source** | LLM training data only | LLM + Real-time tools |
| **Capabilities** | Reasoning, analysis, writing | Above + Search, calculate, lookup |
| **Accuracy** | Limited by training cutoff | Access to current information |
| **Use Cases** | Creative tasks, analysis | Data lookup, calculations, search |
| **Pattern** | Simple LLM invocation | ReAct (Reason → Act → Observe) |
| **Examples** | 1, 2, 3, 4, 5, 6 | 2B, 3B |
| **Best For** | Open-ended problems | Specific data requirements |
| **Tool Tracking** | N/A | Shows which tools were used |

**Recommendation**: Mix both approaches! Use tools where you need specific data/calculations, and regular LLM agents for creative/analytical tasks.

---

## 🎯 Learning Outcomes

By the end of Day 3, participants should be able to:

- [ ] Explain LangGraph architecture (nodes, edges, state)
- [ ] Build sequential multi-agent workflows
- [ ] Implement supervisor patterns for coordination
- [ ] Add conditional routing based on state
- [ ] Configure checkpoints for persistence
- [ ] Implement human-in-the-loop patterns
- [ ] Design swarm intelligence systems
- [ ] **Create tool-enabled agents using ReAct pattern (NEW)**
- [ ] **Integrate tools for search, calculations, and data lookup (NEW)**
- [ ] **Mix regular and tool-enabled agents in workflows (NEW)**
- [ ] Choose appropriate patterns for different use cases
- [ ] Debug multi-agent workflows
- [ ] Build production-ready agent systems

---

## 🎉 Ready to Begin!

You now have a complete Day 3 training setup with:
- ✅ **LangGraph fundamentals** - Nodes, edges, state, checkpoints
- ✅ **Multi-agent patterns** - Sequential, supervisor, swarm
- ✅ **Production features** - Checkpointing, human-in-loop
- ✅ **Structured outputs** - JSON responses from all agents
- ✅ **Progressive examples** - From simple to complex
- ✅ **Reusable components** - Core modules for your projects

**Start your training**: `python examples/example_1_simple_two_agents.py`

---

## 📚 Additional Resources

### Official Documentation
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/) - Official LangGraph docs
- [LangGraph Tutorials](https://langchain-ai.github.io/langgraph/tutorials/) - Step-by-step tutorials
- [LangChain Documentation](https://python.langchain.com/) - LangChain framework docs

### Advanced Topics
- Multi-agent collaboration strategies
- State management best practices
- Checkpoint optimization
- Error handling in graphs
- Performance tuning
- Production deployment

### Next Steps (Day 4+)
- Advanced graph patterns
- Custom checkpoint implementations
- Streaming with LangGraph
- Testing multi-agent systems
- Monitoring and observability
- Deployment strategies
