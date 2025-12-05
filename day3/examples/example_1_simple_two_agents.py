"""
Example 1: Simple Two Agents - Introduction to LangGraph

This example demonstrates the GRAPH FLOW for a basic two-agent system:
1. Creating a LangGraph with two sequential agents
2. Adding nodes and edges to define workflow
3. Sequential agent communication (Researcher → Writer)
4. State management between agents
5. Graph compilation and execution

Graph Structure:
┌──────────────┐      ┌──────────────┐
│  Researcher  │─────▶│   Writer     │───▶ END
│  (Research)  │      │  (Create)    │
└──────────────┘      └──────────────┘

Focus: Graph structure, node connections, and execution flow.
Agents are created using config-based factory pattern (covered in core/agents.py).

Usage:
    python examples/example_1_simple_two_agents.py
"""

import os
import sys
from typing import Any
from dotenv import load_dotenv
import json
import traceback

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# LangGraph imports
from langgraph.graph import StateGraph, END

# Local imports - use config-based agents
from core.graph_state import MultiAgentState, create_initial_state
from core import ResearchAgent, WriterAgent

# Load environment
load_dotenv()


def build_graph():
    """
    Build the two-agent LangGraph.
    
    Agents are instantiated using config-based classes that load all
    settings (prompts, tools, models) from config/agents_config.yaml.
    
    Returns:
        The compiled StateGraph object
    """
    print("🏗️  BUILDING LANGGRAPH - Two Agent Sequential Flow")
    
    # Step 1: Create graph with state schema
    graph = StateGraph(MultiAgentState)
    
    # Step 2: Create agent instances (config-based - no hardcoded settings!)
    researcher = ResearchAgent()  # Loads from config/agents_config.yaml
    writer = WriterAgent()        # Loads from config/agents_config.yaml
    
    # Step 3: Add nodes to graph
    graph.add_node("researcher", researcher)
    graph.add_node("writer", writer)
    
    # Step 4: Define the flow with edges
    graph.add_edge("researcher", "writer")
    graph.add_edge("writer", END)
    
    # Step 5: Set entry point
    graph.set_entry_point("researcher")
            
    return graph


def run_graph(graph: StateGraph, initial_state: dict[str, Any]):
    """
    Execute the compiled graph.
    
    Args:
        graph: The compiled LangGraph to execute
        initial_state: The initial state dictionary
    """    
    print(f"  Task: {initial_state['task']}")
    
    # recursion_limit prevents runaway execution (best practice even for sequential flows)
    config = {"recursion_limit": 10}

    # Execute the graph - state flows through: researcher → writer → END
    result = graph.invoke(initial_state, config)

    print("\n1️⃣  Research Output:")
    if "researcher" in result["agent_outputs"]:
        researcher_data = result["agent_outputs"]["researcher"]
        # Extract nested output structure
        output = researcher_data.get("output", researcher_data)
        print(json.dumps(output, indent=2))
    
    print("\n2️⃣  Writer Output:")
    if "writer" in result["agent_outputs"]:
        writer_data = result["agent_outputs"]["writer"]
        # Extract nested output structure
        output = writer_data.get("output", writer_data)
        
        if "draft" in output:
            print(f"Draft:\n{output['draft']}\n")
            print(f"Metadata:")
            print(f"  - Word Count: {output.get('word_count', 'N/A')}")
            print(f"  - Sections: {output.get('sections', [])}")
            print(f"  - Tone: {output.get('tone', 'N/A')}")
        else:
            print(json.dumps(output, indent=2))
    
    print()
    print("✅ EXECUTION COMPLETE")


def interactive_mode():
    """Interactive mode for custom tasks."""
    
    # Build and compile graph once
    graph = build_graph()
    compiled_graph = graph.compile()
    
    while True:
        print("─"*70)
        task = input("📝 Enter task (or 'quit'): ").strip()
        
        if not task:
            continue
        
        if task.lower() == 'quit':
            print("\n👋 Goodbye!")
            break
        
        try:
            initial_state = create_initial_state(MultiAgentState, task=task)
            run_graph(compiled_graph, initial_state)
        except Exception as e:
            print(f"\n❌ Error: {e}")
            traceback.print_exc()


def main():
    """Main entry point."""    
    interactive_mode()


if __name__ == "__main__":
    main()
