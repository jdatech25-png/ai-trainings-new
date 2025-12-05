"""
Example 2: Three Agents Sequential - Pipeline Pattern

This example demonstrates the GRAPH FLOW for a three-agent pipeline:
1. Building a sequential pipeline with multiple agents
2. Creating a linear workflow chain: A → B → C → END
3. State flow through multiple processing stages
4. Each agent processes and enriches the shared state

Graph Structure:
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  Researcher  │──▶│   Writer     │──▶│  Reviewer    │──▶ END
└──────────────┘   └──────────────┘   └──────────────┘

Focus: Sequential edge connections and multi-stage processing.
Agents are config-based (settings in config/agents_config.yaml).

Usage:
    python examples/example_2_three_agents_sequential.py
"""

import os
import sys
from dotenv import load_dotenv
import json
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.graph import StateGraph, END
from core.graph_state import MultiAgentState, create_initial_state
from core import ResearchAgent, WriterAgent, ReviewerAgent

load_dotenv()


def build_graph():
    """
    Build a three-agent sequential pipeline graph.
    
    Returns:
        Compiled StateGraph ready for execution
    """
    print("🏗️  Building Three-Agent Sequential Pipeline")
    
    # Step 1: Initialize graph with state schema
    graph = StateGraph(MultiAgentState)
    
    # Step 2: Create agent instances (config-based)
    researcher = ResearchAgent()  # Loads all settings from config
    writer = WriterAgent()
    reviewer = ReviewerAgent()
    
    # Step 3: Add nodes
    graph.add_node("researcher", researcher)
    graph.add_node("writer", writer)
    graph.add_node("reviewer", reviewer)
    
    # Step 4: Create sequential pipeline with edges
    graph.add_edge("researcher", "writer")
    graph.add_edge("writer", "reviewer")
    graph.add_edge("reviewer", END)
    
    # Step 5: Set entry point
    graph.set_entry_point("researcher")
    
    return graph


def run_graph(graph: StateGraph, initial_state: dict):
    """
    Execute the three-agent pipeline.
    
    Args:
        graph: The compiled LangGraph to execute
        initial_state: The initial state dictionary
    """
    print(f"  Task: {initial_state['task']}")

    # recursion_limit prevents runaway execution (best practice even for sequential flows)
    config = {"recursion_limit": 10}

    # Execute - watch state flow through all three stages
    result = graph.invoke(initial_state, config)

    # Display final state after pipeline completion
    for agent_name in ["researcher", "writer", "reviewer"]:
        if agent_name in result["agent_outputs"]:
            print(f"\n{agent_name.upper()} OUTPUT:")
            print(json.dumps(result["agent_outputs"][agent_name], indent=2))
    
    print()
    print("✅ Pipeline execution complete!")


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
