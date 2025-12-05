"""
Example 4: Conditional Edges - Quality Gates & Retry Logic

This example demonstrates GRAPH FLOW patterns for conditional routing:
1. State-driven routing decisions (routing based on agent outputs)
2. Quality gates with approval/rejection flows
3. Retry logic with iteration tracking
4. Branching workflows with multiple routing paths
5. Combining conditional edges with state updates

Graph Structure:
┌────────┐   ┌────────┐   ┌──────────┐
│Research│──▶│ Writer │──▶│ Reviewer │
└────────┘   └───▲────┘   └────┬─────┘
                 │              │
                 │       ┌──────┴──────┐
                 │       │             │
                 │    Approved?    Rejected?
                 │       │             │
                 │       ▼             ▼
                 │      END      ┌──────────┐
                 │               │Increment │
                 └───────────────│Iteration │
                                 └──────────┘

Focus: Conditional routing, quality gates, retry patterns, state-based decisions.
Agents are config-based (settings in config/agents_config.yaml).

Usage:
    python examples/example_4_conditional_edges.py
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


def increment_iteration(state: MultiAgentState) -> dict:
    """
    Utility node to increment iteration counter.
    
    This is a simple function node (not an agent) that updates state.
    Used in retry logic to track how many attempts have been made.
    """
    current_iteration = state.get("iteration", 0)
    new_iteration = current_iteration + 1
    
    print(f"\n🔄 Retry Logic: Incrementing iteration {current_iteration} → {new_iteration}")
    
    return {"iteration": new_iteration}


def build_graph():
    """
    Build graph with conditional routing and retry logic.
    
    Returns:
        Compiled StateGraph with conditional routing and retry logic
    """
    print("🏗️  Building Graph with Conditional Edges & Retry Logic")
    
    # Step 1: Initialize graph
    graph = StateGraph(MultiAgentState)
    
    # Step 2: Create agents (config-based)
    researcher = ResearchAgent()
    writer = WriterAgent()
    reviewer = ReviewerAgent()
    
    # Step 3: Add agent nodes
    graph.add_node("researcher", researcher)
    graph.add_node("writer", writer)
    graph.add_node("reviewer", reviewer)
    
    # Step 4: Add utility node for state updates
    graph.add_node("increment_iteration", increment_iteration)
    
    # Step 5: Add simple (unconditional) edges
    graph.add_edge("researcher", "writer")
    graph.add_edge("writer", "reviewer")
    graph.add_edge("increment_iteration", "writer")
    
    # Step 6: Define routing function - with multiple paths
    def route_after_review(state: MultiAgentState) -> str:
        """
        Routing function - decides next node based on reviewer output.
        
        This implements a quality gate with retry logic:
        - If approved → END (success path)
        - If rejected but attempts remain → increment_iteration (retry path)
        - If max iterations reached → END (forced end path)
        """
        reviewer_output = state["agent_outputs"].get("reviewer", {})
        approved = reviewer_output.get("output", {}).get("approved", False)
        iteration = state.get("iteration", 0)
        max_iterations = state.get("max_iterations", 3)
        
        print(f"\n🔀 Routing Decision:")
        print(f"   Approved: {approved}")
        print(f"   Iteration: {iteration}/{max_iterations}")
        
        if approved:
            print("   → Route: END (content approved ✅)")
            return "end"
        elif iteration >= max_iterations:
            print("   → Route: END (max iterations reached ⚠️)")
            return "end"
        else:
            print("   → Route: increment_iteration (retry needed 🔄)")
            return "increment"
    
    # Step 7: Add conditional edge with edge mapping
    graph.add_conditional_edges(
        "reviewer",              # Source node
        route_after_review,      # Routing function
        {
            "end": END,                      # If approved or max iterations
            "increment": "increment_iteration"  # If rejected and can retry
        }
    )
    
    # Step 8: Set entry point
    graph.set_entry_point("researcher")
    
    return graph


def run_graph(graph: StateGraph, initial_state: dict, verbose: bool = False):
    """
    Execute workflow with conditional routing and retry logic.
    
    Args:
        graph: The compiled LangGraph to execute
        initial_state: The initial state dictionary
        verbose: Show detailed debug output
    """
    print(f"  Task: {initial_state['task']}")
        
    # recursion_limit prevents infinite loops (default: 25, we set lower for safety)
    config = {"recursion_limit": 15}

    # Execute - may loop multiple times based on reviewer decisions
    result = graph.invoke(initial_state, config)
    
    if verbose:
        print("\n📋 DEBUG: Full State")
        print(f"Agent Outputs: {json.dumps(result.get('agent_outputs', {}), indent=2)}")
        print()
    
    # Display final results
    print(f"\nIterations completed: {result.get('iteration', 0)}")
    print(f"\nReview Status:")
    reviewer_output = result["agent_outputs"].get("reviewer", {})
    review_data = reviewer_output.get("output", {})
    print(f"  Approved: {review_data.get('approved', False)}")
    print(f"  Rating: {review_data.get('rating', 'N/A')}/10")
    
    if review_data.get("approved", False):
        print(f"\n✅ Content approved by quality gate!")
    else:
        print(f"\n⚠️  Content not approved after {result.get('iteration', 0)} iterations")
        if result.get('iteration', 0) >= result.get('max_iterations', 3):
            print(f"   (Max iterations limit reached - forced termination)")
    
    print()
    print("✅ Conditional workflow complete!")


def interactive_mode():
    """Interactive mode for custom tasks."""
    
    # Build and compile graph once
    graph = build_graph()
    compiled_graph = graph.compile()
    
    while True:
        print("─"*70)
        task = input("� Enter task (or 'quit'): ").strip()
        
        if not task:
            continue
        
        if task.lower() == 'quit':
            print("\n👋 Goodbye!")
            break
        
        verbose = input("Show debug output? (y/n): ").lower().strip() == 'y'
        
        try:
            initial_state = create_initial_state(
                MultiAgentState,
                task=task,
                max_iterations=3
            )
            run_graph(compiled_graph, initial_state, verbose=verbose)
        except Exception as e:
            print(f"\n❌ Error: {e}")
            traceback.print_exc()


def main():
    """Main entry point."""
    interactive_mode()


if __name__ == "__main__":
    main()
