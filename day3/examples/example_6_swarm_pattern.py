"""
Example 6: Swarm Pattern - Parallel Execution & Result Aggregation

This example demonstrates GRAPH FLOW patterns for parallel processing:
1. Multiple entry points (parallel agent execution)
2. Simultaneous node execution (swarm behavior)
3. Result aggregation from multiple sources
4. Shared state updates from parallel agents
5. Convergence patterns (many → one)

Graph Structure:
          ┌─────────────────────┐
          │  Shared Knowledge   │
          └──────────┬──────────┘
                     │
         ┌───────────┼──────────┐
         │           │          │
         ▼           ▼          ▼
    ┌────────┐  ┌────────┐  ┌────────┐
    │Analyst1│  │Analyst2│  │Analyst3│  (Parallel Execution)
    └───┬────┘  └───┬────┘  └───┬────┘
        │           │           │
        └───────────┴───────────┘
                    │
                    ▼
             ┌──────────────┐
             │  Aggregator  │──▶ END
             └──────────────┘

Focus: Parallel execution, multiple entry points, result aggregation, convergence.
Note: This example uses custom analyst agents to demonstrate swarm behavior.

Usage:
    python examples/example_6_swarm_pattern.py
"""

import os
import sys
from dotenv import load_dotenv
import json
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from core.graph_state import SwarmState, create_initial_state

load_dotenv()


def create_analyst_agent(analyst_id: int, specialty: str):
    """
    Create a specialized analyst agent for swarm.
    
    Note: In a real swarm pattern, you might use config-based agents.
    Here we create custom analysts to demonstrate parallel execution
    with different specialties contributing to shared knowledge.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    
    def analyst(state: SwarmState) -> dict:
        """Analyst agent - contributes specialist insight to swarm."""
        print(f"\n🔬 ANALYST {analyst_id} ({specialty}) - Analyzing in parallel...")
        print("-"*70)
        
        task = state["task"]
        shared_knowledge = state.get("shared_knowledge", {})
        
        # Analyze from specialist perspective
        messages = [
            SystemMessage(content=f"""You are Analyst {analyst_id}, specializing in {specialty}.
Analyze the task and provide insights from your specialist perspective.

Respond with JSON:
{{
    "insight": "your main insight",
    "supporting_evidence": ["evidence 1", "evidence 2"],
    "confidence": 0.0-1.0,
    "recommendations": "your recommendations"
}}"""),
            HumanMessage(content=f"Task: {task}\n\nShared Knowledge: {json.dumps(shared_knowledge, indent=2)}")
        ]
        
        response = llm.invoke(messages)
        
        try:
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            analysis = json.loads(content)
        except:
            analysis = {
                "insight": response.content,
                "confidence": 0.7,
                "supporting_evidence": [],
                "recommendations": "See insight"
            }
        
        print(f"Insight: {analysis['insight'][:100]}...")
        print(f"Confidence: {analysis['confidence']}")
        
        # Contribute to shared swarm knowledge
        contribution = {
            "agent": f"analyst_{analyst_id}",
            "specialty": specialty,
            "analysis": analysis
        }
        
        return {
            "agent_contributions": [contribution],
            "shared_knowledge": {f"analyst_{analyst_id}": analysis}
        }
    
    return analyst


def create_aggregator():
    """
    Create aggregator that synthesizes parallel swarm outputs.
    
    This demonstrates the convergence node pattern - taking multiple
    parallel inputs and producing a single synthesized output.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    def aggregator(state: SwarmState) -> dict:
        """Aggregator - synthesizes all parallel analyst contributions."""
        print("\n🎯 AGGREGATOR - Synthesizing Parallel Swarm Intelligence")
        print("="*70)
        
        contributions = state.get("agent_contributions", [])
        
        # Synthesize all parallel contributions
        all_insights = [c["analysis"]["insight"] for c in contributions]
        
        messages = [
            SystemMessage(content="""You are an aggregator synthesizing insights from multiple parallel analysts.
Create a consensus view that integrates all perspectives.

Respond with JSON:
{
    "consensus": "the agreed-upon conclusion",
    "key_points": ["point 1", "point 2", "point 3"],
    "confidence": 0.0-1.0,
    "diverse_perspectives": "summary of different viewpoints"
}"""),
            HumanMessage(content=f"Insights to synthesize:\n{json.dumps(all_insights, indent=2)}")
        ]
        
        response = llm.invoke(messages)
        
        try:
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            consensus = json.loads(content)
        except:
            consensus = {
                "consensus": response.content,
                "key_points": [],
                "confidence": 0.8,
                "diverse_perspectives": "See consensus"
            }
        
        print(f"\nConsensus reached from {len(contributions)} parallel analysts:")
        print(json.dumps(consensus, indent=2))
        
        return {
            "final_consensus": consensus,
            "convergence_score": consensus.get("confidence", 0.8)
        }
    
    return aggregator


def build_graph():
    """
    Build swarm intelligence graph with parallel execution.
    
    This demonstrates parallel processing patterns:
    - Multiple entry points (all analysts start simultaneously)
    - Parallel node execution (swarm behavior)
    - All parallel nodes converge to one aggregator
    - Shared state accumulation from multiple sources
    - Many-to-one convergence pattern
    
    Returns:
        Compiled StateGraph with parallel execution pattern
    """
    print("🏗️  Building Swarm Intelligence Graph (Parallel Pattern)")
    
    # Step 1: Initialize graph
    graph = StateGraph(SwarmState)
    
    # Step 2: Create specialist analyst agents
    analysts = [
        ("analyst_1", create_analyst_agent(1, "technical analysis")),
        ("analyst_2", create_analyst_agent(2, "business strategy")),
        ("analyst_3", create_analyst_agent(3, "user experience"))
    ]
    
    # Step 3: Create aggregator (convergence node)
    aggregator = create_aggregator()
    
    # Step 4: Add all analyst nodes
    for name, agent in analysts:
        graph.add_node(name, agent)
    
    # Step 5: Add aggregator node
    graph.add_node("aggregator", aggregator)
    
    # Step 6: All analysts feed into aggregator (convergence edges)
    for name, _ in analysts:
        graph.add_edge(name, "aggregator")
    
    # Step 7: Aggregator to END
    graph.add_edge("aggregator", END)
    
    # Step 8: Set multiple entry points (PARALLEL EXECUTION!)
    graph.set_entry_point("analyst_1")
    graph.set_entry_point("analyst_2")
    graph.set_entry_point("analyst_3")
    
    return graph


def run_graph(app, initial_state):
    """Execute swarm analysis with compiled graph."""
    try:
        config = {"recursion_limit": 15}
        result = app.invoke(initial_state, config)
        
        print("\n" + "="*70)
        print("� SWARM ANALYSIS RESULT")
        print("="*70)
        
        # Display individual contributions
        print("\n� Individual Contributions:")
        for contribution in result.get("agent_contributions", []):
            print(f"\n{contribution['agent']} ({contribution['specialty']}):")
            print(f"  {contribution['analysis']['insight'][:150]}...")
        
        # Display consensus
        print("\n🤝 Final Consensus:")
        consensus = result.get("final_consensus", {})
        print(json.dumps(consensus, indent=2))
        
        print(f"\n📈 Convergence Score: {result.get('convergence_score', 0):.2f}")
        print("\n" + "="*70)
        
    except Exception as e:
        print(f"\n❌ Error in swarm execution: {str(e)}")
        traceback.print_exc()


def interactive_mode():
    """Interactive mode for swarm intelligence analysis."""
    print("🚀 Swarm Pattern - Parallel Execution")
    print("="*70)
    
    # Build and compile graph once
    graph = build_graph()
    app = graph.compile()
    print("✅ Graph compiled and ready\n")
    
    while True:
        task = input("📝 Enter analysis task (or 'quit'): ").strip()
        if task.lower() == 'quit':
            print("\n👋 Goodbye!")
            break
        if task:
            initial_state = create_initial_state(
                SwarmState,
                task=task,
                active_agents=["analyst_1", "analyst_2", "analyst_3"]
            )
            run_graph(app, initial_state)


def main():
    """Main entry point."""
    interactive_mode()


if __name__ == "__main__":
    main()
