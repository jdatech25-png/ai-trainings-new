"""
Example 3: Supervisor Pattern - Conditional Routing & Dynamic Workflow

This example demonstrates the GRAPH FLOW for a supervisor-coordinated system:
1. Supervisor agent making dynamic routing decisions
2. Worker agents reporting back to supervisor (cycle pattern)
3. Loop until completion condition is met
4. Mapping routing decisions to different graph paths
5. Tool-based handoff mechanism for agent coordination

Graph Structure:
                  ┌────────────┐
             ┌────│ Supervisor │◀────┐
             │    └────────────┘     │
             │                       │
    ┌────────┼────────┬──────────────┤
    ▼        ▼        ▼              │
┌────────┐┌────────┐┌──────────┐    │
│Research││ Writer ││ Reviewer │────┘
└────────┘└────────┘└──────────┘
(All workers report back to supervisor)

Focus: Conditional routing, routing functions, and cyclic graph patterns.
Agents are config-based (settings in config/agents_config.yaml).

Usage:
    python examples/example_3_supervisor_pattern.py
"""

import os
import sys
from dotenv import load_dotenv
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import SupervisorAgent, ResearchAgent, WriterAgent, ReviewerAgent

load_dotenv()


def build_graph():
    """
    Build a supervisor-controlled graph using SupervisorAgent.
        
    Returns:
        SupervisorAgent instance with compiled workflow
    """
    print("🏗️  Building Supervisor-Controlled Graph")
    
    # Step 1: Create worker agents (config-based)
    researcher = ResearchAgent()
    writer = WriterAgent()
    reviewer = ReviewerAgent()
    
    # Step 2: Create supervisor using SupervisorAgent class
    supervisor = SupervisorAgent(
        available_agents=[researcher, writer, reviewer]
    )
    
    return supervisor


def run_graph(supervisor: SupervisorAgent, task: str):
    """
    Execute supervisor-controlled workflow.
    
    Args:
        supervisor: The SupervisorAgent instance
        task: The task description to process
    """
    print(f"  Task: {task}")
    
    try:
        inputs = [{"role": "user", "content": task}]        
        config = {"recursion_limit": 20}    # Limits supervisor coordination cycles

        # Execute with messages-based input and recursion_limit to prevent infinite loops
        result = supervisor.invoke(inputs, config)

    except Exception as e:
        print(f"\n⚠️  Error during execution: {e}")
        return
    
    # Display results
    print("\n📨 Message History:")
    for i, msg in enumerate(result.get("messages", []), 1):
        role = getattr(msg, 'type', getattr(msg, 'role', 'unknown'))
        content = getattr(msg, 'content', str(msg))
        
        # Extract agent name if available
        agent_name = getattr(msg, 'name', None)
        
        # Format the header with agent name if available
        if agent_name:
            print(f"\n{i}. [{role.upper()}] - Node: {agent_name}")
        else:
            print(f"\n{i}. [{role.upper()}]")
        
        # Display content with truncation for long messages
        if len(content) > 200:
            print(f"{content[:200]}...")
        else:
            print(content)
    
    print()
    print("✅ Supervisor pattern executed successfully!")


def interactive_mode():
    """Interactive mode for custom tasks."""
    
    # Build supervisor once
    supervisor = build_graph()
    
    while True:
        print("─"*70)
        task = input("� Enter task (or 'quit'): ").strip()
        
        if not task:
            continue
        
        if task.lower() == 'quit':
            print("\n👋 Goodbye!")
            break
        
        try:
            run_graph(supervisor, task)
        except Exception as e:
            print(f"\n❌ Error: {e}")
            traceback.print_exc()


def main():
    """Main entry point."""
    interactive_mode()


if __name__ == "__main__":
    main()
