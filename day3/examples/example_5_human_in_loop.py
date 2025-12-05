"""
Example 5: Human-in-the-Loop - PROPER Interrupt Patterns & Checkpointing

This example demonstrates CORRECT LangGraph interrupt patterns:
1. Using interrupt_after to pause execution after nodes complete
2. Checkpoint-based workflow persistence across program runs
3. Using update_state() + invoke(None, config) to provide feedback and resume
4. Thread-based session management  
5. Human approval gates with feedback integration

Graph Structure:
┌────────┐   ┌──────────────┐   ┌──────────┐
│ Writer │──▶│Human Approval│──▶│Conditional│
└────────┘   └──────────────┘   └─────┬─────┘
     ▲                                 │
     │                            ┌────┴─────┐
     │                         Approved  Rejected
     │                            │         │
     │                            ▼         ▼
     └────────────────────────── END    (loop back)

Focus: PROPER interrupt patterns using interrupt_after, update_state(), invoke(None, config).
Agents are config-based (settings in config/agents_config.yaml).

Interactive Mode Flow:
1. First run: Creates workflow → Writer executes → Pauses → State saved → Program exits
2. Second run: Loads checkpoint → Shows content → Gets approval → update_state() → invoke(None, config) → Resumes
3. Continues until approved or max revisions reached

Key Pattern with interrupt_after:
    graph.compile(interrupt_after=["writer"])  # Pause AFTER writer
    result = graph.invoke(initial_state, config)  # Runs until interrupt
    # Graph pauses, state saved
    graph.update_state(config, {"approved": True, ...})  # Human input
    result = graph.invoke(None, config)  # Resume from checkpoint

Usage:
    python examples/example_5_human_in_loop.py
    # Run first time: Creates workflow and pauses
    # Run again with same thread ID: Resumes with update_state() + invoke(None, config)
"""

import os
import sys
from dotenv import load_dotenv
import json
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from core.graph_state import HumanLoopState, create_initial_state
from core import WriterAgent

load_dotenv()


def build_graph():
    """Build graph with human-in-the-loop interrupt pattern."""
    print("🏗️  Building Human-in-the-Loop Graph")
    
    graph = StateGraph(HumanLoopState)
    base_writer = WriterAgent()
    
    def writer_node(state: HumanLoopState) -> dict:
        """Writer creates or revises content based on feedback."""
        result = base_writer(state)
        writer_data = result.get("agent_outputs", {}).get("writer", {})
        output = writer_data.get("output", {})
        
        return {
            "agent_output": output,
            "current_node": "writer",
            "pending_approval": True
        }
    
    def human_approval_node(state: HumanLoopState) -> dict:
        """Process human approval decision."""
        return {
            "approved": state.get('approved', False),
            "revision_count": state.get('revision_count', 0),
            "human_feedback": state.get('human_feedback'),
            "pending_approval": False,
            "current_node": "human_approval"
        }
    
    def route_after_approval(state: HumanLoopState) -> str:
        """Route based on human decision."""
        approved = state.get("approved", False)
        revision_count = state.get("revision_count", 0)
        max_revisions = state.get("max_revisions", 3)
        
        if approved:
            return "end"
        elif revision_count >= max_revisions:
            return "end"
        else:
            return "writer"
    
    graph.add_node("writer", writer_node)
    graph.add_node("human_approval", human_approval_node)
    
    graph.add_edge("writer", "human_approval")
    graph.add_conditional_edges(
        "human_approval",
        route_after_approval,
        {
            "end": END,
            "writer": "writer"
        }
    )
    
    graph.set_entry_point("writer")
    
    return graph


def interactive_mode():
    """Interactive mode with checkpoint-based persistence."""
    print("🚀 Human-in-the-Loop with Checkpoint Persistence")
    print("="*70)
    
    thread_id = input("🆔 Thread ID (default: interactive_1): ").strip() or "interactive_1"
    
    graph_builder = build_graph()
    os.makedirs("checkpoints", exist_ok=True)
    
    try:
        with SqliteSaver.from_conn_string("checkpoints/human_loop.db") as checkpointer:
            graph = graph_builder.compile(
                checkpointer=checkpointer,
                interrupt_after=["writer"]
            )
            
            config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 20}
            snapshot = graph.get_state(config)
            
            if snapshot.next:
                # Resume mode: checkpoint exists at an interrupt
                print(f"\n🔄 Resuming from checkpoint (Thread: {thread_id})")
                print(f"Task: {snapshot.values.get('task', 'N/A')}")
                print(f"Revision: {snapshot.values.get('revision_count', 0)}\n")
                
                agent_output = snapshot.values.get("agent_output", {})
                draft = agent_output.get("draft", "No draft")
                print(f"📄 Content:\n{draft}\n")
                
                approval = input("👍 Approve? (yes/no): ").strip().lower()
                feedback = input("💬 Feedback (optional): ").strip()
                
                state_update = {
                    "approved": approval == "yes",
                    "human_feedback": feedback if feedback else None,
                    "pending_approval": False
                }
                
                if approval != "yes":
                    state_update["revision_count"] = snapshot.values.get("revision_count", 0) + 1
                
                graph.update_state(config, state_update)
                result = graph.invoke(None, config=config)
                
                final_snapshot = graph.get_state(config)
                if final_snapshot.next:
                    print("\n⏸️  Paused - Run again to continue")
                else:
                    status = "✅ Approved" if final_snapshot.values.get("approved") else "⚠️  Max revisions"
                    print(f"\n🎯 Complete - {status}")
                    print(f"Revisions: {final_snapshot.values.get('revision_count', 0)}")
            
            elif snapshot.values:
                # Workflow already complete
                print(f"\n✅ Workflow complete for thread: {thread_id}")
                print(f"Approved: {snapshot.values.get('approved', False)}")
                print(f"Revisions: {snapshot.values.get('revision_count', 0)}")
                print("\n💡 Use different thread ID for new workflow")
            
            else:
                # New workflow
                print(f"\n🆕 New workflow (Thread: {thread_id})")
                task = input("📝 Enter task: ").strip() or "Write a blog post about AI"
                
                initial_state = create_initial_state(
                    HumanLoopState,
                    task=task,
                    max_revisions=3,
                    pending_approval=False,
                    approved=False,
                    revision_count=0
                )
                
                result = graph.invoke(initial_state, config=config)
                snapshot = graph.get_state(config)
                
                if snapshot.next:
                    agent_output = snapshot.values.get("agent_output", {})
                    draft = agent_output.get("draft", "No draft")
                    preview = draft[:300] + "..." if len(draft) > 300 else draft
                    print(f"\n⏸️  Paused - Content preview:\n{preview}\n")
                    print(f"💾 State saved (Thread: {thread_id})")
                    print("🔄 Run again to approve/reject")
                
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        traceback.print_exc()


def main():
    """Main entry point."""
    interactive_mode()


if __name__ == "__main__":
    main()
