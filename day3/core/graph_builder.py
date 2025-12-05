"""
Graph Builder - Utilities for Building LangGraph Workflows

This module provides helpers for constructing LangGraph workflows:
- Graph creation and configuration
- Node and edge management
- Conditional routing
- Checkpoint configuration

Key Concepts:
- Graphs are DAGs (Directed Acyclic Graphs) of agent nodes
- Edges connect nodes and control flow
- Conditional edges enable dynamic routing
- Checkpoints enable persistence and human-in-loop
"""

from typing import Dict, Any, List, Callable, Optional, Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import InMemorySaver
import os


class GraphBuilder:
    """
    Helper class for building LangGraph workflows.
    
    Provides a fluent API for constructing complex agent graphs.
    
    Usage:
        builder = GraphBuilder(AgentState)
        builder.add_node("researcher", research_agent)
        builder.add_node("writer", writer_agent)
        builder.add_edge("researcher", "writer")
        builder.add_edge("writer", END)
        graph = builder.compile()
    """
    
    def __init__(self, state_schema: type, checkpoint: bool = False):
        """
        Initialize graph builder.
        
        Args:
            state_schema: TypedDict state schema
            checkpoint: Whether to enable checkpointing
        """
        self.graph = StateGraph(state_schema)
        self.state_schema = state_schema
        self.enable_checkpoint = checkpoint
        self.checkpointer = None
        
        if checkpoint:
            self.checkpointer = self.get_checkpointer()
    
    def get_checkpointer(
        self, 
        use_sqlite: bool = True, 
        checkpoint_dir: str = "checkpoints",
        db_name: str = "checkpoint.db"
    ):
        """
        Get a checkpointer instance for graph persistence.
        
        This method creates and returns a checkpointer that can persist graph state
        across executions, enabling features like human-in-the-loop workflows.
        
        Note:
        - SqliteSaver: Persists state to disk, survives restarts, slower
        - InMemorySaver: Stores state in memory, faster but lost on restart
        - Default is SqliteSaver for reliability and human-in-loop support

        Args:
            use_sqlite: If True, use SqliteSaver (persistent). If False, use InMemorySaver (in-memory only)
            checkpoint_dir: Directory for SQLite database (only used if use_sqlite=True)
            db_name: Name of SQLite database file (only used if use_sqlite=True)
        
        Returns:
            Checkpointer instance (SqliteSaver or InMemorySaver)
        """
        if use_sqlite:
            # Create checkpoint directory if it doesn't exist
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Initialize SQLite checkpointer with persistent storage
            db_path = os.path.join(checkpoint_dir, db_name)
            return SqliteSaver.from_conn_string(db_path)
        else:
            # Initialize in-memory checkpointer (faster but not persistent)
            return InMemorySaver()
    
    def add_node(self, name: str, func: Callable) -> 'GraphBuilder':
        """
        Add a node to the graph.
        
        Args:
            name: Node name
            func: Function to execute (takes state, returns updates)
        
        Returns:
            Self for chaining
        """
        self.graph.add_node(name, func)
        return self
    
    def add_edge(self, source: str, target: str) -> 'GraphBuilder':
        """
        Add a direct edge between nodes.
        
        Args:
            source: Source node name
            target: Target node name (or END)
        
        Returns:
            Self for chaining
        """
        self.graph.add_edge(source, target)
        return self
    
    def add_conditional_edge(
        self,
        source: str,
        condition: Callable[[Dict[str, Any]], str],
        edge_mapping: Optional[Dict[str, str]] = None
    ) -> 'GraphBuilder':
        """
        Add a conditional edge that routes based on state.
        
        Args:
            source: Source node name
            condition: Function that returns next node name
            edge_mapping: Optional mapping of condition outputs to nodes
        
        Returns:
            Self for chaining
        """
        if edge_mapping:
            self.graph.add_conditional_edges(source, condition, edge_mapping)
        else:
            self.graph.add_conditional_edges(source, condition)
        return self
    
    def set_entry_point(self, node: str) -> 'GraphBuilder':
        """
        Set the entry point of the graph.
        
        Args:
            node: Name of the starting node
        
        Returns:
            Self for chaining
        """
        self.graph.set_entry_point(node)
        return self
    
    def set_finish_point(self, node: str) -> 'GraphBuilder':
        """
        Convenience method to add edge to END.
        
        Args:
            node: Node that should end the graph
        
        Returns:
            Self for chaining
        """
        self.graph.add_edge(node, END)
        return self
    
    def compile(self, **kwargs):
        """
        Compile the graph into an executable workflow.
        
        Args:
            **kwargs: Additional compilation options
        
        Returns:
            Compiled graph
        """
        compile_kwargs = {}
        
        if self.enable_checkpoint and self.checkpointer:
            compile_kwargs["checkpointer"] = self.checkpointer
        
        compile_kwargs.update(kwargs)
        
        return self.graph.compile(**compile_kwargs)
    
    def visualize(self, filename: str = "graph.png"):
        """
        Visualize the graph (requires graphviz).
        
        Args:
            filename: Output filename for graph image
        
        Note:
            Requires: pip install pygraphviz
        """
        try:
            from IPython.display import Image, display
            compiled_graph = self.compile()
            display(Image(compiled_graph.get_graph().draw_mermaid_png()))
        except Exception as e:
            print(f"Visualization failed: {e}")
            print("Install pygraphviz: pip install pygraphviz")


# Convenience functions for common patterns

def create_sequential_graph(
    state_schema: type,
    agents: List[tuple[str, Callable]],
    checkpoint: bool = False
):
    """
    Create a sequential graph where agents execute in order.
    
    Args:
        state_schema: State TypedDict class
        agents: List of (name, function) tuples
        checkpoint: Enable checkpointing
    
    Returns:
        Compiled graph
    """
    builder = GraphBuilder(state_schema, checkpoint=checkpoint)
    
    # Add all nodes
    for name, func in agents:
        builder.add_node(name, func)
    
    # Connect sequentially
    for i in range(len(agents) - 1):
        builder.add_edge(agents[i][0], agents[i+1][0])
    
    # Set entry and exit
    builder.set_entry_point(agents[0][0])
    builder.set_finish_point(agents[-1][0])
    
    return builder.compile()


def create_supervisor_graph(
    state_schema: type,
    supervisor: Callable,
    workers: Dict[str, Callable],
    checkpoint: bool = False
):
    """
    Create a supervisor-pattern graph.
    
    The supervisor decides which worker to execute next.
    
    Args:
        state_schema: State TypedDict class
        supervisor: Supervisor function
        workers: Dict of worker name to function
        checkpoint: Enable checkpointing
    
    Returns:
        Compiled graph
    """
    builder = GraphBuilder(state_schema, checkpoint=checkpoint)
    
    # Add supervisor node
    builder.add_node("supervisor", supervisor)
    
    # Add worker nodes
    for name, func in workers.items():
        builder.add_node(name, func)
        # Each worker reports back to supervisor
        builder.add_edge(name, "supervisor")
    
    # Supervisor conditionally routes to workers or END
    def route_supervisor(state):
        selected = state.get("selected_agent", "END")
        if selected.upper() == "FINISH" or state.get("is_complete", False):
            return "end"
        return selected
    
    edge_mapping = {name: name for name in workers.keys()}
    edge_mapping["end"] = END
    
    builder.add_conditional_edge("supervisor", route_supervisor, edge_mapping)
    
    # Set supervisor as entry point
    builder.set_entry_point("supervisor")
    
    return builder.compile()


def create_parallel_graph(
    state_schema: type,
    agents: List[tuple[str, Callable]],
    aggregator: Callable,
    checkpoint: bool = False
):
    """
    Create a graph where agents run in parallel, then aggregate results.
    
    Args:
        state_schema: State TypedDict class
        agents: List of (name, function) tuples to run in parallel
        aggregator: Function to aggregate results
        checkpoint: Enable checkpointing
    
    Returns:
        Compiled graph
    """
    builder = GraphBuilder(state_schema, checkpoint=checkpoint)
    
    # Add a dispatcher node that triggers all agents
    def dispatch(state):
        return state
    
    builder.add_node("dispatch", dispatch)
    builder.set_entry_point("dispatch")
    
    # Add all parallel agents
    for name, func in agents:
        builder.add_node(name, func)
        builder.add_edge("dispatch", name)
        builder.add_edge(name, "aggregator")
    
    # Add aggregator
    builder.add_node("aggregator", aggregator)
    builder.set_finish_point("aggregator")
    
    return builder.compile()


def create_human_in_loop_graph(
    state_schema: type,
    agent: Callable,
    approval_node: str = "human_approval",
    checkpoint: bool = True  # Checkpointing required for human-in-loop
):
    """
    Create a graph with human-in-the-loop approval.
    
    Args:
        state_schema: State TypedDict class
        agent: Agent function to execute
        approval_node: Name of human approval checkpoint
        checkpoint: Enable checkpointing (required)
    
    Returns:
        Compiled graph
    """
    if not checkpoint:
        raise ValueError("Human-in-loop requires checkpointing to be enabled")
    
    builder = GraphBuilder(state_schema, checkpoint=True)
    
    # Add agent node
    builder.add_node("agent", agent)
    
    # Add human approval checkpoint
    def human_approval(state):
        # This node waits for human input
        state["pending_approval"] = True
        return state
    
    builder.add_node(approval_node, human_approval)
    
    # Route: agent -> approval -> conditional (approved/revise)
    builder.set_entry_point("agent")
    builder.add_edge("agent", approval_node)
    
    def approval_route(state):
        if state.get("approved", False):
            return "end"
        else:
            return "agent"
    
    builder.add_conditional_edge(
        approval_node,
        approval_route,
        {"end": END, "agent": "agent"}
    )
    
    return builder.compile()
