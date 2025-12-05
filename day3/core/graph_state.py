"""
Graph State Management - TypedDict State Classes for LangGraph

This module defines state classes used in LangGraph workflows.
LangGraph uses TypedDict to define the schema of state that flows through the graph.

Key Concepts:
- State is shared across all nodes in a graph
- Each node can read from and write to the state
- TypedDict provides type safety and IDE autocomplete
- Operator annotations control how state updates are merged
- All custom states extend MessagesState from langgraph.graph (MessagesState is the base class)

State Update Operators:
- No annotation: Replace the value
- Annotated[list, operator.add]: Append to list
- Annotated[dict, lambda x, y: {**x, **y}]: Merge dictionaries

Best Practice:
- Custom state schemas should extend MessagesState (which provides 'messages' field)
- MessagesState uses add_messages reducer for message accumulation
- Use custom states like MultiAgentState, SupervisorState for specialized workflows
- State is automatically updated based on annotated operators
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langgraph.graph import MessagesState
import operator


# Note: MessagesState is imported from langgraph.graph
# It provides the 'messages' field automatically with add_messages reducer
# All custom states in this module extend MessagesState to inherit message handling


class MultiAgentState(MessagesState):
    """
    Extended state for multi-agent workflows with structured outputs.
    
    Extends langgraph.graph.MessagesState (which provides 'messages' field with add_messages reducer).
    This state supports multiple agents working on different aspects of a task, with each
    agent producing structured JSON output that gets merged into shared state.
    
    Fields (inherited from MessagesState):
        messages: List[BaseMessage] - Conversation history (automatically appended via add_messages)
    
    Additional Fields:
        task: str - The original task/question from user
        current_agent: str - Name of agent currently processing
        agent_outputs: Dict[str, Any] - Structured JSON outputs from each agent (merges via lambda)
        next: str - Name of next node to execute
        iteration: int - Current iteration count (replaced on update)
        max_iterations: int - Maximum allowed iterations (replaced on update)
    
    State Update Behavior:
        - messages: Appended automatically (MessagesState reducer)
        - agent_outputs: Merged using lambda x, y: {**x, **y}
        - Other fields: Replaced with new values
    """
    task: str
    current_agent: str
    # Agent outputs merge instead of replace (merges dicts from multiple agents)
    agent_outputs: Annotated[Dict[str, Any], lambda x, y: {**x, **y}]
    next: str
    iteration: int
    max_iterations: int


class SupervisorState(MessagesState):
    """
    State for supervisor-controlled workflows.
    
    Extends langgraph.graph.MessagesState (which provides 'messages' field with add_messages reducer).
    The supervisor acts as a coordinator that decides which agent should execute next based on
    the current state and task requirements. Used with SupervisorAgent and langgraph_supervisor.
    
    Fields (inherited from MessagesState):
        messages: List[BaseMessage] - Conversation history (automatically appended via add_messages)
    
    Additional Fields:
        task: str - Original user request
        current_step: str - Description of current processing step
        available_agents: List[str] - List of agent names that can be called
        selected_agent: str - Agent selected by supervisor for next execution
        agent_results: Dict[str, Any] - Structured results from each agent (merges via lambda)
        is_complete: bool - Whether the task is finished
        final_output: Optional[str] - Final result when complete
        supervisor_reasoning: Optional[str] - Supervisor's decision-making rationale
        remaining_steps: int - Number of steps remaining (required by create_react_agent)
        structured_response: Optional[Any] - Structured response from supervisor (required by create_react_agent)
    
    State Update Behavior:
        - messages: Appended automatically (MessagesState reducer)
        - agent_results: Merged using lambda x, y: {**x, **y}
        - Other fields: Replaced with new values
    """
    task: str
    current_step: str
    available_agents: List[str]
    selected_agent: str
    # Agent results merge instead of replace (merges results from multiple agents)
    agent_results: Annotated[Dict[str, Any], lambda x, y: {**x, **y}]
    is_complete: bool
    final_output: Optional[str]
    supervisor_reasoning: Optional[str]
    # Required by create_react_agent (used internally by create_supervisor)
    remaining_steps: int
    structured_response: Optional[Any]


class SwarmState(MessagesState):
    """
    State for swarm intelligence patterns.
    
    Extends langgraph.graph.MessagesState (which provides 'messages' field with add_messages reducer).
    Swarm patterns involve multiple agents collaborating dynamically, sharing knowledge and results
    to collectively solve complex problems through parallel execution.
    
    Fields (inherited from MessagesState):
        messages: List[BaseMessage] - Conversation history (automatically appended via add_messages)
    
    Additional Fields:
        task: str - Original task description
        shared_knowledge: Dict[str, Any] - Knowledge base accessible to all agents (merges via lambda)
        agent_contributions: List[Dict[str, Any]] - Individual agent contributions (appends via operator.add)
        active_agents: List[str] - List of currently active agents
        completed_agents: List[str] - List of agents that finished their work
        coordination_messages: List[str] - Inter-agent communication (appends via operator.add)
        convergence_score: float - How close agents are to agreement (0-1)
        final_consensus: Optional[Dict[str, Any]] - Agreed-upon final result
    
    State Update Behavior:
        - messages: Appended automatically (MessagesState reducer)
        - shared_knowledge: Merged using lambda x, y: {**x, **y}
        - agent_contributions: Appended using operator.add
        - coordination_messages: Appended using operator.add
        - Other fields: Replaced with new values
    """
    task: str
    # Shared knowledge merges contributions from all agents
    shared_knowledge: Annotated[Dict[str, Any], lambda x, y: {**x, **y}]
    # Agent contributions append (each agent adds to the list)
    agent_contributions: Annotated[List[Dict[str, Any]], operator.add]
    active_agents: List[str]
    completed_agents: List[str]
    # Coordination messages append (agents communicate via messages)
    coordination_messages: Annotated[List[str], operator.add]
    convergence_score: float
    final_consensus: Optional[Dict[str, Any]]


class HumanLoopState(MessagesState):
    """
    State for human-in-the-loop workflows with checkpoints.
    
    Extends langgraph.graph.MessagesState (which provides 'messages' field with add_messages reducer).
    This state supports patterns where human approval or input is required at specific points in
    the workflow. Used with interrupt_after, checkpointers, and update_state() + invoke(None, config).
    
    Fields (inherited from MessagesState):
        messages: List[BaseMessage] - Conversation history (automatically appended via add_messages)
    
    Additional Fields:
        task: str - Original user task
        current_node: str - Name of current processing node
        pending_approval: bool - Whether human approval is needed
        approval_prompt: Optional[str] - Message to show to human for approval
        human_feedback: Optional[str] - Feedback provided by human
        agent_output: Optional[Dict[str, Any]] - Current agent's output awaiting approval
        approved: bool - Whether the human approved the output
        revision_count: int - Number of times output was revised
        max_revisions: int - Maximum allowed revisions
    
    State Update Behavior:
        - messages: Appended automatically (MessagesState reducer)
        - All other fields: Replaced with new values
        - Human provides input via update_state() after interrupt
    
    Workflow Pattern:
        1. Graph compiled with interrupt_after=["node"]
        2. Graph executes until interrupt point
        3. State saved to checkpoint
        4. Human reviews and provides decision
        5. update_state(config, {"approved": True, ...})
        6. invoke(None, config) to resume
    """
    task: str
    current_node: str
    pending_approval: bool
    approval_prompt: Optional[str]
    human_feedback: Optional[str]
    agent_output: Optional[Dict[str, Any]]
    approved: bool
    revision_count: int
    max_revisions: int


# Utility functions for working with state

def create_initial_state(
    state_class: type,
    task: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Create an initial state with default values.
    
    Args:
        state_class: The TypedDict state class to instantiate
        task: The task description
        **kwargs: Additional fields to override defaults
    
    Returns:
        Dictionary with initial state
    """
    defaults = {
        "messages": [],
        "task": task,
    }
    
    # Add class-specific defaults
    if state_class == MultiAgentState:
        defaults.update({
            "current_agent": "",
            "agent_outputs": {},
            "next": "start",
            "iteration": 0,
            "max_iterations": kwargs.get("max_iterations", 3)
        })
    elif state_class == SupervisorState:
        defaults.update({
            "current_step": "initial",
            "available_agents": kwargs.get("available_agents", []),
            "selected_agent": "",
            "agent_results": {},
            "is_complete": False,
            "final_output": None,
            "supervisor_reasoning": None
        })
    elif state_class == SwarmState:
        defaults.update({
            "shared_knowledge": {},
            "agent_contributions": [],
            "active_agents": kwargs.get("active_agents", []),
            "completed_agents": [],
            "coordination_messages": [],
            "convergence_score": 0.0,
            "final_consensus": None
        })
    elif state_class == HumanLoopState:
        defaults.update({
            "current_node": "start",
            "pending_approval": False,
            "approval_prompt": None,
            "human_feedback": None,
            "agent_output": None,
            "approved": False,
            "revision_count": 0,
            "max_revisions": kwargs.get("max_revisions", 3)
        })
    
    # Override with provided kwargs
    defaults.update(kwargs)
    
    return defaults


def update_state(
    current_state: Dict[str, Any],
    updates: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Update state with new values, respecting annotated operators.
    
    Args:
        current_state: Current state dictionary
        updates: Updates to apply
    
    Returns:
        Updated state dictionary
    """
    new_state = current_state.copy()
    
    for key, value in updates.items():
        if key in new_state:
            # Handle list append
            if isinstance(new_state[key], list) and isinstance(value, list):
                new_state[key] = new_state[key] + value
            # Handle dict merge
            elif isinstance(new_state[key], dict) and isinstance(value, dict):
                new_state[key] = {**new_state[key], **value}
            # Handle replace
            else:
                new_state[key] = value
        else:
            new_state[key] = value
    
    return new_state

