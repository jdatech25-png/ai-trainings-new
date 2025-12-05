"""
Agent Definitions - ReAct-based Agent System with Tool Support

This module provides a unified agent architecture based on LangChain's create_agent.
All agents use the ReAct (Reasoning + Acting) pattern for consistent behavior and tool usage.

Key Concepts:
- All agents are built using create_agent from langchain.agents
- Agents use tools to gather information and perform actions
- Agents return structured JSON outputs for consistency
- Supports custom state schemas extending AgentState
- Supports structured output via response_format parameter
- Factory functions create specialized agents (research, writing, review, etc.)
- ReActAgent class provides the core implementation

Architecture:
    ReActAgent (core implementation)
        ↓
    Factory Functions (create_research_agent, create_writer_agent, etc.)
        ↓
    Business-specific agents with appropriate tools and prompts

Best Practices (from LangChain docs):
- Use state_schema to extend AgentState for custom state fields
- Use response_format with ToolStrategy/ProviderStrategy for structured output
- Custom states must extend AgentState from langchain.agents
"""

from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
import json
import os
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent, AgentState
from langchain.agents.structured_output import ToolStrategy, ProviderStrategy
from langgraph_supervisor import create_supervisor
from langgraph.graph import StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import InMemorySaver
from core.agent_config import AgentConfigLoader
from core import MultiAgentState, SupervisorState
from core.tools import get_tools_by_name

# ============================================================================
# Base Agent Class
# ============================================================================

class BaseAgent(ABC):
    """
    Abstract base class for all agents.
    
    Provides a common interface that all agents must implement.
    Ensures consistency across different agent types.
    
    Attributes:
        name: The agent's name
        _agent: The underlying ReActAgent or implementation
    """
    
    def __init__(self):
        """Initialize base agent. Subclasses must set name and _agent."""
        self.name: str = ""
        self._agent: Optional[ReActAgent] = None
    
    @abstractmethod
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent with the given state.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state dictionary
        """
        pass


# ============================================================================
# ReAct Agent - Core Implementation
# ============================================================================

class ReActAgent:
    """
    Agent that can use tools via ReAct (Reasoning + Acting) pattern.
    
    This agent uses create_agent from langchain.agents which returns
    a compiled graph ready for execution.
    
    Features:
    - Uses tools to gather information or perform actions
    - Reasons about which tools to use
    - Returns structured JSON outputs
    - Compatible with multi-agent graphs
    
    Usage:
        from core.tools import get_research_tools
        
        agent = ReActAgent(
            name="researcher",
            role="Research topics using available tools",
            tools=get_research_tools()
        )
        
        result = agent(state)
    """
    
    def __init__(
        self,
        name: str,
        role: str,
        tools: List[Any],
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_iterations: int = 5,
        verbose: bool = False,
        llm: Optional[ChatOpenAI] = None,
        state_schema: Optional[type] = None,
        response_format: Optional[BaseModel] = None,
        checkpoint: bool = False
    ):
        """
        Initialize a ReAct agent with tools.
        
        Args:
            name: Agent name
            role: Agent's role description
            tools: List of tools the agent can use
            model: OpenAI model to use (ignored if llm is provided)
            temperature: LLM temperature (ignored if llm is provided)
            max_iterations: Maximum reasoning iterations
            verbose: Whether to show agent reasoning
            llm: Pre-configured ChatOpenAI instance (optional, uses shared instance if None)
            state_schema: Custom state schema extending AgentState (optional)
            response_format: Response format strategy for structured output (ToolStrategy/ProviderStrategy)
            checkpoint: Whether to enable checkpointing (default: False)
        """
        self.name = name
        self.role = role
        self.tools = tools
        self.model = model
        self.temperature = temperature
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.state_schema = state_schema
        self.response_format = response_format
        self.enable_checkpoint = checkpoint
        self.checkpointer = None
        
        # Initialize checkpointer if enabled
        if checkpoint:
            self.checkpointer = self.get_checkpointer()
        
        # Use provided LLM or get shared instance from config
        if llm is not None:
            self.llm = llm
        else:
            config = AgentConfigLoader()
            self.llm = config.get_llm(model=model, temperature=temperature)
        
        # Create ReAct agent using langchain.agents.create_agent
        # This returns a compiled graph ready for execution        
        self.agent = create_agent(
            name=self.name,
            model=self.llm,
            tools=self.tools,
            system_prompt=self.role,
            state_schema=self.state_schema,
            response_format=self.response_format,
            checkpointer=self.checkpointer,
            debug=self.verbose
        )
    
    def get_checkpointer(
        self, 
        use_sqlite: bool = True, 
        checkpoint_dir: str = "checkpoints",
        db_name: str = "checkpoint.db"
    ):
        """
        Get a checkpointer instance for agent state persistence.
        
        This method creates and returns a checkpointer that can persist agent state
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

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the ReAct agent.
        
        Args:
            state: Current graph state (must contain 'task' key)
            
        Returns:
            State update with agent output
        """
        task = state.get("task", "")
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"🔧 {self.name.upper()} AGENT (with tools)")
            print(f"{'='*70}")
            print(f"Task: {task}\n")
        
        try:
            # Execute agent using compiled graph
            result = self.agent.invoke({"messages": [{"role": "user", "content": task}]})
            
            # Extract the final message from the result
            messages = result.get("messages", [])
            if messages:
                # Get the last AI message
                last_message = messages[-1]
                output_text = getattr(last_message, 'content', str(last_message))
            else:
                output_text = "No output generated"
            
            # Try to parse JSON from output
            try:
                if "```json" in output_text:
                    output_text = output_text.split("```json")[1].split("```")[0].strip()
                elif "```" in output_text:
                    output_text = output_text.split("```")[1].split("```")[0].strip()
                
                output_json = json.loads(output_text)
            except json.JSONDecodeError:
                # If not JSON, wrap in structure
                output_json = {
                    "result": output_text,
                    "raw_output": True
                }
            
            # Extract tool calls from messages
            tools_used = []
            for msg in messages:
                if hasattr(msg, 'type') and msg.type == 'tool':
                    tools_used.append(getattr(msg, 'name', 'unknown'))
            
            # Create standardized output
            agent_output = {
                "agent_name": self.name,
                "status": "success",
                "output": output_json,
                "reasoning": f"Used {len(self.tools)} available tools to complete task",
                "tools_used": tools_used,
                "next_action": None
            }
            
            if self.verbose:
                print(f"\n✅ {self.name} completed")
                print(f"Tools used: {agent_output['tools_used']}")
                print(f"Output: {json.dumps(output_json, indent=2)}\n")
            
            # Return state update - check which state type we're using
            if "agent_results" in state:
                return {
                    "agent_results": {
                        self.name: agent_output
                    }
                }
            else:
                return {
                    "agent_outputs": {
                        self.name: agent_output
                    }
                }
        
        except Exception as e:
            error_output = {
                "agent_name": self.name,
                "status": "error",
                "output": {},
                "error": str(e),
                "reasoning": f"Error during execution: {str(e)}",
                "next_action": None
            }
            
            if self.verbose:
                print(f"\n❌ {self.name} error: {str(e)}\n")
            
            # Return state update - check which state type we're using
            if "agent_results" in state:
                return {
                    "agent_results": {
                        self.name: error_output
                    }
                }
            else:
                return {
                    "agent_outputs": {
                        self.name: error_output
                    }
                }


# ============================================================================
# Business-Specific Agent Factory Functions
# ============================================================================

def create_research_agent(
    name: Optional[str] = None,
    tools: Optional[List[Any]] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    verbose: bool = False,
    checkpoint: bool = False
) -> ReActAgent:
    """
    Create a research agent that gathers information and analyzes topics.
    
    Configuration is loaded from config/agents_config.yaml.
    The agent uses tools to search documents, access company info, and perform web searches.
    
    Args:
        name: Agent name (uses config default if None)
        tools: List of tools (uses config default tools if None)
        model: OpenAI model to use (uses config default if None)
        temperature: LLM temperature (uses config default if None)
        verbose: Whether to show agent reasoning
        checkpoint: Whether to enable checkpointing (default: False)
    
    Returns:
        ReActAgent configured for research tasks with structured output
    
    Output Schema:
        ResearcherOutput with fields:
        - findings: list of key findings
        - sources: list of sources consulted
        - confidence: float (0-1, how confident the agent is)
        - recommendations: next steps or recommendations    
    """
    # Define the output schema as a Pydantic model
    class ResearcherOutput(BaseModel):
        """Structured output for research agent"""
        findings: List[str] = Field(description="list of key findings or insights discovered")
        sources: List[str] = Field(description="list of sources or tools used")
        confidence: float = Field(description="float between 0-1 indicating confidence level")
        recommendations: str = Field(description="string with next steps or recommendations")
    
    config = AgentConfigLoader()
    agent_type = "research_agent"
    
    # Load configuration from YAML
    name = name or config.get_name(agent_type)
    role = config.get_role(agent_type)
    model = model or config.get_model(agent_type)
    temperature = temperature if temperature is not None else config.get_temperature(agent_type)
    max_iterations = config.get_max_iterations(agent_type)
    
    # Load default tools if not provided
    if tools is None:
        tool_names = config.get_default_tools(agent_type)
        tools = get_tools_by_name(tool_names)
    
    # Get shared LLM instance
    llm = config.get_llm(model=model, temperature=temperature)
    
    return ReActAgent(
        name=name,
        role=role,
        tools=tools,
        model=model,
        temperature=temperature,
        max_iterations=max_iterations,
        verbose=verbose,
        llm=llm,
        state_schema=MultiAgentState,
        response_format=ResearcherOutput,
        checkpoint=checkpoint
    )


def create_writer_agent(
    name: Optional[str] = None,
    tools: Optional[List[Any]] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    verbose: bool = False,
    checkpoint: bool = False
) -> ReActAgent:
    """
    Create a writer agent that creates well-structured content.
    
    Configuration is loaded from config/agents_config.yaml.
    The agent uses tools to analyze text quality and ensure proper structure.
    
    Args:
        name: Agent name (uses config default if None)
        tools: List of tools (uses config default tools if None)
        model: OpenAI model to use (uses config default if None)
        temperature: LLM temperature (uses config default if None)
        verbose: Whether to show agent reasoning
        checkpoint: Whether to enable checkpointing (default: False)
    
    Returns:
        ReActAgent configured for writing tasks with structured output
    
    Output Schema:
        WriterOutput with fields:
        - draft: the written content
        - word_count: number of words
        - sections: list of section names
        - tone: professional/casual/technical
        - revision_notes: areas that might need improvement
    """
    # Define the output schema as a Pydantic model
    class WriterOutput(BaseModel):
        """Structured output for writer agent"""
        draft: str = Field(description="string containing the full written content")
        word_count: int = Field(description="integer count of words in the draft")
        sections: List[str] = Field(description="list of section names/headings used")
        tone: str = Field(description="string describing the tone (professional/casual/technical)")
        revision_notes: str = Field(description="string with suggestions for improvement")
    
    config = AgentConfigLoader()
    agent_type = "writer_agent"
    
    # Load configuration from YAML
    name = name or config.get_name(agent_type)
    role = config.get_role(agent_type)
    model = model or config.get_model(agent_type)
    temperature = temperature if temperature is not None else config.get_temperature(agent_type)
    max_iterations = config.get_max_iterations(agent_type)
    
    # Load default tools if not provided
    if tools is None:
        tool_names = config.get_default_tools(agent_type)
        tools = get_tools_by_name(tool_names)
    
    # Get shared LLM instance
    llm = config.get_llm(model=model, temperature=temperature)
    
    return ReActAgent(
        name=name,
        role=role,
        tools=tools,
        model=model,
        temperature=temperature,
        max_iterations=max_iterations,
        verbose=verbose,
        llm=llm,
        state_schema=MultiAgentState,
        response_format=WriterOutput,
        checkpoint=checkpoint
    )


def create_reviewer_agent(
    name: Optional[str] = None,
    tools: Optional[List[Any]] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    verbose: bool = False,
    checkpoint: bool = False
) -> ReActAgent:
    """
    Create a reviewer agent that evaluates content quality and provides feedback.
    
    Configuration is loaded from config/agents_config.yaml.
    The agent uses tools to analyze text and provide structured feedback.
    
    Args:
        name: Agent name (uses config default if None)
        tools: List of tools (uses config default tools if None)
        model: OpenAI model to use (uses config default if None)
        temperature: LLM temperature (uses config default if None, typically lower for consistency)
        verbose: Whether to show agent reasoning
        checkpoint: If True, use SqliteSaver for persistence; if False, use InMemorySaver
    
    Returns:
        ReActAgent configured for review tasks with structured output
    
    Output Schema:
        ReviewerOutput with fields:
        - rating: 1-10 quality rating
        - strengths: list of content strengths
        - weaknesses: list of areas needing improvement
        - suggestions: list of specific suggestions
        - approved: whether content is approved
        - summary: brief summary of the review    
    """
    # Define the output schema as a Pydantic model
    class ReviewerOutput(BaseModel):
        """Structured output for reviewer agent"""
        rating: int = Field(description="integer from 1-10 indicating overall quality")
        strengths: List[str] = Field(description="list of positive aspects of the content")
        weaknesses: List[str] = Field(description="list of areas that need improvement")
        suggestions: List[str] = Field(description="list of specific actionable suggestions")
        approved: bool = Field(description="boolean indicating if content meets quality standards")
        summary: str = Field(description="string with a brief summary of the review")
    
    config = AgentConfigLoader()
    agent_type = "reviewer_agent"
    
    # Load configuration from YAML
    name = name or config.get_name(agent_type)
    role = config.get_role(agent_type)
    model = model or config.get_model(agent_type)
    temperature = temperature if temperature is not None else config.get_temperature(agent_type)
    max_iterations = config.get_max_iterations(agent_type)
    
    # Load default tools if not provided
    if tools is None:
        tool_names = config.get_default_tools(agent_type)
        tools = get_tools_by_name(tool_names)
    
    # Get shared LLM instance
    llm = config.get_llm(model=model, temperature=temperature)
    
    return ReActAgent(
        name=name,
        role=role,
        tools=tools,
        model=model,
        temperature=temperature,
        max_iterations=max_iterations,
        verbose=verbose,
        llm=llm,
        state_schema=MultiAgentState,
        response_format=ReviewerOutput,
        checkpoint=checkpoint
    )


def create_supervisor_agent(
    available_agents: List[Any],  # Changed from List[str] to List[Any] to accept agent instances
    name: Optional[str] = None,
    tools: Optional[List[Any]] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    prompt: Optional[str] = None
) -> StateGraph:
    """
    Create a supervisor workflow that coordinates other agents.
    
    This uses the official langgraph_supervisor.create_supervisor implementation.
    Configuration is loaded from config/agents_config.yaml for defaults.
    
    Args:
        available_agents: List of agent instances (compiled graphs) to coordinate
        name: Supervisor name (uses config default if None)
        tools: Additional custom tools for the supervisor (optional)
        model: OpenAI model to use (uses config default if None)
        temperature: LLM temperature (uses config default if None)
        prompt: Custom prompt for supervisor (uses config default if None)
    
    Returns:
        StateGraph (uncompiled) that coordinates the provided agents with structured output
    
    Output Schema:
        SupervisorOutput with fields:
        - selected_agent: name of next agent to execute OR 'FINISH' when complete
        - reasoning: explanation for selecting this agent or finishing
        - is_complete: true if task is complete and selecting FINISH
        - progress: current progress description
        - instructions: specific instructions for the selected agent
    
    """
    # Define the output schema as a Pydantic model
    class SupervisorOutput(BaseModel):
        """Structured output for supervisor agent"""
        selected_agent: str = Field(description="string - name of next agent to execute OR 'FINISH' when complete")
        reasoning: str = Field(description="string explaining why this agent should go next or why finishing")
        is_complete: bool = Field(description="boolean - true if task is complete and selecting FINISH")
        progress: str = Field(description="string describing current progress")
        instructions: str = Field(description="string with specific instructions for the selected agent")
    
    config = AgentConfigLoader()
    agent_type = "supervisor_agent"
    
    # Load configuration from YAML
    name = name or config.get_name(agent_type)
    model_name = model or config.get_model(agent_type)
    temperature_val = temperature if temperature is not None else config.get_temperature(agent_type)
    
    # Load default tools if not provided
    if tools is None:
        tool_names = config.get_default_tools(agent_type)
        tools = get_tools_by_name(tool_names)

    # Get custom prompt or use config default
    if prompt is None:
        # Get role from config (contains available agents template)
        if available_agents:
            agent_names = [getattr(agent, 'name', f'agent_{i}') for i, agent in enumerate(available_agents)]
            agents_str = ', '.join(agent_names)
            prompt = config.get_role(agent_type, available_agents=agents_str)
        else:
            prompt = config.get_role(agent_type, available_agents="")
    
    # Get shared LLM instance
    llm = config.get_llm(model=model_name, temperature=temperature_val)
    
    # Create supervisor using langgraph_supervisor
    workflow = create_supervisor(
        supervisor_name=name,
        agents=available_agents,
        model=llm,
        prompt=prompt,
        tools=tools or [],  # Additional tools if provided
        state_schema=SupervisorState,
        response_format=SupervisorOutput
    )
    
    return workflow


# ============================================================================
# Agent Classes - Inherit from BaseAgent
# ============================================================================
# These classes provide a clean interface for creating agents.
# All configuration is loaded from config/agents_config.yaml.

class ResearchAgent(BaseAgent):
    """
    Research agent that gathers information and analyzes topics.
    
    Configuration is loaded from config/agents_config.yaml.
    Inherits from BaseAgent for type safety and consistency.
    
    Usage:
        researcher = ResearchAgent(name="researcher")
        result = researcher(state)
    """
    def __init__(self, name: Optional[str] = None, verbose: bool = False):
        super().__init__()
        self._agent = create_research_agent(name=name, verbose=verbose)
        self.name = self._agent.name
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return self._agent(state)


class WriterAgent(BaseAgent):
    """
    Writer agent that creates well-structured content.
    
    Configuration is loaded from config/agents_config.yaml.
    Inherits from BaseAgent for type safety and consistency.
    
    Usage:
        writer = WriterAgent(name="writer")
        result = writer(state)
    """
    def __init__(self, name: Optional[str] = None, verbose: bool = False):
        super().__init__()
        self._agent = create_writer_agent(name=name, verbose=verbose)
        self.name = self._agent.name
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return self._agent(state)


class ReviewerAgent(BaseAgent):
    """
    Reviewer agent that evaluates content quality and provides feedback.
    
    Configuration is loaded from config/agents_config.yaml.
    Inherits from BaseAgent for type safety and consistency.
    
    Usage:
        reviewer = ReviewerAgent(name="reviewer")
        result = reviewer(state)
    """
    def __init__(self, name: Optional[str] = None, verbose: bool = False):
        super().__init__()
        self._agent = create_reviewer_agent(name=name, verbose=verbose)
        self.name = self._agent.name
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return self._agent(state)


class SupervisorAgent:
    """
    Supervisor agent that coordinates multiple worker agents.
    
    Configuration is loaded from config/agents_config.yaml.
    This class internally uses create_supervisor_agent() which leverages
    the official langgraph_supervisor.create_supervisor implementation.
    
    The supervisor accepts class-based agents (ResearchAgent, WriterAgent, etc.)
    that inherit from BaseAgent and converts them internally to the format needed 
    by langgraph_supervisor.
    
    Usage:
        # Create worker agents using classes (loads from config)
        researcher = ResearchAgent()
        writer = WriterAgent()
        reviewer = ReviewerAgent()
        
        # Create supervisor (all params come from config)
        supervisor = SupervisorAgent(
            available_agents=[researcher, writer, reviewer]
        )
        
        # The supervisor provides a compiled graph
        result = supervisor.app.invoke({"messages": [{"role": "user", "content": "task"}]})
    """
    def __init__(
        self,
        available_agents: List[BaseAgent]
    ):
        """
        Initialize supervisor agent.
        
        Args:
            available_agents: List of BaseAgent instances (ResearchAgent, WriterAgent, etc.)
        """
        # Store configuration
        self.available_agents = available_agents
        
        config = AgentConfigLoader()
        
        # Get supervisor name from config
        self.name = config.get_name("supervisor_agent")
        
        compiled_agents = []
        for available_agent in available_agents:
            # Each BaseAgent has a _agent (ReActAgent) with tools
            if hasattr(available_agent, '_agent') and available_agent._agent is not None:
                react_agent = available_agent._agent
                # Create compiled graph from the agent's tools and name
                compiled_agents.append(react_agent.agent)
            else:
                # If already a compiled agent, use as-is
                compiled_agents.append(available_agent)
        
        # Create supervisor workflow using factory function
        workflow = create_supervisor_agent(
            available_agents=compiled_agents
        )
        
        # Compile the workflow
        self.app = workflow.compile()

            
    def invoke(self, messages: List[Dict[str, str]], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Invoke the supervisor with a list of messages.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            config: Optional config dict (e.g., {"recursion_limit": 20})
            
        Returns:
            Result dict with 'messages' key containing conversation history
        """
        if config is None:
            config = {"recursion_limit": 25}  # Default LangGraph recursion limit
        return self.app.invoke({"messages": messages}, config)
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make the supervisor callable for compatibility.
        
        This allows SupervisorAgent to be used in graphs like other agents.
        Automatically converts state to messages format for the supervisor.
        """
        # If state already has messages, use them
        if "messages" in state:
            return self.app.invoke(state)
        
        # Otherwise, construct messages from task
        task = state.get("task", "")
        messages = [{"role": "user", "content": task}]
        
        return self.app.invoke({"messages": messages})

# ============================================================================