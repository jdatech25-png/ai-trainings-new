"""
Core Module for Day 3: Multi-Agent Systems with LangGraph

This module provides reusable components for building multi-agent workflows:
- graph_state: State management for graphs (all states extend AgentState)
- agents: ReAct-based agent system with factory functions and backward compatibility
- agent_config: YAML-based configuration loader with LLM caching
- graph_builder: Utilities for building LangGraph workflows
- tools: Tool definitions for agents (calculator, search, weather, etc.)

All agents now use the ReAct (Reasoning + Acting) pattern via create_agent.
Agent configurations are centralized in config/agents_config.yaml.

Best Practices:
- Use state_schema parameter in create_agent for custom state
- Use response_format with ToolStrategy/ProviderStrategy for structured output
- All custom states must extend AgentState from langchain.agents
"""

from .graph_state import MultiAgentState, SupervisorState
from .agent_config import AgentConfigLoader, get_config
from .agents import (
    BaseAgent,
    ReActAgent,
    ResearchAgent,
    WriterAgent,
    ReviewerAgent,
    SupervisorAgent,
    create_research_agent,
    create_writer_agent,
    create_reviewer_agent,
    create_supervisor_agent
)
from .graph_builder import GraphBuilder
from .tools import (
    calculator,
    get_current_time,
    weather_info,
    search_documents,
    company_info,
    web_search,
    text_analyzer,
    get_basic_tools,
    get_research_tools,
    get_writer_tools,
    get_all_tools,
    get_tools_by_category,
    get_tools_by_name
)

# Import from langchain.agents for external use
from langchain.agents import AgentState
from langchain.agents.structured_output import ToolStrategy, ProviderStrategy

__all__ = [
    # State (AgentState imported from langchain.agents)
    "AgentState",
    "MultiAgentState",
    "SupervisorState",
    # Structured Output
    "ToolStrategy",
    "ProviderStrategy",
    # Configuration
    "AgentConfigLoader",
    "get_config",
    # Base Agent
    "BaseAgent",
    # Agent Classes
    "ResearchAgent",
    "WriterAgent",
    "ReviewerAgent",
    "SupervisorAgent",
    "ReActAgent",
    # Agent Factory Functions
    "create_research_agent",
    "create_writer_agent",
    "create_reviewer_agent",
    "create_supervisor_agent",
    # Graph Builder
    "GraphBuilder",
    # Tools
    "calculator",
    "get_current_time",
    "weather_info",
    "search_documents",
    "company_info",
    "web_search",
    "text_analyzer",
    "get_basic_tools",
    "get_research_tools",
    "get_writer_tools",
    "get_all_tools",
    "get_tools_by_category",
    "get_tools_by_name",
]
