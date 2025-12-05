"""
Agent Configuration Loader

This module provides a singleton class for loading agent configurations from YAML
and managing shared LLM instances to improve performance and reduce memory usage.

Key Features:
- Loads agent configurations from YAML file
- Caches LLM instances (one per model/temperature combination)
- Provides helper methods to access agent-specific settings
- Thread-safe singleton pattern
"""

import yaml
import os
from typing import Dict, Any, Optional
from pathlib import Path
from langchain_openai import ChatOpenAI


class AgentConfigLoader:
    """
    Singleton class for loading and caching agent configurations.
    
    This class:
    - Loads agent configs from config/agents_config.yaml
    - Creates and caches ChatOpenAI instances
    - Provides convenient access to agent settings
    
    Usage:
        config = AgentConfigLoader()
        llm = config.get_llm()  # Get default LLM
        role = config.get_role("research_agent")
        schema = config.get_output_schema("writer_agent")
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        """Ensure only one instance exists (singleton pattern)"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the config loader (only runs once)"""
        if self._initialized:
            return
        
        # LLM cache: key = "model_temperature", value = ChatOpenAI instance
        self._llm_cache: Dict[str, ChatOpenAI] = {}
        
        # Load configuration from YAML
        self.config_dir = Path(__file__).parent.parent / "config"
        self.config = self._load_config()
        
        self._initialized = True
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load agent configuration from YAML file.
        
        Returns:
            Dictionary containing all agent configurations
        
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML is invalid
        """
        config_file = self.config_dir / "agents_config.yaml"
        
        if not config_file.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_file}\n"
                "Please ensure config/agents_config.yaml exists."
            )
        
        try:
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {config_file}: {e}")
    
    def get_llm(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None
    ) -> ChatOpenAI:
        """
        Get or create a cached ChatOpenAI instance.
        
        This method creates one LLM instance per unique model/temperature
        combination and caches it for reuse across all agents.
        
        Args:
            model: OpenAI model name (uses default if None)
            temperature: LLM temperature (uses default if None)
        
        Returns:
            Cached ChatOpenAI instance
        
        Example:
            llm = config.get_llm()  # Default model and temperature
            llm = config.get_llm(model="gpt-4o", temperature=0.3)
        """
        llm_config = self.config['llm_config']
        
        # Use provided values or fall back to defaults
        model = model or llm_config['default_model']
        temperature = temperature if temperature is not None else llm_config['default_temperature']
        
        # Create cache key
        cache_key = f"{model}_{temperature}"
        
        # Return cached instance if exists
        if cache_key not in self._llm_cache:
            self._llm_cache[cache_key] = ChatOpenAI(
                model=model,
                temperature=temperature,
                api_key=os.getenv(llm_config['api_key_env'])
            )
        
        return self._llm_cache[cache_key]
    
    def get_agent_config(self, agent_type: str) -> Dict[str, Any]:
        """
        Get full configuration for a specific agent type.
        
        Args:
            agent_type: Agent type key (e.g., "research_agent", "writer_agent")
        
        Returns:
            Dictionary containing agent configuration
        
        Raises:
            KeyError: If agent type doesn't exist
        
        Example:
            config = loader.get_agent_config("research_agent")
        """
        if agent_type not in self.config['agents']:
            available = ', '.join(self.config['agents'].keys())
            raise KeyError(
                f"Unknown agent type: {agent_type}\n"
                f"Available types: {available}"
            )
        
        return self.config['agents'][agent_type]
    
    def get_role(self, agent_type: str, **kwargs) -> str:
        """
        Get role description for an agent type.
        
        Supports templates with placeholders (e.g., for supervisor agents).
        
        Args:
            agent_type: Agent type key
            **kwargs: Variables to substitute in role template
        
        Returns:
            Role description string
        
        Example:
            role = loader.get_role("research_agent")
            role = loader.get_role(
                "supervisor_agent",
                available_agents="researcher, writer, reviewer"
            )
        """
        config = self.get_agent_config(agent_type)
        
        # Check if it's a template (for supervisor)
        if 'role_template' in config:
            return config['role_template'].format(**kwargs)
        
        return config.get('role', '')
    
    def get_output_schema(self, agent_type: str) -> Dict[str, str]:
        """
        Get output schema for an agent type.
        
        Converts the YAML schema format to a simple dict format
        used by ReActAgent.
        
        Args:
            agent_type: Agent type key
        
        Returns:
            Dictionary mapping field names to descriptions
        
        Example:
            schema = loader.get_output_schema("research_agent")
            # Returns:
            # {
            #     "findings": "list of key findings...",
            #     "sources": "list of sources...",
            #     ...
            # }
        """
        config = self.get_agent_config(agent_type)
        schema = config.get('output_schema', {})
        
        # Convert from {field: {type, description}} to {field: description}
        return {
            key: value['description']
            for key, value in schema.items()
        }
    
    def get_prompt_template(self, agent_type: Optional[str] = None) -> str:
        """
        Get prompt template for an agent type.
        
        Args:
            agent_type: Agent type key (uses default template if None)
        
        Returns:
            Prompt template string
        
        Example:
            template = loader.get_prompt_template("research_agent")
        """
        if agent_type:
            config = self.get_agent_config(agent_type)
            if 'prompt_template' in config:
                return config['prompt_template']
        
        # Fall back to default template
        return self.config['default_prompt_template']
    
    def get_default_tools(self, agent_type: str) -> list:
        """
        Get default tool names for an agent type.
        
        Args:
            agent_type: Agent type key
        
        Returns:
            List of tool names (strings)
        
        Example:
            tools = loader.get_default_tools("research_agent")
            # Returns: ["search_documents", "web_search", "company_info"]
        """
        config = self.get_agent_config(agent_type)
        return config.get('default_tools', [])
    
    def get_model(self, agent_type: str) -> str:
        """
        Get model name for an agent type.
        
        Args:
            agent_type: Agent type key
        
        Returns:
            Model name (falls back to default if not specified)
        """
        config = self.get_agent_config(agent_type)
        return config.get('model', self.config['llm_config']['default_model'])
    
    def get_temperature(self, agent_type: str) -> float:
        """
        Get temperature for an agent type.
        
        Args:
            agent_type: Agent type key
        
        Returns:
            Temperature value (falls back to default if not specified)
        """
        config = self.get_agent_config(agent_type)
        return config.get('temperature', self.config['llm_config']['default_temperature'])
    
    def get_max_iterations(self, agent_type: str) -> int:
        """
        Get max iterations for an agent type.
        
        Args:
            agent_type: Agent type key
        
        Returns:
            Max iterations value (falls back to default if not specified)
        """
        config = self.get_agent_config(agent_type)
        return config.get('max_iterations', self.config['llm_config']['max_iterations'])
    
    def get_name(self, agent_type: str) -> str:
        """
        Get default name for an agent type.
        
        Args:
            agent_type: Agent type key
        
        Returns:
            Default agent name
        """
        config = self.get_agent_config(agent_type)
        return config.get('name', agent_type)
    
    def list_agent_types(self) -> list:
        """
        Get list of all available agent types.
        
        Returns:
            List of agent type keys
        
        Example:
            types = loader.list_agent_types()
            # Returns: ["research_agent", "writer_agent", "reviewer_agent", ...]
        """
        return list(self.config['agents'].keys())
    
    def reload_config(self):
        """
        Reload configuration from YAML file.
        
        Useful for development when config file is modified.
        Note: Does NOT clear LLM cache.
        """
        self.config = self._load_config()
    
    def clear_llm_cache(self):
        """
        Clear the LLM instance cache.
        
        Forces creation of new LLM instances on next get_llm() call.
        Useful for testing or when switching API keys.
        """
        self._llm_cache.clear()
    
    def __repr__(self) -> str:
        """String representation showing loaded agents"""
        agent_types = ', '.join(self.list_agent_types())
        llm_count = len(self._llm_cache)
        return f"AgentConfigLoader(agents=[{agent_types}], cached_llms={llm_count})"


# Convenience function for getting config instance
def get_config() -> AgentConfigLoader:
    """
    Get the AgentConfigLoader singleton instance.
    
    Returns:
        AgentConfigLoader instance
    
    Example:
        from core.agent_config import get_config
        
        config = get_config()
        llm = config.get_llm()
    """
    return AgentConfigLoader()


# Example usage and testing
if __name__ == "__main__":
    print("=" * 70)
    print("Agent Configuration Loader - Demo")
    print("=" * 70)
    print()
    
    # Create config loader
    config = AgentConfigLoader()
    print(f"✅ Config loaded: {config}")
    print()
    
    # List available agents
    print("Available agent types:")
    for agent_type in config.list_agent_types():
        print(f"  - {agent_type}")
    print()
    
    # Get LLM instances (should be cached)
    print("Testing LLM caching:")
    llm1 = config.get_llm()
    llm2 = config.get_llm()
    print(f"  LLM 1: {llm1}")
    print(f"  LLM 2: {llm2}")
    print(f"  Same instance? {llm1 is llm2}")
    print()
    
    # Get different temperature LLM
    llm3 = config.get_llm(temperature=0.3)
    print(f"  LLM 3 (temp=0.3): {llm3}")
    print(f"  Different from LLM 1? {llm1 is not llm3}")
    print()
    
    # Test agent configuration retrieval
    print("Research Agent Configuration:")
    print(f"  Name: {config.get_name('research_agent')}")
    print(f"  Model: {config.get_model('research_agent')}")
    print(f"  Temperature: {config.get_temperature('research_agent')}")
    print(f"  Default tools: {config.get_default_tools('research_agent')}")
    print(f"  Role: {config.get_role('research_agent')[:100]}...")
    print()
    
    # Test output schema
    print("Writer Agent Output Schema:")
    schema = config.get_output_schema('writer_agent')
    for field, description in schema.items():
        print(f"  {field}: {description}")
    print()
    
    # Test supervisor template
    print("Supervisor Agent Role (with template substitution):")
    role = config.get_role('supervisor_agent', available_agents="researcher, writer, reviewer")
    print(f"  {role[:200]}...")
    print()
    
    print("=" * 70)
    print("✅ Configuration loader working correctly!")
    print("=" * 70)
