"""
Memory Manager - Utilities for Managing Conversational Memory

This module provides utilities for working with different types of
conversational memory in LangChain applications.

Memory Types Supported:
- ConversationBufferMemory: Stores all messages
- ConversationBufferWindowMemory: Keeps last N messages
- ConversationEntityMemory: Tracks entities
- ConversationSummaryMemory: Summarizes old conversations
"""

from typing import Optional, List, Dict, Any
from enum import Enum
import warnings

# Suppress LangChain deprecation warnings for training purposes
warnings.filterwarnings("ignore", message=".*LangChain.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_classic.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationEntityMemory,
    ConversationSummaryMemory
)
from langchain_openai import ChatOpenAI


class MemoryType(str, Enum):
    """Enumeration of supported memory types."""
    BUFFER = "buffer"
    WINDOW = "window"
    ENTITY = "entity"
    SUMMARY = "summary"


class MemoryManager:
    """
    Manages different types of conversational memory.
    
    Usage:
        manager = MemoryManager()
        memory = manager.create_memory(MemoryType.BUFFER)
        
        # Use with a chain
        chain = LLMChain(llm=llm, prompt=prompt, memory=memory)
    """
    
    @staticmethod
    def create_memory(
        memory_type: MemoryType = MemoryType.BUFFER,
        llm: Optional[ChatOpenAI] = None,
        **kwargs
    ):
        """
        Create a memory instance of the specified type.
        
        Args:
            memory_type: Type of memory to create
            llm: LLM instance (required for entity and summary memory)
            **kwargs: Additional arguments for memory configuration
            
        Returns:
            Memory instance
            
        Examples:
            # Buffer memory
            memory = MemoryManager.create_memory(MemoryType.BUFFER)
            
            # Window memory (keeps last 5 messages)
            memory = MemoryManager.create_memory(
                MemoryType.WINDOW,
                k=5
            )
            
            # Entity memory (requires LLM)
            llm = ChatOpenAI(model="gpt-4o-mini")
            memory = MemoryManager.create_memory(
                MemoryType.ENTITY,
                llm=llm
            )
        """
        memory_key = kwargs.get("memory_key", "chat_history")
        return_messages = kwargs.get("return_messages", True)
        
        if memory_type == MemoryType.BUFFER:
            return ConversationBufferMemory(
                memory_key=memory_key,
                return_messages=return_messages
            )
        
        elif memory_type == MemoryType.WINDOW:
            k = kwargs.get("k", 5)  # Default window size
            return ConversationBufferWindowMemory(
                k=k,
                memory_key=memory_key,
                return_messages=return_messages
            )
        
        elif memory_type == MemoryType.ENTITY:
            if llm is None:
                raise ValueError("LLM is required for EntityMemory")
            
            input_key = kwargs.get("input_key", "input")
            return ConversationEntityMemory(
                llm=llm,
                memory_key=memory_key,
                input_key=input_key
            )
        
        elif memory_type == MemoryType.SUMMARY:
            if llm is None:
                raise ValueError("LLM is required for SummaryMemory")
            
            return ConversationSummaryMemory(
                llm=llm,
                memory_key=memory_key,
                return_messages=return_messages
            )
        
        else:
            raise ValueError(f"Unknown memory type: {memory_type}")
    
    @staticmethod
    def get_memory_info(memory) -> Dict[str, Any]:
        """
        Get information about a memory instance.
        
        Args:
            memory: Memory instance
            
        Returns:
            Dictionary with memory info
        """
        info = {
            "type": memory.__class__.__name__,
            "memory_key": getattr(memory, "memory_key", "unknown")
        }
        
        # Add type-specific info
        if isinstance(memory, ConversationBufferWindowMemory):
            info["window_size"] = memory.k
        
        if hasattr(memory, "buffer"):
            info["has_buffer"] = True
            info["buffer_length"] = len(str(memory.buffer))
        
        if hasattr(memory, "chat_memory"):
            info["message_count"] = len(memory.chat_memory.messages)
        
        return info
    
    @staticmethod
    def clear_memory(memory):
        """
        Clear all contents of a memory instance.
        
        Args:
            memory: Memory instance to clear
        """
        memory.clear()
    
    @staticmethod
    def export_memory(memory) -> Dict[str, Any]:
        """
        Export memory contents for persistence.
        
        Args:
            memory: Memory instance
            
        Returns:
            Dictionary with memory contents
        """
        export_data = {
            "type": memory.__class__.__name__,
            "memory_key": getattr(memory, "memory_key", "unknown")
        }
        
        if hasattr(memory, "chat_memory"):
            export_data["messages"] = [
                {
                    "type": msg.type,
                    "content": msg.content
                }
                for msg in memory.chat_memory.messages
            ]
        
        if hasattr(memory, "buffer"):
            export_data["buffer"] = str(memory.buffer)
        
        if hasattr(memory, "entity_store"):
            export_data["entities"] = dict(memory.entity_store.store)
        
        return export_data


class ConversationMemoryWrapper:
    """
    High-level wrapper for working with conversational memory.
    
    Provides convenience methods for common memory operations.
    
    Usage:
        wrapper = ConversationMemoryWrapper(memory_type=MemoryType.BUFFER)
        wrapper.add_user_message("Hello")
        wrapper.add_ai_message("Hi there!")
        history = wrapper.get_conversation_history()
    """
    
    def __init__(
        self,
        memory_type: MemoryType = MemoryType.BUFFER,
        llm: Optional[ChatOpenAI] = None,
        **kwargs
    ):
        """
        Initialize the memory wrapper.
        
        Args:
            memory_type: Type of memory to use
            llm: LLM instance (for entity/summary memory)
            **kwargs: Additional memory configuration
        """
        self.memory = MemoryManager.create_memory(
            memory_type=memory_type,
            llm=llm,
            **kwargs
        )
        self.memory_type = memory_type
    
    def add_user_message(self, message: str):
        """Add a user message to memory."""
        if hasattr(self.memory, "chat_memory"):
            self.memory.chat_memory.add_user_message(message)
    
    def add_ai_message(self, message: str):
        """Add an AI message to memory."""
        if hasattr(self.memory, "chat_memory"):
            self.memory.chat_memory.add_ai_message(message)
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        Get the conversation history.
        
        Returns:
            List of message dictionaries with 'role' and 'content'
        """
        if not hasattr(self.memory, "chat_memory"):
            return []
        
        return [
            {
                "role": msg.type,
                "content": msg.content
            }
            for msg in self.memory.chat_memory.messages
        ]
    
    def get_summary(self) -> str:
        """
        Get a summary of the conversation.
        
        Returns:
            Summary string
        """
        if hasattr(self.memory, "buffer"):
            return str(self.memory.buffer)
        
        # For other memory types, generate a simple summary
        history = self.get_conversation_history()
        if not history:
            return "No conversation yet"
        
        return f"Conversation with {len(history)} messages"
    
    def get_entities(self) -> Dict[str, str]:
        """
        Get extracted entities (only for EntityMemory).
        
        Returns:
            Dictionary of entities
        """
        if hasattr(self.memory, "entity_store"):
            return dict(self.memory.entity_store.store)
        return {}
    
    def clear(self):
        """Clear the memory."""
        self.memory.clear()
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about the memory."""
        return MemoryManager.get_memory_info(self.memory)
    
    def export(self) -> Dict[str, Any]:
        """Export memory contents."""
        return MemoryManager.export_memory(self.memory)


# Convenience functions

def create_simple_memory(window_size: Optional[int] = None):
    """
    Create a simple memory instance.
    
    Args:
        window_size: If provided, creates window memory; otherwise buffer memory
        
    Returns:
        Memory instance
        
    Example:
        # Buffer memory (unlimited)
        memory = create_simple_memory()
        
        # Window memory (last 5 messages)
        memory = create_simple_memory(window_size=5)
    """
    if window_size:
        return MemoryManager.create_memory(MemoryType.WINDOW, k=window_size)
    else:
        return MemoryManager.create_memory(MemoryType.BUFFER)


def create_smart_memory(llm, use_entities: bool = False):
    """
    Create an intelligent memory instance.
    
    Args:
        llm: LLM instance
        use_entities: If True, creates entity memory; otherwise summary memory
        
    Returns:
        Memory instance
        
    Example:
        llm = ChatOpenAI(model="gpt-4o-mini")
        
        # Entity memory (tracks people, places, etc.)
        memory = create_smart_memory(llm, use_entities=True)
        
        # Summary memory (compacts old messages)
        memory = create_smart_memory(llm, use_entities=False)
    """
    memory_type = MemoryType.ENTITY if use_entities else MemoryType.SUMMARY
    return MemoryManager.create_memory(memory_type, llm=llm)


# Example usage
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    print("=" * 70)
    print("Memory Manager - Demo")
    print("=" * 70)
    print()
    
    # Test buffer memory
    print("1. Buffer Memory")
    wrapper = ConversationMemoryWrapper(memory_type=MemoryType.BUFFER)
    wrapper.add_user_message("Hello, my name is Alice")
    wrapper.add_ai_message("Hi Alice! Nice to meet you.")
    wrapper.add_user_message("What's my name?")
    wrapper.add_ai_message("Your name is Alice.")
    
    print("Conversation history:")
    for msg in wrapper.get_conversation_history():
        print(f"  {msg['role']}: {msg['content']}")
    print()
    
    print("Memory info:")
    print(wrapper.get_info())
    print()
    
    # Test window memory
    print("2. Window Memory (k=2)")
    wrapper = ConversationMemoryWrapper(
        memory_type=MemoryType.WINDOW,
        k=2
    )
    wrapper.add_user_message("Message 1")
    wrapper.add_ai_message("Response 1")
    wrapper.add_user_message("Message 2")
    wrapper.add_ai_message("Response 2")
    wrapper.add_user_message("Message 3")  # Message 1 should be forgotten
    
    history = wrapper.get_conversation_history()
    print(f"History length: {len(history)} (should be 4: last 2 exchanges)")
    print()
    
    # Test entity memory
    print("3. Entity Memory")
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    wrapper = ConversationMemoryWrapper(
        memory_type=MemoryType.ENTITY,
        llm=llm
    )
    
    # Note: Entity memory requires actual chain execution to extract entities
    print("Entity memory created (requires chain execution to populate)")
    print()
    
    print("✅ Memory manager demo completed!")
