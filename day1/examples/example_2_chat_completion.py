"""
Example 2: Chat Completion API with Streaming

This example demonstrates:
1. Using OpenAI's Chat Completion API (more flexible than Assistants)
2. Setting system prompts to control AI behavior
3. Maintaining conversation history manually
4. Streaming responses token by token
5. Comparing with Assistants API approach

Chat Completion vs Assistants:
╔════════════════════╦═══════════════════╦══════════════════════╗
║ Feature            ║ Chat Completion   ║ Assistants API       ║
╠════════════════════╬═══════════════════╬══════════════════════╣
║ State Management   ║ Manual (you code) ║ Automatic (OpenAI)   ║
║ Cost               ║ Lower             ║ Higher               ║
║ Latency            ║ Lower             ║ Higher               ║
║ Control            ║ Full              ║ Limited              ║
║ Built-in Tools     ║ No                ║ Yes                  ║
║ Best For           ║ Custom RAG, APIs  ║ Quick prototypes     ║
╚════════════════════╩═══════════════════╩══════════════════════╝

Prerequisites:
- Set OPENAI_API_KEY in .env file
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Dict

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class ChatSession:
    """
    Manages a chat session with conversation history.
    
    This class demonstrates:
    - Manual conversation state management
    - System prompt configuration
    - Message history maintenance
    - Token counting awareness
    
    Attributes:
        system_prompt (str): The system message that defines AI behavior
        messages (List[Dict]): Conversation history
        model (str): OpenAI model to use
        temperature (float): Creativity level (0-2, lower = more deterministic)
    """
    
    def __init__(
        self,
        system_prompt: str,
        model: str = "gpt-4o-mini",
        temperature: float = 1.5
    ):
        """
        Initialize a new chat session.
        
        Args:
            system_prompt: Instructions that define the AI's behavior and role
            model: OpenAI model name (gpt-4o-mini is cost-effective)
            temperature: Controls randomness (0=deterministic, 2=very creative)
        """
        self.model = model
        self.temperature = temperature
        self.messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]
        
        print(f"🤖 Chat session initialized with {model}")
        print(f"📋 System Prompt: {system_prompt[:100]}...")
        print()
    
    def add_user_message(self, message: str):
        """
        Add a user message to the conversation history.
        
        Args:
            message: The user's input text
        """
        self.messages.append({"role": "user", "content": message})
    
    def add_assistant_message(self, message: str):
        """
        Add an assistant response to the conversation history.
        
        Args:
            message: The AI's response text
        """
        self.messages.append({"role": "assistant", "content": message})
    
    def get_streaming_response(self, user_message: str) -> str:
        """
        Send a message and get a streaming response.
        
        This method:
        1. Adds user message to history
        2. Calls OpenAI API with streaming enabled
        3. Prints response tokens as they arrive
        4. Adds complete response to history
        5. Returns full response text
        
        Args:
            user_message: The user's question or statement
        
        Returns:
            str: The complete AI response
        """
        # Add user message to history
        self.add_user_message(user_message)
        
        print(f"👤 You: {user_message}")
        print(f"🤖 Assistant: ", end="", flush=True)
        
        # Call OpenAI API with streaming
        stream = client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            temperature=self.temperature,
            stream=True  # Enable streaming
        )
        
        # Collect and display streaming response
        full_response = ""
        
        for chunk in stream:
            # Extract content from stream chunk
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                full_response += content
        
        print("\n")  # New line after response
        
        # Add assistant response to history
        self.add_assistant_message(full_response)
        
        return full_response
    
    def get_response_non_streaming(self, user_message: str) -> str:
        """
        Send a message and get a non-streaming response.
        
        Use this when you need the complete response at once
        (e.g., for processing, not displaying to users).
        
        Args:
            user_message: The user's question or statement
        
        Returns:
            str: The complete AI response
        """
        self.add_user_message(user_message)
        
        response = client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            temperature=self.temperature,
            stream=False
        )
        
        assistant_message = response.choices[0].message.content
        self.add_assistant_message(assistant_message)
        
        return assistant_message
    
    def get_message_count(self) -> int:
        """
        Get the number of messages in the conversation.
        
        Returns:
            int: Total message count (including system message)
        """
        return len(self.messages)
    
    def clear_history(self, keep_system_prompt: bool = True):
        """
        Clear conversation history.
        
        Useful when starting a new topic or managing token limits.
        
        Args:
            keep_system_prompt: If True, keeps the system message
        """
        if keep_system_prompt:
            self.messages = [self.messages[0]]  # Keep only system prompt
        else:
            self.messages = []
        
        print("🗑️  Conversation history cleared")


def demo_with_different_system_prompts():
    """
    Demonstrate how system prompts change AI behavior.
    
    This shows the power of system prompts in controlling:
    - Tone and style
    - Response format
    - Domain expertise
    - Personality
    """
    print("=" * 70)
    print("Demo: System Prompt Comparison")
    print("=" * 70)
    print()
    
    question = "What is machine learning?"
    
    # Professional system prompt
    print("🎯 Scenario 1: Professional Business Consultant")
    print("-" * 70)
    chat1 = ChatSession(
        system_prompt="""You are a professional business consultant. 
        Provide clear, executive-level explanations suitable for C-level executives.
        Use business terminology and focus on ROI and business value."""
    )
    chat1.get_streaming_response(question)
    
    # Technical system prompt
    print("🎯 Scenario 2: Technical Expert for Developers")
    print("-" * 70)
    chat2 = ChatSession(
        system_prompt="""You are a senior machine learning engineer.
        Provide technical explanations with examples and code when relevant.
        Assume the audience has programming knowledge."""
    )
    chat2.get_streaming_response(question)
    
    # Simple system prompt
    print("🎯 Scenario 3: Teacher for Beginners")
    print("-" * 70)
    chat3 = ChatSession(
        system_prompt="""You are a friendly teacher explaining concepts to beginners.
        Use simple language, analogies, and real-world examples.
        Avoid jargon and technical terms."""
    )
    chat3.get_streaming_response(question)


def interactive_chat():
    """
    Run an interactive chat session with customizable system prompt.
    
    This demonstrates:
    - User-defined system prompts
    - Conversation continuity
    - Manual history management
    - Graceful exit handling
    """
    print("=" * 70)
    print("Chat Completion API - Interactive Chat Demo")
    print("=" * 70)
    print()
    
    # Get custom system prompt or use default
    print("📝 Enter a system prompt (or press Enter for default):")
    print("   Examples:")
    print("   - 'You are a helpful Python programming tutor'")
    print("   - 'You are a marketing expert specializing in social media'")
    print("   - 'You are a financial advisor providing investment guidance'")
    print()
    
    custom_prompt = input("System Prompt: ").strip()
    
    if not custom_prompt:
        custom_prompt = """You are a helpful AI assistant. Provide clear, 
        accurate, and thoughtful responses to user questions."""
        print(f"📋 Using default system prompt")
    
    print()
    
    # Initialize chat session
    chat = ChatSession(system_prompt=custom_prompt)
    
    print("=" * 70)
    print("Chat Session Started")
    print("=" * 70)
    print("💬 Type your messages below")
    print("   Commands: 'exit', 'quit' - end chat")
    print("            'clear' - clear conversation history")
    print("            'count' - show message count")
    print()
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                print("⚠️  Please enter a message\n")
                continue
            
            # Handle commands
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("\n👋 Thanks for chatting! Goodbye!")
                break
            
            if user_input.lower() == 'clear':
                chat.clear_history()
                print()
                continue
            
            if user_input.lower() == 'count':
                count = chat.get_message_count()
                print(f"📊 Total messages in history: {count}")
                print(f"   (1 system + {count-1} conversation messages)\n")
                continue
            
            # Get streaming response
            chat.get_streaming_response(user_input)
            
        except KeyboardInterrupt:
            print("\n\n👋 Chat interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
            continue


def demo_conversation_memory():
    """
    Demonstrate how conversation history provides context.
    
    This shows that the AI "remembers" previous messages
    because we maintain the conversation history.
    """
    print("=" * 70)
    print("Demo: Conversation Memory")
    print("=" * 70)
    print()
    
    chat = ChatSession(
        system_prompt="You are a helpful assistant with good memory."
    )
    
    # First message - establish context
    print("🔹 Message 1: Setting context")
    print("-" * 70)
    chat.get_streaming_response("My name is Alex and I'm learning about AI.")
    
    # Second message - reference previous context
    print("🔹 Message 2: Referencing previous context")
    print("-" * 70)
    chat.get_streaming_response("What was my name again?")
    
    # Third message - build on conversation
    print("🔹 Message 3: Building on conversation")
    print("-" * 70)
    chat.get_streaming_response("What topic did I say I was learning about?")
    
    print("=" * 70)
    print(f"📊 Total messages exchanged: {chat.get_message_count() - 1}")
    print("   (excluding system prompt)")
    print("=" * 70)


if __name__ == "__main__":
    """
    Main execution with multiple demo modes.
    """
    import sys
    
    print("\n🚀 Example 2: Chat Completion API\n")
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found in environment variables")
        print("   Please add it to your .env file")
        sys.exit(1)
    
    # Choose mode based on command line argument
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        
        if mode == "--prompts":
            demo_with_different_system_prompts()
        elif mode == "--memory":
            demo_conversation_memory()
        elif mode == "--interactive":
            interactive_chat()
        else:
            print(f"❌ Unknown mode: {mode}")
            print("\nAvailable modes:")
            print("  --prompts      : Compare different system prompts")
            print("  --memory       : Demonstrate conversation memory")
            print("  --interactive  : Interactive chat (default)")
            sys.exit(1)
    else:
        # Default to interactive mode
        interactive_chat()
    
    print("\n✅ Example 2 completed!")
    print("\n💡 Key Takeaways:")
    print("   • Chat Completion gives you full control over conversation")
    print("   • System prompts are powerful for behavior customization")
    print("   • You manage conversation history manually")
    print("   • Streaming improves user experience")
    print("   • Lower cost and latency than Assistants API")
    print("\n🔄 Comparison with Example 1:")
    print("   • Example 1 (Assistants): OpenAI manages state, higher cost")
    print("   • Example 2 (Chat): You manage state, more control, lower cost")
    print("\n📚 Next: Run build_faiss_store.py then example_3_rag_faiss.py for RAG!")
