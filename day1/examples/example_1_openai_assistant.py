"""
Example 1: OpenAI Assistant API with Streaming

This example demonstrates:
1. How to use OpenAI's Assistants API
2. Connecting to a pre-configured assistant
3. Streaming responses in real-time
4. Managing conversation threads

Assistants vs Chat Completion:
- Assistants: Managed by OpenAI, stateful, can use tools (Code Interpreter, Retrieval)
- Chat Completion: Stateless, more control, lower latency, no built-in tools

When to use Assistants:
- Need built-in code execution
- Want OpenAI to manage conversation state
- Using file search/knowledge retrieval features

Prerequisites:
1. Set OPENAI_API_KEY in .env file
2. (Optional) Create an assistant at https://platform.openai.com/assistants
   and set OPENAI_ASSISTANT_ID in .env
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
import time

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def create_assistant():
    """
    Create a new OpenAI Assistant programmatically.
    
    This function creates an assistant with custom instructions.
    In production, you might want to create assistants through the
    OpenAI dashboard for easier management and updates.
    
    Returns:
        str: The assistant ID
    """
    print("🤖 Creating a new AI Assistant...")
    
    assistant = client.beta.assistants.create(
        name="Business Helper",
        instructions="""You are a helpful business assistant. 
        You provide clear, concise answers to business-related questions.
        Be professional but friendly. If you don't know something, say so.
        Always structure your responses for clarity.""",
        model="gpt-4o-mini",  # Using mini for cost efficiency
        tools=[]  # No tools for this simple example
    )
    
    print(f"✅ Assistant created with ID: {assistant.id}")
    print(f"💡 Tip: Save this ID in your .env file as OPENAI_ASSISTANT_ID")
    print()
    
    return assistant.id


def get_or_create_assistant():
    """
    Get existing assistant from .env or create a new one.
    
    Returns:
        str: The assistant ID
    """
    assistant_id = os.getenv("OPENAI_ASSISTANT_ID")
    
    if assistant_id:
        print(f"🔗 Using existing assistant: {assistant_id}")
        try:
            # Verify the assistant exists
            assistant = client.beta.assistants.retrieve(assistant_id)
            print(f"✅ Assistant verified: {assistant.name}")
            print()
            return assistant_id
        except Exception as e:
            print(f"❌ Error: Could not retrieve assistant {assistant_id}")
            print(f"   {e}")
            print("   Creating a new assistant instead...")
            print()
            return create_assistant()
    else:
        print("ℹ️  No OPENAI_ASSISTANT_ID found in .env")
        return create_assistant()


def chat_with_assistant_streaming(assistant_id: str, user_message: str, thread_id: str = None):
    """
    Send a message to the assistant and stream the response.
    
    This function demonstrates:
    - Creating or using an existing conversation thread
    - Sending messages to the assistant
    - Streaming responses token by token
    - Handling the assistant's run lifecycle
    
    Args:
        assistant_id: The OpenAI assistant ID
        user_message: The user's question or message
        thread_id: Optional existing thread ID for conversation continuity
    
    Returns:
        tuple: (response_text, thread_id) for continued conversation
    """
    # Create a new thread or use existing one
    if thread_id is None:
        thread = client.beta.threads.create()
        thread_id = thread.id
        print(f"🆕 Created new conversation thread: {thread_id}")
    else:
        print(f"🔄 Continuing thread: {thread_id}")
    
    # Add user message to the thread
    client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=user_message
    )
    
    print(f"\n👤 You: {user_message}")
    print(f"🤖 Assistant: ", end="", flush=True)
    
    # Run the assistant with streaming
    full_response = ""
    
    with client.beta.threads.runs.stream(
        thread_id=thread_id,
        assistant_id=assistant_id,
    ) as stream:
        for event in stream:
            # Handle text delta events (streaming response chunks)
            if event.event == 'thread.message.delta':
                for content in event.data.delta.content:
                    if content.type == 'text':
                        text_delta = content.text.value
                        print(text_delta, end="", flush=True)
                        full_response += text_delta
    
    print("\n")  # New line after streaming completes
    
    return full_response, thread_id


def interactive_chat():
    """
    Run an interactive chat session with the assistant.
    
    This demonstrates a complete conversation flow:
    - Get or create an assistant
    - Maintain conversation context across multiple messages
    - Handle streaming responses
    - Allow user to exit gracefully
    """
    print("=" * 70)
    print("OpenAI Assistant API - Interactive Chat Demo")
    print("=" * 70)
    print()
    
    # Get or create assistant
    assistant_id = get_or_create_assistant()
    
    print("=" * 70)
    print("Chat Session Started")
    print("=" * 70)
    print("💬 Type your messages below (type 'exit' or 'quit' to end)")
    print()
    
    thread_id = None  # Will be created on first message
    
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            if not user_input:
                print("⚠️  Please enter a message\n")
                continue
            
            # Check for exit commands
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("\n👋 Thanks for chatting! Goodbye!")
                break
            
            # Chat with assistant (streaming)
            response, thread_id = chat_with_assistant_streaming(
                assistant_id=assistant_id,
                user_message=user_input,
                thread_id=thread_id
            )
            
        except KeyboardInterrupt:
            print("\n\n👋 Chat interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
            continue


def demo_single_question():
    """
    Demonstrate a single question-answer interaction.
    
    This is useful for showing the basic flow without
    requiring user interaction.
    """
    print("=" * 70)
    print("OpenAI Assistant API - Single Question Demo")
    print("=" * 70)
    print()
    
    # Get or create assistant
    assistant_id = get_or_create_assistant()
    
    # Example question
    question = "What are the key benefits of using AI in business operations?"
    
    print("=" * 70)
    print("Demo Question")
    print("=" * 70)
    
    # Get streaming response
    response, thread_id = chat_with_assistant_streaming(
        assistant_id=assistant_id,
        user_message=question
    )
    
    print("=" * 70)
    print(f"Thread ID: {thread_id}")
    print("You can use this thread ID to continue the conversation")
    print("=" * 70)


if __name__ == "__main__":
    """
    Main execution block with multiple demo modes.
    """
    import sys
    
    print("\n🚀 Example 1: OpenAI Assistant API\n")
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found in environment variables")
        print("   Please add it to your .env file")
        sys.exit(1)
    
    # Choose mode based on command line argument
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        # Single question demo mode
        demo_single_question()
    else:
        # Interactive chat mode (default)
        interactive_chat()
