"""
Example 2: Conversational Memory with LangChain

This example demonstrates:
1. ConversationBufferMemory - stores all conversation history
2. ConversationEntityMemory - tracks and extracts entities
3. ConversationBufferWindowMemory - keeps recent N messages
4. ConversationSummaryMemory - summarizes old conversations
5. Memory integration with chains

What is Conversational Memory?
- Allows chatbots to remember previous interactions
- Essential for multi-turn conversations
- Different types optimize for different use cases

Memory Types Comparison:
╔═══════════════════╦══════════════════╦════════════╦═══════════════╗
║ Memory Type       ║ Use Case         ║ Pros       ║ Cons          ║
╠═══════════════════╬══════════════════╬════════════╬═══════════════╣
║ Buffer            ║ Short chats      ║ Complete   ║ Grows forever ║
║ BufferWindow      ║ Fixed context    ║ Fixed size ║ Forgets old   ║
║ Summary           ║ Long chats       ║ Compact    ║ Loses details ║
║ Entity            ║ Track people/etc ║ Extracts   ║ Complex       ║
╚═══════════════════╩══════════════════╩════════════╩═══════════════╝

Prerequisites:
- Set OPENAI_API_KEY in .env file

Usage:
    python examples/example_2_memory.py              # Interactive mode
    python examples/example_2_memory.py --demo       # Demo mode
    python examples/example_2_memory.py --memory buffer
"""

import os
import sys
from dotenv import load_dotenv
import warnings

# Suppress all LangChain deprecation warnings for training purposes
warnings.filterwarnings("ignore", message=".*LangChain.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.chains import LLMChain

# Import centralized managers
from core.memory_manager import MemoryManager, MemoryType

# Load environment variables
load_dotenv()


def demo_buffer_memory():
    """
    Demonstrate ConversationBufferMemory.
    
    This memory type stores ALL messages in the conversation.
    - Simple and accurate
    - Can grow large over time
    - Best for short conversations
    """
    print("=" * 70)
    print("Demo 1: ConversationBufferMemory")
    print("=" * 70)
    print()
    print("📋 Stores ALL conversation history")
    print("✅ Pros: Complete context, accurate")
    print("⚠️  Cons: Can exceed token limits in long conversations")
    print()
    
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    
    # Create memory using MemoryManager
    memory = MemoryManager.create_memory(
        memory_type=MemoryType.BUFFER,
        memory_key="chat_history",
        return_messages=True
    )
    
    # Create prompt with memory placeholder
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. Remember our conversation."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    
    # Create chain with memory
    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        memory=memory,
        verbose=False
    )
    
    # Simulate conversation
    conversations = [
        "Hi! My name is Alex and I'm a software engineer.",
        "What's my name?",
        "What's my profession?",
        "Tell me a joke about my profession"
    ]
    
    for i, user_input in enumerate(conversations, 1):
        print(f"Turn {i}:")
        print(f"👤 You: {user_input}")
        
        response = chain.predict(input=user_input)
        print(f"🤖 Assistant: {response}")
        print()
    
    # Show memory contents
    print("─" * 70)
    print("📝 Memory Contents:")
    print(memory.chat_memory.messages)
    print()
    
    print("💡 Key Concept:")
    print("   BufferMemory stores everything - perfect recall but can grow large.")
    print()


def demo_window_memory():
    """
    Demonstrate ConversationBufferWindowMemory.
    
    This memory type keeps only the last N messages.
    - Fixed size
    - Good for long conversations
    - Forgets older context
    """
    print("=" * 70)
    print("Demo 2: ConversationBufferWindowMemory")
    print("=" * 70)
    print()
    print("📋 Keeps only the last K messages (window)")
    print("✅ Pros: Fixed token usage, good for long chats")
    print("⚠️  Cons: Forgets older conversation parts")
    print()
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    
    # Create memory with window size = 2 (keeps last 2 exchanges) using MemoryManager
    memory = MemoryManager.create_memory(
        memory_type=MemoryType.WINDOW,
        k=2,  # Keep last 2 conversation turns
        memory_key="chat_history",
        return_messages=True
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    
    chain = LLMChain(llm=llm, prompt=prompt, memory=memory, verbose=False)
    
    # Simulate conversation
    conversations = [
        "My name is Sarah.",
        "My favorite color is blue.",
        "My favorite food is pizza.",
        "What's my name?",  # Should not remember (outside window)
        "What's my favorite food?"  # Should remember (in window)
    ]
    
    for i, user_input in enumerate(conversations, 1):
        print(f"Turn {i}:")
        print(f"👤 You: {user_input}")
        
        response = chain.predict(input=user_input)
        print(f"🤖 Assistant: {response}")
        print()
    
    print("💡 Key Concept:")
    print("   WindowMemory keeps only recent context - efficient but forgetful.")
    print()


def demo_entity_memory():
    """
    Demonstrate ConversationEntityMemory.
    
    This memory type extracts and tracks entities (people, places, things).
    - Intelligent extraction
    - Tracks important entities
    - More complex but powerful
    """
    print("=" * 70)
    print("Demo 3: ConversationEntityMemory")
    print("=" * 70)
    print()
    print("📋 Extracts and tracks entities (names, places, etc.)")
    print("✅ Pros: Remembers important facts, intelligent")
    print("⚠️  Cons: More complex, requires LLM calls")
    print()
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    
    # Create entity memory using MemoryManager
    memory = MemoryManager.create_memory(
        memory_type=MemoryType.ENTITY,
        llm=llm
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant with good memory about entities."),
        ("human", "Entities: {entities}\nHistory: {history}\n\nUser: {input}")
    ])
    
    chain = LLMChain(llm=llm, prompt=prompt, memory=memory, verbose=False)
    
    # Simulate conversation
    conversations = [
        "I work at TechCorp in San Francisco.",
        "My colleague John is from New York.",
        "Where do I work?",
        "Where is John from?"
    ]
    
    for i, user_input in enumerate(conversations, 1):
        print(f"Turn {i}:")
        print(f"👤 You: {user_input}")
        
        response = chain.predict(input=user_input)
        print(f"🤖 Assistant: {response}")
        print()
    
    # Show extracted entities
    print("─" * 70)
    print("📝 Extracted Entities:")
    if hasattr(memory, 'entity_store') and hasattr(memory.entity_store, 'store'):
        for entity, info in memory.entity_store.store.items():
            print(f"   • {entity}: {info}")
    print()
    
    print("💡 Key Concept:")
    print("   EntityMemory extracts and tracks important entities automatically.")
    print()


def demo_summary_memory():
    """
    Demonstrate ConversationSummaryMemory.
    
    This memory type summarizes old conversations.
    - Compact representation
    - Good for very long chats
    - May lose some details
    """
    print("=" * 70)
    print("Demo 4: ConversationSummaryMemory")
    print("=" * 70)
    print()
    print("📋 Summarizes old conversation to save tokens")
    print("✅ Pros: Very efficient for long conversations")
    print("⚠️  Cons: May lose specific details in summary")
    print()
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    
    # Create summary memory using MemoryManager
    memory = MemoryManager.create_memory(
        memory_type=MemoryType.SUMMARY,
        llm=llm,
        memory_key="chat_history",
        return_messages=True
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    
    chain = LLMChain(llm=llm, prompt=prompt, memory=memory, verbose=False)
    
    # Simulate a longer conversation
    conversations = [
        "I'm planning a trip to Japan next month.",
        "I want to visit Tokyo, Kyoto, and Osaka.",
        "I'm interested in temples and traditional food.",
        "What should I pack for the trip?",
        "How many days should I spend in each city?"
    ]
    
    for i, user_input in enumerate(conversations, 1):
        print(f"Turn {i}:")
        print(f"👤 You: {user_input}")
        
        response = chain.predict(input=user_input)
        print(f"🤖 Assistant: {response}")
        print()
    
    # Show summary
    print("─" * 70)
    print("📝 Conversation Summary:")
    print(memory.buffer)
    print()
    
    print("💡 Key Concept:")
    print("   SummaryMemory condenses long conversations into compact summaries.")
    print()


def interactive_chat_with_memory():
    """
    Interactive mode: Chat with memory-enabled assistant.
    """
    print("=" * 70)
    print("Interactive Chat with Memory")
    print("=" * 70)
    print()
    
    # Ask user to choose memory type
    print("Choose memory type:")
    print("  1. Buffer (stores everything)")
    print("  2. Window (keeps last N messages)")
    print("  3. Entity (tracks entities)")
    print("  4. Summary (summarizes old messages)")
    print()
    
    choice = input("Enter choice (1-4) [default: 1]: ").strip() or "1"
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    
    # Create appropriate memory based on choice using MemoryManager
    if choice == "1":
        memory = MemoryManager.create_memory(
            memory_type=MemoryType.BUFFER,
            memory_key="chat_history",
            return_messages=True
        )
        memory_name = "Buffer Memory"
    elif choice == "2":
        memory = MemoryManager.create_memory(
            memory_type=MemoryType.WINDOW,
            k=3,
            memory_key="chat_history",
            return_messages=True
        )
        memory_name = "Window Memory (k=3)"
    elif choice == "3":
        memory = MemoryManager.create_memory(
            memory_type=MemoryType.ENTITY,
            llm=llm
        )
        memory_name = "Entity Memory"
    else:
        memory = MemoryManager.create_memory(
            memory_type=MemoryType.SUMMARY,
            llm=llm,
            memory_key="chat_history",
            return_messages=True
        )
        memory_name = "Summary Memory"
    
    print(f"\n✅ Using: {memory_name}")
    print()
    
    # Create prompt
    if choice == "3":  # Entity memory uses different format
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant with good memory about entities."),
            ("human", "Entities: {entities}\nHistory: {history}\n\nUser: {input}")
        ])
    else:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant. Remember our conversation."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
    
    chain = LLMChain(llm=llm, prompt=prompt, memory=memory, verbose=False)
    
    print("💬 Start chatting! The assistant will remember context.")
    print()
    print("Commands:")
    print("  • 'memory' - View memory contents")
    print("  • 'clear' - Clear memory")
    print("  • 'quit' - Exit")
    print()
    
    while True:
        print("─" * 70)
        user_input = input("You: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() == 'quit':
            print("👋 Goodbye!")
            break
        
        if user_input.lower() == 'memory':
            print("\n📝 Current Memory:")
            if hasattr(memory, 'buffer'):
                print(memory.buffer)
            elif hasattr(memory, 'chat_memory'):
                for msg in memory.chat_memory.messages:
                    print(f"  {msg.type}: {msg.content[:100]}...")
            print()
            continue
        
        if user_input.lower() == 'clear':
            memory.clear()
            print("🗑️  Memory cleared!\n")
            continue
        
        # Get response
        print()
        try:
            response = chain.predict(input=user_input)
            print(f"Assistant: {response}")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print()


def compare_memory_types():
    """
    Run the same conversation with different memory types to compare.
    """
    print("=" * 70)
    print("Memory Type Comparison")
    print("=" * 70)
    print()
    print("Running the same conversation with different memory types...\n")
    
    conversations = [
        "My name is Alice.",
        "I live in Seattle.",
        "I work as a data scientist.",
        "I have a dog named Max.",
        "What's my name and where do I live?",
        "Tell me about my job and pet."
    ]
    
    # Create memory types using MemoryManager
    memory_types = [
        ("Buffer", MemoryManager.create_memory(MemoryType.BUFFER, memory_key="chat_history", return_messages=True)),
        ("Window (k=2)", MemoryManager.create_memory(MemoryType.WINDOW, k=2, memory_key="chat_history", return_messages=True))
    ]
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    for memory_name, memory in memory_types:
        print("─" * 70)
        print(f"Testing: {memory_name}")
        print("─" * 70)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        
        chain = LLMChain(llm=llm, prompt=prompt, memory=memory, verbose=False)
        
        for msg in conversations:
            response = chain.predict(input=msg)
            
            # Only print the recall questions
            if "?" in msg:
                print(f"\n👤 {msg}")
                print(f"🤖 {response}")
        
        print()
    
    print("💡 Observations:")
    print("   • Buffer Memory: Remembers everything")
    print("   • Window Memory: Only remembers recent context")
    print()


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="LangChain Memory Examples")
    parser.add_argument("--demo", action="store_true", help="Run demo mode")
    parser.add_argument("--memory", choices=["buffer", "window", "entity", "summary"],
                        help="Run specific memory demo")
    parser.add_argument("--compare", action="store_true", help="Compare memory types")
    args = parser.parse_args()
    
    print("\n🚀 Example 2: Conversational Memory\n")
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found in environment variables")
        print("   Please add it to your .env file")
        sys.exit(1)
    
    if args.compare:
        compare_memory_types()
    elif args.memory:
        if args.memory == "buffer":
            demo_buffer_memory()
        elif args.memory == "window":
            demo_window_memory()
        elif args.memory == "entity":
            demo_entity_memory()
        elif args.memory == "summary":
            demo_summary_memory()
    elif args.demo:
        # Run all demos
        demo_buffer_memory()
        input("Press Enter to continue...\n")
        
        demo_window_memory()
        input("Press Enter to continue...\n")
        
        demo_entity_memory()
        input("Press Enter to continue...\n")
        
        demo_summary_memory()
        
        print("=" * 70)
        print("✅ All demos completed!")
        print()
        print("💡 Key Takeaways:")
        print("   • Buffer: Complete memory, can grow large")
        print("   • Window: Fixed size, forgets old context")
        print("   • Entity: Tracks important entities intelligently")
        print("   • Summary: Compact, good for long conversations")
        print()
        print("📚 Next: Run example_3_tools.py to add tool integration!")
    else:
        # Interactive mode
        interactive_chat_with_memory()


if __name__ == "__main__":
    main()
