"""
Example 4: RAG with Milvus Vector Store (Using ChatService)

This example demonstrates using ChatService with Milvus - a production-grade
distributed vector database.

Key Differences from FAISS:
- FAISS: Local file-based, great for prototypes
- Milvus: Server-based, scalable, production-ready

The ChatService abstracts these differences - same API for both!

Prerequisites:
- Milvus server running (docker run -d -p 19530:19530 milvusdb/milvus:latest)
- Run build_milvus_store.py to create the collection
- Set OPENAI_API_KEY in .env file

Usage:
    python examples/example_4_rag_milvus.py          # Interactive mode
    python examples/example_4_rag_milvus.py --demo   # Demo mode
"""

import sys
import os
import argparse
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.chat_service import ChatService, VectorStoreType

# Load environment variables
load_dotenv()


def run_demo():
    """Run demo mode with sample questions"""
    print("=" * 70)
    print("Example 4: RAG with Milvus (Using ChatService)")
    print("=" * 70)
    print()
    
    # Get Milvus config from env
    milvus_uri = os.getenv("MILVUS_URI", "tcp://localhost:19530")
    milvus_token = os.getenv("MILVUS_TOKEN", None)
    milvus_collection = os.getenv("MILVUS_COLLECTION_NAME", "training_demo")
    
    # Initialize service
    print(f"🔧 Initializing ChatService with Milvus...")
    print(f"   URI: {milvus_uri}")
    print(f"   Collection: {milvus_collection}")
    
    service = ChatService(
        vector_store_type=VectorStoreType.MILVUS,
        milvus_uri=milvus_uri,
        milvus_token=milvus_token,
        milvus_collection=milvus_collection,
        k=3,
        temperature=0
    )
    
    try:
        service.initialize()
        print("✅ Connected to Milvus successfully!")
        print()
    except Exception as e:
        print(f"❌ Failed to connect to Milvus: {e}")
        print("\nPlease ensure:")
        print("  1. Milvus server is running")
        print("  2. Collection exists (run: python scripts/build_milvus_store.py)")
        print("  3. OPENAI_API_KEY is set in .env file")
        return
    
    # Sample questions
    questions = [
        "What are the company benefits?",
        "What is the leave policy?",
        "How does the API authentication work?"
    ]
    
    for i, question in enumerate(questions, 1):
        print("─" * 70)
        print(f"Question {i}: {question}")
        print("─" * 70)
        print()
        
        # Get answer
        result = service.get_answer(question)
        
        print("💡 Answer:")
        print(result["answer"])
        print()
        
        print(f"📚 Sources (Retrieved {len(result['sources'])} documents):")
        for j, doc in enumerate(result['sources'], 1):
            print(f"  {j}. {doc['source']} (page {doc['page']})")
            print(f"     Preview: {doc['content'][:100]}...")
        print()
    
    print("=" * 70)
    print("✅ Demo completed!")
    print()
    print("Key Takeaway:")
    print("  Notice how the code is identical to Example 3 (FAISS)!")
    print("  ChatService provides a unified interface for both vector stores.")
    print()
    print("Run: python examples/example_4_rag_milvus.py  # for interactive mode")
    print("=" * 70)


def run_interactive():
    """Run interactive mode"""
    print("=" * 70)
    print("Example 4: RAG with Milvus - Interactive Mode")
    print("=" * 70)
    print()
    
    # Get Milvus config
    milvus_uri = os.getenv("MILVUS_URI", "tcp://localhost:19530")
    milvus_token = os.getenv("MILVUS_TOKEN", None)
    milvus_collection = os.getenv("MILVUS_COLLECTION_NAME", "training_demo")

    # Initialize service
    print("🔧 Initializing ChatService...")
    service = ChatService(
        vector_store_type=VectorStoreType.MILVUS,
        milvus_uri=milvus_uri,
        milvus_token=milvus_token,
        milvus_collection=milvus_collection,
    )
    
    try:
        service.initialize()
        print("✅ Connected to Milvus!")
        print()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTroubleshooting:")
        print("  1. Start Milvus: docker run -d -p 19530:19530 milvusdb/milvus:latest")
        print("  2. Build collection: python scripts/build_milvus_store.py")
        print("  3. Set OPENAI_API_KEY in .env")
        return
    
    # Show config
    config = service.get_config()
    print("📋 Current Configuration:")
    print(f"  • Vector Store: {config['vector_store_type']}")
    print(f"  • LLM Model: {config['llm_model']}")
    print(f"  • Temperature: {config['temperature']}")
    print(f"  • K (documents): {config['k']}")
    print()
    
    print("💬 Ask questions about the documents!")
    print("Commands:")
    print("  • 'quit' or 'exit' - Exit the program")
    print("  • 'stream' - Toggle streaming mode")
    print("  • 'config' - View configuration")
    print("  • 'sources' - Toggle source display")
    print("  • 'health' - Check service health")
    print()
    
    streaming = True
    show_sources = True
    
    while True:
        print("─" * 70)
        user_input = input("You: ").strip()
        
        if not user_input:
            continue
        
        # Handle commands
        if user_input.lower() in ['quit', 'exit']:
            print("👋 Goodbye!")
            break
        
        elif user_input.lower() == 'stream':
            streaming = not streaming
            print(f"{'✅' if streaming else '❌'} Streaming mode: {streaming}")
            continue
        
        elif user_input.lower() == 'config':
            config = service.get_config()
            print("\n📋 Configuration:")
            for key, value in config.items():
                print(f"  • {key}: {value}")
            print()
            continue
        
        elif user_input.lower() == 'sources':
            show_sources = not show_sources
            print(f"{'✅' if show_sources else '❌'} Show sources: {show_sources}")
            continue
        
        elif user_input.lower() == 'health':
            health = service.health_check()
            print(f"\n🏥 Health Status: {health['status']}")
            print(f"   Message: {health['message']}")
            print()
            continue
        
        # Process question
        print()
        print("Assistant: ", end="", flush=True)
        
        try:
            if streaming:
                # Stream the answer
                full_answer = ""
                for chunk in service.get_answer_stream(user_input):
                    print(chunk, end="", flush=True)
                    full_answer += chunk
                print("\n")
                
                # Get sources separately if needed
                if show_sources:
                    docs = service.retrieve_documents(user_input)
                    print(f"📚 Sources ({len(docs)} documents):")
                    for i, doc in enumerate(docs, 1):
                        print(f"  {i}. {doc['source']} (page {doc['page']})")
            else:
                # Get complete answer
                result = service.get_answer(user_input)
                print(result['answer'])
                print()
                
                if show_sources:
                    print(f"📚 Sources ({len(result['sources'])} documents):")
                    for i, doc in enumerate(result['sources'], 1):
                        print(f"  {i}. {doc['source']} (page {doc['page']})")
                        print(f"     {doc['content'][:100]}...")
        
        except Exception as e:
            print(f"\n❌ Error: {e}")
        
        print()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="RAG with Milvus using ChatService")
    parser.add_argument("--demo", action="store_true", help="Run in demo mode")
    args = parser.parse_args()
    
    if args.demo:
        run_demo()
    else:
        run_interactive()


if __name__ == "__main__":
    main()
