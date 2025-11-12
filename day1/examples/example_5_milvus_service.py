"""
Example 5: Using MilvusService with DocumentRetriever - Interactive Mode

This example demonstrates:
1. Connecting to Milvus using the simplified MilvusService
2. Creating a collection with simple schema (id, text, vector)
3. Inserting documents directly
4. Using DocumentRetriever for searching (recommended approach)
5. Interactive chat based on Milvus data
6. Getting collection statistics

The MilvusService provides a simple, training-friendly interface
to Milvus vector database operations, integrated with DocumentRetriever.

Prerequisites:
- Milvus server running (or Milvus Cloud)
- MILVUS_URI in .env
- OPENAI_API_KEY in .env

Usage:
    python examples/example_5_milvus_service.py          # Interactive mode
    python examples/example_5_milvus_service.py --demo   # Demo mode
"""

import os
import sys
import argparse
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.milvus_service import MilvusService
from core.chat_service import ChatService, VectorStoreType

# Load environment variables
load_dotenv()


def run_demo():
    """
    Run demo mode - demonstrates MilvusService operations.
    """
    print("=" * 70)
    print("Example 5: MilvusService + DocumentRetriever Demo")
    print("=" * 70)
    print()
    
    # Check prerequisites
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found in .env file")
        return
    
    if not os.getenv("MILVUS_URI"):
        print("❌ Error: MILVUS_URI not found in .env file")
        print("   Using default: tcp://localhost:19530")
    
    print("📋 Step 1: Initialize MilvusService (Singleton)")
    print("-" * 70)
    
    # Get singleton instance
    milvus_service = MilvusService()
    print("✅ MilvusService instance created")
    print()
    
    print("📋 Step 2: Connect to Milvus")
    print("-" * 70)
    
    # Connect to Milvus
    milvus_uri = os.getenv("MILVUS_URI", "tcp://localhost:19530")
    collection_name = os.getenv("MILVUS_COLLECTION_NAME", "training_demo")
    
    try:
        milvus_service.connect(
            uri=milvus_uri,
            collection_name=collection_name
        )
        print(f"✅ Connected to Milvus at {milvus_uri}")
        print(f"   Collection: {collection_name}")
        print()
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        print("\n💡 Make sure Milvus server is running:")
        print("   Docker: docker run -d -p 19530:19530 milvusdb/milvus:latest")
        return
    
    print("📋 Step 3: Create/Load Collection")
    print("-" * 70)
    print("Schema:")
    print("   • id: INT64 (auto-generated)")
    print("   • text: VARCHAR (document content)")
    print("   • vector: FLOAT_VECTOR (embeddings)")
    print()
    
    try:
        milvus_service.create_collection(collection_name)
        print("✅ Collection ready (created or loaded existing)")
        print()
    except Exception as e:
        print(f"❌ Failed to create/load collection: {e}")
        return
    
    print("📋 Step 4: Insert Sample Documents")
    print("-" * 70)
    
    # Sample documents about a company
    sample_docs = [
        "The company offers 20 days of vacation per year to all employees.",
        "Health insurance includes comprehensive medical, dental, and vision coverage.",
        "Remote work is allowed up to 3 days per week for all positions.",
        "Annual performance reviews are conducted in December with merit increases.",
        "Employee training budget is $2000 per year for professional development.",
        "The company matches 401k contributions up to 6% of salary.",
        "Flexible working hours are available with core hours from 10am to 3pm.",
        "Parental leave includes 12 weeks of paid time off for new parents.",
    ]
    
    print(f"Inserting {len(sample_docs)} documents...")
    
    try:
        success = milvus_service.insert_documents(
            texts=sample_docs,
            collection_name=collection_name
        )
        
        if success:
            print(f"✅ Successfully inserted {len(sample_docs)} documents")
            print()
        else:
            print("❌ Failed to insert documents")
            return
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    print("📋 Step 5: Use DocumentRetriever for Search")
    print("-" * 70)
    print("Creating DocumentRetriever instance...")
    print()
    
    try:
        # Get DocumentRetriever from MilvusService
        retriever = milvus_service.get_retriever(
            collection_name=collection_name,
            k=3
        )
        print("✅ DocumentRetriever created")
        print()
        
        # Test queries
        queries = [
            "What is the vacation policy?",
            "Tell me about health benefits",
            "Can I work from home?"
        ]
        
        for query in queries:
            print(f"🔍 Query: '{query}'")
            print("-" * 70)
            
            # Use DocumentRetriever to search
            docs = retriever.retrieve(query, k=2)
            
            # Pretty print results
            for i, doc in enumerate(docs, 1):
                content = doc.page_content[:70] + "..." if len(doc.page_content) > 70 else doc.page_content
                print(f"   [{i}] {content}")
            print()
        
    except Exception as e:
        print(f"❌ Retriever failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("📋 Step 6: Get Collection Statistics")
    print("-" * 70)
    
    try:
        stats = milvus_service.get_collection_stats(collection_name)
        
        if stats.get('exists'):
            print(f"Collection: {stats['name']}")
            print(f"Documents: {stats['num_entities']}")
            print(f"Description: {stats['description']}")
        else:
            print("Collection not found or error occurred")
    except Exception as e:
        print(f"❌ Failed to get stats: {e}")
    
    print()
    
    # Summary
    print("=" * 70)
    print("✅ Demo Complete!")
    print("=" * 70)
    print()
    print("What you learned:")
    print("   ✓ How to initialize MilvusService (singleton pattern)")
    print("   ✓ How to connect to Milvus server")
    print("   ✓ Collection is reused if it already exists")
    print("   ✓ How to insert documents with embeddings")
    print("   ✓ How to use DocumentRetriever with Milvus")
    print("   ✓ How to get collection statistics")
    print()
    print("💡 Key Takeaway:")
    print("   Use DocumentRetriever for consistent retrieval across FAISS and Milvus!")
    print()
    print("Next: Run interactive mode without --demo flag")
    print()


def run_interactive():
    """
    Run interactive mode - chat with documents in Milvus.
    """
    print("=" * 70)
    print("Example 5: MilvusService - Interactive Chat Mode")
    print("=" * 70)
    print()
    
    # Check prerequisites
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found in .env file")
        return
    
    # Get Milvus config
    milvus_uri = os.getenv("MILVUS_URI", "tcp://localhost:19530")
    collection_name = os.getenv("MILVUS_COLLECTION_NAME", "training_demo")
    milvus_token = os.getenv("MILVUS_TOKEN", None)
    
    print("🔧 Step 1: Initialize and Load Collection")
    print("-" * 70)
    
    # Initialize MilvusService
    milvus_service = MilvusService()
    
    try:
        milvus_service.connect(uri=milvus_uri, token=milvus_token, collection_name=collection_name)
        milvus_service.create_collection(collection_name)
        print(f"✅ Connected to collection '{collection_name}'")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTroubleshooting:")
        print("  1. Start Milvus: docker run -d -p 19530:19530 milvusdb/milvus:latest")
        print("  2. Run demo first: python examples/example_5_milvus_service.py --demo")
        return
    
    # Get collection stats
    try:
        stats = milvus_service.get_collection_stats(collection_name)
        doc_count = stats.get('num_entities', 0)
        
        if doc_count == 0:
            print(f"⚠️  Collection is empty! Run demo mode first to insert sample data.")
            print(f"   Command: python examples/example_5_milvus_service.py --demo")
            return
        
        print(f"📊 Collection has {doc_count} documents")
    except Exception as e:
        print(f"⚠️  Could not get stats: {e}")
    
    print()
    
    print("🔧 Step 2: Initialize ChatService with Milvus")
    print("-" * 70)
    
    # Initialize ChatService
    try:
        service = ChatService(
            vector_store_type=VectorStoreType.MILVUS,
            milvus_uri=milvus_uri,
            milvus_token=milvus_token,
            milvus_collection=collection_name,
            k=3,
            temperature=0.7
        )
        service.initialize()
        print("✅ ChatService initialized with Milvus backend")
        print()
    except Exception as e:
        print(f"❌ Failed to initialize ChatService: {e}")
        return
    
    # Show config
    config = service.get_config()
    print("📋 Configuration:")
    print(f"  • Vector Store: {config['vector_store_type']}")
    print(f"  • Collection: {collection_name}")
    print(f"  • LLM Model: {config['llm_model']}")
    print(f"  • Temperature: {config['temperature']}")
    print(f"  • K (documents): {config['k']}")
    print()
    
    print("💬 Chat with your Milvus documents!")
    print("Commands:")
    print("  • 'quit' or 'exit' - Exit the program")
    print("  • 'stream' - Toggle streaming mode")
    print("  • 'config' - View configuration")
    print("  • 'sources' - Toggle source display")
    print("  • 'stats' - Show collection statistics")
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
        
        elif user_input.lower() == 'stats':
            try:
                stats = service.get_milvus_stats()
                print(f"\n📊 Collection Statistics:")
                print(f"  • Name: {stats['name']}")
                print(f"  • Documents: {stats['num_entities']}")
                print(f"  • Description: {stats['description']}")
                print()
            except Exception as e:
                print(f"❌ Error getting stats: {e}\n")
            continue
        
        # Process question
        print()
        print("Assistant: ", end="", flush=True)
        
        try:
            if streaming:
                # Stream the answer
                for chunk in service.get_answer_stream(user_input):
                    print(chunk, end="", flush=True)
                print("\n")
                
                # Get sources separately if needed
                if show_sources:
                    docs = service.retrieve_documents(user_input)
                    print(f"📚 Sources ({len(docs)} documents):")
                    for i, doc in enumerate(docs, 1):
                        print(f"  {i}. {doc['source']} (page {doc.get('page', 'N/A')})")
            else:
                # Get complete answer
                result = service.get_answer(user_input)
                print(result['answer'])
                print()
                
                if show_sources:
                    print(f"📚 Sources ({len(result['sources'])} documents):")
                    for i, doc in enumerate(result['sources'], 1):
                        print(f"  {i}. {doc['source']} (page {doc.get('page', 'N/A')})")
                        content = doc['content'][:100] + "..." if len(doc['content']) > 100 else doc['content']
                        print(f"     {content}")
        
        except Exception as e:
            print(f"\n❌ Error: {e}")
        
        print()


def main():
    """Main entry point"""
def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="MilvusService with DocumentRetriever - Interactive Chat"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run in demo mode (shows MilvusService operations)"
    )
    args = parser.parse_args()
    
    if args.demo:
        run_demo()
    else:
        run_interactive()


if __name__ == "__main__":
    main()
