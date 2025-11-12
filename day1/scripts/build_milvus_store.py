"""
Build Milvus Vector Store for Production RAG

This script demonstrates:
1. Connecting to Milvus vector database
2. Creating a collection for document embeddings
3. Inserting documents with metadata
4. Production-ready vector storage

Milvus vs FAISS Comparison:
╔═══════════════════╦══════════════════╦═══════════════════════╗
║ Feature           ║ FAISS            ║ Milvus                ║
╠═══════════════════╬══════════════════╬═══════════════════════╣
║ Deployment        ║ Local file       ║ Server/Cloud          ║
║ Scalability       ║ Limited (GB)     ║ Billions of vectors   ║
║ Performance       ║ Fast for small   ║ Optimized for scale   ║
║ Concurrent Users  ║ Limited          ║ Production-ready      ║
║ Persistence       ║ Manual save      ║ Automatic             ║
║ Filtering         ║ Basic            ║ Advanced metadata     ║
║ Best For          ║ Prototypes/Demos ║ Production Apps       ║
╚═══════════════════╩══════════════════╩═══════════════════════╝

Milvus Features:
- Distributed architecture for high availability
- GPU acceleration support
- Advanced indexing algorithms (HNSW, IVF, etc.)
- Rich metadata filtering capabilities
- Production-grade monitoring and metrics

Prerequisites:
- Milvus server running (configure URI in .env)
- Set OPENAI_API_KEY in .env file
- Have PDF documents in docs/ folder
"""

import os
import sys
from dotenv import load_dotenv
from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.document_loader import load_local_pdfs, get_document_stats
from core.milvus_service import MilvusService

# Load environment variables
load_dotenv()


def build_milvus_collection(
    data_dir: str = "docs",
    collection_name: str = None,
    embedding_model: str = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    drop_existing: bool = False
):
    """
    Build a Milvus collection from PDF documents.
    
    This function:
    1. Connects to Milvus server
    2. Creates/connects to a collection
    3. Loads and chunks documents
    4. Generates embeddings
    5. Inserts into Milvus
    
    Args:
        data_dir: Directory containing PDF files
        collection_name: Name for the Milvus collection (from .env if None)
        embedding_model: OpenAI embedding model name (uses env var if None)
        chunk_size: Size of text chunks in characters
        chunk_overlap: Overlap between chunks
        drop_existing: If True, drops existing collection before creating
    
    Returns:
        Milvus: The vector store object
    """
    # Get embedding model from environment if not provided
    if embedding_model is None:
        embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    
    print("=" * 70)
    print("Building Milvus Vector Store for Production RAG")
    print("=" * 70)
    print()
    
    # Get Milvus configuration from environment
    milvus_uri = os.getenv("MILVUS_URI", "tcp://localhost:19530")
    if collection_name is None:
        collection_name = os.getenv("MILVUS_COLLECTION_NAME", "training_demo")
    
    print("🔧 Milvus Configuration:")
    print(f"   URI: {milvus_uri}")
    print(f"   Collection: {collection_name}")
    print()
    
    # Step 1: Load and chunk PDF documents
    print("📂 Step 1: Loading PDF documents...")
    print(f"   Source directory: {data_dir}")
    print(f"   Chunk size: {chunk_size} characters")
    print(f"   Chunk overlap: {chunk_overlap} characters")
    print()
    
    docs = load_local_pdfs(data_dir, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    if not docs:
        print("\n❌ Error: No documents loaded. Please add PDF files to the docs/ folder")
        return None
    
    # Display statistics
    stats = get_document_stats(docs)
    print("\n📊 Document Statistics:")
    print(f"   Total chunks: {stats['total_chunks']}")
    print(f"   Unique sources: {stats['unique_sources']}")
    print(f"   Average chunk length: {stats['avg_chunk_length']} characters")
    print(f"   Total characters: {stats['total_characters']:,}")
    print()
    
    # Step 2: Convert to LangChain Document format
    print("📝 Step 2: Converting to LangChain Document format...")
    
    langchain_docs = []
    for doc in docs:
        langchain_docs.append(
            Document(
                page_content=doc["content"],
                metadata={
                    "source": doc["source"],
                    "filename": doc["filename"],
                    "chunk_id": str(doc["chunk_id"]),  # Convert to string for Milvus
                    "total_chunks": str(doc["total_chunks"])
                }
            )
        )
    
    print(f"   ✅ Created {len(langchain_docs)} LangChain documents")
    print()
    
    # Step 3: Initialize embedding model
    print("🧮 Step 3: Initializing embedding model...")
    print(f"   Model: {embedding_model}")
    print(f"   Provider: OpenAI")
    
    embeddings = OpenAIEmbeddings(
        model=embedding_model,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    print("   ✅ Embedding model initialized")
    print()
    
    # Step 4: Connect to Milvus and create collection
    print("🔢 Step 4: Connecting to Milvus using MilvusService...")
    print("   ⏳ This may take a few moments...")
    print()
    
    try:
        # Initialize MilvusService (singleton)
        milvus_service = MilvusService()
        
        # Extract token from URI if present
        token = None
        uri = milvus_uri
        if "?token=" in milvus_uri:
            uri, token = milvus_uri.split("?token=")
        elif milvus_token := os.getenv("MILVUS_TOKEN"):
            token = milvus_token
        
        # Connect to Milvus
        milvus_service.connect(
            uri=uri,
            token=token,
            collection_name=collection_name
        )
        
        # Drop existing collection if requested
        if drop_existing:
            print("   ⚠️  Dropping existing collection (if exists)...")
            try:
                from pymilvus import utility
                if utility.has_collection(collection_name):
                    utility.drop_collection(collection_name)
                    print(f"   ✅ Dropped existing collection: {collection_name}")
            except Exception as e:
                print(f"   ⚠️  Could not drop collection: {e}")
        
        # Create collection with simple schema
        milvus_service.create_collection(collection_name)
        
        # Extract text content from LangChain documents
        texts = [doc.page_content for doc in langchain_docs]
        
        # Insert documents into Milvus
        print(f"   📝 Inserting {len(texts)} documents...")
        success = milvus_service.insert_documents(
            texts=texts,
            collection_name=collection_name
        )
        
        if not success:
            raise Exception("Failed to insert documents")
        
        print("   ✅ Documents inserted into Milvus successfully")
        print()
        
        # Get the LangChain vectorstore for compatibility
        vectorstore = milvus_service.get_langchain_vectorstore(collection_name)
        
    except Exception as e:
        print(f"\n❌ Error creating Milvus collection: {e}")
        print("\nTroubleshooting:")
        print("   • Ensure Milvus server is running")
        print("   • Check MILVUS_URI in .env file")
        print("   • Verify network connectivity")
        print("   • Check Milvus server logs")
        print("\n💡 Quick Milvus Setup:")
        print("   • Docker: docker run -d -p 19530:19530 milvusdb/milvus:latest")
        print("   • Milvus Lite: pip install milvus")
        return None
    
    # Summary
    print("=" * 70)
    print("✅ Milvus Vector Store Build Complete!")
    print("=" * 70)
    print()
    print("📋 Summary:")
    print(f"   • Milvus URI: {milvus_uri}")
    print(f"   • Collection: {collection_name}")
    print(f"   • Documents processed: {stats['unique_sources']}")
    print(f"   • Chunks inserted: {stats['total_chunks']}")
    print(f"   • Embeddings generated: {stats['total_chunks']}")
    print(f"   • Index type: HNSW (high performance)")
    print(f"   • Similarity metric: COSINE")
    print()
    print("🚀 Next Steps:")
    print(f"   Run: python example_4_rag_milvus.py")
    print("   This will let you query the Milvus knowledge base!")
    print()
    
    return vectorstore


def test_milvus_connection():
    """
    Test connection to Milvus server.
    
    This helps diagnose connection issues before attempting to build.
    """
    print("=" * 70)
    print("Testing Milvus Connection")
    print("=" * 70)
    print()
    
    milvus_uri = os.getenv("MILVUS_URI", "tcp://localhost:19530")
    print(f"🔍 Attempting to connect to: {milvus_uri}")
    print()
    
    try:
        from pymilvus import connections, utility
        
        # Connect to Milvus
        connections.connect(uri=milvus_uri)
        print("✅ Successfully connected to Milvus!")
        print()
        
        # List existing collections
        collections = utility.list_collections()
        print(f"📚 Existing collections ({len(collections)}):")
        if collections:
            for col in collections:
                print(f"   • {col}")
        else:
            print("   (No collections found)")
        print()
        
        # Disconnect
        connections.disconnect("default")
        print("✅ Connection test completed successfully")
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print()
        print("Troubleshooting:")
        print("   1. Is Milvus server running?")
        print("   2. Is the URI correct in .env?")
        print("   3. Check firewall/network settings")
        print()
        return False


if __name__ == "__main__":
    """
    Main execution - Build Milvus collection.
    """
    import sys
    
    print("\n🚀 Milvus Vector Store Builder\n")
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found in environment variables")
        print("   Please add it to your .env file")
        sys.exit(1)
    
    # Check if docs directory exists
    if not os.path.exists("docs"):
        print("❌ Error: 'docs/' directory not found")
        print("   Please create it and add PDF files")
        sys.exit(1)
    
    # Test connection mode
    if len(sys.argv) > 1 and sys.argv[1] == "--test-connection":
        success = test_milvus_connection()
        sys.exit(0 if success else 1)
    
    # Drop and rebuild mode
    drop_existing = len(sys.argv) > 1 and sys.argv[1] == "--rebuild"
    
    if drop_existing:
        print("⚠️  REBUILD MODE: Will drop existing collection\n")
    
    # Build the vector store
    vectorstore = build_milvus_collection(drop_existing=drop_existing)
    
    if vectorstore is None:
        print("\n❌ Failed to build Milvus collection")
        print("\n💡 Try: python build_milvus_store.py --test-connection")
        sys.exit(1)
    
    print("\n💡 Tips:")
    print("   • Milvus persists data automatically (no manual save needed)")
    print("   • Use --rebuild flag to drop and recreate collection")
    print("   • Monitor Milvus performance through its dashboard")
    print("   • Production: Use Milvus Cloud for managed service")
    print("\n🔍 Compare with FAISS:")
    print("   • FAISS: Local file, simple, good for prototypes")
    print("   • Milvus: Server-based, scalable, production-ready")
