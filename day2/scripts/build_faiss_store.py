"""
Build FAISS Vector Store for RAG

This script demonstrates:
1. Loading PDF documents from a folder
2. Generating embeddings using OpenAI
3. Storing vectors in FAISS (local vector database)
4. Saving the index for later use

FAISS (Facebook AI Similarity Search):
- Fast and efficient similarity search
- Stores vectors locally (no server needed)
- Great for prototypes and small-medium datasets
- Limitations: Not suitable for production scale (billions of vectors)

Vector Store Concepts:
- Embeddings: Text converted to numerical vectors (1536 dimensions for OpenAI)
- Similarity Search: Find vectors close to query vector (semantic similarity)
- Chunks: Documents split into smaller pieces for better retrieval

Prerequisites:
- Set OPENAI_API_KEY in .env file
- Have PDF documents in docs/ folder
"""

import os
import sys
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.document_loader import load_local_pdfs, get_document_stats

# Load environment variables
load_dotenv()


def build_faiss_index(
    data_dir: str = "docs",
    index_path: str = "faiss_index",
    embedding_model: str = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
):
    """
    Build a FAISS vector index from PDF documents.
    
    This function performs the complete RAG pipeline setup:
    1. Load PDFs from directory
    2. Split into chunks
    3. Generate embeddings for each chunk
    4. Create FAISS index
    5. Save to disk
    
    Args:
        data_dir: Directory containing PDF files
        index_path: Where to save the FAISS index
        embedding_model: OpenAI embedding model name (uses env var if None)
                        Options: 
                        - text-embedding-3-small (cheaper, 1536 dims)
                        - text-embedding-3-large (better, 3072 dims)
        chunk_size: Size of text chunks in characters
        chunk_overlap: Overlap between chunks (maintains context)
    
    Returns:
        FAISS: The vector store object
    """
    # Get embedding model from environment if not provided
    if embedding_model is None:
        embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    
    print("=" * 70)
    print("Building FAISS Vector Store for RAG")
    print("=" * 70)
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
                    "chunk_id": doc["chunk_id"],
                    "total_chunks": doc["total_chunks"]
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
    
    # Step 4: Create FAISS vector store
    print("🔢 Step 4: Generating embeddings and building FAISS index...")
    print("   ⏳ This may take a few moments...")
    print()
    
    try:
        # Create FAISS index from documents
        # This calls OpenAI API to generate embeddings for each chunk
        vectorstore = FAISS.from_documents(
            documents=langchain_docs,
            embedding=embeddings
        )
        
        print("   ✅ FAISS index created successfully")
        print()
        
    except Exception as e:
        print(f"\n❌ Error creating FAISS index: {e}")
        print("\nTroubleshooting:")
        print("   • Check OPENAI_API_KEY is set correctly")
        print("   • Ensure you have API credits available")
        print("   • Verify internet connection")
        return None
    
    # Step 5: Save the index to disk
    print("💾 Step 5: Saving FAISS index to disk...")
    print(f"   Location: {index_path}/")
    
    try:
        vectorstore.save_local(index_path)
        print("   ✅ Index saved successfully")
        print()
        
    except Exception as e:
        print(f"\n❌ Error saving index: {e}")
        return None
    
    # Summary
    print("=" * 70)
    print("✅ FAISS Vector Store Build Complete!")
    print("=" * 70)
    print()
    print("📋 Summary:")
    print(f"   • Documents processed: {stats['unique_sources']}")
    print(f"   • Chunks created: {stats['total_chunks']}")
    print(f"   • Embeddings generated: {stats['total_chunks']}")
    print(f"   • Index location: {index_path}/")
    print()
    print("🚀 Next Steps:")
    print(f"   Run: python example_3_rag_faiss.py")
    print("   This will let you query the knowledge base!")
    print()
    
    return vectorstore


def test_vector_store(vectorstore: FAISS):
    """
    Quick test of the vector store with a sample query.
    
    This demonstrates basic similarity search functionality.
    
    Args:
        vectorstore: The FAISS vector store to test
    """
    print("=" * 70)
    print("Testing Vector Store - Sample Query")
    print("=" * 70)
    print()
    
    test_query = "What are the company benefits?"
    print(f"🔍 Test Query: {test_query}")
    print()
    
    # Perform similarity search (returns top k most similar chunks)
    results = vectorstore.similarity_search(test_query, k=3)
    
    print(f"📄 Top {len(results)} Most Relevant Chunks:")
    print()
    
    for i, doc in enumerate(results, 1):
        print(f"   Result {i}:")
        print(f"   Source: {doc.metadata.get('filename', 'Unknown')}")
        print(f"   Chunk: {doc.metadata.get('chunk_id')}/{doc.metadata.get('total_chunks')}")
        print(f"   Content Preview:")
        print(f"   {doc.page_content[:200]}...")
        print()
    
    print("=" * 70)


if __name__ == "__main__":
    """
    Main execution - Build FAISS index and optionally test it.
    """
    import sys
    
    print("\n🚀 FAISS Vector Store Builder\n")
    
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
    
    # Build the vector store
    vectorstore = build_faiss_index()
    
    if vectorstore is None:
        print("\n❌ Failed to build vector store")
        sys.exit(1)
    
    # Run test if requested
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_vector_store(vectorstore)
    
    print("\n💡 Tips:")
    print("   • The index is saved to disk - you don't need to rebuild it")
    print("   • To rebuild with different settings, just run this script again")
    print("   • Try different chunk_size values (500-2000) for optimization")
    print("   • Larger chunks = more context but less precise retrieval")
