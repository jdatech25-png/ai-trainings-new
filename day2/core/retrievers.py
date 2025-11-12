"""
Document Retriever Module

This module handles document retrieval from vector stores using semantic search.
It provides a clean interface for finding relevant context given a query.

Usage:
    from core.retrievers import DocumentRetriever
    from core.vector_stores import FAISSVectorStore
    
    retriever = DocumentRetriever(vector_store, k=3)
    docs = retriever.retrieve("What are the benefits?")
"""

from typing import List
from langchain_core.documents import Document


class DocumentRetriever:
    """
    Retrieves relevant documents from a vector store based on query similarity.
    
    This class wraps vector store retrieval with additional utilities
    for formatting, filtering, and managing retrieval results.
    
    Attributes:
        vector_store: Vector store instance (FAISS or Milvus)
        k: Number of documents to retrieve
        retriever: LangChain retriever interface
    """
    
    def __init__(self, vector_store, k: int = 3):
        """
        Initialize the document retriever.
        
        Args:
            vector_store: FAISSVectorStore or MilvusVectorStore instance
            k: Number of documents to retrieve (default: 3)
        """
        self.vector_store = vector_store
        self.k = k
        self.retriever = vector_store.as_retriever(k=k)
        
        print(f"🔍 Document Retriever initialized (k={k})")
    
    def retrieve(self, query: str, k: int = None) -> List[Document]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query text
            k: Number of documents to retrieve (uses self.k if None)
        
        Returns:
            List of LangChain Document objects
        
        Example:
            >>> retriever = DocumentRetriever(store, k=3)
            >>> docs = retriever.retrieve("What are benefits?")
            >>> print(f"Found {len(docs)} relevant documents")
        """
        k = k or self.k
        
        print(f"🔍 Searching for: '{query}'")
        print(f"   Retrieving top {k} documents...")
        
        docs = self.vector_store.similarity_search(query, k=k)
        
        print(f"   ✅ Found {len(docs)} relevant documents")
        
        return docs
    
    def retrieve_with_scores(self, query: str, k: int = None):
        """
        Retrieve documents with similarity scores.
        
        Args:
            query: Search query text
            k: Number of documents to retrieve
        
        Returns:
            List of tuples (Document, score)
        
        Note: Not all vector stores support scores. Falls back to retrieve().
        """
        k = k or self.k
        
        try:
            # Try to get similarity scores
            if hasattr(self.vector_store.vectorstore, 'similarity_search_with_score'):
                return self.vector_store.vectorstore.similarity_search_with_score(query, k=k)
            else:
                # Fallback to regular retrieval
                docs = self.retrieve(query, k=k)
                return [(doc, 0.0) for doc in docs]  # Score unavailable
        except Exception as e:
            print(f"⚠️  Could not retrieve scores: {e}")
            docs = self.retrieve(query, k=k)
            return [(doc, 0.0) for doc in docs]
    
    def format_docs_for_context(self, docs: List[Document]) -> str:
        """
        Format retrieved documents into a context string for LLM.
        
        Args:
            docs: List of Document objects
        
        Returns:
            Formatted string with all document contents
        
        Example:
            >>> docs = retriever.retrieve("benefits")
            >>> context = retriever.format_docs_for_context(docs)
            >>> # Use context in LLM prompt
        """
        formatted = []
        
        for i, doc in enumerate(docs, 1):
            filename = doc.metadata.get('filename', 'Unknown')
            chunk_id = doc.metadata.get('chunk_id', '?')
            total_chunks = doc.metadata.get('total_chunks', '?')
            
            formatted.append(
                f"[Document {i} - {filename} (Chunk {chunk_id}/{total_chunks})]\n"
                f"{doc.page_content}"
            )
        
        return "\n\n---\n\n".join(formatted)
    
    def get_source_info(self, docs: List[Document]) -> List[dict]:
        """
        Extract source information from documents for citation.
        
        Args:
            docs: List of Document objects
        
        Returns:
            List of dictionaries with source information
        
        Example:
            >>> docs = retriever.retrieve("benefits")
            >>> sources = retriever.get_source_info(docs)
            >>> for source in sources:
            ...     print(f"Source: {source['filename']}")
        """
        sources = []
        
        for i, doc in enumerate(docs, 1):
            sources.append({
                "doc_num": i,
                "filename": doc.metadata.get('filename', 'Unknown'),
                "chunk": f"{doc.metadata.get('chunk_id', '?')}/{doc.metadata.get('total_chunks', '?')}",
                "source": doc.metadata.get('source', ''),
                "preview": doc.page_content[:150] + "..."
            })
        
        return sources
    
    def print_results(self, docs: List[Document], show_content: bool = True):
        """
        Pretty print retrieval results.
        
        Args:
            docs: List of Document objects
            show_content: Whether to show full content (default: True)
        """
        print(f"\n📄 Retrieved {len(docs)} Documents:")
        print("=" * 70)
        
        for i, doc in enumerate(docs, 1):
            filename = doc.metadata.get('filename', 'Unknown')
            chunk_id = doc.metadata.get('chunk_id', '?')
            total_chunks = doc.metadata.get('total_chunks', '?')
            
            print(f"\n[Document {i}]")
            print(f"Source: {filename} (Chunk {chunk_id}/{total_chunks})")
            
            if show_content:
                print(f"\nContent:")
                print("-" * 70)
                print(doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else ""))
        
        print("\n" + "=" * 70)


if __name__ == "__main__":
    """
    Test the document retriever.
    """
    from dotenv import load_dotenv
    from core.embeddings import EmbeddingManager
    from core.vector_stores import FAISSVectorStore
    
    load_dotenv()
    
    print("=" * 70)
    print("Testing Document Retriever")
    print("=" * 70)
    print()
    
    # Setup
    embeddings = EmbeddingManager()
    store = FAISSVectorStore(embeddings)
    
    # Add sample documents
    docs = [
        {
            "content": "The company offers 20 days of paid vacation per year.",
            "filename": "handbook.pdf",
            "chunk_id": 1,
            "total_chunks": 5
        },
        {
            "content": "Health insurance includes dental and vision coverage.",
            "filename": "handbook.pdf",
            "chunk_id": 2,
            "total_chunks": 5
        },
        {
            "content": "Remote work is allowed up to 3 days per week.",
            "filename": "handbook.pdf",
            "chunk_id": 3,
            "total_chunks": 5
        }
    ]
    
    store.add_documents(docs)
    print()
    
    # Create retriever
    retriever = DocumentRetriever(store, k=2)
    print()
    
    # Test retrieval
    query = "What is the vacation policy?"
    results = retriever.retrieve(query)
    
    # Print results
    retriever.print_results(results, show_content=True)
    
    # Test context formatting
    print("\n" + "=" * 70)
    print("Formatted Context for LLM:")
    print("=" * 70)
    context = retriever.format_docs_for_context(results)
    print(context)
    
    print("\n✅ Retriever tests completed!")
