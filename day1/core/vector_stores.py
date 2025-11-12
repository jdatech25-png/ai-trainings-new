"""
Vector Store Managers Module

This module provides unified interfaces for different vector databases:
- FAISSVectorStore: Local file-based vector storage
- MilvusVectorStore: Production-grade distributed vector database

Both stores implement a common interface for easy switching.

Usage:
    from core.vector_stores import FAISSVectorStore, MilvusVectorStore
    from core.embeddings import EmbeddingManager
    from core.document_loader import load_local_pdfs
    
    # Load documents
    docs = load_local_pdfs("docs/")
    
    # Create embedding manager
    embeddings = EmbeddingManager()
    
    # Option 1: FAISS (local)
    store = FAISSVectorStore(embeddings)
    store.add_documents(docs)
    store.save("faiss_index")
    
    # Option 2: Milvus (production)
    store = MilvusVectorStore(embeddings, collection_name="docs")
    store.add_documents(docs)
"""

import os
from typing import List, Dict, Optional
from abc import ABC, abstractmethod
from langchain_community.vectorstores import FAISS
from langchain_milvus import Milvus
from langchain_core.documents import Document


class VectorStoreBase(ABC):
    """
    Abstract base class for vector stores.
    
    This defines a common interface that all vector stores must implement,
    making it easy to switch between FAISS, Milvus, or other vector databases.
    """
    
    @abstractmethod
    def add_documents(self, documents: List[Dict]) -> None:
        """Add documents to the vector store."""
        pass
    
    @abstractmethod
    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """Search for similar documents."""
        pass
    
    @abstractmethod
    def as_retriever(self, k: int = 3):
        """Get a LangChain retriever interface."""
        pass


class FAISSVectorStore(VectorStoreBase):
    """
    FAISS-based vector store for local development and prototyping.
    
    Advantages:
    - No server required
    - Fast for small to medium datasets
    - Simple setup
    - Good for demos and prototypes
    
    Limitations:
    - Limited scalability (< 1M vectors)
    - Manual persistence required
    - No built-in filtering
    - Single-machine only
    
    Attributes:
        embeddings: Embedding manager instance
        vectorstore: FAISS vectorstore instance
    """
    
    def __init__(self, embeddings):
        """
        Initialize FAISS vector store.
        
        Args:
            embeddings: EmbeddingManager instance
        """
        self.embeddings = embeddings.embeddings  # Get LangChain embeddings
        self.vectorstore = None
        print("📦 FAISS Vector Store initialized")
    
    def add_documents(self, documents: List[Dict]) -> None:
        """
        Add documents to FAISS index.
        
        Args:
            documents: List of document dictionaries with 'content' and metadata
        """
        # Convert to LangChain Document format
        langchain_docs = []
        for doc in documents:
            langchain_docs.append(
                Document(
                    page_content=doc["content"],
                    metadata={
                        "source": doc.get("source", ""),
                        "filename": doc.get("filename", ""),
                        "chunk_id": str(doc.get("chunk_id", "")),
                        "total_chunks": str(doc.get("total_chunks", ""))
                    }
                )
            )
        
        print(f"📝 Adding {len(langchain_docs)} documents to FAISS...")
        
        if self.vectorstore is None:
            # Create new index
            self.vectorstore = FAISS.from_documents(
                documents=langchain_docs,
                embedding=self.embeddings
            )
        else:
            # Add to existing index
            self.vectorstore.add_documents(langchain_docs)
        
        print(f"✅ Documents added successfully")
    
    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            k: Number of results to return
        
        Returns:
            List of most similar documents
        """
        if self.vectorstore is None:
            raise ValueError("No documents in store. Call add_documents first.")
        
        return self.vectorstore.similarity_search(query, k=k)
    
    def as_retriever(self, k: int = 3):
        """
        Get LangChain retriever interface.
        
        Args:
            k: Number of documents to retrieve
        
        Returns:
            LangChain retriever object
        """
        if self.vectorstore is None:
            raise ValueError("No documents in store. Call add_documents first.")
        
        return self.vectorstore.as_retriever(search_kwargs={"k": k})
    
    def save(self, path: str) -> None:
        """
        Save FAISS index to disk.
        
        Args:
            path: Directory path to save index
        """
        if self.vectorstore is None:
            raise ValueError("No vector store to save")
        
        self.vectorstore.save_local(path)
        print(f"💾 FAISS index saved to: {path}/")
    
    @classmethod
    def load(cls, path: str, embeddings) -> 'FAISSVectorStore':
        """
        Load FAISS index from disk.
        
        Args:
            path: Directory path to load index from
            embeddings: EmbeddingManager instance
        
        Returns:
            FAISSVectorStore instance
        """
        instance = cls(embeddings)
        instance.vectorstore = FAISS.load_local(
            path,
            embeddings.embeddings,
            allow_dangerous_deserialization=True
        )
        print(f"📂 FAISS index loaded from: {path}/")
        return instance


class MilvusVectorStore(VectorStoreBase):
    """
    Milvus-based vector store for production deployments.
    
    Advantages:
    - Scalable to billions of vectors
    - Distributed architecture
    - Advanced indexing algorithms
    - Rich metadata filtering
    - GPU acceleration support
    - Production monitoring
    
    Requirements:
    - Milvus server running
    - Network connectivity
    
    Attributes:
        embeddings: Embedding manager instance
        vectorstore: Milvus vectorstore instance
        collection_name: Name of Milvus collection
    """
    
    def __init__(
        self,
        embeddings,
        collection_name: str = "documents",
        uri: str = None,
        token: str = None
    ):
        """
        Initialize Milvus vector store.
        
        Args:
            embeddings: EmbeddingManager instance
            collection_name: Name for the collection
            uri: Milvus server URI (from env if None)
            token: Milvus token for authentication (from env if None)
        """
        self.embeddings = embeddings.embeddings  # Get LangChain embeddings
        self.collection_name = collection_name
        
        # Get connection details
        self.uri = uri or os.getenv("MILVUS_URI", "tcp://localhost:19530")
        self.token = token or os.getenv("MILVUS_TOKEN")
        
        # Build connection args
        self.connection_args = {"uri": self.uri}
        if self.token:
            self.connection_args["token"] = self.token
        
        self.vectorstore = None
        print(f"🗄️  Milvus Vector Store initialized")
        print(f"   URI: {self.uri}")
        print(f"   Collection: {collection_name}")
    
    def add_documents(self, documents: List[Dict]) -> None:
        """
        Add documents to Milvus collection.
        
        Args:
            documents: List of document dictionaries with 'content' and metadata
        """
        # Convert to LangChain Document format
        langchain_docs = []
        for doc in documents:
            langchain_docs.append(
                Document(
                    page_content=doc["content"],
                    metadata={
                        "source": doc.get("source", ""),
                        "filename": doc.get("filename", ""),
                        "chunk_id": str(doc.get("chunk_id", "")),
                        "total_chunks": str(doc.get("total_chunks", ""))
                    }
                )
            )
        
        print(f"📝 Adding {len(langchain_docs)} documents to Milvus...")
        
        if self.vectorstore is None:
            # Create new collection
            self.vectorstore = Milvus.from_documents(
                documents=langchain_docs,
                embedding=self.embeddings,
                collection_name=self.collection_name,
                connection_args=self.connection_args,
                index_params={
                    "metric_type": "COSINE",
                    "index_type": "HNSW",
                    "params": {"M": 8, "efConstruction": 64}
                },
                search_params={"metric_type": "COSINE", "params": {"ef": 64}}
            )
        else:
            # Add to existing collection
            self.vectorstore.add_documents(langchain_docs)
        
        print(f"✅ Documents added to Milvus collection: {self.collection_name}")
    
    def similarity_search(
        self,
        query: str,
        k: int = 3,
        filter_expr: str = None
    ) -> List[Document]:
        """
        Search for similar documents with optional filtering.
        
        Args:
            query: Search query
            k: Number of results to return
            filter_expr: Milvus filter expression (e.g., 'filename == "doc.pdf"')
        
        Returns:
            List of most similar documents
        """
        if self.vectorstore is None:
            raise ValueError("No documents in store. Call add_documents first.")
        
        if filter_expr:
            return self.vectorstore.similarity_search(
                query,
                k=k,
                expr=filter_expr
            )
        else:
            return self.vectorstore.similarity_search(query, k=k)
    
    def as_retriever(self, k: int = 3):
        """
        Get LangChain retriever interface.
        
        Args:
            k: Number of documents to retrieve
        
        Returns:
            LangChain retriever object
        """
        if self.vectorstore is None:
            raise ValueError("No documents in store. Call add_documents first.")
        
        return self.vectorstore.as_retriever(search_kwargs={"k": k})
    
    @classmethod
    def connect(
        cls,
        embeddings,
        collection_name: str,
        uri: str = None,
        token: str = None
    ) -> 'MilvusVectorStore':
        """
        Connect to existing Milvus collection.
        
        Args:
            embeddings: EmbeddingManager instance
            collection_name: Name of existing collection
            uri: Milvus server URI (from env if None)
            token: Milvus token (from env if None)
        
        Returns:
            MilvusVectorStore instance
        """
        instance = cls(embeddings, collection_name, uri, token)
        
        # Connect to existing collection
        instance.vectorstore = Milvus(
            embedding_function=embeddings.embeddings,
            collection_name=collection_name,
            connection_args=instance.connection_args
        )
        
        print(f"🔌 Connected to existing collection: {collection_name}")
        return instance


if __name__ == "__main__":
    """
    Test vector stores.
    """
    from dotenv import load_dotenv
    from core.embeddings import EmbeddingManager
    
    load_dotenv()
    
    print("=" * 70)
    print("Testing Vector Stores")
    print("=" * 70)
    print()
    
    # Initialize embeddings
    embeddings = EmbeddingManager()
    print()
    
    # Sample documents
    docs = [
        {
            "content": "The company offers 20 days of vacation",
            "filename": "handbook.pdf",
            "chunk_id": 1,
            "total_chunks": 5
        },
        {
            "content": "Health insurance is provided to all employees",
            "filename": "handbook.pdf",
            "chunk_id": 2,
            "total_chunks": 5
        }
    ]
    
    # Test FAISS
    print("Testing FAISS Vector Store...")
    print("-" * 70)
    faiss_store = FAISSVectorStore(embeddings)
    faiss_store.add_documents(docs)
    
    results = faiss_store.similarity_search("vacation policy", k=1)
    print(f"\n✅ FAISS Search Results:")
    print(f"   Query: 'vacation policy'")
    print(f"   Top result: {results[0].page_content[:50]}...")
    print()
    
    print("=" * 70)
    print("✅ Vector store tests completed!")
