"""
Chat Service - Central RAG Processing Logic

This module provides a unified service layer for all RAG operations.
All chat functionality is centralized here and can be consumed by:
- FastAPI controllers
- Examples
- Scripts
- Tests

The ChatService handles:
- Vector store initialization (FAISS/Milvus)
- Document retrieval
- Answer generation
- Streaming responses
- Configuration management
"""

import os
from typing import List, Dict, Any, Iterator, Optional
from enum import Enum

from langchain_community.vectorstores import FAISS
from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from core.milvus_service import MilvusService


class VectorStoreType(str, Enum):
    """Supported vector store types"""
    FAISS = "faiss"
    MILVUS = "milvus"


class ChatService:
    """
    Centralized service for RAG chat operations.
    
    This service encapsulates all RAG logic including:
    - Vector store management
    - Document retrieval
    - LLM answer generation
    - Streaming support
    
    Usage:
        # Initialize service
        service = ChatService(
            vector_store_type=VectorStoreType.FAISS,
            faiss_index_path="faiss_index"
        )
        
        # Get answer
        answer = service.get_answer("What are the company benefits?")
        
        # Get streaming answer
        for chunk in service.get_answer_stream("What is the leave policy?"):
            print(chunk, end="", flush=True)
    """
    
    def __init__(
        self,
        vector_store_type: VectorStoreType = VectorStoreType.FAISS,
        faiss_index_path: str = "faiss_index",
        milvus_uri: str = None,
        milvus_token: str = None,
        milvus_collection: str = None,
        embedding_model: str = None,
        llm_model: str = None,
        temperature: float = 0,
        k: int = 3,
        api_key: Optional[str] = None
    ):
        """
        Initialize the Chat Service.
        
        Args:
            vector_store_type: Type of vector store to use
            faiss_index_path: Path to FAISS index directory
            milvus_uri: Milvus server URI (uses env var if None)
            milvus_collection: Milvus collection name (uses env var if None)
            embedding_model: OpenAI embedding model name (uses env var if None)
            llm_model: OpenAI LLM model name (uses env var if None)
            temperature: LLM temperature (0 = deterministic, 2 = creative)
            k: Number of documents to retrieve
            api_key: OpenAI API key (defaults to env variable)
        """
        self.vector_store_type = vector_store_type
        self.faiss_index_path = faiss_index_path
        self.milvus_uri = milvus_uri or os.getenv("MILVUS_URI", "tcp://localhost:19530")
        self.milvus_token = milvus_token or os.getenv("MILVUS_TOKEN", None)
        self.milvus_collection = milvus_collection or os.getenv("MILVUS_COLLECTION_NAME", "training_demo")
        self.embedding_model_name = embedding_model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.llm_model_name = llm_model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.temperature = temperature
        self.k = k
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        # Initialize components
        self.embeddings = None
        self.vector_store = None
        self.llm = None
        self.rag_chain = None
        self._is_initialized = False
        self.milvus_service = None
        
    def initialize(self) -> None:
        """
        Initialize the service components.
        This lazy-loads the vector store and LLM.
        
        Raises:
            ValueError: If API key is not provided
            FileNotFoundError: If FAISS index not found
            ConnectionError: If Milvus connection fails
        """
        if self._is_initialized:
            return
            
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model=self.embedding_model_name,
            openai_api_key=self.api_key
        )
        
        # Initialize vector store
        if self.vector_store_type == VectorStoreType.FAISS:
            self._initialize_faiss()
        else:
            self._initialize_milvus()
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=self.llm_model_name,
            temperature=self.temperature,
            openai_api_key=self.api_key
        )
        
        # Create RAG chain
        self._create_rag_chain()
        
        self._is_initialized = True
        
    def _initialize_faiss(self) -> None:
        """Initialize FAISS vector store"""
        if not os.path.exists(self.faiss_index_path):
            raise FileNotFoundError(
                f"FAISS index not found at {self.faiss_index_path}. "
                f"Please run build_faiss_store.py first."
            )
        
        self.vector_store = FAISS.load_local(
            self.faiss_index_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        
    def _initialize_milvus(self) -> None:
        """Initialize Milvus vector store using MilvusService"""
        try:
            # Get singleton MilvusService instance
            self.milvus_service = MilvusService()
            
            # Extract token from URI if present (format: https://...?token=xxx)
            uri = self.milvus_uri
            
            # Connect to Milvus
            self.milvus_service.connect(
                uri=uri,
                token=os.getenv("MILVUS_TOKEN"),
                collection_name=self.milvus_collection
            )
            
            # Ensure collection exists
            self.milvus_service.create_collection(self.milvus_collection)
            
            # Get LangChain vectorstore for compatibility with existing code
            self.vector_store = self.milvus_service.get_langchain_vectorstore(
                self.milvus_collection
            )
            
            if self.vector_store is None:
                raise ValueError(f"Failed to get vectorstore for collection '{self.milvus_collection}'")
            
        except Exception as e:
            raise ConnectionError(
                f"Failed to connect to Milvus at {self.milvus_uri}: {e}"
            )
    
    def _create_rag_chain(self) -> None:
        """Create the RAG processing chain"""
        # RAG prompt template
        template = """You are a helpful AI assistant. Use the following context to answer the question.
If you cannot answer the question based on the context, say so.

Context:
{context}

Question: {question}

Answer:"""

        prompt = ChatPromptTemplate.from_template(template)
        
        # Create retrieval chain
        self.rag_chain = (
            {
                "context": lambda x: self._format_docs(
                    self.vector_store.similarity_search(x["question"], k=self.k)
                ),
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
    
    @staticmethod
    def _format_docs(docs: List[Document]) -> str:
        """Format documents for context"""
        return "\n\n".join([
            f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}"
            for doc in docs
        ])
    
    def retrieve_documents(self, query: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            k: Number of documents to retrieve (overrides default)
            
        Returns:
            List of documents with content and metadata
        """
        if not self._is_initialized:
            self.initialize()
        
        k = k or self.k
        docs = self.vector_store.similarity_search(query, k=k)
        
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "Unknown")
            }
            for doc in docs
        ]
    
    def get_answer(self, query: str) -> Dict[str, Any]:
        """
        Get a complete answer for a query.
        
        Args:
            query: User question
            
        Returns:
            Dictionary containing answer and source documents
        """
        if not self._is_initialized:
            self.initialize()
        
        # Retrieve documents
        docs = self.retrieve_documents(query)
        
        # Generate answer
        answer = self.rag_chain.invoke({"question": query})
        
        return {
            "answer": answer,
            "sources": docs,
            "model": self.llm_model_name,
            "vector_store": self.vector_store_type.value
        }
    
    def get_answer_stream(self, query: str) -> Iterator[str]:
        """
        Get a streaming answer for a query.
        
        Args:
            query: User question
            
        Yields:
            Chunks of the answer as they're generated
        """
        if not self._is_initialized:
            self.initialize()
        
        # Stream the answer
        for chunk in self.rag_chain.stream({"question": query}):
            yield chunk
    
    def get_answer_with_stream(self, query: str) -> Dict[str, Any]:
        """
        Get answer with streaming support and source documents.
        
        Args:
            query: User question
            
        Returns:
            Dictionary with stream iterator and source documents
        """
        if not self._is_initialized:
            self.initialize()
        
        # Retrieve documents first
        docs = self.retrieve_documents(query)
        
        # Create stream
        stream = self.rag_chain.stream({"question": query})
        
        return {
            "stream": stream,
            "sources": docs,
            "model": self.llm_model_name,
            "vector_store": self.vector_store_type.value
        }
    
    def update_config(
        self,
        temperature: Optional[float] = None,
        k: Optional[int] = None,
        llm_model: Optional[str] = None
    ) -> None:
        """
        Update service configuration dynamically.
        
        Args:
            temperature: New temperature value
            k: New k value
            llm_model: New LLM model name
        """
        needs_reinit = False
        
        if temperature is not None and temperature != self.temperature:
            self.temperature = temperature
            needs_reinit = True
        
        if k is not None and k != self.k:
            self.k = k
        
        if llm_model is not None and llm_model != self.llm_model_name:
            self.llm_model_name = llm_model
            needs_reinit = True
        
        # Reinitialize LLM if needed
        if needs_reinit and self._is_initialized:
            self.llm = ChatOpenAI(
                model=self.llm_model_name,
                temperature=self.temperature,
                openai_api_key=self.api_key
            )
            self._create_rag_chain()
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get current service configuration.
        
        Returns:
            Dictionary with current configuration
        """
        return {
            "vector_store_type": self.vector_store_type.value,
            "embedding_model": self.embedding_model_name,
            "llm_model": self.llm_model_name,
            "temperature": self.temperature,
            "k": self.k,
            "is_initialized": self._is_initialized
        }
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check service health.
        
        Returns:
            Dictionary with health status
        """
        try:
            if not self._is_initialized:
                return {
                    "status": "not_initialized",
                    "message": "Service not initialized"
                }
            
            # Test retrieval
            test_docs = self.vector_store.similarity_search("test", k=1)
            
            # Add Milvus-specific stats if using Milvus
            milvus_stats = None
            if self.vector_store_type == VectorStoreType.MILVUS and self.milvus_service:
                milvus_stats = self.milvus_service.get_collection_stats(
                    self.milvus_collection
                )
            
            return {
                "status": "healthy",
                "message": "Service is operational",
                "config": self.get_config(),
                "vector_store_docs": len(test_docs) > 0,
                "milvus_stats": milvus_stats
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "message": f"Service error: {str(e)}"
            }
    
    def insert_documents_to_milvus(self, texts: List[str]) -> bool:
        """
        Insert documents directly into Milvus.
        Only works when using Milvus vector store.
        
        Args:
            texts: List of text strings to insert
            
        Returns:
            True if successful, False otherwise
        """
        if self.vector_store_type != VectorStoreType.MILVUS:
            raise ValueError("This method only works with Milvus vector store")
        
        if not self._is_initialized or not self.milvus_service:
            raise ValueError("Service not initialized with Milvus")
        
        return self.milvus_service.insert_documents(
            texts=texts,
            collection_name=self.milvus_collection
        )
    
    def search_milvus(self, query: str, k: int = None) -> List[Dict[str, Any]]:
        """
        Direct search on Milvus collection.
        Only works when using Milvus vector store.
        
        Args:
            query: Search query
            k: Number of results (uses default if None)
            
        Returns:
            List of search results from Milvus
        """
        if self.vector_store_type != VectorStoreType.MILVUS:
            raise ValueError("This method only works with Milvus vector store")
        
        if not self._is_initialized or not self.milvus_service:
            raise ValueError("Service not initialized with Milvus")
        
        return self.milvus_service.search(
            query=query,
            k=k or self.k,
            collection_name=self.milvus_collection
        )
    
    def get_milvus_stats(self) -> Dict[str, Any]:
        """
        Get Milvus collection statistics.
        Only works when using Milvus vector store.
        
        Returns:
            Dictionary with collection stats
        """
        if self.vector_store_type != VectorStoreType.MILVUS:
            raise ValueError("This method only works with Milvus vector store")
        
        if not self._is_initialized or not self.milvus_service:
            raise ValueError("Service not initialized with Milvus")
        
        return self.milvus_service.get_collection_stats(
            collection_name=self.milvus_collection
        )
    
    def get_document_retriever(self, k: Optional[int] = None):
        """
        Get a DocumentRetriever instance for the current vector store.
        Works with both FAISS and Milvus.
        
        Args:
            k: Number of documents to retrieve (uses default if None)
            
        Returns:
            DocumentRetriever instance
            
        Example:
            >>> service = ChatService(vector_store_type=VectorStoreType.MILVUS)
            >>> service.initialize()
            >>> retriever = service.get_document_retriever(k=5)
            >>> docs = retriever.retrieve("vacation policy")
        """
        from core.retrievers import DocumentRetriever
        
        if not self._is_initialized:
            self.initialize()
        
        # For Milvus, use the MilvusService's get_retriever method
        if self.vector_store_type == VectorStoreType.MILVUS and self.milvus_service:
            return self.milvus_service.get_retriever(
                collection_name=self.milvus_collection,
                k=k or self.k
            )
        
        # For FAISS, create directly from vector_store
        # Assuming vector_store is a VectorStoreBase implementation
        return DocumentRetriever(self.vector_store, k=k or self.k)


# Example usage
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    print("=" * 60)
    print("Chat Service - Centralized RAG Logic")
    print("=" * 60)
    
    # Initialize service
    service = ChatService(
        vector_store_type=VectorStoreType.FAISS,
        faiss_index_path="faiss_index"
    )
    
    print("\n1. Initializing service...")
    service.initialize()
    print("✅ Service initialized successfully")
    
    # Health check
    print("\n2. Health check...")
    health = service.health_check()
    print(f"Status: {health['status']}")
    print(f"Config: {health['config']}")
    
    # Test query
    print("\n3. Testing query...")
    query = "What are the company benefits?"
    print(f"Query: {query}")
    
    result = service.get_answer(query)
    print(f"\nAnswer: {result['answer']}")
    print(f"\nSources ({len(result['sources'])}):")
    for i, doc in enumerate(result['sources'], 1):
        print(f"  {i}. {doc['source']} (page {doc['page']})")
    
    # Test streaming
    print("\n4. Testing streaming...")
    print("Answer (streaming): ", end="", flush=True)
    for chunk in service.get_answer_stream(query):
        print(chunk, end="", flush=True)
    print("\n")
    
    print("✅ All tests passed!")
