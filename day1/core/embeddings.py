"""
Embedding Manager Module

This module provides a wrapper around OpenAI embeddings for generating
vector representations of text. It handles model configuration, batching,
and error handling.

Usage:
    from core.embeddings import EmbeddingManager
    
    manager = EmbeddingManager(model="text-embedding-3-small")
    vectors = manager.embed_documents(["Hello world", "Another doc"])
    query_vector = manager.embed_query("Search query")
"""

import os
from typing import List
from langchain_openai import OpenAIEmbeddings


class EmbeddingManager:
    """
    Manages OpenAI embedding model for text-to-vector conversion.
    
    This class provides a clean interface for generating embeddings
    with proper configuration and error handling.
    
    Attributes:
        model (str): OpenAI embedding model name
        embeddings (OpenAIEmbeddings): LangChain embeddings instance
        dimension (int): Vector dimension size
    """
    
    def __init__(
        self,
        model: str = None,
        api_key: str = None
    ):
        """
        Initialize the embedding manager.
        
        Args:
            model: OpenAI embedding model name (uses env var if None)
                  Options:
                  - text-embedding-3-small (1536 dims, cost-effective)
                  - text-embedding-3-large (3072 dims, higher quality)
            api_key: OpenAI API key (uses env var if None)
        """
        # Get model from environment if not provided
        if model is None:
            model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        
        self.model = model
        
        # Get dimension from environment or determine from model name
        dimension_env = os.getenv("EMBEDDING_DIMENSIONS")
        if dimension_env:
            self.dimension = int(dimension_env)
        else:
            self.dimension = 3072 if "large" in model else 1536
        
        # Initialize OpenAI embeddings
        self.embeddings = OpenAIEmbeddings(
            model=model,
            openai_api_key=api_key or os.getenv("OPENAI_API_KEY")
        )
        
        print(f"✅ Embedding Manager initialized")
        print(f"   Model: {model}")
        print(f"   Dimension: {self.dimension}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple documents.
        
        Args:
            texts: List of text strings to embed
        
        Returns:
            List of embedding vectors (each is a list of floats)
        
        Example:
            >>> manager = EmbeddingManager()
            >>> vectors = manager.embed_documents(["doc1", "doc2"])
            >>> len(vectors)  # 2
            >>> len(vectors[0])  # 1536 (for small model)
        """
        try:
            return self.embeddings.embed_documents(texts)
        except Exception as e:
            print(f"❌ Error embedding documents: {e}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a single query.
        
        This uses a slightly different prompt than embed_documents
        to optimize for search queries.
        
        Args:
            text: Query text to embed
        
        Returns:
            Embedding vector (list of floats)
        
        Example:
            >>> manager = EmbeddingManager()
            >>> vector = manager.embed_query("What are benefits?")
            >>> len(vector)  # 1536 (for small model)
        """
        try:
            return self.embeddings.embed_query(text)
        except Exception as e:
            print(f"❌ Error embedding query: {e}")
            raise
    
    def get_model_info(self) -> dict:
        """
        Get information about the current embedding model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model": self.model,
            "dimension": self.dimension,
            "provider": "OpenAI"
        }


if __name__ == "__main__":
    """
    Test the embedding manager.
    """
    from dotenv import load_dotenv
    load_dotenv()
    
    print("=" * 70)
    print("Testing Embedding Manager")
    print("=" * 70)
    print()
    
    # Initialize manager
    manager = EmbeddingManager()
    print()
    
    # Test document embedding
    print("Testing document embedding...")
    docs = ["This is a test document", "Another test document"]
    vectors = manager.embed_documents(docs)
    print(f"✅ Generated {len(vectors)} document embeddings")
    print(f"   Vector dimension: {len(vectors[0])}")
    print()
    
    # Test query embedding
    print("Testing query embedding...")
    query = "test query"
    vector = manager.embed_query(query)
    print(f"✅ Generated query embedding")
    print(f"   Vector dimension: {len(vector)}")
    print()
    
    # Show model info
    print("Model Information:")
    info = manager.get_model_info()
    for key, value in info.items():
        print(f"   {key}: {value}")
