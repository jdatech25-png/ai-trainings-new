"""
MilvusService: Simplified Singleton Service for Milvus Vector Database

This module provides a basic Milvus service for training purposes with:
- Singleton connection management
- Simple collection with id, text, and vector fields
- Document insertion and retrieval
- Integration with OpenAI embeddings
- Vector search capabilities
- Integration with DocumentRetriever class

Designed for educational purposes - kept simple and straightforward.
"""

import os
import logging
from typing import List, Dict, Any, Optional

from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility,
)
from langchain_openai import OpenAIEmbeddings
from langchain_milvus import Milvus as LangchainMilvus

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MilvusService:
    """
    Simplified singleton service for Milvus vector database.
    
    Features:
    - Maintains single connection to Milvus
    - Simple schema: id (auto), text, vector
    - Document insertion and search
    - Integration with LangChain
    """
    
    _instance = None
    _collection = None
    
    def __new__(cls):
        """Singleton pattern - only one instance allowed"""
        if cls._instance is None:
            logger.info("Creating MilvusService singleton instance")
            cls._instance = super(MilvusService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def connect(
        self,
        uri: str = None,
        token: str = None,
        collection_name: str = "training_demo"
    ):
        """
        Connect to Milvus server.
        
        Args:
            uri: Milvus URI (defaults to env variable)
            token: Milvus token (defaults to env variable)
            collection_name: Name of collection to use
        """
        if self._initialized:
            logger.info("MilvusService already initialized")
            return
        
        self.uri = uri or os.getenv("MILVUS_URI", "tcp://localhost:19530")
        self.token = token or os.getenv("MILVUS_TOKEN", "")
        self.collection_name = collection_name
        
        # Get embedding configuration
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.embedding_dimensions = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))
        
        try:
            logger.info(f"Connecting to Milvus at {self.uri}")
            
            # Connect to Milvus
            connections.connect(
                alias="default",
                uri=self.uri,
                token=self.token
            )
            
            logger.info("✅ Connected to Milvus successfully")
            self._initialized = True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to Milvus: {e}")
            raise
    
    def create_collection(self, collection_name: str = None):
        """
        Create a simple collection with id, text, and vector fields.
        
        Schema:
        - id: INT64, primary key, auto-generated
        - text: VARCHAR, stores document text
        - vector: FLOAT_VECTOR, stores embeddings
        
        Args:
            collection_name: Name for the collection (uses default if None)
        """
        collection_name = collection_name or self.collection_name
        
        # Check if collection already exists
        if utility.has_collection(collection_name):
            logger.info(f"Collection '{collection_name}' already exists")
            self._collection = Collection(name=collection_name)
            self._collection.load()
            return
        
        logger.info(f"Creating collection '{collection_name}'")
        
        # Define simple schema
        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=True
            ),
            FieldSchema(
                name="text",
                dtype=DataType.VARCHAR,
                max_length=65535  # Max text length
            ),
            FieldSchema(
                name="vector",
                dtype=DataType.FLOAT_VECTOR,
                dim=self.embedding_dimensions
            )
        ]
        
        schema = CollectionSchema(
            fields=fields,
            description="Simple training documents collection"
        )
        
        # Create collection
        self._collection = Collection(
            name=collection_name,
            schema=schema
        )
        
        # Create index for vector search
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 128}
        }
        
        self._collection.create_index(
            field_name="vector",
            index_params=index_params
        )
        
        logger.info(f"✅ Collection '{collection_name}' created successfully")
        
        # Load collection into memory
        self._collection.load()
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for texts using OpenAI.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in environment")
        
        embeddings = OpenAIEmbeddings(
            model=self.embedding_model,
            openai_api_key=api_key
        )
        
        return embeddings.embed_documents(texts)
    
    def insert_documents(
        self,
        texts: List[str],
        collection_name: str = None
    ) -> bool:
        """
        Insert documents into Milvus collection.
        
        Args:
            texts: List of text strings to insert
            collection_name: Target collection (uses default if None)
            
        Returns:
            True if successful, False otherwise
        """
        collection_name = collection_name or self.collection_name
        
        try:
            # Ensure collection exists and is loaded
            if self._collection is None or self._collection.name != collection_name:
                self._collection = Collection(name=collection_name)
                self._collection.load()
            
            logger.info(f"Inserting {len(texts)} documents into '{collection_name}'")
            
            # Generate embeddings
            vectors = self.get_embeddings(texts)
            
            # Prepare entities (text and vector only, id is auto-generated)
            entities = [
                texts,     # text field
                vectors    # vector field
            ]
            
            # Insert into collection
            self._collection.insert(entities)
            self._collection.flush()
            
            logger.info(f"✅ Inserted {len(texts)} documents successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to insert documents: {e}")
            return False
    
    def search(
        self,
        query: str,
        k: int = 3,
        collection_name: str = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query: Search query text
            k: Number of results to return
            collection_name: Collection to search (uses default if None)
            
        Returns:
            List of search results with text and distance
        """
        collection_name = collection_name or self.collection_name
        
        try:
            # Ensure collection is loaded
            if self._collection is None or self._collection.name != collection_name:
                self._collection = Collection(name=collection_name)
                self._collection.load()
            
            # Generate query embedding
            query_vector = self.get_embeddings([query])[0]
            
            # Search parameters
            search_params = {
                "metric_type": "L2",
                "params": {"nprobe": 10}
            }
            
            # Perform search
            results = self._collection.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=k,
                output_fields=["text"]
            )
            
            # Format results
            formatted_results = []
            for hits in results:
                for hit in hits:
                    formatted_results.append({
                        "id": hit.id,
                        "text": hit.entity.get("text"),
                        "distance": hit.distance
                    })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            return []
    
    def get_langchain_vectorstore(
        self,
        collection_name: str = None
    ) -> LangchainMilvus:
        """
        Get LangChain-compatible Milvus vectorstore.
        Can be used with DocumentRetriever class.
        
        Args:
            collection_name: Collection to use (uses default if None)
            
        Returns:
            LangChain Milvus vectorstore instance
        """
        collection_name = collection_name or self.collection_name
        
        try:
            # Ensure collection exists
            if not utility.has_collection(collection_name):
                logger.warning(f"Collection '{collection_name}' does not exist")
                return None
            
            # Create OpenAI embeddings
            embeddings = OpenAIEmbeddings(
                model=self.embedding_model,
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
            
            # Create LangChain Milvus vectorstore
            vectorstore = LangchainMilvus(
                embedding_function=embeddings,
                collection_name=collection_name,
                connection_args={
                    "uri": self.uri,
                    "token": self.token
                },
                auto_id=True
            )
            
            logger.info(f"✅ LangChain vectorstore created for '{collection_name}'")
            return vectorstore
            
        except Exception as e:
            logger.error(f"❌ Failed to create LangChain vectorstore: {e}")
            return None
    
    def get_retriever(self, collection_name: str = None, k: int = 3):
        """
        Get a DocumentRetriever instance for this Milvus collection.
        This is the recommended way to retrieve documents.
        
        Args:
            collection_name: Collection to use (uses default if None)
            k: Number of documents to retrieve
            
        Returns:
            DocumentRetriever instance
            
        Example:
            >>> milvus = MilvusService()
            >>> milvus.connect(collection_name="my_docs")
            >>> retriever = milvus.get_retriever(k=5)
            >>> docs = retriever.retrieve("vacation policy")
        """
        from core.retrievers import DocumentRetriever
        from core.vector_stores import MilvusVectorStore
        from core.embeddings import EmbeddingManager
        
        collection_name = collection_name or self.collection_name
        
        # Create embedding manager
        embeddings = EmbeddingManager()
        
        # Connect to existing Milvus collection
        vector_store = MilvusVectorStore.connect(
            embeddings=embeddings,
            collection_name=collection_name,
            uri=self.uri,
            token=self.token
        )
        
        # Create and return DocumentRetriever
        retriever = DocumentRetriever(vector_store, k=k)
        
        logger.info(f"✅ DocumentRetriever created for '{collection_name}' with k={k}")
        return retriever
    
    def get_collection_stats(self, collection_name: str = None) -> Dict[str, Any]:
        """
        Get statistics about a collection.
        
        Args:
            collection_name: Collection to check (uses default if None)
            
        Returns:
            Dictionary with collection statistics
        """
        collection_name = collection_name or self.collection_name
        
        try:
            if not utility.has_collection(collection_name):
                return {
                    "exists": False,
                    "message": f"Collection '{collection_name}' does not exist"
                }
            
            collection = Collection(name=collection_name)
            collection.load()
            
            return {
                "exists": True,
                "name": collection_name,
                "num_entities": collection.num_entities,
                "description": collection.description
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {"exists": False, "error": str(e)}
    
    def disconnect(self):
        """Disconnect from Milvus and cleanup resources"""
        try:
            if self._collection:
                self._collection.release()
                self._collection = None
            
            connections.disconnect(alias="default")
            self._initialized = False
            logger.info("✅ Disconnected from Milvus")
            
        except Exception as e:
            logger.warning(f"Disconnect warning: {e}")


# Example usage
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    print("=" * 60)
    print("Milvus Service - Simple Training Example")
    print("=" * 60)
    
    # Initialize service
    service = MilvusService()
    service.connect(collection_name="training_demo")
    
    # Create collection
    service.create_collection()
    
    # Insert some sample documents
    sample_docs = [
        "The company offers 20 days of vacation per year.",
        "Health insurance includes dental and vision coverage.",
        "Remote work is allowed up to 3 days per week.",
        "Annual performance reviews are conducted in December.",
        "Employee training budget is $2000 per year."
    ]
    
    print("\nInserting sample documents...")
    service.insert_documents(sample_docs)
    
    # Search
    print("\nSearching for: 'vacation policy'")
    results = service.search("vacation policy", k=2)
    
    print("\nSearch Results:")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['text'][:60]}... (distance: {result['distance']:.4f})")
    
    # Get stats
    stats = service.get_collection_stats()
    print(f"\nCollection Stats:")
    print(f"  Documents: {stats.get('num_entities', 0)}")
    
    print("\n✅ Milvus service test completed!")
