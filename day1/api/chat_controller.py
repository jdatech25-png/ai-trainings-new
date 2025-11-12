"""
Chat Controller - FastAPI REST Endpoints

This module provides REST API endpoints for the chat service.
All business logic is handled by ChatService - this controller
just handles HTTP requests/responses.

Endpoints:
- POST /chat - Send a query and get an answer
- POST /chat/stream - Send a query and get a streaming answer
- GET /health - Check service health
- GET /config - Get current configuration
- POST /config - Update configuration
- POST /documents - Retrieve relevant documents

Usage:
    # Start the server
    uvicorn api.chat_controller:app --reload
    
    # Or use the startup script
    python scripts/start_api.py
"""

import sys
import os
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.chat_service import ChatService, VectorStoreType
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# Request/Response Models
class ChatRequest(BaseModel):
    """Chat request model"""
    query: str = Field(..., description="User question", min_length=1)
    k: Optional[int] = Field(None, description="Number of documents to retrieve", ge=1, le=10)
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are the company benefits?",
                "k": 3
            }
        }


class ChatResponse(BaseModel):
    """Chat response model"""
    answer: str = Field(..., description="Generated answer")
    sources: List[Dict[str, Any]] = Field(..., description="Source documents")
    model: str = Field(..., description="LLM model used")
    vector_store: str = Field(..., description="Vector store type")
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "The company offers comprehensive health insurance...",
                "sources": [
                    {
                        "content": "Health insurance coverage includes...",
                        "source": "company_handbook.pdf",
                        "page": 5
                    }
                ],
                "model": "gpt-4o-mini",
                "vector_store": "faiss"
            }
        }


class DocumentRequest(BaseModel):
    """Document retrieval request"""
    query: str = Field(..., description="Search query", min_length=1)
    k: Optional[int] = Field(3, description="Number of documents", ge=1, le=10)


class ConfigUpdate(BaseModel):
    """Configuration update model"""
    temperature: Optional[float] = Field(None, ge=0, le=2)
    k: Optional[int] = Field(None, ge=1, le=10)
    llm_model: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "temperature": 0.7,
                "k": 5,
                "llm_model": "gpt-4"
            }
        }


class InsertDocumentsRequest(BaseModel):
    """Request model for inserting documents into Milvus"""
    texts: List[str] = Field(..., description="List of text documents to insert", min_items=1)
    
    class Config:
        json_schema_extra = {
            "example": {
                "texts": [
                    "The company offers 20 days of vacation per year.",
                    "Health insurance includes dental and vision coverage."
                ]
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    message: str
    config: Optional[Dict[str, Any]] = None
    vector_store_docs: Optional[bool] = None


# Global chat service instance
chat_service: Optional[ChatService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Initializes service on startup, cleans up on shutdown.
    """
    global chat_service
    
    # Startup
    print("🚀 Initializing Chat Service...")
    
    # Get configuration from environment
    vector_store_type = os.getenv("VECTOR_STORE_TYPE", "faiss")
    faiss_index_path = os.getenv("FAISS_INDEX_PATH", "faiss_index")
    milvus_uri = os.getenv("MILVUS_URI", "tcp://localhost:19530")
    milvus_token = os.getenv("MILVUS_TOKEN", None)
    milvus_collection = os.getenv("MILVUS_COLLECTION_NAME", "training_demo")
    
    try:
        chat_service = ChatService(
            vector_store_type=VectorStoreType(vector_store_type),
            faiss_index_path=faiss_index_path,
            milvus_uri=milvus_uri,
            milvus_token=milvus_token,
            milvus_collection=milvus_collection
        )
        chat_service.initialize()
        print("✅ Chat Service initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize Chat Service: {e}")
        raise
    
    yield
    
    # Shutdown
    print("👋 Shutting down Chat Service...")


# Create FastAPI app
app = FastAPI(
    title="RAG Chat API",
    description="REST API for Retrieval Augmented Generation chat service",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Endpoints
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "name": "RAG Chat API",
        "version": "1.0.0",
        "description": "REST API for Retrieval Augmented Generation",
        "endpoints": {
            "stream": "/chat/stream",
        }
    }


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Get an answer for a query using RAG.
    
    This endpoint:
    1. Retrieves relevant documents from the vector store
    2. Generates an answer using the LLM with retrieved context
    3. Returns the answer along with source documents
    """
    if not chat_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat service not initialized"
        )
    
    try:
        # Update k if provided
        if request.k:
            original_k = chat_service.k
            chat_service.k = request.k
        
        result = chat_service.get_answer(request.query)
        
        # Restore original k
        if request.k:
            chat_service.k = original_k
        
        return ChatResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing chat request: {str(e)}"
        )


@app.post("/chat/stream", tags=["Chat"])
async def chat_stream(request: ChatRequest):
    """
    Get a streaming answer for a query using RAG.
    
    Returns a stream of text chunks as they're generated by the LLM.
    """
    if not chat_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat service not initialized"
        )
    
    try:
        # Update k if provided
        if request.k:
            original_k = chat_service.k
            chat_service.k = request.k
        
        async def generate():
            try:
                for chunk in chat_service.get_answer_stream(request.query):
                    yield chunk
            finally:
                # Restore original k
                if request.k:
                    chat_service.k = original_k
        
        return StreamingResponse(
            generate(),
            media_type="text/plain"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing streaming request: {str(e)}"
        )


@app.post("/documents", tags=["Documents"])
async def retrieve_documents(request: DocumentRequest):
    """
    Retrieve relevant documents for a query.
    
    Returns only the documents without generating an answer.
    Useful for debugging or showing users what context is being used.
    """
    if not chat_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat service not initialized"
        )
    
    try:
        docs = chat_service.retrieve_documents(request.query, k=request.k)
        return {
            "query": request.query,
            "documents": docs,
            "count": len(docs)
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving documents: {str(e)}"
        )


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """
    Check service health status.
    
    Returns information about service status and configuration.
    """
    if not chat_service:
        return HealthResponse(
            status="not_initialized",
            message="Chat service not initialized"
        )
    
    health_status = chat_service.health_check()
    return HealthResponse(**health_status)


@app.get("/config", tags=["Configuration"])
async def get_config():
    """
    Get current service configuration.
    
    Returns settings like model names, temperature, k value, etc.
    """
    if not chat_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat service not initialized"
        )
    
    return chat_service.get_config()


@app.post("/config", tags=["Configuration"])
async def update_config(config: ConfigUpdate):
    """
    Update service configuration dynamically.
    
    Allows changing temperature, k, and LLM model without restarting.
    """
    if not chat_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat service not initialized"
        )
    
    try:
        chat_service.update_config(
            temperature=config.temperature,
            k=config.k,
            llm_model=config.llm_model
        )
        return {
            "message": "Configuration updated successfully",
            "new_config": chat_service.get_config()
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating configuration: {str(e)}"
        )


@app.post("/milvus/insert", tags=["Milvus"])
async def insert_documents_to_milvus(request: InsertDocumentsRequest):
    """
    Insert documents directly into Milvus collection.
    
    Only works when using Milvus as the vector store.
    """
    if not chat_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat service not initialized"
        )
    
    try:
        success = chat_service.insert_documents_to_milvus(request.texts)
        return {
            "success": success,
            "message": f"Inserted {len(request.texts)} documents into Milvus",
            "count": len(request.texts)
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error inserting documents: {str(e)}"
        )


@app.post("/milvus/search", tags=["Milvus"])
async def search_milvus(request: DocumentRequest):
    """
    Direct search on Milvus collection.
    
    Returns raw Milvus search results with distances.
    Only works when using Milvus as the vector store.
    """
    if not chat_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat service not initialized"
        )
    
    try:
        results = chat_service.search_milvus(request.query, k=request.k)
        return {
            "query": request.query,
            "results": results,
            "count": len(results)
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error searching Milvus: {str(e)}"
        )


@app.get("/milvus/stats", tags=["Milvus"])
async def get_milvus_stats():
    """
    Get Milvus collection statistics.
    
    Returns information about the collection including number of documents.
    Only works when using Milvus as the vector store.
    """
    if not chat_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat service not initialized"
        )
    
    try:
        stats = chat_service.get_milvus_stats()
        return stats
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting Milvus stats: {str(e)}"
        )


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {
        "error": "Not Found",
        "message": "The requested endpoint does not exist",
        "available_endpoints": {
            "chat": "/chat",
            "stream": "/chat/stream",
            "documents": "/documents",
            "health": "/health",
            "config": "/config"
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("Starting RAG Chat API Server")
    print("=" * 60)
    print("\nEndpoints:")
    print("  - POST /chat - Get an answer")
    print("  - POST /chat/stream - Get streaming answer")
    print("\nDocs: http://localhost:8000/docs")
    print("=" * 60)
    
    uvicorn.run(
        "api.chat_controller:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
