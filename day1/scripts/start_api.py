"""
Start API Server Script

This script starts the FastAPI server for the RAG Chat service.

Usage:
    python scripts/start_api.py
    
    # Custom host/port
    python scripts/start_api.py --host 0.0.0.0 --port 8080
    
    # Production mode (no reload)
    python scripts/start_api.py --no-reload

The API will be available at:
- API: http://localhost:8000
- Interactive Docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

Endpoints:
- POST /chat - Get an answer for a query
- POST /chat/stream - Get streaming answer
- POST /documents - Retrieve relevant documents
- GET /health - Check service health
- GET /config - Get configuration
- POST /config - Update configuration
"""

import os
import sys
import argparse
import uvicorn
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Start the RAG Chat API server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--no-reload", action="store_true", help="Disable auto-reload")
    parser.add_argument("--log-level", default="info", choices=["critical", "error", "warning", "info", "debug"], help="Log level")
    args = parser.parse_args()
    
    print("=" * 70)
    print("🚀 Starting RAG Chat API Server")
    print("=" * 70)
    print()
    print("Configuration:")
    print(f"  • Host: {args.host}")
    print(f"  • Port: {args.port}")
    print(f"  • Reload: {not args.no_reload}")
    print(f"  • Log Level: {args.log_level}")
    print()
    print("Endpoints:")
    print(f"  • API: http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{args.port}")
    print(f"  • Docs: http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{args.port}/docs")
    print(f"  • ReDoc: http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{args.port}/redoc")
    print()
    print("Vector Store Configuration:")
    
    vector_store_type = os.getenv("VECTOR_STORE_TYPE", "faiss")
    print(f"  • Type: {vector_store_type}")
    
    if vector_store_type == "faiss":
        faiss_path = os.getenv("FAISS_INDEX_PATH", "faiss_index")
        print(f"  • FAISS Index: {faiss_path}")
        if not os.path.exists(faiss_path):
            print(f"\n⚠️  WARNING: FAISS index not found at {faiss_path}")
            print("    Run: python scripts/build_faiss_store.py")
    else:
        milvus_uri = os.getenv("MILVUS_URI", "tcp://localhost:19530")
        milvus_collection = os.getenv("MILVUS_COLLECTION_NAME", "training_demo")
        print(f"  • Milvus URI: {milvus_uri}")
        print(f"  • Collection: {milvus_collection}")
    
    print()
    print("=" * 70)
    print("Press Ctrl+C to stop the server")
    print("=" * 70)
    print()
    
    # Start the server
    uvicorn.run(
        "api.chat_controller:app",
        host=args.host,
        port=args.port,
        reload=not args.no_reload,
        log_level=args.log_level
    )


if __name__ == "__main__":
    main()
