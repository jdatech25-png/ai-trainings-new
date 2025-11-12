"""
Document Loader Utility Module

This module provides utilities for loading and processing PDF documents
for use in RAG (Retrieval Augmented Generation) applications.

Key Features:
- Load PDF files from a local directory
- Extract text content from PDFs
- Split documents into manageable chunks for embeddings
- Preserve document metadata (source file path)

Usage Example:
    # Example usage
    from core.document_loader import load_local_pdfs
    
    docs = load_local_pdfs("docs/")
    print(f"Loaded {len(docs)} document chunks")
"""

from pathlib import Path
from typing import List, Dict
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_local_pdfs(
    folder_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Dict[str, str]]:
    """
    Load and split local PDF files into text chunks.
    
    This function reads all PDF files from the specified folder, extracts their
    text content, and splits them into smaller chunks suitable for embedding
    and vector storage. Each chunk maintains a reference to its source file.
    
    Args:
        folder_path (str): Path to the folder containing PDF files
        chunk_size (int): Maximum size of each text chunk in characters.
                         Default is 1000. Larger chunks provide more context
                         but may exceed embedding model token limits.
        chunk_overlap (int): Number of characters to overlap between chunks.
                            Default is 200. Overlap helps maintain context
                            across chunk boundaries.
    
    Returns:
        List[Dict[str, str]]: List of dictionaries, each containing:
            - 'content': The text content of the chunk
            - 'source': Path to the source PDF file
    
    Example:
        >>> docs = load_local_pdfs("docs/", chunk_size=500, chunk_overlap=100)
        >>> print(f"Loaded {len(docs)} chunks")
        >>> print(f"First chunk: {docs[0]['content'][:100]}...")
    
    Notes:
        - Only processes files with .pdf extension
        - Skips pages that cannot be extracted
        - Empty PDFs or pages are handled gracefully
        - File paths are preserved as absolute paths for traceability
    """
    # Initialize text splitter with specified parameters
    # RecursiveCharacterTextSplitter attempts to split on natural boundaries
    # (paragraphs, sentences) rather than arbitrary character positions
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,  # Use character count for chunk size
        separators=["\n\n", "\n", " ", ""]  # Try these separators in order
    )
    
    docs = []
    pdf_files = list(Path(folder_path).glob("*.pdf"))
    
    if not pdf_files:
        print(f"⚠️  Warning: No PDF files found in {folder_path}")
        return docs
    
    print(f"📚 Found {len(pdf_files)} PDF file(s) to process")
    
    # Process each PDF file
    for pdf_file in pdf_files:
        try:
            print(f"   📄 Processing: {pdf_file.name}...")
            
            # Read PDF using PyPDF2
            reader = PdfReader(str(pdf_file))
            num_pages = len(reader.pages)
            
            # Extract text from all pages
            text_parts = []
            for page_num, page in enumerate(reader.pages, 1):
                try:
                    page_text = page.extract_text()
                    if page_text:  # Only add non-empty pages
                        text_parts.append(page_text)
                except Exception as e:
                    print(f"      ⚠️  Warning: Could not extract page {page_num}: {e}")
            
            # Combine all pages into single text
            full_text = "\n".join(text_parts)
            
            if not full_text.strip():
                print(f"      ⚠️  Warning: No text extracted from {pdf_file.name}")
                continue
            
            # Split text into chunks
            chunks = text_splitter.split_text(full_text)
            
            # Create document dictionaries with metadata
            for i, chunk in enumerate(chunks, 1):
                docs.append({
                    "content": chunk,
                    "source": str(pdf_file.absolute()),
                    "chunk_id": i,
                    "total_chunks": len(chunks),
                    "filename": pdf_file.name
                })
            
            print(f"      ✅ Extracted {len(chunks)} chunks from {num_pages} pages")
            
        except Exception as e:
            print(f"      ❌ Error processing {pdf_file.name}: {e}")
            continue
    
    print(f"\n✅ Total chunks created: {len(docs)}")
    return docs


def get_document_stats(docs: List[Dict[str, str]]) -> Dict:
    """
    Calculate statistics about loaded documents.
    
    Useful for understanding the dataset size and distribution
    before building vector stores.
    
    Args:
        docs: List of document chunks from load_local_pdfs()
    
    Returns:
        Dictionary containing:
            - total_chunks: Total number of chunks
            - unique_sources: Number of unique source files
            - avg_chunk_length: Average characters per chunk
            - total_characters: Total characters across all chunks
    
    Example:
        >>> docs = load_local_pdfs("docs/")
        >>> stats = get_document_stats(docs)
        >>> print(f"Average chunk size: {stats['avg_chunk_length']} characters")
    """
    if not docs:
        return {
            "total_chunks": 0,
            "unique_sources": 0,
            "avg_chunk_length": 0,
            "total_characters": 0
        }
    
    total_chars = sum(len(doc["content"]) for doc in docs)
    unique_sources = len(set(doc["source"] for doc in docs))
    
    return {
        "total_chunks": len(docs),
        "unique_sources": unique_sources,
        "avg_chunk_length": total_chars // len(docs),
        "total_characters": total_chars
    }


if __name__ == "__main__":
    """
    Test the document loader with sample PDFs.
    This section runs when the module is executed directly.
    """
    print("=" * 60)
    print("Testing Document Loader")
    print("=" * 60)
    print()
    
    # Test with default parameters
    docs = load_local_pdfs("docs/")
    
    if docs:
        # Display statistics
        stats = get_document_stats(docs)
        print("\n" + "=" * 60)
        print("Document Statistics")
        print("=" * 60)
        print(f"Total chunks: {stats['total_chunks']}")
        print(f"Unique sources: {stats['unique_sources']}")
        print(f"Average chunk length: {stats['avg_chunk_length']} characters")
        print(f"Total characters: {stats['total_characters']:,}")
        
        # Display sample chunk
        print("\n" + "=" * 60)
        print("Sample Chunk (First Document)")
        print("=" * 60)
        print(f"Source: {docs[0]['filename']}")
        print(f"Chunk ID: {docs[0]['chunk_id']}/{docs[0]['total_chunks']}")
        print(f"\nContent Preview:")
        print("-" * 60)
        print(docs[0]['content'][:300] + "...")
    else:
        print("\n❌ No documents loaded. Please add PDF files to the 'docs/' folder.")
