"""
Answer Generator Module

This module handles LLM-based answer generation using retrieved context.
It provides both streaming and non-streaming response generation with RAG.

Usage:
    from core.generators import AnswerGenerator
    from core.retrievers import DocumentRetriever
    
    generator = AnswerGenerator(model="gpt-4o-mini")
    retriever = DocumentRetriever(store)
    
    # Get relevant documents
    docs = retriever.retrieve("What are benefits?")
    
    # Generate answer
    answer = generator.generate_with_stream(
        query="What are benefits?",
        documents=docs
    )
"""

import os
from typing import List, Iterator
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


class AnswerGenerator:
    """
    Generates answers using LLM with retrieved context (RAG).
    
    This class handles:
    - Prompt construction with context
    - LLM invocation
    - Streaming responses
    - Error handling
    
    Attributes:
        model: LLM model name
        temperature: Creativity level (0-2)
        llm: ChatOpenAI instance
        prompt: RAG prompt template
    """
    
    def __init__(
        self,
        model: str = None,
        temperature: float = 0.0,
        api_key: str = None
    ):
        """
        Initialize the answer generator.
        
        Args:
            model: OpenAI model name (uses env var if None)
                  Options: gpt-4o-mini, gpt-4o, gpt-4-turbo, etc.
            temperature: Controls randomness (0=deterministic, 2=creative)
            api_key: OpenAI API key (uses env var if None)
        """
        # Get model from environment if not provided
        if model is None:
            model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        
        self.model = model
        self.temperature = temperature
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            openai_api_key=api_key or os.getenv("OPENAI_API_KEY"),
            streaming=True  # Enable streaming by default
        )
        
        # Create RAG prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant answering questions based on provided context.

Instructions:
• Use ONLY the information from the context to answer questions
• If the context doesn't contain relevant information, say so clearly
• Be concise but thorough
• Cite specific parts of the context when applicable
• If you're not certain, express appropriate uncertainty
• Do not make up information or use external knowledge"""),
            ("human", """Context from knowledge base:
{context}

Question: {question}

Answer based on the context above:""")
        ])
        
        print(f"🤖 Answer Generator initialized")
        print(f"   Model: {model}")
        print(f"   Temperature: {temperature}")
    
    def format_documents(self, documents: List[Document]) -> str:
        """
        Format documents into context string.
        
        Args:
            documents: List of LangChain Document objects
        
        Returns:
            Formatted context string
        """
        formatted = []
        
        for i, doc in enumerate(documents, 1):
            filename = doc.metadata.get('filename', 'Unknown')
            chunk_info = f"{doc.metadata.get('chunk_id', '?')}/{doc.metadata.get('total_chunks', '?')}"
            formatted.append(
                f"[Document {i} - {filename} (Chunk {chunk_info})]\n{doc.page_content}"
            )
        
        return "\n\n---\n\n".join(formatted)
    
    def generate(self, query: str, documents: List[Document]) -> str:
        """
        Generate answer without streaming (returns complete response).
        
        Args:
            query: User's question
            documents: Retrieved context documents
        
        Returns:
            Complete answer string
        
        Example:
            >>> generator = AnswerGenerator()
            >>> docs = retriever.retrieve("benefits")
            >>> answer = generator.generate("What are benefits?", docs)
            >>> print(answer)
        """
        # Format context
        context = self.format_documents(documents)
        
        # Build chain
        chain = (
            {
                "context": lambda x: context,
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        
        # Generate answer
        answer = chain.invoke(query)
        
        return answer
    
    def generate_with_stream(
        self,
        query: str,
        documents: List[Document],
        print_stream: bool = True
    ) -> Iterator[str]:
        """
        Generate answer with streaming (yields chunks as they arrive).
        
        Args:
            query: User's question
            documents: Retrieved context documents
            print_stream: Whether to print chunks to console
        
        Yields:
            Answer chunks as they are generated
        
        Example:
            >>> generator = AnswerGenerator()
            >>> docs = retriever.retrieve("benefits")
            >>> for chunk in generator.generate_with_stream("What are benefits?", docs):
            ...     print(chunk, end="", flush=True)
        """
        # Format context
        context = self.format_documents(documents)
        
        # Build chain
        chain = (
            {
                "context": lambda x: context,
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        
        # Stream answer
        for chunk in chain.stream(query):
            if print_stream:
                print(chunk, end="", flush=True)
            yield chunk
        
        if print_stream:
            print()  # New line after streaming
    
    def generate_simple(self, query: str, context: str) -> str:
        """
        Generate answer from pre-formatted context string.
        
        Useful when you already have formatted context and don't
        need document formatting.
        
        Args:
            query: User's question
            context: Pre-formatted context string
        
        Returns:
            Complete answer string
        """
        # Build chain with string context
        chain = (
            {
                "context": lambda x: context,
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        
        return chain.invoke(query)
    
    def update_system_prompt(self, new_prompt: str):
        """
        Update the system prompt for answer generation.
        
        Args:
            new_prompt: New system prompt text
        
        Example:
            >>> generator = AnswerGenerator()
            >>> generator.update_system_prompt(
            ...     "You are a technical expert. Use code examples."
            ... )
        """
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", new_prompt),
            ("human", """Context from knowledge base:
{context}

Question: {question}

Answer based on the context above:""")
        ])
        
        print(f"✅ System prompt updated")


if __name__ == "__main__":
    """
    Test the answer generator.
    """
    from dotenv import load_dotenv
    from core.embeddings import EmbeddingManager
    from core.vector_stores import FAISSVectorStore
    from core.retrievers import DocumentRetriever
    
    load_dotenv()
    
    print("=" * 70)
    print("Testing Answer Generator")
    print("=" * 70)
    print()
    
    # Setup components
    embeddings = EmbeddingManager()
    store = FAISSVectorStore(embeddings)
    
    # Add sample documents
    docs = [
        {
            "content": "TechCorp offers 20 days of paid vacation annually. Employees can carry forward up to 5 unused days to the next year.",
            "filename": "handbook.pdf",
            "chunk_id": 1,
            "total_chunks": 3
        },
        {
            "content": "Health insurance coverage includes medical, dental, and vision care for employees and their dependents.",
            "filename": "handbook.pdf",
            "chunk_id": 2,
            "total_chunks": 3
        },
        {
            "content": "The company provides a 401(k) retirement plan with up to 5% matching contribution.",
            "filename": "handbook.pdf",
            "chunk_id": 3,
            "total_chunks": 3
        }
    ]
    
    store.add_documents(docs)
    print()
    
    # Create retriever and generator
    retriever = DocumentRetriever(store, k=2)
    generator = AnswerGenerator(model="gpt-4o-mini", temperature=0)
    print()
    
    # Test question
    query = "What vacation benefits does the company offer?"
    
    print("=" * 70)
    print(f"Question: {query}")
    print("=" * 70)
    print()
    
    # Retrieve context
    retrieved_docs = retriever.retrieve(query)
    print()
    
    # Generate answer with streaming
    print("🤖 Answer (streaming):")
    print("-" * 70)
    
    full_answer = ""
    for chunk in generator.generate_with_stream(query, retrieved_docs, print_stream=True):
        full_answer += chunk
    
    print()
    print("=" * 70)
    print("✅ Answer generation completed!")
    print(f"\nFull answer length: {len(full_answer)} characters")
