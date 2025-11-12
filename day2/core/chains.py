"""
Chains Module - Reusable LangChain Chain Patterns

This module provides ready-to-use chain patterns for common tasks:
- Basic LLM chains
- Sequential chains
- RAG chains with memory
- Tool-enabled chains

These patterns can be imported and customized for your specific use cases.
"""

from typing import List, Dict, Any, Optional
import warnings

# Suppress LangChain deprecation warnings for training purposes
warnings.filterwarnings("ignore", message=".*LangChain.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_classic.chains import LLMChain
from langchain_classic.memory import ConversationBufferMemory


class ChainFactory:
    """
    Factory class for creating common chain patterns.
    
    Usage:
        factory = ChainFactory(model="gpt-4o-mini")
        chain = factory.create_basic_chain("Translate {text} to {language}")
        result = chain.invoke({"text": "Hello", "language": "French"})
    """
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        api_key: Optional[str] = None
    ):
        """
        Initialize the chain factory.
        
        Args:
            model: OpenAI model name
            temperature: LLM temperature (0-2)
            api_key: OpenAI API key (uses env var if None)
        """
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            openai_api_key=api_key
        )
        self.model = model
        self.temperature = temperature
    
    def create_basic_chain(self, template: str):
        """
        Create a basic chain: Prompt → LLM → Parser
        
        Args:
            template: Prompt template string with {variable} placeholders
            
        Returns:
            Runnable chain
            
        Example:
            chain = factory.create_basic_chain("Explain {concept} in simple terms")
            result = chain.invoke({"concept": "quantum physics"})
        """
        prompt = ChatPromptTemplate.from_template(template)
        output_parser = StrOutputParser()
        
        chain = prompt | self.llm | output_parser
        return chain
    
    def create_chain_with_system_prompt(
        self,
        system_prompt: str,
        user_template: str
    ):
        """
        Create a chain with separate system and user prompts.
        
        Args:
            system_prompt: System instruction for the AI
            user_template: User message template
            
        Returns:
            Runnable chain
            
        Example:
            chain = factory.create_chain_with_system_prompt(
                system_prompt="You are a helpful Python tutor",
                user_template="Explain {concept}"
            )
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", user_template)
        ])
        output_parser = StrOutputParser()
        
        chain = prompt | self.llm | output_parser
        return chain
    
    def create_json_output_chain(self, template: str):
        """
        Create a chain that returns structured JSON output.
        
        Args:
            template: Prompt template that requests JSON output
            
        Returns:
            Chain that outputs parsed JSON
            
        Example:
            chain = factory.create_json_output_chain(
                "Analyze sentiment of: {text}. Return JSON with 'sentiment' and 'confidence'"
            )
        """
        prompt = ChatPromptTemplate.from_template(template)
        output_parser = JsonOutputParser()
        
        chain = prompt | self.llm | output_parser
        return chain
    
    def create_sequential_chain(self, steps: List[Dict[str, str]]):
        """
        Create a multi-step sequential chain.
        
        Args:
            steps: List of step definitions, each with 'prompt' and 'output_key'
            
        Returns:
            Sequential chain
            
        Example:
            chain = factory.create_sequential_chain([
                {"prompt": "Generate a topic about {subject}", "output_key": "topic"},
                {"prompt": "Write a summary of: {topic}", "output_key": "summary"}
            ])
        """
        # This is a simplified example - full implementation would be more complex
        chains = []
        for step in steps:
            prompt = ChatPromptTemplate.from_template(step['prompt'])
            chain = prompt | self.llm | StrOutputParser()
            chains.append(chain)
        
        return chains  # Caller can chain them together
    
    def create_rag_chain(self, vector_store, k: int = 3):
        """
        Create a RAG (Retrieval Augmented Generation) chain.
        
        Args:
            vector_store: LangChain vector store instance
            k: Number of documents to retrieve
            
        Returns:
            RAG chain
            
        Example:
            from langchain_community.vectorstores import FAISS
            vector_store = FAISS.load_local(...)
            chain = factory.create_rag_chain(vector_store, k=3)
            result = chain.invoke({"question": "What are the benefits?"})
        """
        # RAG prompt template
        template = """You are a helpful AI assistant. Use the following context to answer the question.
If you cannot answer based on the context, say so.

Context:
{context}

Question: {question}

Answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # Create retrieval chain
        chain = (
            {
                "context": lambda x: self._format_docs(
                    vector_store.similarity_search(x["question"], k=k)
                ),
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return chain
    
    @staticmethod
    def _format_docs(docs) -> str:
        """Format documents for RAG context."""
        return "\n\n".join([
            f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}"
            for doc in docs
        ])


class MemoryChainFactory:
    """
    Factory for creating chains with conversational memory.
    
    Usage:
        factory = MemoryChainFactory()
        chain = factory.create_conversational_chain()
        result = chain.predict(input="Hello, my name is Alice")
    """
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        api_key: Optional[str] = None
    ):
        """Initialize the memory chain factory."""
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            openai_api_key=api_key
        )
    
    def create_conversational_chain(
        self,
        system_prompt: str = "You are a helpful AI assistant.",
        memory_key: str = "chat_history"
    ):
        """
        Create a chain with conversation buffer memory.
        
        Args:
            system_prompt: System instruction
            memory_key: Key for memory in the chain
            
        Returns:
            LLMChain with memory
            
        Example:
            chain = factory.create_conversational_chain()
            response1 = chain.predict(input="My name is Alice")
            response2 = chain.predict(input="What's my name?")
        """
        # Create memory
        memory = ConversationBufferMemory(
            memory_key=memory_key,
            return_messages=True
        )
        
        # Create prompt with memory placeholder
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name=memory_key),
            ("human", "{input}")
        ])
        
        # Create chain
        chain = LLMChain(
            llm=self.llm,
            prompt=prompt,
            memory=memory,
            verbose=False
        )
        
        return chain
    
    def create_rag_chain_with_memory(
        self,
        vector_store,
        k: int = 3,
        system_prompt: str = "You are a helpful AI assistant with access to documents."
    ):
        """
        Create a RAG chain with conversational memory.
        
        This combines document retrieval with conversation context.
        
        Args:
            vector_store: LangChain vector store
            k: Number of documents to retrieve
            system_prompt: System instruction
            
        Returns:
            Chain with RAG and memory
        """
        from langchain_classic.memory import ConversationBufferMemory
        
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key="question"
        )
        
        # RAG prompt with memory
        template = """You are a helpful AI assistant with access to documents.

Chat History:
{chat_history}

Retrieved Context:
{context}

Question: {question}

Answer based on the context and our conversation history:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # Note: Full implementation would require custom chain logic
        # This is a simplified structure
        return {
            "vector_store": vector_store,
            "memory": memory,
            "prompt": prompt,
            "llm": self.llm,
            "k": k
        }


# Convenience functions for quick chain creation

def create_simple_qa_chain(
    question_template: str = "Answer this question: {question}",
    model: str = "gpt-4o-mini"
):
    """
    Quick helper to create a simple Q&A chain.
    
    Example:
        chain = create_simple_qa_chain()
        answer = chain.invoke({"question": "What is AI?"})
    """
    factory = ChainFactory(model=model)
    return factory.create_basic_chain(question_template)


def create_translation_chain(model: str = "gpt-4o-mini"):
    """
    Create a translation chain.
    
    Example:
        chain = create_translation_chain()
        result = chain.invoke({"text": "Hello", "language": "Spanish"})
    """
    factory = ChainFactory(model=model)
    return factory.create_basic_chain("Translate {text} to {language}")


def create_summarization_chain(model: str = "gpt-4o-mini"):
    """
    Create a text summarization chain.
    
    Example:
        chain = create_summarization_chain()
        summary = chain.invoke({"text": "Long text here..."})
    """
    factory = ChainFactory(model=model)
    return factory.create_basic_chain(
        "Summarize the following text concisely:\n\n{text}"
    )


# Example usage
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    print("=" * 70)
    print("Chains Module - Demo")
    print("=" * 70)
    print()
    
    # Test basic chain
    factory = ChainFactory(api_key=os.getenv("OPENAI_API_KEY"))
    
    print("1. Basic Chain - Translation")
    chain = create_translation_chain()
    result = chain.invoke({"text": "Good morning", "language": "French"})
    print(f"Result: {result}")
    print()
    
    print("2. Chain with System Prompt")
    chain = factory.create_chain_with_system_prompt(
        system_prompt="You are a Python programming expert",
        user_template="Explain {concept} in one sentence"
    )
    result = chain.invoke({"concept": "list comprehension"})
    print(f"Result: {result}")
    print()
    
    print("3. Conversational Chain with Memory")
    mem_factory = MemoryChainFactory(api_key=os.getenv("OPENAI_API_KEY"))
    chain = mem_factory.create_conversational_chain()
    
    response1 = chain.predict(input="My favorite color is blue")
    print(f"Response 1: {response1}")
    
    response2 = chain.predict(input="What's my favorite color?")
    print(f"Response 2: {response2}")
    print()
    
    print("✅ Chains module demo completed!")
