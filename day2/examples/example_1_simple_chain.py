"""
Example 1: Simple LLMChain - Understanding Chain Basics

This example demonstrates:
1. Basic LLMChain composition with LangChain Expression Language (LCEL)
2. PromptTemplate → LLM → OutputParser pattern
3. Sequential chains for multi-step processing
4. Input/output handling and structured responses

What is a Chain?
- Chains are composable units that link prompts, models, and processing steps
- They make workflows repeatable, testable, and maintainable
- LCEL uses the `|` operator to chain components together

Chain Pattern:
┌──────────────┐     ┌─────────┐     ┌──────────────┐     ┌────────┐
│ PromptTemplate│────▶│   LLM   │────▶│ OutputParser │────▶│ Result │
└──────────────┘     └─────────┘     └──────────────┘     └────────┘

Prerequisites:
- Set OPENAI_API_KEY in .env file

Usage:
    python examples/example_1_simple_chain.py          # Interactive mode
    python examples/example_1_simple_chain.py --demo   # Demo mode
"""

import os
import sys
from dotenv import load_dotenv
from typing import List, Dict

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

# Load environment variables
load_dotenv()


def demo_basic_chain():
    """
    Demonstrate the simplest possible chain: Prompt → LLM → Parser
    
    This is the fundamental pattern for all LangChain workflows.
    """
    print("=" * 70)
    print("Demo 1: Basic Chain Pattern")
    print("=" * 70)
    print()
    
    # Initialize LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Create a prompt template
    prompt = ChatPromptTemplate.from_template(
        "Tell me a {length} joke about {topic}."
    )
    
    # Create output parser
    output_parser = StrOutputParser()
    
    # Compose the chain using LCEL (LangChain Expression Language)
    chain = prompt | llm | output_parser
    
    # Invoke the chain
    print("🔗 Chain: Prompt → LLM → Parser")
    print()
    print("Input: {'length': 'short', 'topic': 'programming'}")
    print()
    print("Output:")
    result = chain.invoke({"length": "short", "topic": "programming"})
    print(result)
    print()
    
    print("💡 Key Concept:")
    print("   The `|` operator chains components together.")
    print("   Data flows left to right through the chain.")
    print()


def demo_streaming_chain():
    """
    Demonstrate streaming output from a chain.
    
    Streaming improves UX by showing results as they're generated.
    """
    print("=" * 70)
    print("Demo 2: Streaming Chain")
    print("=" * 70)
    print()
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    prompt = ChatPromptTemplate.from_template(
        "Write a {length} story about {topic}."
    )
    output_parser = StrOutputParser()
    
    chain = prompt | llm | output_parser
    
    print("🔗 Streaming Chain")
    print()
    print("Input: {'length': '3-sentence', 'topic': 'a robot learning to code'}")
    print()
    print("Output (streaming):")
    
    # Stream the output
    for chunk in chain.stream({"length": "3-sentence", "topic": "a robot learning to code"}):
        print(chunk, end="", flush=True)
    
    print("\n")
    print("💡 Key Concept:")
    print("   Use .stream() instead of .invoke() for streaming output.")
    print("   Great for chat interfaces and long responses.")
    print()


def demo_structured_output_chain():
    """
    Demonstrate parsing structured output (JSON) from LLM.
    
    This is useful when you need machine-readable data from the LLM.
    """
    print("=" * 70)
    print("Demo 3: Structured Output Chain (JSON)")
    print("=" * 70)
    print()
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Prompt that requests JSON output
    prompt = ChatPromptTemplate.from_template(
        """Analyze the sentiment of this text and return a JSON object with:
        - sentiment: "positive", "negative", or "neutral"
        - confidence: a number between 0 and 1
        - key_phrases: list of important phrases (max 3)
        
        Text: {text}
        
        Return only valid JSON, nothing else."""
    )
    
    # JSON output parser
    output_parser = JsonOutputParser()
    
    chain = prompt | llm | output_parser
    
    print("🔗 Chain: Prompt → LLM → JSON Parser")
    print()
    
    test_text = "I absolutely love this product! It's amazing and works perfectly."
    print(f"Input text: '{test_text}'")
    print()
    
    result = chain.invoke({"text": test_text})
    
    print("Output (parsed JSON):")
    import json
    print(json.dumps(result, indent=2))
    print()
    
    print("💡 Key Concept:")
    print("   JsonOutputParser automatically extracts JSON from LLM response.")
    print("   Useful for structured data extraction and API integration.")
    print()


def demo_sequential_chain():
    """
    Demonstrate a sequential chain with multiple steps.
    
    This shows how to build multi-step workflows where output of one
    step becomes input to the next.
    """
    print("=" * 70)
    print("Demo 4: Sequential Chain (Multi-Step)")
    print("=" * 70)
    print()
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    
    # Step 1: Generate a topic
    topic_prompt = ChatPromptTemplate.from_template(
        "Suggest a creative topic for a {category} blog post. "
        "Return only the topic title, nothing else."
    )
    
    # Step 2: Generate an outline based on the topic
    outline_prompt = ChatPromptTemplate.from_template(
        "Create a 3-point outline for a blog post titled: {topic}"
    )
    
    output_parser = StrOutputParser()
    
    # Create sequential chain
    # Step 1: Get topic
    topic_chain = topic_prompt | llm | output_parser
    
    # Step 2: Get outline (takes output from step 1)
    outline_chain = outline_prompt | llm | output_parser
    
    print("🔗 Sequential Chain:")
    print("   Step 1: Generate topic")
    print("   Step 2: Create outline from topic")
    print()
    
    category = "technology"
    print(f"Input: category = '{category}'")
    print()
    
    # Execute step 1
    print("Step 1 - Generated Topic:")
    topic = topic_chain.invoke({"category": category})
    print(f"  {topic}")
    print()
    
    # Execute step 2 with output from step 1
    print("Step 2 - Generated Outline:")
    outline = outline_chain.invoke({"topic": topic})
    print(outline)
    print()
    
    print("💡 Key Concept:")
    print("   Sequential chains pass data from one step to the next.")
    print("   Each step can transform the data for the next step.")
    print()


def demo_parallel_chain():
    """
    Demonstrate parallel chain execution.
    
    Multiple chains can run in parallel when they don't depend on each other.
    """
    print("=" * 70)
    print("Demo 5: Parallel Chain Execution")
    print("=" * 70)
    print()
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    
    # Create multiple independent chains
    pros_prompt = ChatPromptTemplate.from_template(
        "List 2 pros of {topic}. Be concise."
    )
    cons_prompt = ChatPromptTemplate.from_template(
        "List 2 cons of {topic}. Be concise."
    )
    summary_prompt = ChatPromptTemplate.from_template(
        "Write a 1-sentence summary of {topic}."
    )
    
    output_parser = StrOutputParser()
    
    # Create parallel chains using RunnableParallel
    parallel_chain = RunnableParallel(
        pros=pros_prompt | llm | output_parser,
        cons=cons_prompt | llm | output_parser,
        summary=summary_prompt | llm | output_parser
    )
    
    print("🔗 Parallel Chain:")
    print("   • Generate pros (parallel)")
    print("   • Generate cons (parallel)")
    print("   • Generate summary (parallel)")
    print()
    
    topic = "remote work"
    print(f"Input: topic = '{topic}'")
    print()
    
    result = parallel_chain.invoke({"topic": topic})
    
    print("Results:")
    print(f"\nSummary:\n{result['summary']}")
    print(f"\nPros:\n{result['pros']}")
    print(f"\nCons:\n{result['cons']}")
    print()
    
    print("💡 Key Concept:")
    print("   RunnableParallel executes multiple chains simultaneously.")
    print("   Faster than sequential when chains are independent.")
    print()


def interactive_chain_builder():
    """
    Interactive mode: Let users build and test chains.
    """
    print("=" * 70)
    print("Interactive Chain Builder")
    print("=" * 70)
    print()

    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0.7)
    output_parser = StrOutputParser()
    
    print("Build custom chains by entering prompts!")
    print()
    print("Commands:")
    print("  • Type your prompt (use {variable_name} for variables)")
    print("  • 'examples' - Show example prompts")
    print("  • 'quit' - Exit")
    print()
    
    while True:
        print("─" * 70)
        prompt_text = input("Enter prompt template (or 'quit'): ").strip()
        
        if not prompt_text:
            continue
        
        if prompt_text.lower() == 'quit':
            print("👋 Goodbye!")
            break
        
        if prompt_text.lower() == 'examples':
            print("\n📝 Example Prompts:")
            print("  • Translate {text} to {language}")
            print("  • Explain {concept} in simple terms")
            print("  • Write a {length} {genre} story")
            print("  • Summarize this in {style} style: {content}")
            print()
            continue
        
        # Extract variables from prompt
        import re
        variables = re.findall(r'\{(\w+)\}', prompt_text)
        
        if not variables:
            print("⚠️  No variables found. Use {variable_name} syntax.")
            continue
        
        print(f"\n📋 Variables found: {', '.join(variables)}")
        print()
        
        # Get values for variables
        inputs = {}
        for var in variables:
            value = input(f"  Enter value for '{var}': ").strip()
            inputs[var] = value
        
        # Create and execute chain
        prompt = ChatPromptTemplate.from_template(prompt_text)
        chain = prompt | llm | output_parser
        
        print()
        print("🤖 Generating response...")
        print()
        
        try:
            # Stream the response
            for chunk in chain.stream(inputs):
                print(chunk, end="", flush=True)
            print("\n")
        except Exception as e:
            print(f"❌ Error: {e}\n")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="LangChain Simple Chain Examples")
    parser.add_argument("--demo", action="store_true", help="Run demo mode")
    args = parser.parse_args()
    
    print("\n🚀 Example 1: Simple LLMChain\n")
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found in environment variables")
        print("   Please add it to your .env file")
        sys.exit(1)
    
    if args.demo:
        # Run all demos
        demo_basic_chain()
        input("Press Enter to continue...\n")
        
        demo_streaming_chain()
        input("Press Enter to continue...\n")
        
        demo_structured_output_chain()
        input("Press Enter to continue...\n")
        
        demo_sequential_chain()
        input("Press Enter to continue...\n")
        
        demo_parallel_chain()
        
        print("=" * 70)
        print("✅ All demos completed!")
        print()
        print("💡 Key Takeaways:")
        print("   • Chains compose prompts, LLMs, and parsers")
        print("   • Use `|` operator (LCEL) to chain components")
        print("   • .invoke() for single execution, .stream() for streaming")
        print("   • Chains can be sequential or parallel")
        print("   • OutputParser extracts structured data")
        print()
        print("📚 Next: Run example_2_memory.py to add conversational memory!")
    else:
        # Interactive mode
        interactive_chain_builder()


if __name__ == "__main__":
    main()
