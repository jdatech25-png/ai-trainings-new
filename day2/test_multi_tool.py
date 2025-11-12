"""
Quick test script to verify multi-tool agent behavior
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate

load_dotenv()

@tool
def calculator(expression: str) -> str:
    """Useful for math calculations. Input should be a valid Python expression."""
    try:
        result = eval(expression)
        return f"The result is {result}"
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def get_word_length(word: str) -> str:
    """Returns the length of a word. Input should be a word or phrase."""
    return f"The word '{word}' has {len(word)} characters"

@tool
def double_number(number: str) -> str:
    """Doubles a number. Input should be a number."""
    try:
        num = float(number)
        result = num * 2
        return f"Double of {number} is {result}"
    except:
        return "Error: Invalid number"

# Setup
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
tools = [calculator, get_word_length, double_number]

react_prompt = PromptTemplate.from_template("""
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}
""")

agent = create_react_agent(llm, tools, react_prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=10,
    handle_parsing_errors=True
)

# Test questions that require multiple tools
test_questions = [
    "Calculate 5 + 7, then double that result",
    "How many letters are in 'hello' and what is that number times 3?",
    "What is 10 + 5, and how many characters are in the word 'answer'?"
]

print("=" * 70)
print("Testing Multi-Tool Agent")
print("=" * 70)
print()

for i, question in enumerate(test_questions, 1):
    print(f"\n{'=' * 70}")
    print(f"Test {i}: {question}")
    print("=" * 70)
    
    try:
        result = agent_executor.invoke({"input": question})
        print(f"\n✅ Final Answer: {result['output']}\n")
    except Exception as e:
        print(f"❌ Error: {e}\n")
    
    if i < len(test_questions):
        input("\nPress Enter for next test...\n")

print("\n" + "=" * 70)
print("Testing complete!")
print("=" * 70)
