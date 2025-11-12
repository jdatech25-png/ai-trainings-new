# Reference Examples from Day 1

This folder contains RAG (Retrieval Augmented Generation) examples from Day 1 training. These are **optional reference material** for Day 2.

---

## Purpose for Day 2

These examples are kept because:

1. **Background Knowledge**: Understanding RAG helps with Day 2's guided project
2. **Guided Project Dependency**: `example_4_guided_project.py` uses FAISS for document search
3. **Optional Advanced Topics**: Shows production vector database patterns

---

## Files in This Folder

### `rag_faiss_example.py`
- **What**: RAG with FAISS (local vector database)
- **When to use**: Quick reference for how RAG works
- **Prerequisites**: 
  - Run `scripts/build_faiss_store.py` first
  - Set `OPENAI_API_KEY` in `.env`

```bash
# Run demo
python examples/reference_from_day1/rag_faiss_example.py --demo

# Interactive mode
python examples/reference_from_day1/rag_faiss_example.py
```

### `rag_milvus_example.py`
- **What**: RAG with Milvus (production vector database)
- **When to use**: Understanding production-scale vector databases
- **Prerequisites**:
  - Milvus server running
  - Run `scripts/build_milvus_store.py` first
  - Set `OPENAI_API_KEY` and `MILVUS_URI` in `.env`

```bash
# Run demo
python examples/reference_from_day1/rag_milvus_example.py --demo

# Interactive mode
python examples/reference_from_day1/rag_milvus_example.py
```

### `milvus_service_example.py`
- **What**: Using MilvusService utility class
- **When to use**: Advanced Milvus operations and statistics
- **Prerequisites**: Same as rag_milvus_example.py

```bash
# Run demo
python examples/reference_from_day1/milvus_service_example.py --demo

# Interactive mode
python examples/reference_from_day1/milvus_service_example.py
```

---

## Should You Cover These in Day 2 Training?

### Recommended Approach:

**✅ Briefly mention (5-10 mins)**
- "Yesterday we learned RAG - retrieving relevant documents to answer questions"
- "The guided project uses this same RAG pattern with FAISS"
- "These reference files show pure RAG if you need a refresher"

**✅ Focus on Day 2 (90% of time)**
- Example 1: Chains
- Example 2: Memory  
- Example 3: Tools
- Example 4: Guided Project (RAG + Memory + Tools)

**❌ Don't re-teach Day 1**
- Don't spend significant time on pure RAG
- Students already learned this on Day 1
- Just reference it as needed

---

## For Students

### If you missed Day 1 or need a RAG refresher:

1. **Quick option**: Run the guided project and observe how `search_documents` tool works
2. **Detailed option**: Review these reference examples to understand RAG fundamentals
3. **Read**: Day 1 README section on RAG (if available)

### Questions these examples answer:

- How does document retrieval work?
- What's the difference between FAISS and Milvus?
- How do embeddings and similarity search work?
- How to build a vector store?

---

## Integration with Day 2

The guided project (`example_4_guided_project.py`) integrates RAG with new Day 2 concepts:

```python
@tool
def search_documents(query: str) -> str:
    """Search company documents using RAG (from Day 1)"""
    # Uses FAISS vector store
    # Returns relevant document chunks
    pass

# Combined with:
# - Memory (Day 2) - Remember conversation context  
# - Tools (Day 2) - Calculator, weather, etc.
# - Chains (Day 2) - Structured workflows
```

---

## Summary

**These are optional background material.** Focus on Day 2's core examples (1-4), but keep these handy for:
- Students who need RAG clarification
- Understanding the document search tool in the guided project
- Advanced students interested in production vector databases

---

*For Day 2 training flow, see main README.md*
