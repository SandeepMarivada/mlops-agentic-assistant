"""
Module 12: Agent Tools
Tools give the agent the ability to take actions beyond just generating text.
Each tool is a function the LLM can decide to call.
"""

from langchain.tools import tool
from rag import get_retriever


@tool
def search_mlops_docs(query: str) -> str:
    """
    Search the MLOps knowledge base to answer questions about:
    - MLOps concepts and best practices
    - Tools like MLflow, Kubeflow, KServe, DVC
    - LLM and Agentic AI concepts
    - RAG systems, LangGraph, CrewAI, FastAPI
    Always use this tool before answering MLOps questions.
    """
    retriever = get_retriever()
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant documents found."
    results = "\n\n---\n\n".join([doc.page_content for doc in docs])
    return f"Retrieved context:\n\n{results}"


@tool
def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression.
    Input must be a valid Python math expression.
    Examples: "2 + 2", "100 * 0.95", "2 ** 10"
    """
    import math
    safe_globals = {
        "__builtins__": {},
        "math": math,
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
    }
    try:
        result = eval(expression, safe_globals)
        return f"Result: {result}"
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"
