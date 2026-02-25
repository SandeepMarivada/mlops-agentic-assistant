"""
Module 10 + 12: LLM Agent with LangGraph
Problem solved: Chatbots fail to reason or take actions autonomously.
Solution: LangGraph agent with tool use enables step-by-step reasoning and action.

Flow:
User message → LLM decides to call tool or respond
    → If tool: execute tool → feed result back to LLM
    → If no tool: return final answer
"""

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from config import MODEL_NAME, OPENROUTER_API_KEY, OPENROUTER_BASE_URL
from tools import calculate, search_mlops_docs

# ── State ────────────────────────────────────────────────────────────────────
# AgentState holds the conversation messages.
# Annotated[list, operator.add] means new messages are appended, not replaced.

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]


# ── LLM setup ────────────────────────────────────────────────────────────────
# OpenRouter is OpenAI-compatible, so we use ChatOpenAI with a custom base_url.

tools = [search_mlops_docs, calculate]

llm = ChatOpenAI(
    model=MODEL_NAME,
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL,
).bind_tools(tools)   # bind_tools tells the LLM about available tools


# ── Nodes ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an MLOps expert assistant with access to an MLOps knowledge base.

Your capabilities:
1. Answer questions about MLOps using the search_mlops_docs tool
2. Perform calculations using the calculate tool
3. Explain concepts clearly with examples

Always search the knowledge base before answering technical MLOps questions.
Be concise, accurate, and educational."""


def call_llm(state: AgentState) -> dict:
    """LLM node: Decide whether to call a tool or respond directly."""
    system = SystemMessage(content=SYSTEM_PROMPT)
    messages = [system] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


# ── Graph ────────────────────────────────────────────────────────────────────
# LangGraph StateGraph defines the agent as a directed graph:
#   llm → (if tool call) → tools → llm → ...
#   llm → (if no tool) → END

def create_agent():
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("llm", call_llm)
    graph.add_node("tools", ToolNode(tools))

    # Set entry point
    graph.set_entry_point("llm")

    # Conditional edge: if LLM called a tool → go to tools node, else → END
    graph.add_conditional_edges("llm", tools_condition)

    # After tools execute, always go back to LLM
    graph.add_edge("tools", "llm")

    return graph.compile()


# Compiled agent - ready to use
agent = create_agent()
