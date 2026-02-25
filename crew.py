"""
Module 12: CrewAI Multi-Agent System
Problem solved: Manual orchestration of agent logic is error-prone.
Solution: CrewAI provides structured multi-agent collaboration with defined roles.

vs LangGraph:
- LangGraph  → Single agent, complex state/tool flow, fine-grained control
- CrewAI     → Multiple specialized agents collaborating on a task

Architecture:
  User Question
      ↓
  Researcher Agent  → searches MLOps knowledge base (RAG tool)
      ↓ (findings passed as context)
  Writer Agent      → synthesizes into a clear, structured answer
      ↓
  Final Answer
"""

from crewai import Agent, Crew, LLM, Process, Task
from crewai.tools import BaseTool

from config import OPENROUTER_API_KEY, MODEL_NAME
from rag import get_retriever

# ── LLM for CrewAI ────────────────────────────────────────────────────────────
# CrewAI uses LiteLLM under the hood.
# For OpenRouter, the model format is: "openrouter/<provider>/<model>"

llm = LLM(
    model=f"openrouter/{MODEL_NAME}",      # e.g. openrouter/openai/gpt-4o-mini
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)


# ── Custom Tool ───────────────────────────────────────────────────────────────
# CrewAI tools inherit from BaseTool (different from LangChain @tool decorator)

class MLOpsSearchTool(BaseTool):
    name: str = "MLOps Knowledge Base Search"
    description: str = (
        "Search the MLOps knowledge base for information about "
        "MLOps concepts, tools (MLflow, Kubeflow, KServe, DVC), "
        "LLM agents, RAG systems, LangGraph, CrewAI, and FastAPI. "
        "Input should be a search query string."
    )

    def _run(self, query: str) -> str:
        retriever = get_retriever()
        docs = retriever.invoke(query)
        if not docs:
            return "No relevant documents found in the knowledge base."
        return "\n\n---\n\n".join([doc.page_content for doc in docs])


search_tool = MLOpsSearchTool()


# ── Agents ─────────────────────────────────────────────────────────────────────
# Each agent has a role, goal, and backstory — this shapes how the LLM behaves.

researcher = Agent(
    role="MLOps Research Specialist",
    goal=(
        "Search the MLOps knowledge base thoroughly and gather all relevant "
        "information to answer the user's question accurately."
    ),
    backstory=(
        "You are a senior MLOps engineer with 10+ years of experience building "
        "production ML systems. You are meticulous about finding accurate information "
        "and always use the knowledge base before providing answers."
    ),
    tools=[search_tool],
    llm=llm,
    verbose=True,
)

writer = Agent(
    role="MLOps Technical Writer",
    goal=(
        "Transform research findings into clear, well-structured, "
        "educational technical explanations for MLOps engineers."
    ),
    backstory=(
        "You are a technical writer who has written documentation for major MLOps "
        "platforms. You specialize in making complex concepts understandable, "
        "using examples and clear structure."
    ),
    llm=llm,
    verbose=True,
)


# ── Crew Runner ───────────────────────────────────────────────────────────────

def run_crew(question: str) -> str:
    """
    Run the CrewAI multi-agent system to answer an MLOps question.

    Process:
    1. Researcher searches knowledge base and compiles findings
    2. Writer takes findings and writes a comprehensive answer
    Both run sequentially (Process.sequential).
    """

    # Task 1: Research
    research_task = Task(
        description=(
            f"Search the MLOps knowledge base and gather all relevant information "
            f"to answer this question: '{question}'\n\n"
            f"Use the MLOps Knowledge Base Search tool with multiple queries if needed. "
            f"Compile all key facts, definitions, and examples."
        ),
        expected_output=(
            "Comprehensive research findings including: key definitions, "
            "how it works, problem it solves, and relevant examples from the knowledge base."
        ),
        agent=researcher,
    )

    # Task 2: Write the answer (receives research_task output as context)
    writing_task = Task(
        description=(
            f"Using the research findings provided, write a comprehensive, "
            f"well-structured answer to: '{question}'\n\n"
            f"Format your response with clear sections, bullet points where appropriate, "
            f"and practical examples."
        ),
        expected_output=(
            "A clear, well-structured technical explanation with: "
            "a brief intro, detailed explanation, practical examples, and a summary."
        ),
        agent=writer,
        context=[research_task],   # Writer receives Researcher's output as context
    )

    # Assemble and run the crew
    crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, writing_task],
        process=Process.sequential,   # researcher runs first, then writer
        verbose=True,
    )

    result = crew.kickoff()
    return str(result)
