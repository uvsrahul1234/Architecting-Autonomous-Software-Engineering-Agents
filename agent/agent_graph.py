from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END

# ==========================================
# 1. DEFINE THE AGENT's MEMORY (STATE)
# ==========================================
class AgentState(TypedDict):
    """
    This dictionary represents the complete memory of the agent during a single run.
    Every node in our graph will read from and write to this state.
    """
    # The LangChain message history (Prompt -> LLM Response -> System Feedback)
    # The 'operator.add' tells LangGraph to append new messages, not overwrite them.
    messages: Annotated[Sequence[BaseMessage], operator.add]
    
    # Issue Context
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    parsed_tests: list[str]
    file_context: str
    
    # Working Memory
    generated_patch: str
    test_output: str
    
    # Safety Guardrail
    iteration_count: int

# ==========================================
# 2. INITIALIZE THE GRAPH
# ==========================================
# We create a new workflow and tell it to use our AgentState
workflow = StateGraph(AgentState)

print("Agent State defined and Workflow initialized successfully.")