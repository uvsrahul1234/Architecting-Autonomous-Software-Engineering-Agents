from langchain_core.messages import SystemMessage, HumanMessage
from .agent_graph import AgentState

# ==========================================
# 1. DEFINE THE AGENT'S PERSONA
# ==========================================
SYSTEM_PROMPT = """You are an autonomous AI software engineer.
Your goal is to fix the software bug described by the user.

You will be provided with:
1. The GitHub Issue description.
2. The specific test paths that are currently failing.

Your task is to analyze the issue and write a Python code patch that fixes the bug and passes the tests.
Focus strictly on logical syntax. Do not output conversational filler.
"""

# ==========================================
# 2. DEFINE THE NODE FUNCTION
# ==========================================
def generate_initial_prompt_node(state: AgentState):
    """
    This node reads the raw issue data from the State and constructs 
    the initial LangChain messages to feed into the LLM.
    """
    print("--- Node Execution: Generating Initial Prompt ---")

    # 1. Read the data from the current State
    issue = state.get("problem_statement", "")
    tests = state.get("parsed_tests", [])
    context = state.get("file_context", "") # Get the downloaded code

    # 2. Format the tests into a readable string
    test_str = "\n".join(tests) if tests else "No specific test paths provided."
    
    # 3. Construct the Human Message (Now includes the real repository code!)
    user_prompt = (
        f"### GitHub Issue:\n{issue}\n\n"
        f"### Repository Context (Target Files):\n{context}\n\n"
        f"### Target Tests to Pass:\n{test_str}"
    )

    # 4. Bundle into LangChain message objects
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt)
    ]

    # 5. Return the updated State
    # We return a dictionary matching the AgentState keys. 
    # LangGraph knows to append these messages to the existing list.
    return {"messages": messages, "iteration_count": 0}