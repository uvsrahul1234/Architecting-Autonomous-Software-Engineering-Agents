import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from .agent_graph import AgentState
from sandbox.docker_manager import DockerSandbox

# Force the API Key into the environment
os.environ["GROQ_API_KEY"] = "your key"

# ==========================================
# 1. INITIALIZE THE GROQ LLM (Cloud Open-Source)
# ==========================================
# We are using Meta's massive 70-Billion parameter Llama-3 model hosted on Groq.
# Temperature remains 0.1 for strict coding syntax.
llm = ChatGroq(
    model="qwen/qwen3-32b", 
    temperature=0.1
)

# ==========================================
# 2. GENERATION NODE (The AI writes the code)
# ==========================================
def generate_code_node(state: AgentState):
    """Sends the current state history to the LLM and extracts the Python code."""
    print("--- Node Execution: Generating Code Patch ---")
    
    messages = state["messages"]
    iteration = state.get("iteration_count", 0)
    
    # 1. Query the LLM
    response = llm.invoke(messages)
    
    # 2. Extract strictly the Python code from the LLM's markdown formatting
    patch_code = response.content
    if "```python" in response.content:
        patch_code = response.content.split("```python")[1].split("```")[0].strip()
    elif "```" in response.content:
        patch_code = response.content.split("```")[1].split("```")[0].strip()
        
    print(f"Code generation complete (Iteration {iteration + 1}).")
        
    # 3. Update the State
    return {
        "messages": [response],          # Append the AI's exact response to history
        "generated_patch": patch_code,   # Save the clean code to memory
        "iteration_count": iteration + 1 # Tick the safety counter
    }

# ==========================================
# 3. EXECUTION NODE (The Sandbox runs the code)
# ==========================================
def execute_tests_node(state: AgentState):
    """Takes the generated code, runs it in a cloned GitHub repo inside Docker."""
    print("--- Node Execution: Running Real-World Repo Tests ---")
    
    patch = state.get("generated_patch", "")
    repo = state.get("repo", "")
    commit = state.get("base_commit", "")
    
    if not patch:
        error_msg = "Error: No valid Python code was generated."
        return {"messages": [HumanMessage(content=error_msg)], "test_output": error_msg}
        
    sandbox = DockerSandbox()
    output = ""
    
    try:
        sandbox.start_container()
        
        # Escape quotes for the bash script
        safe_patch = patch.replace('"', '\\"').replace('$', '\\$')
        
        # 1. THE BASH SCRIPT: Clone, Checkout, Patch, and Test
        bash_script = f"""
        # Install testing tools
        pip install pytest --quiet
        
        # Clone the repository
        git clone https://github.com/{repo}.git /workspace --quiet
        cd /workspace
        
        # Roll back to the exact moment of the bug
        git checkout {commit} --quiet
        
        # Save the AI's patch to a file
        cat << 'EOF' > ai_patch.py
        {safe_patch}
        EOF
        
        # Run the patch inside the repository's ecosystem
        # (PYTHONPATH=. ensures it can import from the cloned repo)
        export PYTHONPATH=.
        python ai_patch.py
        """
        
        print(f"Cloning {repo} at commit {commit[:7]}...")
        # Run the script inside Docker
        output = sandbox.execute_command(f"sh -c '{bash_script}'")
        
        if not output:
            output = "[Execution finished with no terminal output. Syntax is valid.]"
            
    except Exception as e:
        output = f"Sandbox Infrastructure Error: {str(e)}"
    finally:
        sandbox.cleanup()
        
    print(f"Sandbox Output Logged. Length: {len(output)} chars.")
        
    feedback_prompt = (
        f"### Test Execution Results:\n{output}\n\n"
        "If there are Traceback errors, analyze them and rewrite the code to fix the bug. "
        "If the tests passed successfully, output exactly: ALL TESTS PASSED."
    )
    
    return {
        "test_output": output,
        "messages": [HumanMessage(content=feedback_prompt)]
    }