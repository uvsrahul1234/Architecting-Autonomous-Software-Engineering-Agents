# ==========================================
# MAIN CONTROLLER: The LangGraph State Machine
# ==========================================
import re
import requests

from langgraph.graph import END
from agent.agent_graph import AgentState, workflow
from agent.prompt_engineering import generate_initial_prompt_node
from agent.tools import generate_code_node, execute_tests_node

# 1. Register the Nodes (The Actions)
workflow.add_node("prompt_engineer", generate_initial_prompt_node)
workflow.add_node("coder", generate_code_node)
workflow.add_node("sandbox", execute_tests_node)

# 2. Define the Standard Edges (The linear flow)
workflow.set_entry_point("prompt_engineer")
workflow.add_edge("prompt_engineer", "coder")
workflow.add_edge("coder", "sandbox")

# 3. Define the Conditional Edge (The Agentic Loop)
def routing_logic(state: AgentState):
    """
    This is the brain's decision engine. It looks at the Sandbox output 
    and decides whether to loop back and try again, or stop.
    """
    output = state.get("test_output", "")
    iterations = state.get("iteration_count", 0)
    
    # Success Condition
    if "ALL TESTS PASSED" in output or "failed" not in output.lower():
        print("\n🏆 SUCCESS: The AI successfully patched the bug!")
        return "end"
        
    # Safety Guardrail Condition (Stop after 3 tries)
    if iterations >= 3:
        print("\n🛑 MAX ITERATIONS REACHED: The AI could not fix the bug in 3 attempts.")
        return "end"
        
    # Retry Condition
    print(f"\n🔄 TESTS FAILED (Attempt {iterations}). Routing back to LLM to try again...\n")
    return "retry"

# Tell the workflow to use our routing logic after the sandbox finishes
workflow.add_conditional_edges(
    "sandbox",
    routing_logic,
    {
        "retry": "coder", # If routing_logic returns "retry", loop back to the coder node
        "end": END        # If it returns "end", stop the graph
    }
)

# 4. Compile the Application
swe_agent = workflow.compile()

def categorize_failure(test_output: str, iterations: int) -> str:
    """Analyzes the Docker output to determine the root cause of failure."""
    if "ALL TESTS PASSED" in test_output:
        return "None (Success)"
    
    if iterations >= 3 and "failed" in test_output.lower():
        return "Max Iterations Reached"
        
    if "SyntaxError" in test_output or "IndentationError" in test_output:
        return "Syntax Error"
        
    if "AttributeError" in test_output or "NameError" in test_output or "ModuleNotFoundError" in test_output:
        return "Hallucination (Missing Variable/Import)"
        
    if "AssertionError" in test_output or "FAILED" in test_output:
        return "Logic Error (Test Assertion Failed)"
        
    return "Unknown Runtime Error"

# # ==========================================
# # TEST RUN: Issue scikit-learn-25500
# # ==========================================
# if __name__ == "__main__":
#     print("🚀 Initializing Autonomous SWE Agent...\n")
    
#     # We will use the exact same issue we tested in Colab to see if the agent 
#     # can write the code and run it successfully.
#     initial_state = {
#         "messages": [],
#         "instance_id": "scikit-learn-25500",
#         "problem_statement": 'CalibratedClassifierCV with set_config(transform_output="pandas") fails.',
#         "parsed_tests": ["sklearn/calibration.py"],
#         "generated_patch": "",
#         "test_output": "",
#         "iteration_count": 0
#     }
    
#     # Run the graph
#     final_state = swe_agent.invoke(initial_state)
    
#     print("\n==========================================")
#     print("FINAL AGENT STATE:")
#     print(f"Total Iterations: {final_state['iteration_count']}")
#     print("Final Patch Generated:")
#     print(final_state["generated_patch"][:300] + "...\n")
#     print("==========================================")

# ==========================================
# THESIS BATCH RUN: SWE-bench Lite
# ==========================================
if __name__ == "__main__":
    from datasets import load_dataset
    import time
    import csv
    import os
    import re
    import requests
    
    print("🚀 Initializing Autonomous SWE Agent Batch Run...\n")
    
    # 1. Setup the CSV File
    csv_filename = "results_llama3_70b.csv"
    file_exists = os.path.isfile(csv_filename)
    
    with open(csv_filename, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Write headers if the file is brand new
        if not file_exists:
            writer.writerow(['Instance_ID', 'Status', 'Iterations', 'Failure_Reason', 'Generated_Patch', 'Ground_Truth'])
    
    # 2. Download/Load the dataset
    print("Loading SWE-bench Lite dataset...")
    dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    
    # Grab the first 10 issues for our data test
    test_batch = dataset.select(range(10)) 
    
    successful_patches = 0
    
    # 3. Loop through the dataset
    for i, issue in enumerate(test_batch):
        print(f"\n==========================================")
        print(f"🛠️ SOLVING BUG {i+1}/{len(test_batch)}: {issue['instance_id']}")
        print(f"Repo: {issue['repo']} | Base Commit: {issue['base_commit']}")
        print(f"==========================================")
        
        # --- THE SWE-BENCH SHORTCUT (Context Retrieval) ---
        print("Extracting target file paths and downloading context...")
        ground_truth_patch = issue["patch"]
        files_to_edit = set(re.findall(r"--- a/(.+)", ground_truth_patch))
        
        file_context = ""
        for file_path in files_to_edit:
            url = f"https://raw.githubusercontent.com/{issue['repo']}/{issue['base_commit']}/{file_path}"
            resp = requests.get(url)
            if resp.status_code == 200:
                file_context += f"--- {file_path} ---\n```python\n{resp.text}\n```\n\n"
                print(f"✅ Downloaded context: {file_path}")
            else:
                print(f"⚠️ Failed to download: {file_path}")
        # --------------------------------------------------
        
        # Setup the initial state
        initial_state = {
            "messages": [],
            "instance_id": issue["instance_id"],
            "repo": issue["repo"],
            "base_commit": issue["base_commit"],
            "problem_statement": issue["problem_statement"],
            "parsed_tests": ["Run pytest to verify."], 
            "file_context": file_context,
            "generated_patch": "",
            "test_output": "",
            "iteration_count": 0
        }
        
        # Turn the agent loose
        final_state = swe_agent.invoke(initial_state)
        
        # Determine Success or Failure
        # Determine Success or Failure
        output = final_state.get("test_output", "")
        iterations = final_state.get("iteration_count", 0)
        patch = final_state.get("generated_patch", "")
        
        # Call our new taxonomy function
        failure_reason = categorize_failure(output, iterations)
        
        if "ALL TESTS PASSED" in output or "failed" not in output.lower():
            status = "PASS"
            successful_patches += 1
            print(f"✅ {issue['instance_id']} Resolved in {iterations} iterations.")
        else:
            status = "FAIL"
            print(f"❌ {issue['instance_id']} Failed. Reason: {failure_reason}")
            
        # 4. Save the result to the CSV immediately
        with open(csv_filename, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([issue['instance_id'], status, iterations, failure_reason, patch, ground_truth_patch])
        
        # 5. Respect the API Rate Limits (20 seconds between issues)
        print("Sleeping for 20 seconds to respect Groq API limits...")
        time.sleep(60)

    # Final Metric Output
    print("\n==========================================")
    print(f"📊 BATCH RUN COMPLETE")
    print(f"Total Issues Processed: {len(test_batch)}")
    print(f"Successful Patches: {successful_patches}")
    print(f"Pass Rate: {(successful_patches/len(test_batch))*100:.2f}%")
    print(f"Results saved to: {csv_filename}")
    print("==========================================")