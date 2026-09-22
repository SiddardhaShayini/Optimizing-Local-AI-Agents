import re
from framework_engine import FastAgentEngine

# 1. The framework auto-starts the server in the background
engine = FastAgentEngine(model_path="gemma-2b-q4.gguf")

# 2. Register tools with standard Python decorators
@engine.tool("Multiplies two numbers. Pass arguments as (a, b).")
def multiply(a: float, b: float) -> float:
    return a * b

@engine.tool("Searches the local filesystem for a file by name. Pass as 'filename.txt'.")
def search_file(filename: str) -> str:
    # Simulated file search
    return f"Found {filename} in C:/User/Documents/"

# 3. The Execution Loop
print("\n=== Agent Loop Started ===")
current_input = "Multiply 124 by 45, then tell me if you can find 'report.pdf'."

for step in range(1, 5):
    result = engine.run_turn(current_input)
    output = result["output"]
    print(f"\n[Turn {step} | Prompt Eval: {result['prompt_eval_ms']} ms]")
    print(output)

    if "Final Answer:" in output:
        break

    # Parse and execute tool automatically
    match = re.search(r"Action:\s*(.*?)\nAction Input:\s*(.*)", output)
    if match:
        tool_name = match.group(1).strip()
        tool_args = match.group(2).strip()
        
        print(f"--> [System] Executing {tool_name}({tool_args})")
        observation = engine.execute_tool(tool_name, tool_args)
        print(f"--> [System] Result: {observation}")
        
        current_input = f"Observation: {observation}"
    else:
        break