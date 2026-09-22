import re
from ddgs import DDGS
from framework_engine import FastAgentEngine

# 1. Initialize our headless engine
engine = FastAgentEngine(model_path="gemma-2b-q4.gguf")

# 2. Define the Web Search Tool
@engine.tool("Searches the web for current information. Pass a query string like 'latest space news'.")
def web_search(query: str) -> str:
    print(f"    [Fetching web results for: '{query}']")
    try:
        # Fetch the top 3 results silently
        results = DDGS().text(query, max_results=3)
        if not results:
            return "No results found."
        
        # Condense the results so we don't blow up the agent's context window
        formatted = "\n".join([f"- {r['title']}: {r['body']}" for r in results])
        return formatted
    except Exception as e:
        return f"Search failed: {e}"

# 3. Define a local calculator tool just in case it needs to do math on the results
@engine.tool("Evaluates basic math expressions. Pass a string like '2026 - 1990'.")
def calculate(expression: str) -> str:
    try:
        # Safe-ish eval for basic math
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Math error: {e}"

# 4. The Autonomous Loop
print("\n=== Live Research Agent Started ===")
current_input = "Who is the current CEO of Microsoft and how old are they?"

for step in range(1, 6):
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
        
        # Clean up the arguments if the model wrapped them in quotes
        if tool_args.startswith(("'", '"')) and tool_args.endswith(("'", '"')):
            tool_args = tool_args[1:-1]
            
        observation = engine.execute_tool(tool_name, f"'{tool_args}'")
        current_input = f"Observation: {observation}"
    else:
        print("\n[System] Agent stopped without a final answer or valid tool call.")
        break