import ast
import operator
import re
from framework_engine import FastAgentEngine

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS


_SAFE_MATH_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def safe_eval_math(expr: str):
    """Safely evaluates basic arithmetic expressions via AST without code execution."""
    def _eval_node(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        elif isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in _SAFE_MATH_OPERATORS:
                raise ValueError(f"Unsupported operator: {op_type.__name__}")
            return _SAFE_MATH_OPERATORS[op_type](_eval_node(node.left), _eval_node(node.right))
        elif isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in _SAFE_MATH_OPERATORS:
                raise ValueError(f"Unsupported operator: {op_type.__name__}")
            return _SAFE_MATH_OPERATORS[op_type](_eval_node(node.operand))
        raise ValueError(f"Unsupported expression node: {type(node).__name__}")

    tree = ast.parse(expr.strip(), mode="eval")
    return _eval_node(tree.body)


# 1. Initialize headless engine
engine = FastAgentEngine(model_path="gemma-2b-q4.gguf")

# 2. Define the Web Search Tool
@engine.tool("Searches the web for current information. Pass a query string like 'latest space news'.")
def web_search(query: str) -> str:
    print(f"    [Fetching web results for: '{query}']")
    try:
        results = DDGS().text(query, max_results=3)
        if not results:
            return "No results found."

        # Condense results to protect the agent's context window
        formatted = "\n".join([f"- {r.get('title', '')}: {r.get('body', '')}" for r in results])
        return formatted
    except Exception as e:
        return f"Search failed: {e}"

# 3. Define a local calculator tool using safe AST math evaluation
@engine.tool("Evaluates basic math expressions. Pass a string like '2026 - 1990'.")
def calculate(expression: str) -> str:
    try:
        return str(safe_eval_math(expression))
    except Exception as e:
        return f"Math error: {e}"

# 4. The Autonomous Loop
def main():
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

            # Clean up the arguments if wrapped in quotes
            if tool_args.startswith(("'", '"')) and tool_args.endswith(("'", '"')):
                tool_args = tool_args[1:-1]

            observation = engine.execute_tool(tool_name, tool_args)
            print(f"--> [System] Executing {tool_name}({tool_args}) -> {observation}")
            current_input = f"Observation: {observation}"
        else:
            print("\n[System] Agent stopped without a final answer or valid tool call.")
            break


if __name__ == "__main__":
    main()