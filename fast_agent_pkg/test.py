from fast_agent import FastAgent

agent = FastAgent(model_path="gemma-2b-q4.gguf")

@agent.tool("Multiplies two integers: (a, b)")
def multiply(a: int, b: int) -> int:
    return a * b

print("\n--- Test 1: Reflex (Low compute budget) ---")
res1 = agent.ask("Define recursion in one sentence.")
print("Output:", res1)

print("\n--- Test 2: Tool-Assisted (Medium compute budget) ---")
res2 = agent.ask("Multiply 36 by 12 and return the final answer.")
print("Output:", res2)