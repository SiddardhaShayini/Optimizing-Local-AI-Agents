# FastAgent Package

`fast-agent` is a lightweight, zero-overhead local LLM inference orchestrator designed to run autonomous ReAct agents on highly constrained edge devices (e.g., CPU-only, 8 GB RAM).

## Features
- **Longest Common Prefix (LCP) Slot Caching**: Reuses system prompts and tool definitions directly from KV cache to slash TTFT from ~3,500ms to ~70ms.
- **8-bit Quantized KV Cache (`q8_0`)**: Halves active context memory usage.
- **Safe Execution**: Evaluates arguments safely via AST / JSON parsing without `eval()`.
- **Subprocess Deadlock Prevention**: Proactively drains pipe buffers in a daemon thread.
- **GBNF Grammar Constraints**: Mathematically constrains model output into strict ReAct formatting.
- **Adaptive Compute Controller**: Dynamically adjusts token budgets and temperatures across Reflex, Tool-Assisted, and Deliberative tiers.
- **Unified Engine**: Single interface supporting autonomous agent loops (`ask`), single turn steps (`run_turn`), and structured JSON extraction (`run_structured`).

## Quickstart

```python
from fast_agent import FastAgent

agent = FastAgent(model_path="gemma-2b-q4.gguf")

@agent.tool("Multiplies two integers. Pass as (a, b).")
def multiply(a: int, b: int) -> int:
    return a * b

response = agent.ask("Multiply 36 by 12 and return the final answer.")
print(response)
```
