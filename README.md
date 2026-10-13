# FastAgent: Hardware-Adaptive Local LLM Inference Orchestrator

FastAgent is a zero-overhead, lightweight inference orchestrator designed to run autonomous ReAct agents on highly constrained edge devices (e.g., CPU-only, 8 GB RAM). By wrapping the `llama-server` backend with aggressive KV cache management and dynamic compute routing, FastAgent reduces Time-To-First-Token (TTFT) latency from ~3,500ms down to ~70ms.

## The Problem

Running autonomous agents using multi-turn tool loops on constrained hardware typically results in:

1. **Out-Of-Memory (OOM) Crashes:** Unbounded conversation history quickly overflows 8GB of RAM.
2. **Extreme Latency:** Re-evaluating the system prompt and tool definitions on every turn takes 3-4 seconds per cycle on CPU.
3. **Malformed Output:** Small models (<3B parameters) struggle to format ReAct syntax reliably without grammar constraints.
4. **Security Risks:** Naive agent frameworks often use insecure `eval()` calls on model-generated tool arguments.

## The Architecture

FastAgent implements a headless C++ engine wrapper that forces stability and speed through:

* **Longest Common Prefix (LCP) Slot Caching:** Reuses the system prompt and tool definitions directly from RAM via `cache_prompt: True`.
* **8-bit KV Quantization:** Cuts memory usage of the active context window in half (`-ctk q8_0 -ctv q8_0`).
* **Rolling Context Windows:** Automatically prunes the oldest conversational turns while pinning the immutable system prompt.
* **Safe Tool Execution:** Safely parses arguments using AST (`ast.literal_eval`) and JSON without dangerous `eval()` execution.
* **Deadlock Prevention:** Continuously drains subprocess standard output in background daemon threads so OS pipe buffers never saturate.
* **GBNF Grammar Constraints:** Constrains token logits into syntactically valid ReAct formats or JSON schemas.
* **Adaptive Compute Controller:** Dynamically scales token budgets (`n_predict`) and temperatures across `REFLEX`, `TOOL_ASSISTED`, and `DELIBERATIVE` tiers based on prompt complexity.

## Performance Benchmarks

| Metric | Without Framework (Default Server) | With FastAgent |
| :--- | :--- | :--- |
| **Prompt Eval Latency (TTFT)** | ~3,500 ms per turn | **~73 ms per turn** |
| **Context Memory Footprint** | 16-bit floats (f16) | **8-bit quantized (q8_0)** |
| **Multi-Turn Stability** | Fails / OOM on long loops | **Stable (Rolling Window)** |
| **Tool Execution Safety** | Unsafe `eval()` / Fragile | **Safe AST / JSON Parsing** |

## Quickstart

### 1. Prerequisites

You must provide a `llama-server` binary and a quantized `.gguf` model file.

1. Download the latest `llama.cpp` release binaries for your operating system.
2. Place `llama-server.exe` (and its associated `.dll` libraries) in your system PATH or directly in the project directory.
3. Download a quantized model using the included zero-dependency helper:

```bash
python download_model.py --model qwen2.5-1.5b
```

### 2. Installation

Install the package locally in editable mode:

```bash
git clone https://github.com/SiddardhaShayini/Optimizing-Local-AI-Agents.git
cd Optimizing-Local-AI-Agents
pip install -e .
```

To install optional features (such as the Streamlit diagnostic wizard):
```bash
pip install -e ".[wizard]"
```

### 3. Setup Wizard (Recommended)

To ensure your edge environment has the correct C++ binaries and model files, run the included diagnostic dashboard:

```bash
streamlit run fast_agent_wizard.py
```

### 4. Usage Examples

#### Autonomous Agent Loop (`ask`)

```python
from fast_agent import FastAgent

# The runtime automatically locates llama-server.exe and boots the engine
agent = FastAgent(model_path="gemma-2b-q4.gguf")

@agent.tool("Multiplies two integers. Pass as (a, b).")
def multiply(a: int, b: int) -> int:
    return a * b

response = agent.ask("Multiply 36 by 12 and return the final answer.")
print(response)
```

#### Step-by-Step Turn Execution (`run_turn`)

```python
from fast_agent import FastAgent

agent = FastAgent(model_path="gemma-2b-q4.gguf")

turn = agent.run_turn("Calculate 124 * 45")
print(turn["output"])
print(f"Prompt Eval Latency: {turn['prompt_eval_ms']} ms")
```

#### Structured JSON Extraction (`run_structured`)

```python
from fast_agent import FastAgent

agent = FastAgent(model_path="gemma-2b-q4.gguf")

schema = {
    "type": "object",
    "properties": {
        "status": {"type": "string"},
        "result": {"type": "number"}
    },
    "required": ["status", "result"]
}

result = agent.run_structured("Extract status: OK and code: 200", json_schema=schema)
print(result)
```

## Running Benchmarks

Comprehensive benchmarks comparing FastAgent against Ollama baseline are located in `benchmarks/`:

```bash
# Run multi-turn agent loop latency test
python benchmarks/benchmark_agent_loop.py

# Run n-gram / KV cache benchmark
python benchmarks/benchmark_framework.py
```

## Contributing

We welcome community contributions! Because this framework targets resource-constrained edge devices, all pull requests must maintain the zero-overhead philosophy. Please review our [Contributing Guidelines](.github/CONTRIBUTING.md) and [Code of Conduct](.github/CODE_OF_CONDUCT.md) before submitting code.

## License

MIT License.

## Developer

Siddardha Shayini

* GitHub: [https://github.com/SiddardhaShayini](https://github.com/SiddardhaShayini)