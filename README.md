# FastAgent: Hardware-Adaptive Local LLM Inference Orchestrator

FastAgent is a zero-overhead, lightweight inference orchestrator designed to run autonomous ReAct agents on highly constrained edge devices (e.g., CPU-only, 8 GB RAM). By wrapping the `llama-server` backend with aggressive KV cache management and dynamic compute routing, FastAgent reduces Time-To-First-Token (TTFT) latency from ~3,500ms down to ~70ms.

## The Problem

Running autonomous agents using multi-turn tool loops on constrained hardware typically results in:

1. **Out-Of-Memory (OOM) Crashes:** Unbounded conversation history quickly overflows 8GB of RAM.
2. **Extreme Latency:** Re-evaluating the system prompt and tool definitions on every turn takes 3-4 seconds per cycle on CPU.
3. **Malformed JSON:** Small models (<3B parameters) struggle to format ReAct JSON payloads accurately, breaking the agent loop.

## The Architecture

FastAgent implements a headless C++ engine wrapper that forces stability and speed through:

* **Longest Common Prefix (LCP) Slot Caching:** Reuses the system prompt and tool definitions directly from RAM.
* **8-bit KV Quantization:** Cuts memory usage of the active context window in half.
* **Rolling Context Windows:** Automatically prunes the oldest conversational turns while pinning the immutable system prompt.
* **GBNF Grammar Constraints:** Mathematically forces the token logits into syntactically perfect JSON.
* **Adaptive Compute Controller:** Dynamically scales token budgets (`n_predict`) and temperatures based on prompt complexity, saving CPU cycles on simple queries.

## Performance Benchmarks

| Metric | Without Framework (Default Server) | With FastAgent |
| :--- | :--- | :--- |
| **Prompt Eval Latency (TTFT)** | ~3,500 ms per turn | **~73 ms per turn** |
| **Context Memory Footprint** | 16-bit floats (f16) | **8-bit quantized (q8_0)** |
| **Multi-Turn Stability** | Fails / OOM on long loops | **Stable (Rolling Window)** |

## Quickstart

### 1. Prerequisites

You must provide your own `llama-server` binary and a `.gguf` model file.

1. Download `llama.cpp` releases and extract the binaries.
2. Download a quantized model (e.g., `gemma-2b-q4.gguf`).
3. Place `llama-server.exe` and its associated `.dll` files in your system PATH or directly in the project directory.

### 2. Installation

Install the package locally in editable mode:

```bash
git clone https://github.com/SiddardhaShayini/Optimizing-Local-AI-Agents.git
cd Optimizing-Local-AI-Agents
pip install -e .
```

### 3. Setup Wizard (Recommended)
To ensure your edge environment has the correct C++ binaries and model files, run the included diagnostic dashboard:

```bash
pip install streamlit
streamlit run fast_agent_wizard.py

### 4. Usage Example

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

## Contributing

We welcome community contributions! Because this framework targets resource-constrained edge devices, all pull requests must maintain the zero-overhead philosophy. Please review our [Contributing Guidelines](CONTRIBUTING.md) and [Code of Conduct](CODE_OF_CONDUCT.md) before submitting code.

## License

MIT License.

## Developer

Siddardha Shayini

* GitHub: [https://github.com/SiddardhaShayini](https://github.com/SiddardhaShayini)