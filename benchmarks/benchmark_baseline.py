import json
import os
import time
from datetime import datetime
import psutil
import requests

OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma:2b"
RESULTS_FILE = "benchmark_results.json"

# Representative workloads: short query, reasoning, and long context / agent-like prompt
TEST_PROMPTS = [
    {
        "category": "Short Query",
        "prompt": "Explain what an operating system kernel is in two sentences.",
    },
    {
        "category": "Reasoning & Math",
        "prompt": "Solve this step by step: A farmer has 15 sheep, and all but 8 die. How many are left? Then explain why.",
    },
    {
        "category": "Code / Agent Task",
        "prompt": (
            "Write a clean Python function using only the standard library that traverses "
            "a nested dictionary and extracts all leaf values into a flat list. Include docstring and type hints."
        ),
    },
    {
        "category": "Context-Heavy / Summarization",
        "prompt": (
            "Summarize the following passage into three key takeaways:\n"
            "Distributed systems consist of autonomous computing entities that communicate "
            "over a network. A core challenge is consensus, where nodes must agree on a "
            "single data value or state despite network partitions or failures. Common "
            "consensus algorithms include Paxos and Raft. Raft simplifies Paxos by decomposing "
            "the consensus problem into leader election, log replication, and safety. While Paxos "
            "is mathematically elegant, Raft is specifically designed to be understandable and "
            "implementable in production software systems."
        ),
    },
]


def run_benchmark_prompt(prompt: str, category: str):
    process = psutil.Process(os.getpid())
    ram_before = psutil.virtual_memory().used / (1024**2)  # in MB

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": True,  # Streaming needed to measure TTFT
        "options": {
            "temperature": 0.2,
            "num_predict": 256,
        },
    }

    cpu_measurements = []
    t_start = time.perf_counter()
    t_first_token = None
    generated_text = ""
    eval_count = 0
    prompt_eval_count = 0
    total_duration_ns = 0
    load_duration_ns = 0
    prompt_eval_duration_ns = 0
    eval_duration_ns = 0

    response = requests.post(OLLAMA_API_URL, json=payload, stream=True)
    response.raise_for_status()

    for line in response.iter_lines():
        if line:
            chunk = json.loads(line.decode("utf-8"))

            if t_first_token is None and chunk.get("response"):
                t_first_token = time.perf_counter()

            generated_text += chunk.get("response", "")
            cpu_measurements.append(psutil.cpu_percent(interval=None))

            if chunk.get("done"):
                total_duration_ns = chunk.get("total_duration", 0)
                load_duration_ns = chunk.get("load_duration", 0)
                prompt_eval_count = chunk.get("prompt_eval_count", 0)
                prompt_eval_duration_ns = chunk.get("prompt_eval_duration", 0)
                eval_count = chunk.get("eval_count", 0)
                eval_duration_ns = chunk.get("eval_duration", 0)

    t_end = time.perf_counter()
    ram_after = psutil.virtual_memory().used / (1024**2)

    # Convert Ollama nanoseconds to human-readable units
    ttft_sec = (t_first_token - t_start) if t_first_token else 0.0
    total_time_sec = (t_end - t_start)
    eval_speed_tok_sec = (
        (eval_count / (eval_duration_ns / 1e9)) if eval_duration_ns > 0 else 0.0
    )
    prompt_speed_tok_sec = (
        (prompt_eval_count / (prompt_eval_duration_ns / 1e9))
        if prompt_eval_duration_ns > 0
        else 0.0
    )
    avg_cpu = sum(cpu_measurements) / len(cpu_measurements) if cpu_measurements else 0.0

    return {
        "timestamp": datetime.now().isoformat(),
        "category": category,
        "prompt": prompt,
        "metrics": {
            "total_time_sec": round(total_time_sec, 3),
            "ttft_sec": round(ttft_sec, 3),
            "prompt_tokens": prompt_eval_count,
            "prompt_speed_tok_sec": round(prompt_speed_tok_sec, 2),
            "generated_tokens": eval_count,
            "eval_speed_tok_sec": round(eval_speed_tok_sec, 2),
            "avg_cpu_percent": round(avg_cpu, 1),
            "system_ram_used_mb": round(ram_after, 1),
            "ram_delta_mb": round(ram_after - ram_before, 1),
        },
        "response_sample": generated_text[:120] + "...",
    }


def main():
    print(f"=== Starting Baseline Benchmarking for '{MODEL_NAME}' ===")
    results = []

    # Warmup pass (prevents cold model load skewing metrics)
    print("Running warmup pass...")
    requests.post(
        OLLAMA_API_URL,
        json={"model": MODEL_NAME, "prompt": "hi", "stream": False},
    )

    for item in TEST_PROMPTS:
        print(f"Testing: {item['category']}...")
        data = run_benchmark_prompt(item["prompt"], item["category"])
        results.append(data)
        time.sleep(1)

    # Save to file
    all_runs = []
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE, "r") as f:
                all_runs = json.load(f)
        except Exception:
            all_runs = []

    all_runs.append({"run_type": "baseline_ollama_raw", "records": results})

    with open(RESULTS_FILE, "w") as f:
        json.dump(all_runs, f, indent=2)

    print(f"\nBenchmark completed! Results appended to '{RESULTS_FILE}'.")
    print_summary_table(results)


def print_summary_table(results):
    print("\n" + "=" * 80)
    print(
        f"{'Category':<28} | {'TTFT (s)':<9} | {'Gen (tok/s)':<11} | {'Tokens':<7} | {'Avg CPU%':<8}"
    )
    print("-" * 80)
    for r in results:
        m = r["metrics"]
        print(
            f"{r['category']:<28} | {m['ttft_sec']:<9.2f} | {m['eval_speed_tok_sec']:<11.2f} | {m['generated_tokens']:<7} | {m['avg_cpu_percent']:<8.1f}%"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
