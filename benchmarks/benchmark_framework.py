import json
import os
import time
from datetime import datetime
import psutil
import requests

# Pointing to llama-server instead of Ollama
API_URL = "http://localhost:8080/completion"
RESULTS_FILE = "benchmark_results.json"

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
    ram_before = psutil.virtual_memory().used / (1024**2)

    # llama-server payload format
    payload = {
        "prompt": prompt,
        "n_predict": 256,
        "temperature": 0.2,
        "stream": True
    }

    cpu_measurements = []
    t_start = time.perf_counter()
    t_first_token = None
    
    # These will be populated by the final STOP chunk
    timings = {}

    response = requests.post(API_URL, json=payload, stream=True)
    response.raise_for_status()

    for line in response.iter_lines():
        if line:
            decoded = line.decode("utf-8")
            if decoded.startswith("data: "):
                try:
                    chunk = json.loads(decoded[6:])
                    
                    if t_first_token is None:
                        t_first_token = time.perf_counter()

                    cpu_measurements.append(psutil.cpu_percent(interval=None))
                    
                    # llama-server sends timings in the final chunk
                    if chunk.get("stop") is True:
                        timings = chunk.get("timings", {})
                except json.JSONDecodeError:
                    continue

    t_end = time.perf_counter()
    ram_after = psutil.virtual_memory().used / (1024**2)

    # Calculate metrics
    ttft_sec = (t_first_token - t_start) if t_first_token else 0.0
    total_time_sec = (t_end - t_start)
    
    # llama-server provides tok/sec directly in the timings object
    eval_speed_tok_sec = timings.get("predicted_per_second", 0.0)
    generated_tokens = timings.get("predicted_n", 0)
    
    avg_cpu = sum(cpu_measurements) / len(cpu_measurements) if cpu_measurements else 0.0

    return {
        "timestamp": datetime.now().isoformat(),
        "category": category,
        "prompt": prompt,
        "metrics": {
            "total_time_sec": round(total_time_sec, 3),
            "ttft_sec": round(ttft_sec, 3),
            "generated_tokens": generated_tokens,
            "eval_speed_tok_sec": round(eval_speed_tok_sec, 2),
            "avg_cpu_percent": round(avg_cpu, 1),
            "system_ram_used_mb": round(ram_after, 1),
            "ram_delta_mb": round(ram_after - ram_before, 1),
        }
    }

def main():
    print("=== Starting Framework N-Gram Benchmarking ===")
    results = []

    # Warmup
    requests.post(API_URL, json={"prompt": "hi", "n_predict": 2, "stream": False})

    for item in TEST_PROMPTS:
        print(f"Testing: {item['category']}...")
        data = run_benchmark_prompt(item["prompt"], item["category"])
        results.append(data)
        time.sleep(1)

    # Append to existing baseline file
    all_runs = []
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as f:
            all_runs = json.load(f)

    all_runs.append({"run_type": "framework_ngram_speculative", "records": results})

    with open(RESULTS_FILE, "w") as f:
        json.dump(all_runs, f, indent=2)

    print("\nBenchmark completed!")
    
    print("\n" + "=" * 80)
    print(f"{'Category':<28} | {'TTFT (s)':<9} | {'Gen (tok/s)':<11} | {'Avg CPU%':<8}")
    print("-" * 80)
    for r in results:
        m = r["metrics"]
        print(f"{r['category']:<28} | {m['ttft_sec']:<9.2f} | {m['eval_speed_tok_sec']:<11.2f} | {m['avg_cpu_percent']:<8.1f}%")
    print("=" * 80)

if __name__ == "__main__":
    main()