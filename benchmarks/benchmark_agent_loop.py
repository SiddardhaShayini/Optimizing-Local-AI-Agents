import time
import requests
import json

API_URL = "http://localhost:8080/completion"

# We simulate a heavy system prompt that an autonomous agent would use
SYSTEM_PROMPT = """You are an autonomous Python coding agent.
You have access to the following tools:
- read_file(path): Reads a file.
- write_file(path, content): Writes to a file.
- run_python(code): Executes python code and returns stdout.

You must follow the ReAct format strictly:
Thought: (What you need to do)
Action: (The tool name)
Action Input: (The tool arguments)

Begin."""

# An agent loop sends the whole history back to the model on every turn
AGENT_TURNS = [
    "User: Write a script to calculate the 100th Fibonacci number.\nAgent:",
    "Observation: File 'fib.py' created.\nAgent:",
    "Observation: Output is 354224848179261915075.\nAgent:"
]

def run_agent_loop():
    print("=== Starting Multi-Turn Agent Loop Benchmark ===")
    
    current_prompt = SYSTEM_PROMPT + "\n\n"
    
    for i, user_input in enumerate(AGENT_TURNS, 1):
        print(f"\n--- Turn {i} ---")
        current_prompt += user_input
        
        payload = {
            "prompt": current_prompt,
            "n_predict": 30,
            "temperature": 0.1,
            "stream": True,
            "cache_prompt": True  # <-- THIS IS THE MAGIC FLAG
        }
        
        t_start = time.perf_counter()
        t_first_token = None
        
        try:
            response = requests.post(API_URL, json=payload, stream=True)
            response.raise_for_status()
            
            generated_text = ""
            for line in response.iter_lines():
                if line:
                    decoded = line.decode("utf-8")
                    if decoded.startswith("data: "):
                        try:
                            chunk = json.loads(decoded[6:])
                            if t_first_token is None:
                                t_first_token = time.perf_counter()
                            if not chunk.get("stop"):
                                generated_text += chunk.get("content", "")
                        except json.JSONDecodeError:
                            continue
                            
            t_end = time.perf_counter()
            
            ttft = t_first_token - t_start if t_first_token else 0
            total_time = t_end - t_start
            
            print(f"Time to First Token (TTFT): {ttft:.3f} seconds")
            print(f"Total Turn Time: {total_time:.3f} seconds")
            print(f"Agent Output: {generated_text.strip()}")
            
            # Append the agent's output to the prompt history for the next turn
            current_prompt += generated_text + "\n\n"
            
        except Exception as e:
            print(f"Request failed: {e}")

if __name__ == "__main__":
    run_agent_loop()
