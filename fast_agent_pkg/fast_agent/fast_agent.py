# fast_agent.py
import atexit
import inspect
import json
import re
import socket
import subprocess
import time
import requests

class FastAgent:
    def __init__(self, model_path: str = "gemma-2b-q4.gguf", port: int = 8080, max_history_turns: int = 6):
        self.port = port
        self.endpoint = f"http://localhost:{port}/completion"
        self.max_history_turns = max_history_turns
        self.system_prompt_text = ""
        self.conversation_history = []
        self.tools = {}
        self.server_process = None

        self._boot_engine(model_path)
        atexit.register(self.shutdown)

    def _boot_engine(self, model_path: str):
        cmd = [
            r".\llama-server.exe",
            "-m", model_path,
            "-c", "2048",
            "-ctk", "q8_0",
            "-ctv", "q8_0",
            "--port", str(self.port)
        ]
        self.server_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        while True:
            try:
                with socket.create_connection(("localhost", self.port), timeout=1):
                    break
            except OSError:
                time.sleep(0.4)

    def shutdown(self):
        if self.server_process:
            self.server_process.terminate()
            self.server_process.wait()

    def tool(self, description: str):
        def decorator(func):
            sig = inspect.signature(func)
            self.tools[func.__name__] = {"callable": func, "description": description, "signature": str(sig)}
            return func
        return decorator

    def _build_system_prompt(self) -> str:
        prompt = "You are a direct, autonomous agent.\nTools:\n"
        for name, data in self.tools.items():
            prompt += f"- {name}{data['signature']}: {data['description']}\n"
        prompt += "\nFormat:\nThought: <reason>\nAction: <tool_name>\nAction Input: <args>\nObservation: <result>\nFinal Answer: <text>\n"
        return prompt

    def _exec_tool(self, name: str, args_str: str) -> str:
        if name not in self.tools:
            return f"Error: Tool '{name}' not found."
        try:
            func = self.tools[name]["callable"]
            args = eval(args_str)
            return str(func(*args) if isinstance(args, tuple) else func(args))
        except Exception as e:
            return f"Error executing tool: {e}"

    def ask(self, user_prompt: str, max_steps: int = 5) -> str:
        """Single entrypoint: handles the entire autonomous loop and returns the final answer."""
        if not self.system_prompt_text:
            self.system_prompt_text = self._build_system_prompt()

        current_input = f"User: {user_prompt.strip()}\nAgent:"

        for _ in range(max_steps):
            self.conversation_history.append(current_input)
            if len(self.conversation_history) > self.max_history_turns:
                self.conversation_history = self.conversation_history[-self.max_history_turns:]

            full_prompt = f"{self.system_prompt_text}\n\n" + "\n".join(self.conversation_history)
            payload = {
                "prompt": full_prompt,
                "n_predict": 128,
                "temperature": 0.1,
                "cache_prompt": True,
                "stop": ["\nObservation:", "\nUser:"]
            }

            resp = requests.post(self.endpoint, json=payload).json()
            output = resp.get("content", "").strip()
            self.conversation_history[-1] += f" {output}"

            if "Final Answer:" in output:
                return output.split("Final Answer:")[-1].strip()

            match = re.search(r"Action:\s*(.*?)\nAction Input:\s*(.*)", output)
            if match:
                t_name, t_args = match.group(1).strip(), match.group(2).strip()
                obs = self._exec_tool(t_name, t_args)
                current_input = f"Observation: {obs}\nAgent:"
            else:
                return output

        return "Max steps reached without a final answer."

'''
How You Use It in Any Future Project
Now, any time you start a new application, you drop fast_agent.py into your folder and write clean, expressive code like this:

Python
from fast_agent import FastAgent

# 1. Initialize
app = FastAgent(model_path="gemma-2b-q4.gguf")

# 2. Add whatever tools your specific project needs
@app.tool("Adds two integers together. Pass as (a, b).")
def add_numbers(a: int, b: int) -> int:
    return a + b

# 3. Ask a question — the framework does all the caching, loops, and parsing
response = app.ask("Add 458 and 921, then state the final sum.")
print("Result:", response)
No server commands, no manual JSON wrangling, and zero memory leaks.
'''
