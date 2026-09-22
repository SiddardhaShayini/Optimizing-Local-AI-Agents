# fast_agent/core.py
import atexit
import inspect
import json
import os
import re
import shutil
import socket
import subprocess
import time
import requests


def find_llama_server() -> str:
    """Locates llama-server.exe in package directory, current directory, or system PATH."""
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(pkg_dir, "llama-server.exe"),
        os.path.abspath("llama-server.exe"),
        os.path.join(os.path.dirname(pkg_dir), "llama-server.exe"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path

    system_bin = shutil.which("llama-server") or shutil.which("llama-server.exe")
    if system_bin:
        return system_bin

    raise FileNotFoundError(
        "Could not find 'llama-server.exe'. Please ensure it exists inside "
        f"'{pkg_dir}' or in your project root."
    )


class AdaptiveComputeController:
    """Evaluates task complexity and dynamically allocates token and compute budgets."""

    @staticmethod
    def inspect_budget(query: str, has_tools: bool) -> dict:
        q = query.lower().strip()

        # Tier 1: Reflex (Definitions, greetings, simple lookups)
        if len(query.split()) < 8 and not any(
            k in q for k in ["why", "how", "solve", "code", "analyze", "find"]
        ):
            return {
                "compute_tier": "REFLEX",
                "max_tokens": 48,
                "temperature": 0.0,
                "allow_tools": False,
            }

        # Tier 2: Analytical / Tool Task
        if any(
            k in q
            for k in [
                "calculate",
                "find",
                "search",
                "run",
                "multiply",
                "+",
                "*",
                "/",
                "math",
            ]
        ):
            return {
                "compute_tier": "TOOL_ASSISTED",
                "max_tokens": 128,
                "temperature": 0.1,
                "allow_tools": True,
            }

        # Tier 3: Deliberative Reasoning
        return {
            "compute_tier": "DELIBERATIVE",
            "max_tokens": 256,
            "temperature": 0.2,
            "allow_tools": has_tools,
        }


class FastAgent:
    """Hardware-adaptive, zero-overhead local LLM agent runtime."""

    def __init__(
        self,
        model_path: str = "gemma-2b-q4.gguf",
        port: int = 8080,
        max_history_turns: int = 6,
    ):
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
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file missing! Could not find: {os.path.abspath(model_path)}"
            )

        server_binary = find_llama_server()
        server_dir = os.path.dirname(os.path.abspath(server_binary))
        print(f"[FastAgent] Booting C++ engine via '{server_binary}' on port {self.port}...")

        cmd = [
            server_binary,
            "-m", os.path.abspath(model_path),
            "-c", "2048",
            "-ctk", "q8_0",
            "-ctv", "q8_0",
            "--port", str(self.port),
        ]

        # cwd=server_dir ensures Windows resolves DLLs located in the server's folder
        self.server_process = subprocess.Popen(
            cmd,
            cwd=server_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Wait for port to open while checking if process exited
        for _ in range(60):  # 30-second timeout
            if self.server_process.poll() is not None:
                # Process crashed, grab whatever output was produced
                output, _ = self.server_process.communicate()
                raise RuntimeError(
                    f"llama-server crashed during startup (Exit code: {self.server_process.returncode})!\n"
                    f"Output:\n{output}"
                )

            try:
                with socket.create_connection(("localhost", self.port), timeout=0.5):
                    print("[FastAgent] Engine online and ready.")
                    return
            except OSError:
                time.sleep(0.5)

        self.shutdown()
        raise TimeoutError("Timed out waiting for llama-server to listen on port.")

    def shutdown(self):
        if self.server_process:
            self.server_process.terminate()
            self.server_process.wait()
            print("[FastAgent] Engine shut down cleanly.")

    def tool(self, description: str):
        def decorator(func):
            sig = inspect.signature(func)
            self.tools[func.__name__] = {
                "callable": func,
                "description": description,
                "signature": str(sig),
            }
            return func

        return decorator

    def _build_system_prompt(self) -> str:
        prompt = "You are a direct, autonomous agent.\nTools:\n"
        for name, data in self.tools.items():
            prompt += f"- {name}{data['signature']}: {data['description']}\n"
        prompt += (
            "\nFormat strictly as:\n"
            "Thought: <reasoning>\n"
            "Action: <tool_name>\n"
            "Action Input: <args>\n"
            "Observation: <will be provided>\n"
            "Final Answer: <conclusion>\n"
        )
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
        """Dynamically budgets compute and runs the agent loop."""
        budget = AdaptiveComputeController.inspect_budget(
            user_prompt, has_tools=bool(self.tools)
        )
        print(f"[FastAgent Router] Dynamic Budget Tier: {budget['compute_tier']}")

        if not self.system_prompt_text:
            self.system_prompt_text = self._build_system_prompt()

        current_input = f"User: {user_prompt.strip()}\nAgent:"

        for step in range(max_steps):
            self.conversation_history.append(current_input)
            if len(self.conversation_history) > self.max_history_turns:
                self.conversation_history = self.conversation_history[
                    -self.max_history_turns :
                ]

            full_prompt = (
                f"{self.system_prompt_text}\n\n"
                + "\n".join(self.conversation_history)
            )

            payload = {
                "prompt": full_prompt,
                "n_predict": budget["max_tokens"],
                "temperature": budget["temperature"],
                "cache_prompt": True,
                "stop": ["\nObservation:", "\nUser:"],
            }

            resp = requests.post(self.endpoint, json=payload).json()
            output = resp.get("content", "").strip()
            self.conversation_history[-1] += f" {output}"

            if "Final Answer:" in output:
                return output.split("Final Answer:")[-1].strip()

            if not budget["allow_tools"]:
                return output

            match = re.search(r"Action:\s*(.*?)\nAction Input:\s*(.*)", output)
            if match:
                t_name, t_args = match.group(1).strip(), match.group(2).strip()
                obs = self._exec_tool(t_name, t_args)
                current_input = f"Observation: {obs}\nAgent:"
            else:
                return output

        return "Max steps reached without a final answer."