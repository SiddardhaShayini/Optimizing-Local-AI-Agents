# fast_agent/core.py
"""
FastAgent: Hardware-Adaptive Local LLM Inference Orchestrator
Optimized for resource-constrained edge devices (CPU-only, 8GB RAM).
"""
import ast
import atexit
import inspect
import json
import os
import re
import shutil
import socket
import subprocess
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import requests

# Standard GBNF grammar to enforce ReAct structured formatting on smaller models (<3B)
REACT_GRAMMAR = r"""
root ::= (thought-action | thought-final)
thought-action ::= "Thought: " [^\n]+ "\nAction: " [a-zA-Z0-9_]+ "\nAction Input: " [^\n]+
thought-final ::= "Thought: " [^\n]+ "\nFinal Answer: " [^\n]+
""".strip()


def find_llama_server() -> str:
    """Locates llama-server.exe in package directory, current working directory, or system PATH."""
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(pkg_dir))
    candidates = [
        os.path.join(pkg_dir, "llama-server.exe"),
        os.path.abspath("llama-server.exe"),
        os.path.join(project_root, "llama-server.exe"),
        os.path.join(os.path.dirname(pkg_dir), "llama-server.exe"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return os.path.abspath(path)

    system_bin = shutil.which("llama-server") or shutil.which("llama-server.exe")
    if system_bin:
        return os.path.abspath(system_bin)

    raise FileNotFoundError(
        "Could not find 'llama-server.exe'. Please ensure it exists inside "
        f"'{pkg_dir}', in your project root, or in system PATH."
    )


def is_port_in_use(port: int, host: str = "localhost") -> bool:
    """Checks whether a TCP port is currently occupied."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def find_free_port(start_port: int = 8080) -> int:
    """Returns start_port if free, otherwise binds an ephemeral available port."""
    if not is_port_in_use(start_port):
        return start_port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def safe_parse_args(args_str: str) -> Any:
    """
    Safely parses arguments passed by LLMs without using dangerous eval().
    Supports JSON, Python literal tuples/lists/dicts, numbers, and unquoted strings.
    """
    cleaned = args_str.strip()
    if not cleaned:
        return ()

    # Strip surrounding quotes if whole string was quoted
    if (cleaned.startswith('"') and cleaned.endswith('"')) or (
        cleaned.startswith("'") and cleaned.endswith("'")
    ):
        inner = cleaned[1:-1].strip()
        # If the inner content is structured, try parsing that
        if inner.startswith(("{", "[", "(")):
            cleaned = inner

    # 1. Try JSON parsing
    if cleaned.startswith(("{", "[")):
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

    # 2. Try ast.literal_eval for safe Python literals (tuples, numbers, booleans, dicts)
    try:
        return ast.literal_eval(cleaned)
    except (ValueError, SyntaxError):
        pass

    # 3. If it looks like comma-separated values without outer parentheses, try wrapping
    if "," in cleaned and not cleaned.startswith("(") and not cleaned.endswith(")"):
        try:
            return ast.literal_eval(f"({cleaned})")
        except (ValueError, SyntaxError):
            pass

    # 4. Fallback: treat as plain raw string (safe string argument)
    return cleaned


class AdaptiveComputeController:
    """Evaluates task complexity and dynamically allocates token and compute budgets."""

    @staticmethod
    def inspect_budget(
        query: str,
        has_tools: bool = False,
        tool_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        q = query.lower().strip()
        words = q.split()

        # Keywords suggesting calculation, tool use, or external data retrieval
        tool_keywords = [
            "calculate", "find", "search", "run", "multiply", "add", "divide",
            "subtract", "+", "*", "/", "-", "%", "math", "lookup", "fetch",
            "check", "read", "write", "open", "execute", "count"
        ]
        if tool_names:
            tool_keywords.extend([t.lower() for t in tool_names])

        has_tool_cue = any(k in q for k in tool_keywords)

        # Tier 1: Reflex (Short definitions, simple greetings without tool indicators)
        if (
            len(words) < 8
            and not has_tool_cue
            and not any(k in q for k in ["why", "how", "solve", "code", "analyze", "summarize"])
        ):
            return {
                "compute_tier": "REFLEX",
                "max_tokens": 64,
                "temperature": 0.0,
                "allow_tools": False,
            }

        # Tier 2: Analytical / Tool Task
        if has_tool_cue and has_tools:
            return {
                "compute_tier": "TOOL_ASSISTED",
                "max_tokens": 160,
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
    """
    Hardware-adaptive, zero-overhead local LLM agent runtime.
    Features: 8-bit quantized KV cache, LCP prompt caching, safe AST tool calling,
    GBNF grammar constraints, rolling context, and subprocess deadlock prevention.
    """

    def __init__(
        self,
        model_path: str = "gemma-2b-q4.gguf",
        port: int = 8080,
        max_history_turns: int = 6,
        auto_port: bool = True,
        context_size: int = 2048,
    ):
        self.requested_port = port
        self.port = find_free_port(port) if auto_port else port
        self.endpoint = f"http://localhost:{self.port}/completion"
        self.max_history_turns = max_history_turns
        self.context_size = context_size
        self.system_prompt_text = ""
        self.conversation_history: List[str] = []
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.server_process: Optional[subprocess.Popen] = None
        self._drain_thread: Optional[threading.Thread] = None
        self._stop_draining = threading.Event()
        self._startup_logs: List[str] = []

        self._boot_engine(model_path)
        atexit.register(self.shutdown)

    def _drain_stdout(self):
        """Continuously drains stdout in a background daemon thread to prevent pipe buffer deadlock."""
        if not self.server_process or not self.server_process.stdout:
            return
        try:
            for line in iter(self.server_process.stdout.readline, ""):
                if self._stop_draining.is_set():
                    break
        except Exception:
            pass

    def _boot_engine(self, model_path: str):
        # Resolve model path across candidate directories
        resolved_model = os.path.abspath(model_path)
        if not os.path.exists(resolved_model):
            # Check package dir and current dir
            pkg_model = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_path)
            if os.path.exists(pkg_model):
                resolved_model = pkg_model
            else:
                raise FileNotFoundError(
                    f"Model file missing! Could not find: {resolved_model}"
                )

        # Check if an existing llama-server is already running on this port
        if is_port_in_use(self.port):
            try:
                test_resp = requests.get(f"http://localhost:{self.port}/health", timeout=1.0)
                if test_resp.status_code == 200:
                    print(f"[FastAgent] Reusing existing healthy engine on port {self.port}.")
                    return
            except Exception:
                pass

        server_binary = find_llama_server()
        server_dir = os.path.dirname(os.path.abspath(server_binary))
        print(f"[FastAgent] Booting C++ engine via '{server_binary}' on port {self.port}...")

        cmd = [
            server_binary,
            "-m", resolved_model,
            "-c", str(self.context_size),
            "-ctk", "q8_0",
            "-ctv", "q8_0",
            "--port", str(self.port),
        ]

        self.server_process = subprocess.Popen(
            cmd,
            cwd=server_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Wait for port to become active, collecting startup logs
        start_time = time.time()
        timeout_seconds = 45.0
        while time.time() - start_time < timeout_seconds:
            # Check if process crashed prematurely
            if self.server_process.poll() is not None:
                remaining_out = ""
                try:
                    remaining_out, _ = self.server_process.communicate(timeout=2.0)
                except Exception:
                    pass
                log_dump = "\n".join(self._startup_logs) + "\n" + remaining_out
                raise RuntimeError(
                    f"llama-server crashed during startup (Exit code: {self.server_process.returncode})!\n"
                    f"Output:\n{log_dump.strip()}"
                )

            # Check if port is receptive
            try:
                with socket.create_connection(("localhost", self.port), timeout=0.5):
                    print(f"[FastAgent] Engine online and ready on port {self.port}.")
                    # Launch background daemon thread to drain output and prevent OS buffer saturation
                    self._drain_thread = threading.Thread(target=self._drain_stdout, daemon=True)
                    self._drain_thread.start()
                    return
            except OSError:
                time.sleep(0.4)

        self.shutdown()
        raise TimeoutError(f"Timed out ({timeout_seconds}s) waiting for llama-server on port {self.port}.")

    def shutdown(self):
        """Safely shuts down the local engine process and frees resources."""
        self._stop_draining.set()
        if self.server_process and self.server_process.poll() is None:
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=3.0)
            except (subprocess.TimeoutExpired, Exception):
                try:
                    self.server_process.kill()
                except Exception:
                    pass
            print("[FastAgent] Engine shut down cleanly.")
        self.server_process = None

    def tool(self, description: str) -> Callable:
        """Decorator to register Python functions as agent tools."""
        def decorator(func: Callable) -> Callable:
            sig = inspect.signature(func)
            self.tools[func.__name__] = {
                "callable": func,
                "description": description,
                "signature": str(sig),
                "param_names": list(sig.parameters.keys()),
            }
            return func

        return decorator

    def _build_system_prompt(self) -> str:
        prompt = "You are an autonomous problem-solving agent.\nAvailable Tools:\n"
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
        """Safely executes a registered tool using ast.literal_eval / JSON parsing."""
        if name not in self.tools:
            return f"Error: Tool '{name}' not found."

        tool_meta = self.tools[name]
        func = tool_meta["callable"]
        param_names = tool_meta["param_names"]

        try:
            parsed_args = safe_parse_args(args_str)

            # Dictionary args (kwargs)
            if isinstance(parsed_args, dict):
                return str(func(**parsed_args))

            # Tuple or List of positional arguments
            if isinstance(parsed_args, (tuple, list)):
                if len(param_names) == 1 and len(parsed_args) > 1:
                    # Target function expects a single collection
                    return str(func(parsed_args))
                return str(func(*parsed_args))

            # Single scalar or string argument
            if len(param_names) == 0:
                return str(func())
            return str(func(parsed_args))

        except TypeError as te:
            # Fallback: if single arg was parsed as scalar but func expects string representation
            try:
                return str(func(args_str.strip()))
            except Exception:
                return f"Tool argument error: {te}"
        except Exception as e:
            return f"Error executing tool '{name}': {e}"

    # Alias for compatibility with framework_engine
    execute_tool = _exec_tool

    def clear_history(self):
        """Clears accumulated conversation turns."""
        self.conversation_history = []

    def run_turn(
        self,
        new_input: str,
        max_tokens: int = 160,
        temperature: float = 0.1,
        grammar: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Runs a single turn with rolling context window and LCP slot caching.
        Returns response text and prompt evaluation timing.
        """
        if not self.system_prompt_text:
            self.system_prompt_text = self._build_system_prompt()

        self.conversation_history.append(f"User: {new_input.strip()}\nAgent:")

        if len(self.conversation_history) > self.max_history_turns:
            self.conversation_history = self.conversation_history[-self.max_history_turns:]

        full_prompt = f"{self.system_prompt_text}\n\n" + "\n".join(self.conversation_history)

        payload: Dict[str, Any] = {
            "prompt": full_prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "cache_prompt": True,
            "stop": ["\nObservation:", "\nUser:"],
        }
        if grammar:
            payload["grammar"] = grammar

        resp = requests.post(self.endpoint, json=payload)
        resp.raise_for_status()

        data = resp.json()
        output_text = data.get("content", "").strip()
        self.conversation_history[-1] += f" {output_text}"

        prompt_eval_ms = round(data.get("timings", {}).get("prompt_ms", 0.0), 2)
        return {
            "output": output_text,
            "prompt_eval_ms": prompt_eval_ms,
        }

    def run_structured(
        self,
        prompt: str,
        json_schema: Dict[str, Any],
        max_tokens: int = 256,
    ) -> Dict[str, Any]:
        """Bypasses history to extract 100% syntactically valid JSON using llama-server schema constraint."""
        payload = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": 0.0,
            "json_schema": json_schema,
            "cache_prompt": False,
        }

        resp = requests.post(self.endpoint, json=payload)
        resp.raise_for_status()
        raw_content = resp.json().get("content", "").strip()
        return json.loads(raw_content)

    def ask(
        self,
        user_prompt: str,
        max_steps: int = 5,
        use_grammar: bool = False,
        custom_grammar: Optional[str] = None,
    ) -> str:
        """
        Autonomous ReAct execution loop with dynamic compute budgeting,
        safe tool calling, and LCP cache reuse.
        """
        budget = AdaptiveComputeController.inspect_budget(
            user_prompt,
            has_tools=bool(self.tools),
            tool_names=list(self.tools.keys()),
        )
        print(f"[FastAgent Router] Dynamic Budget Tier: {budget['compute_tier']}")

        if not self.system_prompt_text:
            self.system_prompt_text = self._build_system_prompt()

        current_input = f"User: {user_prompt.strip()}\nAgent:"

        for _ in range(max_steps):
            self.conversation_history.append(current_input)
            if len(self.conversation_history) > self.max_history_turns:
                self.conversation_history = self.conversation_history[-self.max_history_turns:]

            full_prompt = f"{self.system_prompt_text}\n\n" + "\n".join(self.conversation_history)

            payload: Dict[str, Any] = {
                "prompt": full_prompt,
                "n_predict": budget["max_tokens"],
                "temperature": budget["temperature"],
                "cache_prompt": True,
                "stop": ["\nObservation:", "\nUser:"],
            }

            if custom_grammar:
                payload["grammar"] = custom_grammar
            elif use_grammar and bool(self.tools):
                payload["grammar"] = REACT_GRAMMAR

            try:
                resp = requests.post(self.endpoint, json=payload)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                return f"Error communicating with local inference engine: {e}"

            output = data.get("content", "").strip()
            self.conversation_history[-1] += f" {output}"

            if "Final Answer:" in output:
                return output.split("Final Answer:")[-1].strip()

            # Attempt to parse tool call
            match = re.search(r"Action:\s*(.*?)\nAction Input:\s*(.*)", output, re.DOTALL)
            if match:
                t_name = match.group(1).strip()
                t_args = match.group(2).strip()
                # If there are trailing lines, take the first line or clean block
                if "\nObservation:" in t_args:
                    t_args = t_args.split("\nObservation:")[0].strip()

                obs = self._exec_tool(t_name, t_args)
                current_input = f"Observation: {obs}\nAgent:"
            else:
                # If tier was REFLEX and no tool called, return direct response
                if not budget["allow_tools"] or not self.tools:
                    return output
                return output

        return "Max steps reached without a final answer."


# Direct alias providing 100% backward compatibility with framework_engine
FastAgentEngine = FastAgent