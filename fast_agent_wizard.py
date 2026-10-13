import os
import psutil
import streamlit as st
from fast_agent import find_llama_server

st.set_page_config(page_title="FastAgent Setup Wizard", page_icon="⚡", layout="centered")

st.title("⚡ FastAgent Edge Setup Wizard")
st.markdown(
    "Diagnostic dashboard to verify hardware readiness, C++ engine binaries, and quantized GGUF models for edge AI agents."
)

# Hardware Diagnostics
st.header("1. Hardware & System Resources")
col1, col2 = st.columns(2)
mem = psutil.virtual_memory()
total_ram_gb = round(mem.total / (1024**3), 2)
available_ram_gb = round(mem.available / (1024**3), 2)
cpu_cores = os.cpu_count() or 1

with col1:
    st.metric("Total System RAM", f"{total_ram_gb} GB")
    st.metric("Available RAM", f"{available_ram_gb} GB")

with col2:
    st.metric("CPU Logical Cores", f"{cpu_cores}")
    if total_ram_gb <= 8.5:
        st.info("💡 Target Profile: **Constrained Edge Device (8GB RAM)**")
    else:
        st.success("✅ Ample RAM for 8-bit quantized inference")

# Binary Discovery
st.header("2. Core Engine Binary Check")
try:
    server_path = find_llama_server()
    st.success(f"✅ Found `llama-server` binary at: `{server_path}`")
except FileNotFoundError:
    server_path = None
    st.error("❌ `llama-server` executable not found.")
    st.markdown("""
    **Resolution Steps:**
    1. Download the latest `llama.cpp` Windows/Linux release archive.
    2. Extract `llama-server.exe` and its companion DLLs into this directory or add to your system `PATH`.
    """)

# Model Discovery
st.header("3. Quantized Model Discovery")


def find_gguf_models():
    models = []
    # Check current directory and subdirectories
    for root, _, files in os.walk("."):
        for f in files:
            if f.endswith(".gguf"):
                models.append(os.path.relpath(os.path.join(root, f), "."))
    return sorted(list(set(models)))


models = find_gguf_models()

if models:
    st.success(f"✅ Discovered {len(models)} GGUF Model(s): `{', '.join(models)}`")
    selected_model = st.selectbox("Select model for quickstart test:", models)
else:
    selected_model = None
    st.error("❌ No `.gguf` model files found.")
    st.markdown("""
    **Resolution Steps:**
    Download a quantized GGUF model (e.g. `gemma-2b-q4.gguf` or `Qwen2.5-1.5B-Instruct-Q4_K_M.gguf`).
    You can run `python download_model.py` to fetch a recommended model automatically.
    """)

# Quickstart Verification
st.header("4. Quickstart Verification")
if server_path and selected_model:
    st.markdown("Your edge environment is fully configured! Launch your autonomous agent with:")
    code = f'''from fast_agent import FastAgent

# Boots the C++ backend with 8-bit KV quantization & slot caching
agent = FastAgent(model_path="{selected_model}")

@agent.tool("Adds two numbers together. Pass as (a, b)")
def add_numbers(a: int, b: int) -> int:
    return a + b

response = agent.ask("What is 144 + 256? Use the tool to find out.")
print("Final Answer:", response)
'''
    st.code(code, language="python")
else:
    st.warning("⚠️ Please resolve the engine or model dependencies above before running FastAgent.")