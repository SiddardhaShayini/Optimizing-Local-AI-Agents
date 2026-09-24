import os
import shutil
import streamlit as st

st.set_page_config(page_title="FastAgent Setup Wizard", layout="centered")

st.title("⚡ FastAgent Setup Wizard")
st.markdown("This diagnostic tool ensures your edge environment is ready to run the `FastAgent` inference orchestrator.")

def find_llama_server():
    local_bin = os.path.abspath("llama-server.exe")
    if os.path.exists(local_bin):
        return local_bin
    system_bin = shutil.which("llama-server") or shutil.which("llama-server.exe")
    return system_bin

def find_gguf_models():
    return [f for f in os.listdir('.') if f.endswith('.gguf')]

st.header("1. Core Engine Check")
server_path = find_llama_server()

if server_path:
    st.success(f"✅ Found `llama-server` executable at: `{server_path}`")
else:
    st.error("❌ `llama-server` executable not found.")
    st.markdown("""
    **How to fix:**
    1. Download the latest `llama.cpp` release for your OS.
    2. Extract the archive.
    3. Place `llama-server.exe` and its `.dll` files in this directory.
    """)

st.header("2. Model Check")
models = find_gguf_models()

if models:
    st.success(f"✅ Found Quantized Models: {', '.join(models)}")
    selected_model = st.selectbox("Select model for quickstart test:", models)
else:
    st.error("❌ No `.gguf` model files found in this directory.")
    st.markdown("""
    **How to fix:**
    Download a highly quantized model suitable for 8GB RAM CPUs. 
    *Recommended: [Gemma-2B-Q4_K_M (Hugging Face)](https://huggingface.co/)*
    """)

st.header("3. Quickstart Verification")
if server_path and models:
    st.markdown("Your environment is perfectly configured. Copy this code to launch your first autonomous agent:")
    code = f"""from fast_agent import FastAgent

# Boots the C++ backend and loads the KV cache
agent = FastAgent(model_path="{selected_model}")

@agent.tool("Adds two numbers together. Pass as (a, b)")
def add_numbers(a: int, b: int) -> int:
    return a + b

print(agent.ask("What is 144 + 256? Use the tool to find out."))
"""
    st.code(code, language="python")
else:
    st.warning("⚠️ Please resolve the engine and model dependencies above before running FastAgent.")