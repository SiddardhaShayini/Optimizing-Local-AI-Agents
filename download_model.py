#!/usr/bin/env python3
"""
download_model.py: Convenience script to download quantized GGUF models for FastAgent.
Zero extra dependencies (uses standard library urllib).
"""
import argparse
import os
import sys
import urllib.request

MODELS = {
    "gemma-2b-q4": {
        "url": "https://huggingface.co/google/gemma-2b-it-GGUF/resolve/main/2b_it_v2.gguf",
        "filename": "gemma-2b-q4.gguf",
        "description": "Google Gemma 2B Instruct (Q4 GGUF)",
    },
    "qwen2.5-1.5b": {
        "url": "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "filename": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "description": "Qwen 2.5 1.5B Instruct (Q4_K_M GGUF - Great for ReAct tool use)",
    },
    "llama-3.2-1b": {
        "url": "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "filename": "llama-3.2-1b-instruct-q4_k_m.gguf",
        "description": "Meta Llama 3.2 1B Instruct (Q4_K_M GGUF - Ultra-lightweight)",
    },
}


def download_file(url: str, dest_path: str):
    print(f"Downloading from: {url}")
    print(f"Saving to:       {dest_path}")

    def reporthook(count, block_size, total_size):
        if total_size <= 0:
            return
        downloaded = count * block_size
        percent = min(100.0, downloaded * 100.0 / total_size)
        mb_down = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        sys.stdout.write(f"\rProgress: [{percent:5.1f}%] {mb_down:.1f} MB / {mb_total:.1f} MB")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, dest_path, reporthook=reporthook)
        print("\nDownload complete!")
    except Exception as e:
        print(f"\nDownload failed: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Download GGUF models for FastAgent.")
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        default="qwen2.5-1.5b",
        help="Model identifier to download (default: qwen2.5-1.5b)",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to save the downloaded model (default: current directory)",
    )
    args = parser.parse_args()

    meta = MODELS[args.model]
    dest = os.path.join(args.output_dir, meta["filename"])

    if os.path.exists(dest):
        print(f"Model already exists at '{dest}'.")
        return

    print(f"Selected: {meta['description']}")
    download_file(meta["url"], dest)


if __name__ == "__main__":
    main()
