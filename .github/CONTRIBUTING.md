# Contributing to FastAgent

We welcome pull requests! Because this framework targets resource-constrained edge devices, all contributions must maintain the zero-overhead philosophy.

1. **Fork the repository** and create your branch from `main`.
2. **Test your code** on a local CPU environment (no GPU required).
3. Ensure `fast_agent.py` still successfully orchestrates the `llama-server` subprocess without memory leaks.
4. Issue your Pull Request with a clear description of the latency or feature improvements.