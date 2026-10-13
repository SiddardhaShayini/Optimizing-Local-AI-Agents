# framework_engine.py
"""
Backwards-compatibility wrapper for FastAgentEngine.
Bridges legacy script imports to the unified FastAgent orchestrator.
"""
from fast_agent_pkg.fast_agent import (
    FastAgent,
    FastAgentEngine,
    AdaptiveComputeController,
    find_llama_server,
    find_free_port,
    safe_parse_args,
    REACT_GRAMMAR,
)

__all__ = [
    "FastAgent",
    "FastAgentEngine",
    "AdaptiveComputeController",
    "find_llama_server",
    "find_free_port",
    "safe_parse_args",
    "REACT_GRAMMAR",
]