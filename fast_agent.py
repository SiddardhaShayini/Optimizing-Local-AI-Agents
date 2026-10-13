# fast_agent.py
"""
FastAgent: Hardware-Adaptive Local LLM Inference Orchestrator
Root module re-exporting core implementation.
"""
from fast_agent_pkg.fast_agent import (
    FastAgent,
    FastAgentEngine,
    AdaptiveComputeController,
    find_llama_server,
    find_free_port,
    is_port_in_use,
    safe_parse_args,
    REACT_GRAMMAR,
)

__all__ = [
    "FastAgent",
    "FastAgentEngine",
    "AdaptiveComputeController",
    "find_llama_server",
    "find_free_port",
    "is_port_in_use",
    "safe_parse_args",
    "REACT_GRAMMAR",
]
