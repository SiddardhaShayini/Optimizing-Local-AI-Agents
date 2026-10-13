# fast_agent/fast_agent.py
"""
Backwards-compatibility re-export module.
Imports unified FastAgent implementation from core.
"""
from .core import (
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
