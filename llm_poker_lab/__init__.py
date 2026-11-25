"""Evolve six-max no-limit hold'em bots with a local LLM.

Baselines and LLM-written bots share one ``act(state)`` interface.
A match deals duplicate hands through PokerKit, and an evolution loop
asks Ollama to rewrite the weakest idea after each measured generation.
"""

__version__ = "0.1.0"
