"""
Self-Constructing Meta-Agents (SCMA) module.

This module implements AI agents that can self-design new AI agents
using Generative Architecture Models.
"""

from .meta_agent import MetaAgent
from .agent_factory import AgentFactory
from .evolution import Evolution

__all__ = ['MetaAgent', 'AgentFactory', 'Evolution']
