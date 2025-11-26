"""
Neuro-Symbolic Autonomy Framework (NSAF) - Core Package
======================================================

Author: Bolorerdene Bundgaa
Contact: bolor@ariunbolor.org
Website: https://bolor.me

Core package initialization for the Neuro-Symbolic Autonomy Framework.
Exports all main classes and functions for the framework components.
"""

from .framework import NeuroSymbolicAutonomyFramework
from .task_clustering import QuantumSymbolicTaskClustering, TaskCluster
from .scma import SCMA, AgentGenome, AgentPopulation
from .memory_tuning import HyperSymbolicMemory, MemoryNode, MemoryGraph
from .recursive_intent import (
    RecursiveIntentProjection, 
    Intent, 
    IntentProjection, 
    IntentGraph
)
from .human_ai_synergy import (
    HumanAISynergy,
    CognitiveState,
    SyncEvent,
    CognitiveBuffer
)
from .foundation_models import NSAFFoundationModels, FoundationModelClient, ModelProvider
from .mcp_interface import NSAFMCPServer

__version__ = "1.0.0"
__author__ = "Bolorerdene Bundgaa"
__email__ = "bolor@ariunbolor.org"
__website__ = "https://bolor.me"

__all__ = [
    # Main framework
    "NeuroSymbolicAutonomyFramework",
    
    # Task clustering
    "QuantumSymbolicTaskClustering",
    "TaskCluster",
    
    # Self-Constructing Meta-Agents
    "SCMA",
    "AgentGenome",
    "AgentPopulation",
    
    # Memory tuning
    "HyperSymbolicMemory",
    "MemoryNode",
    "MemoryGraph",
    
    # Recursive intent projection
    "RecursiveIntentProjection",
    "Intent",
    "IntentProjection",
    "IntentGraph",
    
    # Human-AI synergy
    "HumanAISynergy",
    "CognitiveState",
    "SyncEvent",
    "CognitiveBuffer",
    
    # Foundation models
    "NSAFFoundationModels",
    "FoundationModelClient", 
    "ModelProvider",
    
    # MCP interface
    "NSAFMCPServer"
]

# Framework configuration
DEFAULT_CONFIG_PATH = "config/config.yaml"

# Framework components
COMPONENTS = {
    "task_clustering": QuantumSymbolicTaskClustering,
    "scma": SCMA,
    "memory_tuning": HyperSymbolicMemory,
    "recursive_intent": RecursiveIntentProjection,
    "human_ai_synergy": HumanAISynergy
}

def create_framework(config_path: str = DEFAULT_CONFIG_PATH) -> NeuroSymbolicAutonomyFramework:
    """
    Create a new instance of the Neuro-Symbolic Autonomy Framework.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        NeuroSymbolicAutonomyFramework: Framework instance
    """
    return NeuroSymbolicAutonomyFramework(config_path)
