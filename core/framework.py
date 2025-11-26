"""
Neuro-Symbolic Autonomy Framework (NSAF) - Main Framework
=========================================================

Author: Bolorerdene Bundgaa
Contact: bolor@ariunbolor.org
Website: https://bolor.me

Main framework class that integrates all components of the
Neuro-Symbolic Autonomy Framework.
"""

from typing import List, Dict, Any, Optional, Union
import yaml
import logging
from pathlib import Path
import asyncio
import torch
from datetime import datetime

from .task_clustering import QuantumSymbolicTaskClustering, TaskCluster
from .scma import SCMA, AgentGenome
from .memory_tuning import HyperSymbolicMemory, MemoryNode
from .recursive_intent import RecursiveIntentProjection, Intent
from .human_ai_synergy import HumanAISynergy, CognitiveState

class NeuroSymbolicAutonomyFramework:
    """
    Main framework class that integrates all components of the
    Neuro-Symbolic Autonomy Framework.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        self.task_clustering = QuantumSymbolicTaskClustering(config_path)
        self.scma = SCMA(config_path)
        self.memory_tuning = HyperSymbolicMemory(config_path)
        self.recursive_intent = RecursiveIntentProjection(config_path)
        self.human_ai_synergy = HumanAISynergy(config_path)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize state
        self.active_agents: Dict[str, AgentGenome] = {}
        self.task_clusters: Dict[str, TaskCluster] = {}
        self.current_intent: Optional[Intent] = None
        
        self.logger.info("Initialized Neuro-Symbolic Autonomy Framework")

    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a complex task through the framework pipeline.
        
        Args:
            task: Dictionary containing task description and requirements
            
        Returns:
            Dict[str, Any]: Processing results and generated agents
        """
        try:
            # 1. Quantum-Symbolic Task Clustering
            self.logger.info("Starting task clustering...")
            clusters = self.task_clustering.decompose_problem(task)
            for cluster in clusters:
                self.task_clusters[cluster.id] = cluster
            
            # 2. Generate agents through SCMA
            self.logger.info("Generating agents...")
            for cluster in clusters:
                agent = self.scma.generate_optimal_agent({
                    'tasks': cluster.tasks,
                    'constraints': task.get('constraints', {}),
                    'requirements': task.get('requirements', {})
                })
                self.active_agents[agent.id] = agent
            
            # 3. Apply Hyper-Symbolic Memory Tuning
            self.logger.info("Applying memory tuning...")
            for agent_id, agent in self.active_agents.items():
                memory_node = MemoryNode(
                    id=f"agent_{agent_id}_memory",
                    content={
                        'architecture': agent.architecture,
                        'capabilities': agent.capabilities,
                        'performance': agent.fitness
                    },
                    type='agent_state',
                    timestamp=datetime.now().timestamp(),
                    importance=agent.fitness,
                    connections=set()
                )
                self.memory_tuning.add_memory(
                    content=memory_node.content,
                    memory_type=memory_node.type
                )
            
            # 4. Initialize Intent Projection
            self.logger.info("Initializing intent projection...")
            initial_intent = Intent(
                id=f"task_{int(datetime.now().timestamp())}",
                description=task.get('description', ''),
                goals=task.get('goals', []),
                constraints=task.get('constraints', []),
                context={
                    'task_clusters': len(clusters),
                    'active_agents': len(self.active_agents),
                    'complexity': task.get('complexity', 0.5)
                },
                confidence=0.8,
                timestamp=datetime.now().timestamp()
            )
            
            # Optimize intent
            self.current_intent = self.recursive_intent.recursive_optimize(
                initial_intent,
                {
                    'task_progress': 0.0,
                    'agent_performance': 0.0,
                    'system_load': 0.0
                }
            )
            
            # 5. Enable Human-AI Synergy
            self.logger.info("Enabling human-AI synergy...")
            cognitive_state = CognitiveState(
                id=f"cognitive_{int(datetime.now().timestamp())}",
                attention={
                    'task_execution': 0.8,
                    'performance_monitoring': 0.6,
                    'error_handling': 0.4
                },
                context={
                    'task': task,
                    'active_agents': len(self.active_agents),
                    'progress': 0.0
                },
                goals=task.get('goals', []),
                mental_model={
                    'understanding': 0.8,
                    'confidence': 0.7
                },
                timestamp=datetime.now().timestamp(),
                confidence=0.8
            )
            
            self.human_ai_synergy.update_human_state(cognitive_state)
            self.human_ai_synergy.start_sync()
            
            # Return results
            return {
                'status': 'success',
                'task_clusters': [
                    {
                        'id': cluster.id,
                        'size': len(cluster.tasks),
                        'similarity_score': cluster.similarity_score
                    }
                    for cluster in clusters
                ],
                'agents': [
                    {
                        'id': agent.id,
                        'capabilities': agent.capabilities,
                        'fitness': agent.fitness
                    }
                    for agent in self.active_agents.values()
                ],
                'intent': {
                    'id': self.current_intent.id,
                    'confidence': self.current_intent.confidence,
                    'goals': self.current_intent.goals
                },
                'metrics': {
                    'clustering': {
                        'num_clusters': len(clusters),
                        'avg_similarity': np.mean([
                            c.similarity_score for c in clusters
                        ])
                    },
                    'agents': {
                        'num_agents': len(self.active_agents),
                        'avg_fitness': np.mean([
                            a.fitness for a in self.active_agents.values()
                        ])
                    },
                    'memory': self.memory_tuning.get_metrics(),
                    'synergy': self.human_ai_synergy.get_metrics()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error processing task: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }

    async def update_task_state(self, 
                              task_id: str, 
                              state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update the state of a running task.
        
        Args:
            task_id: ID of the task to update
            state: New state information
            
        Returns:
            Dict[str, Any]: Updated task status
        """
        try:
            # Update cognitive state
            cognitive_state = CognitiveState(
                id=f"update_{int(datetime.now().timestamp())}",
                attention={
                    'task_monitoring': 0.7,
                    'state_update': 0.8,
                    'adaptation': 0.6
                },
                context=state,
                goals=state.get('goals', []),
                mental_model={
                    'understanding': state.get('understanding', 0.7),
                    'confidence': state.get('confidence', 0.7)
                },
                timestamp=datetime.now().timestamp(),
                confidence=state.get('confidence', 0.7)
            )
            
            self.human_ai_synergy.update_human_state(cognitive_state)
            
            # Update intent if needed
            if self.current_intent:
                new_intent = self.recursive_intent.optimize_intent(
                    self.current_intent,
                    self.recursive_intent.project_intent(
                        self.current_intent, state
                    )
                )
                
                if new_intent.confidence > self.current_intent.confidence:
                    self.current_intent = new_intent
            
            # Return updated status
            return {
                'status': 'success',
                'task_id': task_id,
                'current_state': state,
                'intent': {
                    'id': self.current_intent.id if self.current_intent else None,
                    'confidence': (self.current_intent.confidence 
                                 if self.current_intent else 0.0)
                },
                'metrics': {
                    'memory': self.memory_tuning.get_metrics(),
                    'synergy': self.human_ai_synergy.get_metrics()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error updating task state: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }

    def get_active_agents(self) -> List[Dict[str, Any]]:
        """Get information about all active agents"""
        return [
            {
                'id': agent.id,
                'capabilities': agent.capabilities,
                'fitness': agent.fitness,
                'generation': agent.generation,
                'parent_ids': agent.parent_ids
            }
            for agent in self.active_agents.values()
        ]

    def get_task_clusters(self) -> List[Dict[str, Any]]:
        """Get information about all task clusters"""
        return [
            {
                'id': cluster.id,
                'tasks': len(cluster.tasks),
                'similarity_score': cluster.similarity_score,
                'quantum_state': (cluster.quantum_state.tolist() 
                                if cluster.quantum_state is not None else None)
            }
            for cluster in self.task_clusters.values()
        ]

    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        return {
            'active_agents': len(self.active_agents),
            'task_clusters': len(self.task_clusters),
            'current_intent': {
                'id': self.current_intent.id if self.current_intent else None,
                'confidence': (self.current_intent.confidence 
                             if self.current_intent else 0.0)
            },
            'memory_status': self.memory_tuning.get_metrics(),
            'synergy_status': self.human_ai_synergy.get_sync_status(),
            'timestamp': datetime.now().timestamp()
        }

    async def shutdown(self) -> None:
        """Gracefully shutdown the framework"""
        try:
            # Stop synchronization
            self.human_ai_synergy.stop_sync()
            
            # Clear active agents
            self.active_agents.clear()
            
            # Clear task clusters
            self.task_clusters.clear()
            
            # Reset intent
            self.current_intent = None
            
            self.logger.info("Framework shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
