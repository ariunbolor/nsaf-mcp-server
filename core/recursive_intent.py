"""
Neuro-Symbolic Autonomy Framework (NSAF) - Recursive Intent Projection
=====================================================================

Author: Bolorerdene Bundgaa
Contact: bolor@ariunbolor.org
Website: https://bolor.me

Recursive Intent Projection (RIP) implementation for continuous self-optimization,
intent-based learning, and multi-step projection planning.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import yaml
import logging
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import networkx as nx
from datetime import datetime
import ray

@dataclass
class Intent:
    """Represents an agent's intent"""
    id: str
    description: str
    goals: List[Dict[str, Any]]
    constraints: List[Dict[str, Any]]
    context: Dict[str, Any]
    confidence: float
    timestamp: float
    parent_id: Optional[str] = None

@dataclass
class IntentProjection:
    """Represents a projected future intent state"""
    intent: Intent
    projected_state: Dict[str, Any]
    success_probability: float
    required_capabilities: List[str]
    adaptation_steps: List[Dict[str, Any]]

class IntentGraph:
    """Graph structure for tracking intent evolution"""
    def __init__(self):
        self.graph = nx.DiGraph()
        self.current_intent: Optional[Intent] = None
        
    def add_intent(self, intent: Intent) -> None:
        """Add intent to graph"""
        self.graph.add_node(intent.id, 
                           intent=intent,
                           timestamp=intent.timestamp)
        
        if intent.parent_id and intent.parent_id in self.graph:
            self.graph.add_edge(intent.parent_id, intent.id)
        
        self.current_intent = intent
    
    def get_intent_history(self, steps: int = 5) -> List[Intent]:
        """Get recent intent history"""
        if not self.current_intent:
            return []
        
        history = []
        current = self.current_intent
        
        while current and len(history) < steps:
            history.append(current)
            predecessors = list(self.graph.predecessors(current.id))
            current = (self.graph.nodes[predecessors[0]]['intent'] 
                      if predecessors else None)
        
        return history

class RecursiveIntentProjection:
    """
    Implements Recursive Intent Projection (RIP) for continuous agent optimization
    through intent-based learning and adaptation.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['recursive_intent']
        
        # Initialize parameters
        self.projection_depth = self.config['projection_depth']
        self.confidence_threshold = self.config['confidence_threshold']
        self.update_frequency = self.config['update_frequency']
        self.max_iterations = self.config['max_iterations']
        
        # Initialize components
        self.intent_graph = IntentGraph()
        self.intent_encoder = self._build_intent_encoder()
        self.state_predictor = self._build_state_predictor()
        self.adaptation_generator = self._build_adaptation_generator()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize Ray for distributed computing if not already initialized
        if not ray.is_initialized():
            ray.init()

    def _build_intent_encoder(self) -> nn.Module:
        """Build neural encoder for intent representations"""
        return nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    def _build_state_predictor(self) -> nn.Module:
        """Build neural network for state prediction"""
        return nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

    def _build_adaptation_generator(self) -> nn.Module:
        """Build neural network for generating adaptation steps"""
        return nn.Sequential(
            nn.Linear(320, 256),  # 64 (intent) + 256 (state)
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    def _encode_intent(self, intent: Intent) -> torch.Tensor:
        """Encode intent into a vector representation"""
        # Combine intent components into a feature vector
        features = []
        
        # Encode goals
        goal_features = []
        for goal in intent.goals:
            # Convert goal dictionary to feature vector
            # This is a simplified encoding - could be enhanced
            goal_vec = torch.tensor([
                float(goal.get('priority', 0.5)),
                float(goal.get('complexity', 0.5)),
                float(goal.get('progress', 0.0))
            ])
            goal_features.append(goal_vec)
        
        if goal_features:
            goals_tensor = torch.stack(goal_features).mean(dim=0)
            features.append(goals_tensor)
        
        # Encode constraints
        constraint_features = []
        for constraint in intent.constraints:
            # Convert constraint dictionary to feature vector
            constraint_vec = torch.tensor([
                float(constraint.get('importance', 0.5)),
                float(constraint.get('strictness', 0.5))
            ])
            constraint_features.append(constraint_vec)
        
        if constraint_features:
            constraints_tensor = torch.stack(constraint_features).mean(dim=0)
            features.append(constraints_tensor)
        
        # Add context features
        context_vec = torch.tensor([
            float(intent.context.get('complexity', 0.5)),
            float(intent.context.get('urgency', 0.5)),
            intent.confidence,
            float(intent.timestamp) / 1e9  # Normalize timestamp
        ])
        features.append(context_vec)
        
        # Combine all features
        combined_features = torch.cat(features)
        
        # Pad or truncate to expected input dimension (512)
        target_dim = 512
        current_dim = combined_features.shape[0]
        
        if current_dim < target_dim:
            # Pad with zeros
            padding = torch.zeros(target_dim - current_dim)
            combined_features = torch.cat([combined_features, padding])
        elif current_dim > target_dim:
            # Truncate to target dimension
            combined_features = combined_features[:target_dim]
        
        # Ensure we have the correct shape and add batch dimension
        combined_features = combined_features.unsqueeze(0)  # Shape: [1, 512]
        
        # Encode through neural network
        return self.intent_encoder(combined_features.float())

    def _predict_state(self, current_state: Dict[str, Any], 
                      intent_encoding: torch.Tensor, 
                      steps: int) -> List[Dict[str, Any]]:
        """Predict future states based on current state and intent"""
        # Convert state values to floats, handling nested dicts
        state_values = []
        for v in current_state.values():
            if isinstance(v, dict):
                # For nested dicts, take a representative numeric value
                if 'available' in v:
                    state_values.append(1.0 if v['available'] else 0.0)
                else:
                    # Take the first numeric value or default to 0.5
                    numeric_val = next((val for val in v.values() if isinstance(val, (int, float))), 0.5)
                    state_values.append(float(numeric_val))
            else:
                state_values.append(float(v))
        
        state_tensor = torch.tensor(state_values)
        
        # Ensure intent_encoding is properly squeezed to 1D for concatenation
        if intent_encoding.dim() > 1:
            intent_encoding = intent_encoding.squeeze(0)
        
        # Combine state and intent
        combined = torch.cat([state_tensor, intent_encoding])
        
        # Pad or truncate to expected dimension for state predictor (128)
        target_dim = 128
        current_dim = combined.shape[0]
        
        if current_dim < target_dim:
            # Pad with zeros
            padding = torch.zeros(target_dim - current_dim)
            combined = torch.cat([combined, padding])
        elif current_dim > target_dim:
            # Truncate to target dimension
            combined = combined[:target_dim]
        
        # Add batch dimension for neural network
        combined = combined.unsqueeze(0)  # Shape: [1, 128]
        
        # Generate future states
        future_states = []
        current = combined
        
        for _ in range(steps):
            next_state = self.state_predictor(current)
            future_states.append(next_state)
            
            # Adjust dimensions for next iteration
            if next_state.shape[1] != target_dim:
                if next_state.shape[1] > target_dim:
                    current = next_state[:, :target_dim]  # Truncate
                else:
                    # Pad to target dimension
                    padding = torch.zeros(1, target_dim - next_state.shape[1])
                    current = torch.cat([next_state, padding], dim=1)
            else:
                current = next_state
        
        # Convert tensors back to dictionaries
        # Note: Since we may have padded/truncated, we'll create simplified states
        return [
            {
                f"state_dim_{i}": state.squeeze(0)[i].item() if i < state.shape[1] else 0.0
                for i in range(min(len(current_state), state.shape[1]))
            }
            for state in future_states
        ]

    def _generate_adaptation_steps(self, 
                                 intent_encoding: torch.Tensor,
                                 current_state: torch.Tensor,
                                 target_state: torch.Tensor) -> List[Dict[str, Any]]:
        """Generate steps to adapt from current to target state"""
        # Combine intent and state information
        # Ensure all tensors are 1D for concatenation
        if intent_encoding.dim() > 1:
            intent_encoding = intent_encoding.squeeze(0)
        if current_state.dim() > 1:
            current_state = current_state.squeeze(0)
        if target_state.dim() > 1:
            target_state = target_state.squeeze(0)
            
        combined = torch.cat([
            intent_encoding,
            current_state,
            target_state
        ])
        
        # Pad or truncate to expected dimension for adaptation generator (320)
        target_dim = 320
        current_dim = combined.shape[0]
        
        if current_dim < target_dim:
            # Pad with zeros
            padding = torch.zeros(target_dim - current_dim)
            combined = torch.cat([combined, padding])
        elif current_dim > target_dim:
            # Truncate to target dimension
            combined = combined[:target_dim]
        
        # Add batch dimension for neural network
        combined = combined.unsqueeze(0)  # Shape: [1, 320]
        
        # Generate adaptation steps
        adaptation_encoding = self.adaptation_generator(combined)
        
        # Decode adaptation steps (simplified version)
        # Remove batch dimension for processing
        adaptation_encoding = adaptation_encoding.squeeze(0)
        
        steps = []
        step_size = 16  # Size of each adaptation step encoding
        
        for i in range(0, len(adaptation_encoding), step_size):
            step_encoding = adaptation_encoding[i:i+step_size]
            
            # Convert encoding to adaptation step
            step = {
                'type': 'adjust_parameters',  # Placeholder
                'parameters': {
                    'learning_rate': float(step_encoding[0]),
                    'batch_size': int(step_encoding[1] * 128),
                    'optimization_weight': float(step_encoding[2])
                },
                'estimated_improvement': float(step_encoding[3])
            }
            
            steps.append(step)
        
        return steps

    def project_intent(self, current_intent: Intent, 
                      current_state: Dict[str, Any]) -> List[IntentProjection]:
        """
        Project future intent states and required adaptations.
        
        Args:
            current_intent: Current agent intent
            current_state: Current agent state
            
        Returns:
            List[IntentProjection]: List of projected future states
        """
        # Add current intent to graph
        self.intent_graph.add_intent(current_intent)
        
        # Encode current intent
        intent_encoding = self._encode_intent(current_intent)
        
        # Predict future states
        future_states = self._predict_state(
            current_state, 
            intent_encoding,
            self.projection_depth
        )
        
        # Generate projections
        projections = []
        # Convert state values to floats, handling nested dicts
        state_values = []
        for v in current_state.values():
            if isinstance(v, dict):
                # For nested dicts, take a representative numeric value
                if 'available' in v:
                    state_values.append(1.0 if v['available'] else 0.0)
                else:
                    # Take the first numeric value or default to 0.5
                    numeric_val = next((val for val in v.values() if isinstance(val, (int, float))), 0.5)
                    state_values.append(float(numeric_val))
            else:
                state_values.append(float(v))
        
        current_state_tensor = torch.tensor(state_values)
        
        for i, future_state in enumerate(future_states):
            # Create projected intent
            projected_intent = Intent(
                id=f"{current_intent.id}_proj_{i}",
                description=f"Projected intent {i} steps from {current_intent.id}",
                goals=current_intent.goals,
                constraints=current_intent.constraints,
                context=current_intent.context,
                confidence=current_intent.confidence * (0.9 ** i),  # Decay confidence
                timestamp=datetime.now().timestamp(),
                parent_id=current_intent.id
            )
            
            # Generate adaptation steps
            # Convert future state values to floats, handling nested dicts
            future_state_values = []
            for v in future_state.values():
                if isinstance(v, dict):
                    # For nested dicts, take a representative numeric value
                    if 'available' in v:
                        future_state_values.append(1.0 if v['available'] else 0.0)
                    else:
                        # Take the first numeric value or default to 0.5
                        numeric_val = next((val for val in v.values() if isinstance(val, (int, float))), 0.5)
                        future_state_values.append(float(numeric_val))
                else:
                    future_state_values.append(float(v))
            
            future_state_tensor = torch.tensor(future_state_values)
            adaptation_steps = self._generate_adaptation_steps(
                intent_encoding,
                current_state_tensor,
                future_state_tensor
            )
            
            # Create projection
            projection = IntentProjection(
                intent=projected_intent,
                projected_state=future_state,
                success_probability=max(0.0, 1.0 - (0.1 * i)),  # Simple decay
                required_capabilities=[
                    'learning',
                    'adaptation',
                    'self_optimization'
                ],
                adaptation_steps=adaptation_steps
            )
            
            projections.append(projection)
        
        return projections

    def optimize_intent(self, current_intent: Intent, 
                       projections: List[IntentProjection]) -> Intent:
        """
        Optimize current intent based on projections.
        
        Args:
            current_intent: Current agent intent
            projections: List of intent projections
            
        Returns:
            Intent: Optimized intent
        """
        # Filter projections by confidence threshold
        valid_projections = [
            p for p in projections 
            if p.success_probability >= self.confidence_threshold
        ]
        
        if not valid_projections:
            return current_intent
        
        # Select best projection
        best_projection = max(valid_projections, 
                            key=lambda p: p.success_probability)
        
        # Create optimized intent
        optimized_intent = Intent(
            id=f"{current_intent.id}_opt_{int(datetime.now().timestamp())}",
            description=f"Optimized intent from {current_intent.id}",
            goals=current_intent.goals,
            constraints=current_intent.constraints,
            context={
                **current_intent.context,
                'optimization_steps': best_projection.adaptation_steps
            },
            confidence=best_projection.success_probability,
            timestamp=datetime.now().timestamp(),
            parent_id=current_intent.id
        )
        
        # Add to intent graph
        self.intent_graph.add_intent(optimized_intent)
        
        return optimized_intent

    @ray.remote
    def _parallel_project(self, intent: Intent, 
                         state: Dict[str, Any]) -> List[IntentProjection]:
        """Parallel intent projection"""
        return self.project_intent(intent, state)

    def recursive_optimize(self, initial_intent: Intent,
                         initial_state: Dict[str, Any],
                         max_iterations: Optional[int] = None) -> Intent:
        """
        Recursively optimize intent through multiple iterations.
        
        Args:
            initial_intent: Initial agent intent
            initial_state: Initial agent state
            max_iterations: Maximum number of optimization iterations
            
        Returns:
            Intent: Final optimized intent
        """
        current_intent = initial_intent
        iterations = 0
        max_iter = max_iterations or self.max_iterations
        
        while iterations < max_iter:
            # Generate projections in parallel
            future_refs = [
                self._parallel_project.remote(current_intent, initial_state)
                for _ in range(3)  # Generate multiple projection sets
            ]
            
            # Gather results
            all_projections = []
            for projections in ray.get(future_refs):
                all_projections.extend(projections)
            
            # Optimize intent
            optimized_intent = self.optimize_intent(current_intent, all_projections)
            
            # Check for convergence
            if (optimized_intent.confidence >= self.confidence_threshold or
                optimized_intent.confidence <= current_intent.confidence):
                break
            
            current_intent = optimized_intent
            iterations += 1
        
        return current_intent

    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get history of intent optimizations"""
        history = []
        
        for node_id in nx.topological_sort(self.intent_graph.graph):
            node_data = self.intent_graph.graph.nodes[node_id]
            intent = node_data['intent']
            
            history.append({
                'intent_id': intent.id,
                'description': intent.description,
                'confidence': intent.confidence,
                'timestamp': intent.timestamp,
                'parent_id': intent.parent_id
            })
        
        return history
