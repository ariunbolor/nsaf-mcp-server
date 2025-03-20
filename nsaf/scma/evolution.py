"""
Evolution implementation for the Self-Constructing Meta-Agents (SCMA) module.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Tuple, Callable, Union

from ..utils.config import Config
from .meta_agent import MetaAgent
from .agent_factory import AgentFactory


class Evolution:
    """
    Class for managing the evolutionary process of meta-agents.
    
    This class provides high-level functionality for evolving populations
    of meta-agents, including visualization, logging, and experiment management.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize a new Evolution instance.
        
        Args:
            config: Configuration object containing parameters for evolution.
                   If None, default configuration will be used.
        """
        self.config = config or Config()
        self.factory = AgentFactory(config=self.config)
        self.experiment_name = f"experiment_{int(time.time())}"
        self.experiment_dir = "./experiments"
        self.checkpoints_dir = os.path.join(self.experiment_dir, self.experiment_name, "checkpoints")
        self.logs_dir = os.path.join(self.experiment_dir, self.experiment_name, "logs")
        self.plots_dir = os.path.join(self.experiment_dir, self.experiment_name, "plots")
        self.best_agents: List[MetaAgent] = []
        self.evolution_history: List[Dict[str, Any]] = []
        
    def setup_experiment(self, experiment_name: Optional[str] = None, 
                        base_dir: Optional[str] = None) -> None:
        """
        Set up directories for the experiment.
        
        Args:
            experiment_name: Name of the experiment.
                           If None, a timestamp-based name will be used.
            base_dir: Base directory for the experiment.
                     If None, the default directory will be used.
        """
        if experiment_name:
            self.experiment_name = experiment_name
        
        if base_dir:
            self.experiment_dir = base_dir
        
        self.checkpoints_dir = os.path.join(self.experiment_dir, self.experiment_name, "checkpoints")
        self.logs_dir = os.path.join(self.experiment_dir, self.experiment_name, "logs")
        self.plots_dir = os.path.join(self.experiment_dir, self.experiment_name, "plots")
        
        # Create directories
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Save configuration
        self.config.save_to_file(os.path.join(self.experiment_dir, self.experiment_name, "config.yaml"))
    
    def run_evolution(self, fitness_function: Callable[[MetaAgent], float],
                     generations: Optional[int] = None,
                     target_fitness: Optional[float] = None,
                     checkpoint_interval: Optional[int] = None,
                     population_size: Optional[int] = None,
                     verbose: bool = True) -> Dict[str, Any]:
        """
        Run the evolutionary process.
        
        Args:
            fitness_function: Function that takes a MetaAgent and returns a fitness score.
            generations: Number of generations to evolve.
                        If None, the value from config will be used.
            target_fitness: Target fitness to reach. Evolution will stop if this fitness is reached.
                           If None, evolution will continue for the specified number of generations.
            checkpoint_interval: Number of generations between checkpoints.
                               If None, no checkpoints will be saved.
            population_size: Size of the population to create.
                           If None, the value from config will be used.
            verbose: Whether to print progress information.
            
        Returns:
            Dictionary containing evolution results.
        """
        if generations is None:
            generations = self.config.get('scma.generations', 50)
        
        if checkpoint_interval is None:
            checkpoint_interval = self.config.get('training.checkpoint_interval', 0)
        
        # Set up experiment directories
        self.setup_experiment()
        
        # Initialize population
        if population_size is not None:
            self.factory.initialize_population(size=population_size)
        else:
            self.factory.initialize_population()
        
        start_time = time.time()
        
        # Run evolution
        for generation in range(generations):
            # Evaluate current population
            stats = self.factory.evaluate_population(fitness_function)
            
            # Log progress
            if verbose:
                print(f"Generation {generation}: "
                      f"Mean Fitness = {stats['mean_fitness']:.4f}, "
                      f"Max Fitness = {stats['max_fitness']:.4f}")
            
            # Save checkpoint if needed
            if checkpoint_interval > 0 and generation % checkpoint_interval == 0:
                checkpoint_dir = os.path.join(self.checkpoints_dir, f"generation_{generation}")
                self.factory.save_population(checkpoint_dir)
                
                # Save the best agent separately
                best_agent = self.factory.get_best_agent()
                if best_agent:
                    best_agent.save(os.path.join(self.checkpoints_dir, f"best_agent_gen_{generation}"))
                    self.best_agents.append(best_agent)
            
            # Check if target fitness is reached
            if target_fitness is not None and stats['max_fitness'] >= target_fitness:
                if verbose:
                    print(f"Target fitness {target_fitness} reached at generation {generation}.")
                break
            
            # Create next generation
            if generation < generations - 1:  # Skip for the last generation
                self.factory.create_next_generation()
            
            # Record history
            self.evolution_history.append({
                'generation': generation,
                'stats': stats,
                'elapsed_time': time.time() - start_time
            })
        
        # Final evaluation
        final_stats = self.factory.evaluate_population(fitness_function)
        
        # Save final population
        final_checkpoint_dir = os.path.join(self.checkpoints_dir, "final_generation")
        self.factory.save_population(final_checkpoint_dir)
        
        # Save the best agent
        best_agent = self.factory.get_best_agent()
        if best_agent:
            best_agent.save(os.path.join(self.checkpoints_dir, "best_agent_final"))
            self.best_agents.append(best_agent)
        
        # Generate plots
        self._generate_fitness_plots()
        
        # Compile results
        elapsed_time = time.time() - start_time
        results = {
            'experiment_name': self.experiment_name,
            'generations': generations,
            'actual_generations': len(self.evolution_history),
            'target_fitness': target_fitness,
            'best_fitness': self.factory.best_fitness,
            'best_agent_id': self.factory.best_agent.id if self.factory.best_agent else None,
            'elapsed_time': elapsed_time,
            'final_stats': final_stats,
            'experiment_dir': os.path.join(self.experiment_dir, self.experiment_name)
        }
        
        # Save results
        np.save(os.path.join(self.logs_dir, "evolution_results.npy"), results)
        
        return results
    
    def _generate_fitness_plots(self) -> None:
        """
        Generate plots of fitness over generations.
        """
        if not self.evolution_history:
            return
        
        generations = [entry['generation'] for entry in self.evolution_history]
        mean_fitness = [entry['stats']['mean_fitness'] for entry in self.evolution_history]
        max_fitness = [entry['stats']['max_fitness'] for entry in self.evolution_history]
        min_fitness = [entry['stats']['min_fitness'] for entry in self.evolution_history]
        
        # Plot mean and max fitness
        plt.figure(figsize=(10, 6))
        plt.plot(generations, mean_fitness, label='Mean Fitness')
        plt.plot(generations, max_fitness, label='Max Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Fitness Evolution')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.plots_dir, 'fitness_evolution.png'))
        
        # Plot fitness distribution
        plt.figure(figsize=(10, 6))
        plt.fill_between(generations, min_fitness, max_fitness, alpha=0.3, label='Fitness Range')
        plt.plot(generations, mean_fitness, label='Mean Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Fitness Distribution')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.plots_dir, 'fitness_distribution.png'))
    
    def visualize_agent(self, agent: MetaAgent, save_path: Optional[str] = None) -> None:
        """
        Visualize the architecture of a meta-agent.
        
        Args:
            agent: The meta-agent to visualize.
            save_path: Path to save the visualization.
                      If None, the visualization will be displayed.
        """
        # Create a figure
        plt.figure(figsize=(12, 8))
        
        # Extract architecture parameters
        hidden_layers = agent.architecture['hidden_layers']
        activations = agent.architecture['activations']
        use_batch_norm = agent.architecture['use_batch_normalization']
        dropout_rate = agent.architecture['dropout_rate']
        optimizer = agent.architecture['optimizer']
        learning_rate = agent.architecture['learning_rate']
        connections = agent.architecture.get('connections', {})
        
        # Plot layers
        layer_positions = []
        layer_sizes = [agent.architecture['input_shape'][1]] + hidden_layers + [agent.architecture['output_shape'][1]]
        layer_labels = ['Input'] + [f'Hidden {i+1}\n{size} units\n{act}' 
                                   for i, (size, act) in enumerate(zip(hidden_layers, activations))] + ['Output']
        
        max_size = max(layer_sizes)
        
        for i, (size, label) in enumerate(zip(layer_sizes, layer_labels)):
            x = i / (len(layer_sizes) - 1)
            y = 0.5
            radius = 0.1 * (size / max_size)
            
            circle = plt.Circle((x, y), radius, fill=True, alpha=0.7)
            plt.gca().add_patch(circle)
            
            plt.text(x, y, label, ha='center', va='center', fontsize=10)
            
            layer_positions.append((x, y, radius))
        
        # Plot connections
        for i in range(len(layer_positions) - 1):
            x1, y1, r1 = layer_positions[i]
            x2, y2, r2 = layer_positions[i + 1]
            
            plt.arrow(x1 + r1, y1, x2 - x1 - r1 - r2, 0, 
                     head_width=0.02, head_length=0.02, fc='black', ec='black', alpha=0.5)
        
        # Plot additional connections from the connections dictionary
        for connection_key, strength in connections.items():
            from_layer, to_layer = map(int, connection_key.split('->'))
            
            if from_layer < len(layer_positions) and to_layer < len(layer_positions):
                x1, y1, r1 = layer_positions[from_layer]
                x2, y2, r2 = layer_positions[to_layer]
                
                # Draw a curved arrow for additional connections
                plt.annotate('', 
                            xy=(x2, y2 + r2), 
                            xytext=(x1, y1 + r1),
                            arrowprops=dict(arrowstyle='->', 
                                           connectionstyle='arc3,rad=0.3',
                                           alpha=strength))
        
        # Add agent information
        info_text = (
            f"Agent ID: {agent.id[:8]}...\n"
            f"Fitness: {agent.fitness:.4f}\n"
            f"Generation: {agent.generation}\n"
            f"Optimizer: {optimizer}\n"
            f"Learning Rate: {learning_rate:.6f}\n"
            f"Batch Normalization: {'Yes' if use_batch_norm else 'No'}\n"
            f"Dropout Rate: {dropout_rate:.2f}"
        )
        
        plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))
        
        plt.xlim(-0.1, 1.1)
        plt.ylim(0, 1)
        plt.axis('off')
        plt.title(f"Architecture of Meta-Agent {agent.id[:8]}...")
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def visualize_population(self, save_dir: Optional[str] = None) -> None:
        """
        Visualize all agents in the current population.
        
        Args:
            save_dir: Directory to save the visualizations.
                     If None, the visualizations will be saved to the plots directory.
        """
        if not self.factory.population:
            return
        
        if save_dir is None:
            save_dir = os.path.join(self.plots_dir, "population")
        
        os.makedirs(save_dir, exist_ok=True)
        
        for i, agent in enumerate(self.factory.population):
            save_path = os.path.join(save_dir, f"agent_{i}.png")
            self.visualize_agent(agent, save_path=save_path)
    
    def visualize_best_agent(self, save_path: Optional[str] = None) -> None:
        """
        Visualize the best agent in the current population.
        
        Args:
            save_path: Path to save the visualization.
                      If None, the visualization will be saved to the plots directory.
        """
        best_agent = self.factory.get_best_agent()
        
        if best_agent is None:
            return
        
        if save_path is None:
            save_path = os.path.join(self.plots_dir, "best_agent.png")
        
        self.visualize_agent(best_agent, save_path=save_path)
    
    def compare_agents(self, agents: List[MetaAgent], metrics: List[str], 
                      save_path: Optional[str] = None) -> None:
        """
        Compare multiple agents based on specified metrics.
        
        Args:
            agents: List of agents to compare.
            metrics: List of metrics to compare (e.g., 'fitness', 'generation', 'hidden_layers').
            save_path: Path to save the comparison plot.
                      If None, the plot will be saved to the plots directory.
        """
        if not agents:
            return
        
        num_agents = len(agents)
        num_metrics = len(metrics)
        
        # Create a figure
        fig, axes = plt.subplots(num_metrics, 1, figsize=(10, 3 * num_metrics))
        
        if num_metrics == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            values = []
            labels = []
            
            for j, agent in enumerate(agents):
                if metric == 'fitness':
                    values.append(agent.fitness)
                elif metric == 'generation':
                    values.append(agent.generation)
                elif metric == 'hidden_layers':
                    values.append(len(agent.architecture['hidden_layers']))
                elif metric == 'total_params':
                    # Calculate total parameters in the model
                    total_params = agent.model.count_params()
                    values.append(total_params)
                elif metric == 'learning_rate':
                    values.append(agent.architecture['learning_rate'])
                else:
                    # Try to get the metric from the architecture
                    if metric in agent.architecture:
                        values.append(agent.architecture[metric])
                    else:
                        values.append(0)
                
                labels.append(f"Agent {j+1}\n({agent.id[:6]}...)")
            
            axes[i].bar(labels, values)
            axes[i].set_title(f"{metric.replace('_', ' ').title()}")
            axes[i].grid(axis='y', linestyle='--', alpha=0.7)
            
            # Rotate x-axis labels for better readability
            plt.setp(axes[i].get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.plots_dir, "agent_comparison.png")
        
        plt.savefig(save_path)
        plt.close()
    
    def load_experiment(self, experiment_dir: str) -> Dict[str, Any]:
        """
        Load an experiment from a directory.
        
        Args:
            experiment_dir: Directory containing the experiment data.
            
        Returns:
            Dictionary containing the loaded experiment data.
        """
        # Load configuration
        config_path = os.path.join(experiment_dir, "config.yaml")
        if os.path.exists(config_path):
            self.config.load_from_file(config_path)
        
        # Set experiment directories
        self.experiment_dir = os.path.dirname(experiment_dir)
        self.experiment_name = os.path.basename(experiment_dir)
        self.checkpoints_dir = os.path.join(experiment_dir, "checkpoints")
        self.logs_dir = os.path.join(experiment_dir, "logs")
        self.plots_dir = os.path.join(experiment_dir, "plots")
        
        # Load factory and population
        final_checkpoint_dir = os.path.join(self.checkpoints_dir, "final_generation")
        if os.path.exists(final_checkpoint_dir):
            self.factory = AgentFactory.load_population(final_checkpoint_dir, config=self.config)
        
        # Load best agents
        self.best_agents = []
        best_agent_path = os.path.join(self.checkpoints_dir, "best_agent_final")
        if os.path.exists(f"{best_agent_path}_data.npy"):
            best_agent = MetaAgent.load(best_agent_path, config=self.config)
            self.best_agents.append(best_agent)
        
        # Load evolution history
        results_path = os.path.join(self.logs_dir, "evolution_results.npy")
        if os.path.exists(results_path):
            results = np.load(results_path, allow_pickle=True).item()
        else:
            results = {}
        
        return results
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current experiment.
        
        Returns:
            Dictionary containing experiment summary.
        """
        return {
            'experiment_name': self.experiment_name,
            'experiment_dir': os.path.join(self.experiment_dir, self.experiment_name),
            'generations': len(self.evolution_history),
            'best_fitness': self.factory.best_fitness,
            'best_agent_id': self.factory.best_agent.id if self.factory.best_agent else None,
            'population_size': len(self.factory.population),
            'config': str(self.config)
        }
