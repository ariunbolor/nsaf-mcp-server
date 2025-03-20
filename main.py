"""
Main script for demonstrating the Neuro-Symbolic Autonomy Framework (NSAF) prototype.

This script provides a simple example of using the Self-Constructing Meta-Agents (SCMA)
component of the NSAF framework.
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple

from nsaf.utils.config import Config
from nsaf.scma.meta_agent import MetaAgent
from nsaf.scma.agent_factory import AgentFactory
from nsaf.scma.evolution import Evolution


def create_synthetic_dataset(num_samples: int = 1000, 
                            input_dim: int = 1024, 
                            output_dim: int = 1024,
                            complexity: str = 'medium') -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a synthetic dataset for training and evaluating meta-agents.
    
    Args:
        num_samples: Number of samples to generate.
        input_dim: Dimension of input data.
        output_dim: Dimension of output data.
        complexity: Complexity of the data generation function ('simple', 'medium', or 'complex').
        
    Returns:
        Tuple of (inputs, outputs) arrays.
    """
    # Generate random input data
    inputs = np.random.normal(0, 1, (num_samples, input_dim))
    
    # Generate outputs based on complexity
    if complexity == 'simple':
        # Simple linear transformation
        W = np.random.normal(0, 0.1, (input_dim, output_dim))
        b = np.random.normal(0, 0.1, output_dim)
        outputs = np.dot(inputs, W) + b
        
    elif complexity == 'medium':
        # Non-linear transformation with multiple components
        hidden_dim = min(input_dim, output_dim) // 2
        W1 = np.random.normal(0, 0.1, (input_dim, hidden_dim))
        b1 = np.random.normal(0, 0.1, hidden_dim)
        W2 = np.random.normal(0, 0.1, (hidden_dim, output_dim))
        b2 = np.random.normal(0, 0.1, output_dim)
        
        hidden = np.tanh(np.dot(inputs, W1) + b1)
        outputs = np.dot(hidden, W2) + b2
        
    else:  # complex
        # Highly non-linear transformation with multiple hidden layers
        hidden_dims = [input_dim // 2, input_dim // 4, input_dim // 2]
        
        # First hidden layer
        W1 = np.random.normal(0, 0.1, (input_dim, hidden_dims[0]))
        b1 = np.random.normal(0, 0.1, hidden_dims[0])
        h1 = np.tanh(np.dot(inputs, W1) + b1)
        
        # Second hidden layer
        W2 = np.random.normal(0, 0.1, (hidden_dims[0], hidden_dims[1]))
        b2 = np.random.normal(0, 0.1, hidden_dims[1])
        h2 = np.sin(np.dot(h1, W2) + b2)  # Different activation
        
        # Third hidden layer
        W3 = np.random.normal(0, 0.1, (hidden_dims[1], hidden_dims[2]))
        b3 = np.random.normal(0, 0.1, hidden_dims[2])
        h3 = np.tanh(np.dot(h2, W3) + b3)
        
        # Output layer
        W4 = np.random.normal(0, 0.1, (hidden_dims[2], output_dim))
        b4 = np.random.normal(0, 0.1, output_dim)
        outputs = np.dot(h3, W4) + b4
        
        # Add some non-linear interactions
        mask = np.random.choice([0, 1], size=output_dim, p=[0.7, 0.3])
        outputs += mask * np.sin(np.sum(inputs, axis=1, keepdims=True))
    
    # Add some noise
    outputs += np.random.normal(0, 0.01, outputs.shape)
    
    return inputs, outputs


def evaluate_agent_on_dataset(agent: MetaAgent, 
                             inputs: np.ndarray, 
                             targets: np.ndarray) -> float:
    """
    Evaluate a meta-agent on a dataset.
    
    Args:
        agent: The meta-agent to evaluate.
        inputs: Input data.
        targets: Target data.
        
    Returns:
        Negative mean squared error (higher is better).
    """
    predictions = agent.predict(inputs)
    mse = np.mean((predictions - targets) ** 2)
    
    # Return negative MSE as fitness (higher is better)
    return -mse


def run_scma_example():
    """
    Run an example of the Self-Constructing Meta-Agents (SCMA) component.
    """
    print("Running NSAF Self-Constructing Meta-Agents (SCMA) example...")
    
    # Create a configuration
    config = Config()
    
    # Set configuration parameters (use environment variables if available)
    population_size = int(os.environ.get('NSAF_POPULATION_SIZE', 20))
    generations = int(os.environ.get('NSAF_GENERATIONS', 10))
    config.set('scma.population_size', population_size)
    config.set('scma.generations', generations)
    config.set('scma.mutation_rate', 0.2)
    config.set('scma.crossover_rate', 0.7)
    config.set('scma.architecture_complexity', 'medium')
    config.set('scma.agent_memory_size', 1024)
    
    # Create synthetic dataset
    print("Creating synthetic dataset...")
    input_dim = config.get('scma.agent_memory_size')
    output_dim = config.get('scma.agent_memory_size')
    
    train_inputs, train_targets = create_synthetic_dataset(
        num_samples=500,
        input_dim=input_dim,
        output_dim=output_dim,
        complexity='medium'
    )
    
    test_inputs, test_targets = create_synthetic_dataset(
        num_samples=100,
        input_dim=input_dim,
        output_dim=output_dim,
        complexity='medium'
    )
    
    # Create an Evolution instance
    evolution = Evolution(config=config)
    
    # Set up experiment
    evolution.setup_experiment(experiment_name="scma_example")
    
    # Define fitness function
    def fitness_function(agent):
        return evaluate_agent_on_dataset(agent, train_inputs, train_targets)
    
    # Run evolution
    print("Running evolution...")
    results = evolution.run_evolution(
        fitness_function=fitness_function,
        generations=config.get('scma.generations'),
        population_size=config.get('scma.population_size'),
        checkpoint_interval=2,
        verbose=True
    )
    
    # Get the best agent
    best_agent = evolution.factory.get_best_agent()
    
    if best_agent:
        # Evaluate on test set
        test_fitness = evaluate_agent_on_dataset(best_agent, test_inputs, test_targets)
        print(f"Best agent test fitness: {test_fitness:.6f}")
        
        # Visualize the best agent
        evolution.visualize_best_agent()
        
        # Make predictions with the best agent
        print("Making predictions with the best agent...")
        sample_inputs = test_inputs[:5]
        sample_targets = test_targets[:5]
        sample_predictions = best_agent.predict(sample_inputs)
        
        # Print sample predictions
        for i in range(5):
            print(f"Sample {i+1}:")
            print(f"  Input shape: {sample_inputs[i].shape}")
            print(f"  Target shape: {sample_targets[i].shape}")
            print(f"  Prediction shape: {sample_predictions[i].shape}")
            print(f"  Target mean: {np.mean(sample_targets[i]):.4f}, std: {np.std(sample_targets[i]):.4f}")
            print(f"  Prediction mean: {np.mean(sample_predictions[i]):.4f}, std: {np.std(sample_predictions[i]):.4f}")
            print(f"  MSE: {np.mean((sample_predictions[i] - sample_targets[i]) ** 2):.6f}")
            print()
    
    # Print experiment summary
    print("Experiment summary:")
    summary = evolution.get_experiment_summary()
    for key, value in summary.items():
        if key != 'config':  # Skip printing the full config
            print(f"  {key}: {value}")
    
    print(f"Results saved to: {results['experiment_dir']}")
    print("Done!")


def run_agent_comparison_example():
    """
    Run an example comparing different meta-agents.
    """
    print("Running Meta-Agent comparison example...")
    
    # Create a configuration
    config = Config()
    
    # Create agents with different architectures
    print("Creating agents with different architectures...")
    
    # Simple agent
    config.set('scma.architecture_complexity', 'simple')
    simple_agent = MetaAgent(config=config)
    simple_agent.metadata['type'] = 'simple'
    
    # Medium agent
    config.set('scma.architecture_complexity', 'medium')
    medium_agent = MetaAgent(config=config)
    medium_agent.metadata['type'] = 'medium'
    
    # Complex agent
    config.set('scma.architecture_complexity', 'complex')
    complex_agent = MetaAgent(config=config)
    complex_agent.metadata['type'] = 'complex'
    
    # Create synthetic dataset
    print("Creating synthetic dataset...")
    input_dim = config.get('scma.agent_memory_size')
    output_dim = config.get('scma.agent_memory_size')
    
    train_inputs, train_targets = create_synthetic_dataset(
        num_samples=500,
        input_dim=input_dim,
        output_dim=output_dim,
        complexity='medium'
    )
    
    test_inputs, test_targets = create_synthetic_dataset(
        num_samples=100,
        input_dim=input_dim,
        output_dim=output_dim,
        complexity='medium'
    )
    
    # Train each agent
    print("Training agents...")
    agents = [simple_agent, medium_agent, complex_agent]
    
    for agent in agents:
        print(f"Training {agent.metadata['type']} agent...")
        agent.train(train_inputs, train_targets, validation_split=0.2)
    
    # Evaluate agents
    print("Evaluating agents...")
    for agent in agents:
        train_fitness = evaluate_agent_on_dataset(agent, train_inputs, train_targets)
        test_fitness = evaluate_agent_on_dataset(agent, test_inputs, test_targets)
        
        agent.fitness = test_fitness
        
        print(f"{agent.metadata['type']} agent:")
        print(f"  Train fitness: {train_fitness:.6f}")
        print(f"  Test fitness: {test_fitness:.6f}")
        print(f"  Architecture: {len(agent.architecture['hidden_layers'])} hidden layers")
        print(f"  Parameters: {agent.model.count_params()}")
        print()
    
    # Create an Evolution instance for visualization
    evolution = Evolution(config=config)
    evolution.setup_experiment(experiment_name="agent_comparison")
    
    # Compare agents
    print("Comparing agents...")
    evolution.compare_agents(
        agents=agents,
        metrics=['fitness', 'hidden_layers', 'total_params', 'learning_rate'],
        save_path=os.path.join(evolution.plots_dir, "agent_comparison.png")
    )
    
    # Visualize each agent
    print("Visualizing agents...")
    for agent in agents:
        evolution.visualize_agent(
            agent=agent,
            save_path=os.path.join(evolution.plots_dir, f"{agent.metadata['type']}_agent.png")
        )
    
    print(f"Results saved to: {evolution.experiment_dir}")
    print("Done!")


def main():
    """
    Main function.
    """
    print("=" * 80)
    print("Neuro-Symbolic Autonomy Framework (NSAF) Prototype")
    print("Self-Constructing Meta-Agents (SCMA) Component")
    print("=" * 80)
    
    # Set random seed for reproducibility (use environment variable if available)
    random_seed = int(os.environ.get('NSAF_RANDOM_SEED', 42))
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)
    print(f"Using random seed: {random_seed}")
    
    # Create experiments directory
    os.makedirs("./experiments", exist_ok=True)
    
    # Run examples
    run_scma_example()
    print("\n" + "=" * 80 + "\n")
    run_agent_comparison_example()


if __name__ == "__main__":
    main()
