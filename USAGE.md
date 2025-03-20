# NSAF Prototype Usage Guide

This guide provides instructions for using the Neuro-Symbolic Autonomy Framework (NSAF) prototype in your own projects.

## Overview

The NSAF prototype implements the Self-Constructing Meta-Agents (SCMA) component, which enables AI agents to self-design and evolve, creating optimized versions of themselves dynamically.

## Basic Usage

### 1. Import the Required Modules

```python
from nsaf.utils.config import Config
from nsaf.scma.meta_agent import MetaAgent
from nsaf.scma.agent_factory import AgentFactory
from nsaf.scma.evolution import Evolution
```

### 2. Create a Configuration

```python
# Create a configuration with default parameters
config = Config()

# Customize configuration parameters
config.set('scma.population_size', 20)
config.set('scma.generations', 10)
config.set('scma.mutation_rate', 0.2)
config.set('scma.crossover_rate', 0.7)
config.set('scma.architecture_complexity', 'medium')
config.set('scma.agent_memory_size', 1024)
```

### 3. Create and Use a Single Meta-Agent

```python
# Create a meta-agent
agent = MetaAgent(config=config)

# Train the agent on your data
agent.train(x_train, y_train, x_val=x_val, y_val=y_val)

# Make predictions
predictions = agent.predict(x_test)

# Evaluate the agent
loss, metric = agent.evaluate(x_test, y_test)
```

### 4. Evolve a Population of Meta-Agents

```python
# Define a fitness function
def fitness_function(agent):
    # Evaluate the agent and return a fitness score
    # Higher values are better
    predictions = agent.predict(x_test)
    mse = np.mean((predictions - y_test) ** 2)
    return -mse  # Negative MSE (higher is better)

# Create an Evolution instance
evolution = Evolution(config=config)

# Set up experiment
evolution.setup_experiment(experiment_name="my_experiment")

# Run evolution
results = evolution.run_evolution(
    fitness_function=fitness_function,
    generations=10,
    population_size=20,
    checkpoint_interval=2,
    verbose=True
)

# Get the best agent
best_agent = evolution.factory.get_best_agent()

# Use the best agent
predictions = best_agent.predict(x_test)
```

## Advanced Usage

### 1. Creating Custom Datasets

You can create your own datasets for training and evaluating meta-agents:

```python
import numpy as np

# Create a synthetic dataset
def create_custom_dataset(num_samples, input_dim, output_dim):
    # Generate input data
    inputs = np.random.normal(0, 1, (num_samples, input_dim))
    
    # Generate output data (your custom function)
    outputs = some_transformation_function(inputs)
    
    return inputs, outputs

# Create train and test datasets
x_train, y_train = create_custom_dataset(1000, 1024, 1024)
x_test, y_test = create_custom_dataset(200, 1024, 1024)
```

### 2. Customizing the Fitness Function

The fitness function is crucial for evolution. You can customize it based on your specific requirements:

```python
def custom_fitness_function(agent):
    # Make predictions
    predictions = agent.predict(x_test)
    
    # Calculate multiple metrics
    mse = np.mean((predictions - y_test) ** 2)
    mae = np.mean(np.abs(predictions - y_test))
    
    # Combine metrics into a single fitness score
    # You can weight different aspects based on your priorities
    fitness = -mse - 0.5 * mae
    
    # Add penalties for complexity if desired
    complexity_penalty = 0.001 * len(agent.architecture['hidden_layers'])
    fitness -= complexity_penalty
    
    return fitness
```

### 3. Saving and Loading Agents

You can save and load agents for later use:

```python
# Save an agent
agent.save("path/to/save/agent")

# Load an agent
loaded_agent = MetaAgent.load("path/to/save/agent", config=config)
```

### 4. Visualizing Agents and Evolution

The Evolution class provides several visualization methods:

```python
# Visualize a specific agent
evolution.visualize_agent(agent, save_path="agent_visualization.png")

# Visualize the best agent
evolution.visualize_best_agent(save_path="best_agent.png")

# Visualize the entire population
evolution.visualize_population(save_dir="population_visualizations")

# Compare multiple agents
evolution.compare_agents(
    agents=[agent1, agent2, agent3],
    metrics=['fitness', 'hidden_layers', 'total_params', 'learning_rate'],
    save_path="agent_comparison.png"
)
```

### 5. Customizing Agent Architecture

You can create agents with specific architectures:

```python
# Define a custom architecture
custom_architecture = {
    'hidden_layers': [128, 64, 32],
    'activations': ['relu', 'tanh', 'sigmoid'],
    'use_batch_normalization': True,
    'dropout_rate': 0.2,
    'optimizer': 'adam',
    'learning_rate': 0.001,
    'connections': {},
    'input_shape': (None, 1024),
    'output_shape': (None, 1024)
}

# Create an agent with the custom architecture
agent = MetaAgent(config=config, architecture=custom_architecture)
```

## Example Workflows

### 1. Basic Evolution Workflow

```python
import numpy as np
from nsaf.utils.config import Config
from nsaf.scma.evolution import Evolution

# Create configuration
config = Config()

# Create datasets
x_train, y_train = create_dataset(1000, 1024, 1024)
x_test, y_test = create_dataset(200, 1024, 1024)

# Define fitness function
def fitness_function(agent):
    predictions = agent.predict(x_train)
    mse = np.mean((predictions - y_train) ** 2)
    return -mse

# Create and run evolution
evolution = Evolution(config=config)
evolution.setup_experiment(experiment_name="basic_evolution")
results = evolution.run_evolution(
    fitness_function=fitness_function,
    generations=20,
    population_size=30,
    verbose=True
)

# Get and use the best agent
best_agent = evolution.factory.get_best_agent()
test_predictions = best_agent.predict(x_test)
test_mse = np.mean((test_predictions - y_test) ** 2)
print(f"Test MSE: {test_mse:.6f}")
```

### 2. Transfer Learning Workflow

```python
# Train on one task
evolution1 = Evolution(config=config)
evolution1.setup_experiment(experiment_name="task1")
evolution1.run_evolution(fitness_function_task1, generations=10)
best_agent_task1 = evolution1.factory.get_best_agent()

# Use the best agent from task1 as a starting point for task2
evolution2 = Evolution(config=config)
evolution2.setup_experiment(experiment_name="task2")

# Initialize population with mutations of the best agent from task1
factory = evolution2.factory
factory.population = [best_agent_task1.mutate() for _ in range(20)]

# Continue evolution on task2
evolution2.run_evolution(fitness_function_task2, generations=10)
best_agent_task2 = evolution2.factory.get_best_agent()
```

## Tips for Effective Use

1. **Start Small**: Begin with small populations and few generations to ensure your setup works correctly.

2. **Tune Hyperparameters**: Experiment with different values for population size, mutation rate, and crossover rate.

3. **Design Good Fitness Functions**: The fitness function is crucial for effective evolution. Make sure it accurately reflects your goals.

4. **Use Checkpoints**: Set a checkpoint interval to save intermediate results, which allows you to resume evolution if needed.

5. **Visualize Results**: Use the visualization tools to understand how your agents are evolving.

6. **Balance Exploration and Exploitation**: Adjust mutation and crossover rates to balance exploration of new architectures and exploitation of good ones.

7. **Consider Computational Resources**: Evolution can be computationally intensive. Adjust the complexity and population size based on your available resources.

## Further Reading

For more information on the concepts behind NSAF and SCMA, refer to the following resources:

- The README.md file in this repository
- Code documentation in the source files
- The original paper describing the Neuro-Symbolic Autonomy Framework (NSAF)
