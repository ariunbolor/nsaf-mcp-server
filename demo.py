#!/usr/bin/env python
"""
Demo script for the Neuro-Symbolic Autonomy Framework (NSAF) prototype.

This script demonstrates the core concepts of the Self-Constructing Meta-Agents (SCMA)
component without requiring TensorFlow or other heavy dependencies.
"""

import os
import random
import time
import math
from typing import List, Dict, Any, Tuple


class SimpleMetaAgent:
    """A simplified version of MetaAgent for demonstration purposes."""
    
    def __init__(self, architecture=None):
        """Initialize a new SimpleMetaAgent."""
        self.id = f"agent_{int(time.time() * 1000) % 10000}_{random.randint(1000, 9999)}"
        self.architecture = architecture or self._generate_random_architecture()
        self.fitness = 0.0
        self.generation = 0
        self.parent_ids = []
    
    def _generate_random_architecture(self) -> Dict[str, Any]:
        """Generate a random neural network architecture."""
        # Number of layers
        num_layers = random.randint(1, 5)
        
        # Layer sizes
        hidden_layers = [random.randint(16, 128) for _ in range(num_layers)]
        
        # Activation functions
        activation_options = ['relu', 'tanh', 'sigmoid', 'elu', 'selu']
        activations = [random.choice(activation_options) for _ in range(num_layers)]
        
        # Other parameters
        use_batch_norm = random.choice([True, False])
        dropout_rate = random.uniform(0, 0.5) if random.choice([True, False]) else 0
        
        # Optimizer and learning rate
        optimizer_options = ['adam', 'sgd', 'rmsprop', 'adagrad']
        optimizer = random.choice(optimizer_options)
        learning_rate = random.uniform(0.0001, 0.01)
        
        return {
            'hidden_layers': hidden_layers,
            'activations': activations,
            'use_batch_normalization': use_batch_norm,
            'dropout_rate': dropout_rate,
            'optimizer': optimizer,
            'learning_rate': learning_rate,
        }
    
    def predict(self, x):
        """Simulate making predictions."""
        # In a real implementation, this would use the neural network
        # Here we just return a simple transformation of the input
        return [self._simulate_prediction(val) for val in x]
    
    def _simulate_prediction(self, x):
        """Simulate a prediction for a single input value."""
        # Use the architecture to influence the prediction
        num_layers = len(self.architecture['hidden_layers'])
        avg_layer_size = sum(self.architecture['hidden_layers']) / num_layers if num_layers > 0 else 1
        
        # Create a non-linear transformation based on the architecture
        result = math.sin(x * 0.1 * num_layers) + math.cos(x * 0.05 * avg_layer_size / 50)
        
        # Add some randomness
        result += random.uniform(-0.1, 0.1)
        
        return result
    
    def mutate(self):
        """Create a mutated copy of this agent."""
        # Create a copy of the architecture
        new_architecture = self.architecture.copy()
        new_architecture['hidden_layers'] = self.architecture['hidden_layers'].copy()
        new_architecture['activations'] = self.architecture['activations'].copy()
        
        # Mutate hidden layers
        if random.random() < 0.3:
            # Randomly add, remove, or modify a layer
            mutation_type = random.choice(['add', 'remove', 'modify'])
            
            if mutation_type == 'add' and len(new_architecture['hidden_layers']) < 5:
                # Add a new layer
                position = random.randint(0, len(new_architecture['hidden_layers']))
                new_architecture['hidden_layers'].insert(position, random.randint(16, 128))
                
                # Add a corresponding activation function
                activation_options = ['relu', 'tanh', 'sigmoid', 'elu', 'selu']
                new_architecture['activations'].insert(position, random.choice(activation_options))
                
            elif mutation_type == 'remove' and len(new_architecture['hidden_layers']) > 1:
                # Remove a random layer
                position = random.randint(0, len(new_architecture['hidden_layers']) - 1)
                new_architecture['hidden_layers'].pop(position)
                new_architecture['activations'].pop(position)
                
            elif len(new_architecture['hidden_layers']) > 0:
                # Modify a random layer
                position = random.randint(0, len(new_architecture['hidden_layers']) - 1)
                current_size = new_architecture['hidden_layers'][position]
                change_factor = random.uniform(0.5, 1.5)
                new_size = max(16, min(128, int(current_size * change_factor)))
                new_architecture['hidden_layers'][position] = new_size
        
        # Mutate other parameters
        if random.random() < 0.2:
            new_architecture['use_batch_normalization'] = not new_architecture['use_batch_normalization']
        
        if random.random() < 0.2:
            new_architecture['dropout_rate'] = random.uniform(0, 0.5)
        
        if random.random() < 0.2:
            optimizer_options = ['adam', 'sgd', 'rmsprop', 'adagrad']
            new_architecture['optimizer'] = random.choice(optimizer_options)
        
        if random.random() < 0.2:
            current_lr = new_architecture['learning_rate']
            change_factor = random.uniform(0.5, 2.0)
            new_architecture['learning_rate'] = max(0.0001, min(0.01, current_lr * change_factor))
        
        # Create a new agent with the mutated architecture
        child = SimpleMetaAgent(architecture=new_architecture)
        child.parent_ids = [self.id]
        child.generation = self.generation + 1
        
        return child
    
    def crossover(self, other):
        """Perform crossover with another agent."""
        # Create a new architecture by combining elements from both parents
        new_architecture = {}
        
        # Crossover hidden layers and activations
        if len(self.architecture['hidden_layers']) > 0 and len(other.architecture['hidden_layers']) > 0:
            # Determine crossover points
            self_layers = self.architecture['hidden_layers']
            other_layers = other.architecture['hidden_layers']
            
            self_activations = self.architecture['activations']
            other_activations = other.architecture['activations']
            
            # Randomly choose which parent to start with
            if random.choice([True, False]):
                # Start with self, end with other
                crossover_point_self = random.randint(1, len(self_layers))
                crossover_point_other = random.randint(0, len(other_layers))
                
                new_layers = self_layers[:crossover_point_self] + other_layers[crossover_point_other:]
                new_activations = self_activations[:crossover_point_self] + other_activations[crossover_point_other:]
            else:
                # Start with other, end with self
                crossover_point_other = random.randint(1, len(other_layers))
                crossover_point_self = random.randint(0, len(self_layers))
                
                new_layers = other_layers[:crossover_point_other] + self_layers[crossover_point_self:]
                new_activations = other_activations[:crossover_point_other] + self_activations[crossover_point_self:]
            
            new_architecture['hidden_layers'] = new_layers
            new_architecture['activations'] = new_activations
        else:
            # If one parent has no layers, use the other parent's layers
            parent_with_layers = self if len(self.architecture['hidden_layers']) > 0 else other
            new_architecture['hidden_layers'] = parent_with_layers.architecture['hidden_layers'].copy()
            new_architecture['activations'] = parent_with_layers.architecture['activations'].copy()
        
        # For other parameters, randomly choose from either parent
        for key in ['use_batch_normalization', 'dropout_rate', 'optimizer', 'learning_rate']:
            new_architecture[key] = self.architecture[key] if random.choice([True, False]) else other.architecture[key]
        
        # Create a new agent with the combined architecture
        child = SimpleMetaAgent(architecture=new_architecture)
        child.parent_ids = [self.id, other.id]
        child.generation = max(self.generation, other.generation) + 1
        
        return child
    
    def __str__(self):
        """Get a string representation of the agent."""
        return (f"Agent(id={self.id}, "
                f"fitness={self.fitness:.4f}, "
                f"generation={self.generation}, "
                f"layers={self.architecture['hidden_layers']})")


class SimpleEvolution:
    """A simplified version of Evolution for demonstration purposes."""
    
    def __init__(self, population_size=10, generations=5):
        """Initialize a new SimpleEvolution."""
        self.population_size = population_size
        self.generations = generations
        self.population = []
        self.best_agent = None
        self.best_fitness = float('-inf')
        self.generation = 0
        self.history = []
    
    def initialize_population(self):
        """Initialize a population of agents."""
        self.population = [SimpleMetaAgent() for _ in range(self.population_size)]
        return self.population
    
    def evaluate_population(self, fitness_function):
        """Evaluate the fitness of all agents in the population."""
        fitnesses = []
        
        for agent in self.population:
            fitness = fitness_function(agent)
            agent.fitness = fitness
            fitnesses.append(fitness)
            
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_agent = agent
        
        # Calculate statistics
        stats = {
            'generation': self.generation,
            'mean_fitness': sum(fitnesses) / len(fitnesses) if fitnesses else 0,
            'max_fitness': max(fitnesses) if fitnesses else 0,
            'min_fitness': min(fitnesses) if fitnesses else 0,
            'best_agent_id': self.best_agent.id if self.best_agent else None,
        }
        
        self.history.append(stats)
        return stats
    
    def select_parents(self):
        """Select parents for reproduction using tournament selection."""
        tournament_size = 3
        selected_parents = []
        
        # Select as many parents as the current population size
        for _ in range(len(self.population)):
            # Randomly select tournament_size agents
            tournament = random.sample(self.population, min(tournament_size, len(self.population)))
            
            # Select the agent with the highest fitness
            winner = max(tournament, key=lambda agent: agent.fitness)
            selected_parents.append(winner)
        
        return selected_parents
    
    def create_next_generation(self):
        """Create the next generation of agents."""
        # Select parents
        parents = self.select_parents()
        
        # Parameters
        crossover_rate = 0.7
        mutation_rate = 0.2
        elitism = True
        elite_size = 2 if elitism else 0
        
        # Create new population
        new_population = []
        
        # Add elite agents if enabled
        if elitism and elite_size > 0:
            # Sort population by fitness (descending)
            sorted_population = sorted(self.population, key=lambda agent: agent.fitness, reverse=True)
            
            # Add top agents to new population
            new_population.extend(sorted_population[:elite_size])
        
        # Fill the rest of the population with offspring
        while len(new_population) < self.population_size:
            # Select two parents
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)
            
            # Perform crossover with probability crossover_rate
            if random.random() < crossover_rate and parent1 != parent2:
                child = parent1.crossover(parent2)
            else:
                # No crossover, just clone one parent
                child = SimpleMetaAgent(architecture=parent1.architecture.copy())
                child.parent_ids = [parent1.id]
                child.generation = parent1.generation + 1
            
            # Perform mutation with probability mutation_rate
            if random.random() < mutation_rate:
                child = child.mutate()
            
            new_population.append(child)
        
        # Update population and generation
        self.population = new_population
        self.generation += 1
        
        return self.population
    
    def run_evolution(self, fitness_function):
        """Run the evolutionary process."""
        print(f"Starting evolution with population size {self.population_size} for {self.generations} generations")
        
        # Initialize population if not already initialized
        if not self.population:
            self.initialize_population()
        
        # Run for the specified number of generations
        for generation in range(self.generations):
            # Evaluate current population
            stats = self.evaluate_population(fitness_function)
            
            print(f"Generation {generation}: "
                  f"Mean Fitness = {stats['mean_fitness']:.4f}, "
                  f"Max Fitness = {stats['max_fitness']:.4f}")
            
            # Create next generation (except for the last generation)
            if generation < self.generations - 1:
                self.create_next_generation()
        
        # Final evaluation
        final_stats = self.evaluate_population(fitness_function)
        
        print("\nEvolution completed!")
        print(f"Best agent: {self.best_agent}")
        print(f"Best fitness: {self.best_fitness:.4f}")
        
        return self.best_agent


def simple_fitness_function(agent):
    """A simple fitness function for demonstration."""
    # Use the architecture to calculate fitness
    num_layers = len(agent.architecture['hidden_layers'])
    total_neurons = sum(agent.architecture['hidden_layers'])
    
    # We'll reward architectures with a moderate number of layers and neurons
    layer_fitness = -0.1 * abs(num_layers - 3)  # Optimal is 3 layers
    neuron_fitness = -0.001 * abs(total_neurons - 150)  # Optimal is around 150 neurons
    
    # We'll also consider the learning rate (prefer middle values)
    lr_fitness = -10 * abs(agent.architecture['learning_rate'] - 0.001)
    
    # Combine the components
    fitness = layer_fitness + neuron_fitness + lr_fitness
    
    # Add some noise to make it interesting
    fitness += random.uniform(-0.1, 0.1)
    
    return fitness


def run_demo():
    """Run a demonstration of the NSAF prototype."""
    print("=" * 80)
    print("NSAF Prototype Demonstration")
    print("Self-Constructing Meta-Agents (SCMA) Component")
    print("=" * 80)
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Create and run evolution
    evolution = SimpleEvolution(population_size=10, generations=5)
    best_agent = evolution.run_evolution(simple_fitness_function)
    
    print("\nBest Agent Architecture:")
    for key, value in best_agent.architecture.items():
        print(f"  {key}: {value}")
    
    # Demonstrate the agent's prediction capability
    print("\nDemonstrating agent's prediction capability:")
    test_inputs = [0, 1, 2, 3, 4, 5]
    predictions = best_agent.predict(test_inputs)
    
    for x, y in zip(test_inputs, predictions):
        print(f"  Input: {x}, Prediction: {y:.4f}")
    
    # Show evolution history
    print("\nEvolution History:")
    for i, stats in enumerate(evolution.history):
        print(f"  Generation {i}: Mean Fitness = {stats['mean_fitness']:.4f}, Max Fitness = {stats['max_fitness']:.4f}")
    
    print("\nThis demonstration shows how Self-Constructing Meta-Agents can evolve")
    print("and optimize their architectures through generations.")
    print("In a full implementation with TensorFlow, these agents would be")
    print("actual neural networks that can learn and solve complex problems.")


if __name__ == "__main__":
    run_demo()
