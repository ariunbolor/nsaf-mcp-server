"""
Agent Factory implementation for the Self-Constructing Meta-Agents (SCMA) module.
"""

import os
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable

from ..utils.config import Config
from .meta_agent import MetaAgent


class AgentFactory:
    """
    Factory class for creating and managing populations of meta-agents.
    
    This class is responsible for generating populations of meta-agents,
    evaluating their fitness, and facilitating their evolution.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize a new AgentFactory instance.
        
        Args:
            config: Configuration object containing parameters for the factory.
                   If None, default configuration will be used.
        """
        self.config = config or Config()
        self.population: List[MetaAgent] = []
        self.generation = 0
        self.best_agent: Optional[MetaAgent] = None
        self.best_fitness = float('-inf')
        self.fitness_history: List[Dict[str, float]] = []
        
    def initialize_population(self, size: Optional[int] = None) -> List[MetaAgent]:
        """
        Initialize a population of meta-agents.
        
        Args:
            size: Size of the population to create.
                 If None, the value from config will be used.
                 
        Returns:
            List of created meta-agents.
        """
        if size is None:
            size = self.config.get('scma.population_size', 10)
        
        self.population = [MetaAgent(config=self.config) for _ in range(size)]
        return self.population
    
    def evaluate_population(self, fitness_function: Callable[[MetaAgent], float]) -> Dict[str, Any]:
        """
        Evaluate the fitness of all agents in the population.
        
        Args:
            fitness_function: Function that takes a MetaAgent and returns a fitness score.
            
        Returns:
            Dictionary containing evaluation statistics.
        """
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
            'mean_fitness': np.mean(fitnesses),
            'median_fitness': np.median(fitnesses),
            'min_fitness': np.min(fitnesses),
            'max_fitness': np.max(fitnesses),
            'std_fitness': np.std(fitnesses),
            'best_agent_id': self.best_agent.id if self.best_agent else None,
            'best_fitness': self.best_fitness
        }
        
        self.fitness_history.append(stats)
        return stats
    
    def select_parents(self, selection_strategy: Optional[str] = None) -> List[MetaAgent]:
        """
        Select parents for reproduction based on their fitness.
        
        Args:
            selection_strategy: Strategy to use for parent selection.
                              If None, the value from config will be used.
                              
        Returns:
            List of selected parent agents.
        """
        if selection_strategy is None:
            selection_strategy = self.config.get('scma.evolution_strategy', 'tournament')
        
        if selection_strategy == 'tournament':
            return self._tournament_selection()
        elif selection_strategy == 'roulette':
            return self._roulette_selection()
        elif selection_strategy == 'rank':
            return self._rank_selection()
        else:
            raise ValueError(f"Unknown selection strategy: {selection_strategy}")
    
    def _tournament_selection(self) -> List[MetaAgent]:
        """
        Select parents using tournament selection.
        
        Returns:
            List of selected parent agents.
        """
        tournament_size = self.config.get('scma.tournament_size', 3)
        population_size = len(self.population)
        selected_parents = []
        
        # Select as many parents as the current population size
        for _ in range(population_size):
            # Randomly select tournament_size agents
            tournament = np.random.choice(population_size, tournament_size, replace=False)
            tournament_agents = [self.population[i] for i in tournament]
            
            # Select the agent with the highest fitness
            winner = max(tournament_agents, key=lambda agent: agent.fitness)
            selected_parents.append(winner)
        
        return selected_parents
    
    def _roulette_selection(self) -> List[MetaAgent]:
        """
        Select parents using roulette wheel selection.
        
        Returns:
            List of selected parent agents.
        """
        population_size = len(self.population)
        
        # Get all fitness values
        fitnesses = np.array([agent.fitness for agent in self.population])
        
        # Handle negative fitness values by shifting all values to be positive
        if np.min(fitnesses) < 0:
            fitnesses = fitnesses - np.min(fitnesses) + 1e-6
        
        # Handle zero fitness values
        if np.sum(fitnesses) == 0:
            # If all fitnesses are zero, select randomly
            probabilities = np.ones(population_size) / population_size
        else:
            # Calculate selection probabilities
            probabilities = fitnesses / np.sum(fitnesses)
        
        # Select parents based on probabilities
        selected_indices = np.random.choice(
            population_size,
            size=population_size,
            p=probabilities,
            replace=True
        )
        
        return [self.population[i] for i in selected_indices]
    
    def _rank_selection(self) -> List[MetaAgent]:
        """
        Select parents using rank-based selection.
        
        Returns:
            List of selected parent agents.
        """
        population_size = len(self.population)
        
        # Sort population by fitness
        sorted_population = sorted(self.population, key=lambda agent: agent.fitness)
        
        # Assign ranks (higher rank = higher fitness)
        ranks = np.arange(1, population_size + 1)
        
        # Calculate selection probabilities based on ranks
        selection_pressure = self.config.get('scma.selection_pressure', 0.8)
        probabilities = np.array([
            (2 - selection_pressure) / population_size +
            (2 * rank * (selection_pressure - 1)) / (population_size * (population_size - 1))
            for rank in ranks
        ])
        
        # Ensure probabilities sum to 1
        probabilities = probabilities / np.sum(probabilities)
        
        # Select parents based on probabilities
        selected_indices = np.random.choice(
            population_size,
            size=population_size,
            p=probabilities,
            replace=True
        )
        
        return [sorted_population[i] for i in selected_indices]
    
    def create_next_generation(self) -> List[MetaAgent]:
        """
        Create the next generation of agents through reproduction and mutation.
        
        Returns:
            List of agents in the new generation.
        """
        # Select parents
        parents = self.select_parents()
        
        # Get parameters
        population_size = len(self.population)
        crossover_rate = self.config.get('scma.crossover_rate', 0.7)
        mutation_rate = self.config.get('scma.mutation_rate', 0.1)
        elitism = self.config.get('scma.elitism', True)
        elite_size = self.config.get('scma.elite_size', 2) if elitism else 0
        
        # Create new population
        new_population = []
        
        # Add elite agents if enabled
        if elitism and elite_size > 0:
            # Sort population by fitness (descending)
            sorted_population = sorted(self.population, key=lambda agent: agent.fitness, reverse=True)
            
            # Add top agents to new population
            new_population.extend(sorted_population[:elite_size])
        
        # Fill the rest of the population with offspring
        while len(new_population) < population_size:
            # Select two parents
            parent1 = np.random.choice(parents)
            parent2 = np.random.choice(parents)
            
            # Perform crossover with probability crossover_rate
            if np.random.random() < crossover_rate and parent1 != parent2:
                child = parent1.crossover(parent2)
            else:
                # No crossover, just clone one parent
                child = parent1.mutate(0)  # Create a copy with no mutation
            
            # Perform mutation with probability mutation_rate
            if np.random.random() < mutation_rate:
                child = child.mutate(mutation_rate)
            
            new_population.append(child)
        
        # Update population and generation
        self.population = new_population
        self.generation += 1
        
        return self.population
    
    def evolve(self, fitness_function: Callable[[MetaAgent], float], 
               generations: Optional[int] = None, 
               target_fitness: Optional[float] = None) -> Dict[str, Any]:
        """
        Evolve the population for a specified number of generations or until a target fitness is reached.
        
        Args:
            fitness_function: Function that takes a MetaAgent and returns a fitness score.
            generations: Number of generations to evolve.
                        If None, the value from config will be used.
            target_fitness: Target fitness to reach. Evolution will stop if this fitness is reached.
                           If None, evolution will continue for the specified number of generations.
                           
        Returns:
            Dictionary containing evolution statistics.
        """
        if generations is None:
            generations = self.config.get('scma.generations', 50)
        
        # Initialize population if not already initialized
        if not self.population:
            self.initialize_population()
        
        # Evolve for the specified number of generations
        for _ in range(generations):
            # Evaluate current population
            stats = self.evaluate_population(fitness_function)
            
            # Check if target fitness is reached
            if target_fitness is not None and stats['max_fitness'] >= target_fitness:
                break
            
            # Create next generation
            self.create_next_generation()
        
        # Final evaluation
        final_stats = self.evaluate_population(fitness_function)
        
        return {
            'initial_generation': self.generation - generations,
            'final_generation': self.generation,
            'generations_evolved': generations,
            'best_fitness': self.best_fitness,
            'best_agent_id': self.best_agent.id if self.best_agent else None,
            'fitness_history': self.fitness_history,
            'final_stats': final_stats
        }
    
    def save_population(self, directory: str) -> None:
        """
        Save the current population to a directory.
        
        Args:
            directory: Directory to save the population.
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save each agent
        for i, agent in enumerate(self.population):
            agent.save(os.path.join(directory, f"agent_{i}"))
        
        # Save factory metadata
        factory_data = {
            'generation': self.generation,
            'best_fitness': self.best_fitness,
            'best_agent_id': self.best_agent.id if self.best_agent else None,
            'fitness_history': self.fitness_history
        }
        
        np.save(os.path.join(directory, "factory_data.npy"), factory_data)
    
    @classmethod
    def load_population(cls, directory: str, config: Optional[Config] = None) -> 'AgentFactory':
        """
        Load a population from a directory.
        
        Args:
            directory: Directory to load the population from.
            config: Optional configuration object.
            
        Returns:
            AgentFactory instance with the loaded population.
        """
        factory = cls(config=config)
        
        # Load factory metadata
        factory_data = np.load(os.path.join(directory, "factory_data.npy"), allow_pickle=True).item()
        factory.generation = factory_data['generation']
        factory.best_fitness = factory_data['best_fitness']
        factory.fitness_history = factory_data['fitness_history']
        
        # Load agents
        factory.population = []
        i = 0
        
        while os.path.exists(os.path.join(directory, f"agent_{i}_data.npy")):
            agent = MetaAgent.load(os.path.join(directory, f"agent_{i}"), config=config)
            factory.population.append(agent)
            
            # Check if this is the best agent
            if factory.best_agent is None or agent.id == factory_data['best_agent_id']:
                factory.best_agent = agent
            
            i += 1
        
        return factory
    
    def get_best_agent(self) -> Optional[MetaAgent]:
        """
        Get the agent with the highest fitness in the population.
        
        Returns:
            The agent with the highest fitness, or None if the population is empty.
        """
        if not self.population:
            return None
        
        return max(self.population, key=lambda agent: agent.fitness)
    
    def get_population_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current population.
        
        Returns:
            Dictionary containing population statistics.
        """
        if not self.population:
            return {
                'population_size': 0,
                'generation': self.generation,
                'best_fitness': self.best_fitness
            }
        
        fitnesses = [agent.fitness for agent in self.population]
        
        return {
            'population_size': len(self.population),
            'generation': self.generation,
            'mean_fitness': np.mean(fitnesses),
            'median_fitness': np.median(fitnesses),
            'min_fitness': np.min(fitnesses),
            'max_fitness': np.max(fitnesses),
            'std_fitness': np.std(fitnesses),
            'best_agent_id': self.best_agent.id if self.best_agent else None,
            'best_fitness': self.best_fitness
        }
