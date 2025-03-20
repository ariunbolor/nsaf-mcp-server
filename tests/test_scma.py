"""
Tests for the Self-Constructing Meta-Agents (SCMA) module.
"""

import os
import numpy as np
import pytest
import tensorflow as tf
from typing import List, Dict, Any

from nsaf.utils.config import Config
from nsaf.scma.meta_agent import MetaAgent
from nsaf.scma.agent_factory import AgentFactory
from nsaf.scma.evolution import Evolution


class TestMetaAgent:
    """Test cases for the MetaAgent class."""
    
    def test_initialization(self):
        """Test that a MetaAgent can be initialized with default parameters."""
        agent = MetaAgent()
        
        assert agent is not None
        assert agent.id is not None
        assert agent.architecture is not None
        assert agent.model is not None
        assert isinstance(agent.model, tf.keras.Model)
        assert agent.fitness == 0.0
        assert agent.generation == 0
        assert len(agent.parent_ids) == 0
    
    def test_random_architecture_generation(self):
        """Test that random architectures are generated correctly."""
        config = Config()
        
        # Test with different complexity levels
        for complexity in ['simple', 'medium', 'complex']:
            config.set('scma.architecture_complexity', complexity)
            agent = MetaAgent(config=config)
            
            assert agent.architecture is not None
            assert 'hidden_layers' in agent.architecture
            assert 'activations' in agent.architecture
            assert 'use_batch_normalization' in agent.architecture
            assert 'dropout_rate' in agent.architecture
            assert 'optimizer' in agent.architecture
            assert 'learning_rate' in agent.architecture
            
            # Check that the number of layers is appropriate for the complexity
            if complexity == 'simple':
                assert len(agent.architecture['hidden_layers']) <= 2
            elif complexity == 'medium':
                assert len(agent.architecture['hidden_layers']) <= 4
            else:  # complex
                assert len(agent.architecture['hidden_layers']) <= 8
    
    def test_model_building(self):
        """Test that models are built correctly from architectures."""
        agent = MetaAgent()
        
        # Check that the model has the correct number of layers
        expected_layers = 2 + len(agent.architecture['hidden_layers'])  # Input + hidden + output
        if agent.architecture['use_batch_normalization']:
            expected_layers += len(agent.architecture['hidden_layers'])  # Batch normalization layers
        if agent.architecture['dropout_rate'] > 0:
            expected_layers += len(agent.architecture['hidden_layers'])  # Dropout layers
        
        # Count layers (excluding activation layers which are counted separately in TF)
        model_layers = [layer for layer in agent.model.layers 
                       if not isinstance(layer, tf.keras.layers.Activation)]
        
        assert len(model_layers) > 0
        
        # Check input and output shapes
        input_shape = agent.model.input_shape
        output_shape = agent.model.output_shape
        
        assert input_shape[1] == agent.architecture['input_shape'][1]
        assert output_shape[1] == agent.architecture['output_shape'][1]
    
    def test_prediction(self):
        """Test that the agent can make predictions."""
        agent = MetaAgent()
        
        # Create random input data
        input_shape = agent.architecture['input_shape'][1]
        batch_size = 5
        inputs = np.random.random((batch_size, input_shape))
        
        # Make predictions
        predictions = agent.predict(inputs)
        
        # Check that predictions have the correct shape
        output_shape = agent.architecture['output_shape'][1]
        assert predictions.shape == (batch_size, output_shape)
    
    def test_mutation(self):
        """Test that mutation creates a new agent with modified architecture."""
        parent = MetaAgent()
        child = parent.mutate(mutation_rate=1.0)  # Force mutation
        
        assert child is not None
        assert child.id != parent.id
        assert child.parent_ids == [parent.id]
        assert child.generation == parent.generation + 1
        
        # Check that at least one aspect of the architecture has changed
        architecture_different = False
        
        for key in ['hidden_layers', 'activations', 'use_batch_normalization', 
                   'dropout_rate', 'optimizer', 'learning_rate']:
            if parent.architecture[key] != child.architecture[key]:
                architecture_different = True
                break
        
        assert architecture_different
    
    def test_crossover(self):
        """Test that crossover creates a new agent with combined architecture."""
        parent1 = MetaAgent()
        parent2 = MetaAgent()
        child = parent1.crossover(parent2)
        
        assert child is not None
        assert child.id != parent1.id
        assert child.id != parent2.id
        assert set(child.parent_ids) == {parent1.id, parent2.id}
        assert child.generation == max(parent1.generation, parent2.generation) + 1
        
        # Check that the child's architecture has elements from both parents
        # This is hard to test deterministically due to the random nature of crossover,
        # but we can check that the architecture is valid
        assert 'hidden_layers' in child.architecture
        assert 'activations' in child.architecture
        assert 'use_batch_normalization' in child.architecture
        assert 'dropout_rate' in child.architecture
        assert 'optimizer' in child.architecture
        assert 'learning_rate' in child.architecture
        
        # Check that the number of hidden layers and activations match
        assert len(child.architecture['hidden_layers']) == len(child.architecture['activations'])
    
    def test_save_load(self, tmp_path):
        """Test that an agent can be saved and loaded."""
        agent = MetaAgent()
        
        # Set some properties
        agent.fitness = 0.75
        agent.generation = 5
        agent.parent_ids = ['parent1', 'parent2']
        agent.metadata = {'test': 'value'}
        
        # Save the agent
        filepath = os.path.join(tmp_path, 'test_agent')
        agent.save(filepath)
        
        # Check that the files exist
        assert os.path.exists(f"{filepath}_model.h5")
        assert os.path.exists(f"{filepath}_data.npy")
        
        # Load the agent
        loaded_agent = MetaAgent.load(filepath)
        
        # Check that the properties are preserved
        assert loaded_agent.id == agent.id
        assert loaded_agent.fitness == agent.fitness
        assert loaded_agent.generation == agent.generation
        assert loaded_agent.parent_ids == agent.parent_ids
        assert loaded_agent.metadata == agent.metadata
        
        # Check that the architecture is preserved
        for key in agent.architecture:
            if isinstance(agent.architecture[key], list):
                assert len(loaded_agent.architecture[key]) == len(agent.architecture[key])
                for i in range(len(agent.architecture[key])):
                    assert loaded_agent.architecture[key][i] == agent.architecture[key][i]
            else:
                assert loaded_agent.architecture[key] == agent.architecture[key]


class TestAgentFactory:
    """Test cases for the AgentFactory class."""
    
    def test_initialization(self):
        """Test that an AgentFactory can be initialized with default parameters."""
        factory = AgentFactory()
        
        assert factory is not None
        assert factory.population == []
        assert factory.generation == 0
        assert factory.best_agent is None
        assert factory.best_fitness == float('-inf')
        assert factory.fitness_history == []
    
    def test_initialize_population(self):
        """Test that a population can be initialized."""
        factory = AgentFactory()
        population = factory.initialize_population(size=5)
        
        assert len(population) == 5
        assert factory.population == population
        assert all(isinstance(agent, MetaAgent) for agent in population)
    
    def test_evaluate_population(self):
        """Test that a population can be evaluated."""
        factory = AgentFactory()
        factory.initialize_population(size=5)
        
        # Define a simple fitness function
        def fitness_function(agent):
            return len(agent.architecture['hidden_layers'])
        
        # Evaluate the population
        stats = factory.evaluate_population(fitness_function)
        
        assert 'generation' in stats
        assert 'mean_fitness' in stats
        assert 'median_fitness' in stats
        assert 'min_fitness' in stats
        assert 'max_fitness' in stats
        assert 'std_fitness' in stats
        assert 'best_agent_id' in stats
        assert 'best_fitness' in stats
        
        # Check that the fitness values are set
        for agent in factory.population:
            assert agent.fitness == len(agent.architecture['hidden_layers'])
        
        # Check that the best agent is identified correctly
        best_agent = max(factory.population, key=lambda a: a.fitness)
        assert factory.best_agent == best_agent
        assert factory.best_fitness == best_agent.fitness
        
        # Check that the fitness history is updated
        assert len(factory.fitness_history) == 1
        assert factory.fitness_history[0] == stats
    
    def test_select_parents(self):
        """Test that parents can be selected for reproduction."""
        factory = AgentFactory()
        factory.initialize_population(size=10)
        
        # Set fitness values
        for i, agent in enumerate(factory.population):
            agent.fitness = i
        
        # Test different selection strategies
        for strategy in ['tournament', 'roulette', 'rank']:
            parents = factory.select_parents(selection_strategy=strategy)
            
            assert len(parents) == len(factory.population)
            assert all(parent in factory.population for parent in parents)
    
    def test_create_next_generation(self):
        """Test that a new generation can be created."""
        factory = AgentFactory()
        factory.initialize_population(size=10)
        
        # Set fitness values
        for i, agent in enumerate(factory.population):
            agent.fitness = i
        
        # Create next generation
        new_population = factory.create_next_generation()
        
        assert len(new_population) == len(factory.population)
        assert factory.generation == 1
        assert all(isinstance(agent, MetaAgent) for agent in new_population)
        
        # Check that the new population is different from the old one
        old_ids = [agent.id for agent in factory.population]
        new_ids = [agent.id for agent in new_population]
        
        assert set(old_ids) != set(new_ids)
    
    def test_evolve(self):
        """Test that a population can be evolved over multiple generations."""
        factory = AgentFactory()
        
        # Define a simple fitness function
        def fitness_function(agent):
            return len(agent.architecture['hidden_layers'])
        
        # Evolve the population
        results = factory.evolve(fitness_function, generations=5)
        
        assert 'initial_generation' in results
        assert 'final_generation' in results
        assert 'generations_evolved' in results
        assert 'best_fitness' in results
        assert 'best_agent_id' in results
        assert 'fitness_history' in results
        assert 'final_stats' in results
        
        assert factory.generation == 5
        assert len(factory.fitness_history) == 5
        assert factory.best_agent is not None
        assert factory.best_fitness > 0
    
    def test_save_load_population(self, tmp_path):
        """Test that a population can be saved and loaded."""
        factory = AgentFactory()
        factory.initialize_population(size=5)
        
        # Set some properties
        factory.generation = 10
        factory.best_fitness = 0.75
        factory.best_agent = factory.population[0]
        factory.fitness_history = [{'generation': i, 'mean_fitness': i/10} for i in range(10)]
        
        # Save the population
        directory = os.path.join(tmp_path, 'test_population')
        factory.save_population(directory)
        
        # Check that the files exist
        assert os.path.exists(os.path.join(directory, "factory_data.npy"))
        for i in range(5):
            assert os.path.exists(os.path.join(directory, f"agent_{i}_data.npy"))
            assert os.path.exists(os.path.join(directory, f"agent_{i}_model.h5"))
        
        # Load the population
        loaded_factory = AgentFactory.load_population(directory)
        
        # Check that the properties are preserved
        assert loaded_factory.generation == factory.generation
        assert loaded_factory.best_fitness == factory.best_fitness
        assert len(loaded_factory.population) == len(factory.population)
        assert len(loaded_factory.fitness_history) == len(factory.fitness_history)
        
        # Check that the best agent is preserved
        assert loaded_factory.best_agent is not None
        assert loaded_factory.best_agent.id == factory.best_agent.id


class TestEvolution:
    """Test cases for the Evolution class."""
    
    def test_initialization(self):
        """Test that an Evolution instance can be initialized with default parameters."""
        evolution = Evolution()
        
        assert evolution is not None
        assert evolution.factory is not None
        assert evolution.experiment_name is not None
        assert evolution.experiment_dir is not None
        assert evolution.checkpoints_dir is not None
        assert evolution.logs_dir is not None
        assert evolution.plots_dir is not None
        assert evolution.best_agents == []
        assert evolution.evolution_history == []
    
    def test_setup_experiment(self, tmp_path):
        """Test that experiment directories can be set up."""
        evolution = Evolution()
        
        # Set up experiment
        experiment_name = "test_experiment"
        base_dir = str(tmp_path)
        evolution.setup_experiment(experiment_name=experiment_name, base_dir=base_dir)
        
        # Check that the directories are created
        assert evolution.experiment_name == experiment_name
        assert evolution.experiment_dir == base_dir
        assert os.path.exists(evolution.checkpoints_dir)
        assert os.path.exists(evolution.logs_dir)
        assert os.path.exists(evolution.plots_dir)
        assert os.path.exists(os.path.join(base_dir, experiment_name, "config.yaml"))
    
    def test_run_evolution(self, tmp_path):
        """Test that evolution can be run."""
        evolution = Evolution()
        
        # Set up experiment
        experiment_name = "test_evolution"
        base_dir = str(tmp_path)
        evolution.setup_experiment(experiment_name=experiment_name, base_dir=base_dir)
        
        # Define a simple fitness function
        def fitness_function(agent):
            return len(agent.architecture['hidden_layers'])
        
        # Run evolution
        results = evolution.run_evolution(
            fitness_function=fitness_function,
            generations=3,
            population_size=5,
            checkpoint_interval=1,
            verbose=False
        )
        
        assert 'experiment_name' in results
        assert 'generations' in results
        assert 'actual_generations' in results
        assert 'target_fitness' in results
        assert 'best_fitness' in results
        assert 'best_agent_id' in results
        assert 'elapsed_time' in results
        assert 'final_stats' in results
        assert 'experiment_dir' in results
        
        # Check that the evolution history is recorded
        assert len(evolution.evolution_history) == 3
        
        # Check that checkpoints are saved
        assert os.path.exists(os.path.join(evolution.checkpoints_dir, "generation_0"))
        assert os.path.exists(os.path.join(evolution.checkpoints_dir, "generation_1"))
        assert os.path.exists(os.path.join(evolution.checkpoints_dir, "generation_2"))
        assert os.path.exists(os.path.join(evolution.checkpoints_dir, "final_generation"))
        
        # Check that the best agent is saved
        assert os.path.exists(os.path.join(evolution.checkpoints_dir, "best_agent_final_data.npy"))
        assert os.path.exists(os.path.join(evolution.checkpoints_dir, "best_agent_final_model.h5"))
        
        # Check that plots are generated
        assert os.path.exists(os.path.join(evolution.plots_dir, "fitness_evolution.png"))
        assert os.path.exists(os.path.join(evolution.plots_dir, "fitness_distribution.png"))
    
    def test_load_experiment(self, tmp_path):
        """Test that an experiment can be loaded."""
        # First, run an evolution to create experiment data
        evolution = Evolution()
        
        # Set up experiment
        experiment_name = "test_load"
        base_dir = str(tmp_path)
        evolution.setup_experiment(experiment_name=experiment_name, base_dir=base_dir)
        
        # Define a simple fitness function
        def fitness_function(agent):
            return len(agent.architecture['hidden_layers'])
        
        # Run evolution
        evolution.run_evolution(
            fitness_function=fitness_function,
            generations=2,
            population_size=3,
            checkpoint_interval=1,
            verbose=False
        )
        
        # Now, create a new Evolution instance and load the experiment
        new_evolution = Evolution()
        experiment_dir = os.path.join(base_dir, experiment_name)
        results = new_evolution.load_experiment(experiment_dir)
        
        # Check that the experiment is loaded correctly
        assert new_evolution.experiment_name == experiment_name
        assert new_evolution.experiment_dir == base_dir
        assert new_evolution.factory is not None
        assert len(new_evolution.factory.population) > 0
        assert new_evolution.factory.best_agent is not None
        
        # Check that the results are loaded correctly
        assert 'experiment_name' in results
        assert 'generations' in results
        assert 'best_fitness' in results
