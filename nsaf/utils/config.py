"""
Configuration utilities for the NSAF framework.
"""

import os
import yaml
from typing import Dict, Any, Optional


class Config:
    """
    Configuration class for the NSAF framework.
    
    This class handles loading, saving, and accessing configuration parameters
    for the framework.
    """
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize a new Config instance.
        
        Args:
            config_dict: Optional dictionary containing configuration parameters.
                         If None, default parameters will be used.
        """
        self._config = config_dict or self._get_default_config()
    
    @staticmethod
    def _get_default_config() -> Dict[str, Any]:
        """
        Get the default configuration parameters.
        
        Returns:
            A dictionary containing default configuration parameters.
        """
        return {
            'scma': {
                'population_size': 10,
                'generations': 50,
                'mutation_rate': 0.1,
                'crossover_rate': 0.7,
                'selection_pressure': 0.8,
                'architecture_complexity': 'medium',
                'fitness_evaluation_steps': 100,
                'agent_memory_size': 1024,
                'learning_rate': 0.001,
                'optimizer': 'adam',
                'activation_function': 'relu',
                'hidden_layers': [128, 64, 32],
                'use_batch_normalization': True,
                'dropout_rate': 0.2,
                'evolution_strategy': 'tournament',
                'tournament_size': 3,
                'elitism': True,
                'elite_size': 2,
            },
            'training': {
                'batch_size': 32,
                'epochs': 100,
                'validation_split': 0.2,
                'early_stopping_patience': 10,
                'model_checkpoint': True,
                'checkpoint_path': './checkpoints',
            },
            'logging': {
                'level': 'INFO',
                'log_file': './nsaf.log',
                'console_output': True,
            },
            'system': {
                'random_seed': 42,
                'use_gpu': True,
                'num_threads': 4,
                'memory_limit': 0.8,  # 80% of available memory
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration parameter.
        
        Args:
            key: The parameter key, can be nested using dot notation (e.g., 'scma.population_size').
            default: The default value to return if the key is not found.
            
        Returns:
            The parameter value, or the default value if the key is not found.
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration parameter.
        
        Args:
            key: The parameter key, can be nested using dot notation (e.g., 'scma.population_size').
            value: The parameter value.
        """
        keys = key.split('.')
        config = self._config
        
        for i, k in enumerate(keys[:-1]):
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def load_from_file(self, file_path: str) -> None:
        """
        Load configuration from a YAML file.
        
        Args:
            file_path: Path to the YAML configuration file.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if config:
            self._config = config
    
    def save_to_file(self, file_path: str) -> None:
        """
        Save configuration to a YAML file.
        
        Args:
            file_path: Path to save the YAML configuration file.
        """
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        with open(file_path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False)
    
    def __str__(self) -> str:
        """
        Get a string representation of the configuration.
        
        Returns:
            A YAML string representation of the configuration.
        """
        return yaml.dump(self._config, default_flow_style=False)
