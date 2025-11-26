"""
Configuration Loader Module
-------------------------
Handles loading and managing configuration settings from YAML files and environment variables.
"""

import os
from typing import Dict, Any, Optional
import yaml
from pathlib import Path
from dotenv import load_dotenv

class ConfigurationError(Exception):
    """Custom exception for configuration errors"""
    pass

class ConfigLoader:
    """
    Configuration loader that handles loading and merging configuration from
    multiple sources (YAML files, environment variables, etc.)
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the config loader
        
        Args:
            config_path: Optional path to custom config file
        """
        self.config: Dict[str, Any] = {}
        self.env_prefix = "AGENT_BUILDER_"
        
        # Load environment variables from .env file if it exists
        load_dotenv()
        
        # Load default configuration
        self._load_default_config()
        
        # Load custom configuration if provided
        if config_path:
            self._load_custom_config(config_path)
            
        # Override with environment variables
        self._load_environment_variables()
        
        # Validate configuration
        self._validate_config()

    def _load_default_config(self) -> None:
        """Load default configuration from default_config.yaml"""
        default_config_path = Path(__file__).parent / "default_config.yaml"
        
        try:
            with open(default_config_path) as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            raise ConfigurationError(f"Failed to load default config: {str(e)}")

    def _load_custom_config(self, config_path: str) -> None:
        """
        Load custom configuration and merge with defaults
        
        Args:
            config_path: Path to custom config file
        """
        try:
            with open(config_path) as f:
                custom_config = yaml.safe_load(f)
                
            # Deep merge custom config with defaults
            self.config = self._deep_merge(self.config, custom_config)
            
        except Exception as e:
            raise ConfigurationError(f"Failed to load custom config: {str(e)}")

    def _load_environment_variables(self) -> None:
        """Load configuration from environment variables"""
        for env_var, value in os.environ.items():
            if env_var.startswith(self.env_prefix):
                # Convert environment variable name to config key
                # e.g., AGENT_BUILDER_MODEL_PROVIDER_OPENAI_API_KEY -> model_provider.openai.api_key
                config_key = env_var[len(self.env_prefix):].lower().replace("_", ".")
                
                # Set value in config
                self._set_nested_value(self.config, config_key.split("."), value)

    def _set_nested_value(self, 
                         config: Dict[str, Any], 
                         key_parts: list,
                         value: Any) -> None:
        """
        Set a value in a nested dictionary using a list of key parts
        
        Args:
            config: Configuration dictionary
            key_parts: List of nested key parts
            value: Value to set
        """
        current = config
        for part in key_parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        # Convert value to appropriate type
        current[key_parts[-1]] = self._convert_value(value)

    def _convert_value(self, value: str) -> Any:
        """
        Convert string value to appropriate type
        
        Args:
            value: String value to convert
            
        Returns:
            Converted value
        """
        # Try to convert to boolean
        if value.lower() in ["true", "false"]:
            return value.lower() == "true"
            
        # Try to convert to integer
        try:
            return int(value)
        except ValueError:
            pass
            
        # Try to convert to float
        try:
            return float(value)
        except ValueError:
            pass
            
        # Return as string if no other conversion works
        return value

    def _deep_merge(self, 
                   dict1: Dict[str, Any], 
                   dict2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two dictionaries
        
        Args:
            dict1: Base dictionary
            dict2: Dictionary to merge (overrides dict1)
            
        Returns:
            Merged dictionary
        """
        result = dict1.copy()
        
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
                
        return result

    def _validate_config(self) -> None:
        """
        Validate the configuration
        
        Raises:
            ConfigurationError: If configuration is invalid
        """
        required_settings = [
            "model_provider.default",
            "api.host",
            "api.port"
        ]
        
        for setting in required_settings:
            if not self.get(setting):
                raise ConfigurationError(f"Missing required configuration: {setting}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation
        
        Args:
            key: Configuration key in dot notation (e.g., "model_provider.openai.api_key")
            default: Default value if key doesn't exist
            
        Returns:
            Configuration value or default
        """
        try:
            value = self.config
            for part in key.split("."):
                value = value[part]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value using dot notation
        
        Args:
            key: Configuration key in dot notation
            value: Value to set
        """
        self._set_nested_value(self.config, key.split("."), value)

    def get_all(self) -> Dict[str, Any]:
        """
        Get the entire configuration
        
        Returns:
            Complete configuration dictionary
        """
        return self.config.copy()

    def reload(self) -> None:
        """Reload the configuration"""
        self._load_default_config()
        self._load_environment_variables()
        self._validate_config()

# Global configuration instance
config = ConfigLoader()
