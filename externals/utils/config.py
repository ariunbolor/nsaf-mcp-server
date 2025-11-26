"""
Configuration Module
-----------------
Configuration management utilities for the agent builder system.
"""

from typing import Any, Dict, List, Optional, Union, Type
import os
import json
import yaml
from pathlib import Path
from dataclasses import dataclass, field
import logging
from functools import lru_cache

from .exceptions import ConfigError
from .validation import Schema, Required, Type as TypeValidator

class ConfigSource:
    """Available configuration sources"""
    ENV = "env"
    FILE = "file"
    DEFAULT = "default"

@dataclass
class ConfigValue:
    """Configuration value with metadata"""
    value: Any
    source: str
    timestamp: float = field(default_factory=lambda: time.time())

class ConfigManager:
    """Configuration manager"""
    
    def __init__(self,
                 config_dir: Optional[Union[str, Path]] = None,
                 env_prefix: str = "APP_"):
        self.config_dir = Path(config_dir) if config_dir else Path("config")
        self.env_prefix = env_prefix
        self.values: Dict[str, ConfigValue] = {}
        self.schema: Optional[Schema] = None
        self.logger = logging.getLogger(__name__)
        
    def set_schema(self, schema: Schema) -> None:
        """Set validation schema"""
        self.schema = schema
        
    def load(self,
            default_config: Optional[Dict[str, Any]] = None,
            config_files: Optional[List[Union[str, Path]]] = None) -> None:
        """
        Load configuration from all sources
        
        Args:
            default_config: Default configuration values
            config_files: List of configuration files to load
            
        Raises:
            ConfigError: If configuration loading fails
        """
        try:
            # Start with defaults
            if default_config:
                for key, value in default_config.items():
                    self.values[key] = ConfigValue(
                        value=value,
                        source=ConfigSource.DEFAULT
                    )
                    
            # Load from files
            if config_files:
                for file_path in config_files:
                    self._load_file(file_path)
                    
            # Load from environment
            self._load_env()
            
            # Validate if schema set
            if self.schema:
                config_dict = {
                    key: value.value
                    for key, value in self.values.items()
                }
                self.schema.validate(config_dict)
                
        except Exception as e:
            raise ConfigError(f"Failed to load configuration: {str(e)}")
            
    def _load_file(self, file_path: Union[str, Path]) -> None:
        """Load configuration from file"""
        path = Path(file_path)
        
        if not path.is_absolute():
            path = self.config_dir / path
            
        try:
            with open(path) as f:
                if path.suffix in ['.yaml', '.yml']:
                    config = yaml.safe_load(f)
                elif path.suffix == '.json':
                    config = json.load(f)
                else:
                    raise ConfigError(f"Unsupported file type: {path.suffix}")
                    
            for key, value in config.items():
                self.values[key] = ConfigValue(
                    value=value,
                    source=ConfigSource.FILE
                )
                
        except Exception as e:
            raise ConfigError(f"Failed to load {path}: {str(e)}")
            
    def _load_env(self) -> None:
        """Load configuration from environment variables"""
        for key, value in os.environ.items():
            if key.startswith(self.env_prefix):
                config_key = key[len(self.env_prefix):].lower()
                
                # Try to parse as JSON for complex values
                try:
                    parsed_value = json.loads(value)
                except json.JSONDecodeError:
                    parsed_value = value
                    
                self.values[config_key] = ConfigValue(
                    value=parsed_value,
                    source=ConfigSource.ENV
                )
                
    def get(self,
            key: str,
            default: Any = None,
            required: bool = False) -> Any:
        """
        Get configuration value
        
        Args:
            key: Configuration key
            default: Default value if not found
            required: Whether the key is required
            
        Returns:
            Configuration value
            
        Raises:
            ConfigError: If required key is missing
        """
        try:
            if key in self.values:
                return self.values[key].value
                
            if required:
                raise ConfigError(f"Required configuration key missing: {key}")
                
            return default
            
        except Exception as e:
            if required:
                raise ConfigError(f"Failed to get config value: {str(e)}")
            return default
            
    def set(self,
            key: str,
            value: Any,
            source: str = ConfigSource.DEFAULT) -> None:
        """
        Set configuration value
        
        Args:
            key: Configuration key
            value: Configuration value
            source: Configuration source
        """
        self.values[key] = ConfigValue(
            value=value,
            source=source
        )
        
    def update(self,
               values: Dict[str, Any],
               source: str = ConfigSource.DEFAULT) -> None:
        """
        Update multiple configuration values
        
        Args:
            values: Dictionary of configuration values
            source: Configuration source
        """
        for key, value in values.items():
            self.set(key, value, source)
            
    def save(self, file_path: Union[str, Path]) -> None:
        """
        Save configuration to file
        
        Args:
            file_path: Path to save configuration
            
        Raises:
            ConfigError: If save fails
        """
        try:
            path = Path(file_path)
            
            if not path.is_absolute():
                path = self.config_dir / path
                
            # Ensure directory exists
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert to dictionary
            config = {
                key: value.value
                for key, value in self.values.items()
            }
            
            # Save based on file type
            with open(path, 'w') as f:
                if path.suffix in ['.yaml', '.yml']:
                    yaml.safe_dump(config, f, default_flow_style=False)
                elif path.suffix == '.json':
                    json.dump(config, f, indent=2)
                else:
                    raise ConfigError(f"Unsupported file type: {path.suffix}")
                    
        except Exception as e:
            raise ConfigError(f"Failed to save configuration: {str(e)}")
            
    def get_source(self, key: str) -> Optional[str]:
        """Get source of configuration value"""
        if key in self.values:
            return self.values[key].source
        return None
        
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration values"""
        return {
            key: value.value
            for key, value in self.values.items()
        }
        
    def get_with_source(self) -> Dict[str, Dict[str, Any]]:
        """Get all configuration values with sources"""
        return {
            key: {
                'value': value.value,
                'source': value.source,
                'timestamp': value.timestamp
            }
            for key, value in self.values.items()
        }

class EnvironmentConfig:
    """Environment-specific configuration"""
    
    def __init__(self,
                 environments: Dict[str, Dict[str, Any]],
                 default_env: str = "development"):
        self.environments = environments
        self.default_env = default_env
        self.current_env = self._get_current_env()
        
    def _get_current_env(self) -> str:
        """Get current environment"""
        return os.getenv("APP_ENV", self.default_env)
        
    def get_config(self) -> Dict[str, Any]:
        """Get configuration for current environment"""
        # Get default config
        config = self.environments.get(self.default_env, {}).copy()
        
        # Override with current environment
        if self.current_env != self.default_env:
            config.update(self.environments.get(self.current_env, {}))
            
        return config

@lru_cache()
def get_config_manager(
    config_dir: Optional[Union[str, Path]] = None,
    env_prefix: str = "APP_"
) -> ConfigManager:
    """
    Get cached config manager instance
    
    Args:
        config_dir: Configuration directory
        env_prefix: Environment variable prefix
        
    Returns:
        ConfigManager instance
    """
    return ConfigManager(config_dir, env_prefix)

# Default configuration schema
DEFAULT_SCHEMA = Schema({
    'debug': [TypeValidator(bool)],
    'log_level': [Required(), TypeValidator(str)],
    'secret_key': [Required(), TypeValidator(str)],
    'database_url': [Required(), TypeValidator(str)],
    'redis_url': [TypeValidator(str)],
    'allowed_hosts': [TypeValidator(list)],
    'cors_origins': [TypeValidator(list)],
    'max_workers': [TypeValidator(int)],
    'request_timeout': [TypeValidator(int)],
    'cache_ttl': [TypeValidator(int)]
})

# Global config manager instance
config = get_config_manager()
config.set_schema(DEFAULT_SCHEMA)
