from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
import json
import os
from pathlib import Path

class BaseSkill(ABC):
    def __init__(self, name: str, description: str, config: Dict[str, Any] = None):
        self.name = name
        self.description = description
        self.config = config or {}
        self.required_credentials = []
        self.system_config = {}
        self.user_config = {}
        
        # Load configurations
        self._load_system_config()
        self._load_user_config()

    def _load_system_config(self):
        """Load system-level configuration"""
        config_path = Path(__file__).parent / 'configs' / f'{self.name}.json'
        if config_path.exists():
            with open(config_path) as f:
                self.system_config = json.load(f)

    def _load_user_config(self):
        """Load user-level configuration"""
        user_config_dir = Path.home() / '.ai_agent_builder' / 'skills' / self.name
        user_config_file = user_config_dir / 'config.json'
        if user_config_file.exists():
            with open(user_config_file) as f:
                self.user_config = json.load(f)

    def save_user_config(self):
        """Save user-level configuration"""
        user_config_dir = Path.home() / '.ai_agent_builder' / 'skills' / self.name
        user_config_dir.mkdir(parents=True, exist_ok=True)
        
        user_config_file = user_config_dir / 'config.json'
        with open(user_config_file, 'w') as f:
            json.dump(self.user_config, f, indent=2)

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value, checking user config first then system config"""
        return self.user_config.get(key, self.system_config.get(key, default))

    def set_config(self, key: str, value: Any):
        """Set user-level configuration value"""
        self.user_config[key] = value
        self.save_user_config()

    @abstractmethod
    def execute(self, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute the skill with given parameters"""
        pass

    @abstractmethod
    def validate_params(self, params: Dict[str, Any]) -> List[str]:
        """Validate parameters, return list of error messages if any"""
        pass

    def check_credentials(self) -> List[str]:
        """Check if required credentials are available"""
        missing = []
        for cred in self.required_credentials:
            if not self.get_config(f'credentials.{cred}'):
                missing.append(cred)
        return missing

    def to_dict(self) -> Dict[str, Any]:
        """Convert skill to dictionary representation"""
        return {
            'name': self.name,
            'description': self.description,
            'required_credentials': self.required_credentials,
            'config_schema': self.system_config.get('config_schema', {}),
            'params_schema': self.system_config.get('params_schema', {})
        }
