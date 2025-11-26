from typing import Dict, Type, List
import importlib
import json
from pathlib import Path
from .base_skill import BaseSkill

class SkillRegistry:
    def __init__(self):
        self._skills: Dict[str, Type[BaseSkill]] = {}
        self._instances: Dict[str, BaseSkill] = {}
        self._configs: Dict[str, Dict] = {}
        
        # Load all available skills
        self._load_skills()
        
    def _load_skills(self):
        """Load all available skills from the skills directory"""
        skills_dir = Path(__file__).parent
        
        # Load skill modules
        for skill_file in skills_dir.glob('*_skill.py'):
            if skill_file.stem == 'base_skill':
                continue
                
            try:
                # Import skill module
                module_name = f"skills.{skill_file.stem}"
                module = importlib.import_module(module_name)
                
                # Find skill class (assuming it ends with 'Skill')
                for attr_name in dir(module):
                    if attr_name.endswith('Skill') and attr_name != 'BaseSkill':
                        skill_class = getattr(module, attr_name)
                        if isinstance(skill_class, type) and issubclass(skill_class, BaseSkill):
                            # Create temporary instance to get name
                            temp_instance = skill_class()
                            self._skills[temp_instance.name] = skill_class
                            
            except Exception as e:
                print(f"Error loading skill {skill_file.stem}: {e}")
        
        # Load skill configs
        configs_dir = skills_dir / 'configs'
        if configs_dir.exists():
            for config_file in configs_dir.glob('*.json'):
                try:
                    with open(config_file) as f:
                        self._configs[config_file.stem] = json.load(f)
                except Exception as e:
                    print(f"Error loading config {config_file.stem}: {e}")

    def get_skill(self, name: str) -> BaseSkill:
        """Get or create skill instance"""
        if name not in self._instances:
            if name not in self._skills:
                raise ValueError(f"Unknown skill: {name}")
            
            # Create new instance
            skill_class = self._skills[name]
            self._instances[name] = skill_class()
            
        return self._instances[name]

    def get_skill_config(self, name: str) -> Dict:
        """Get skill configuration"""
        return self._configs.get(name, {})

    def list_skills(self) -> List[Dict]:
        """List all available skills"""
        skills = []
        for name, skill_class in self._skills.items():
            # Create temporary instance to get info
            temp_instance = skill_class()
            skills.append({
                'name': name,
                'description': temp_instance.description,
                'config_schema': self.get_skill_config(name).get('config_schema', {}),
                'params_schema': self.get_skill_config(name).get('params_schema', {})
            })
        return skills

    def validate_skill_config(self, name: str, config: Dict) -> List[str]:
        """Validate skill configuration"""
        if name not in self._skills:
            return [f"Unknown skill: {name}"]
            
        skill_config = self.get_skill_config(name)
        if not skill_config:
            return []
            
        schema = skill_config.get('config_schema', {})
        # TODO: Implement JSON Schema validation
        return []

    def validate_skill_params(self, name: str, params: Dict) -> List[str]:
        """Validate skill parameters"""
        if name not in self._skills:
            return [f"Unknown skill: {name}"]
            
        skill_config = self.get_skill_config(name)
        if not skill_config:
            return []
            
        schema = skill_config.get('params_schema', {})
        # TODO: Implement JSON Schema validation
        return []

    def execute_skill(self, name: str, params: Dict = None) -> Dict:
        """Execute a skill with parameters"""
        try:
            skill = self.get_skill(name)
            
            # Validate parameters
            errors = self.validate_skill_params(name, params or {})
            if errors:
                return {
                    'success': False,
                    'errors': errors
                }
            
            # Execute skill
            return skill.execute(params)
            
        except Exception as e:
            return {
                'success': False,
                'errors': [str(e)]
            }

# Global skill registry instance
skill_registry = SkillRegistry()
