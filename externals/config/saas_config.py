"""
SaaS Configuration
----------------
Configuration for running the AI Agent Builder as a SaaS platform.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import json

class SaaSConfig:
    def __init__(self):
        self.tenant_configs: Dict[str, Dict[str, Any]] = {}
        self.load_configs()
        
    def load_configs(self):
        """Load tenant configurations"""
        config_dir = Path("config/tenants")
        config_dir.mkdir(parents=True, exist_ok=True)
        
        for config_file in config_dir.glob("*.json"):
            with open(config_file, "r") as f:
                tenant_id = config_file.stem
                self.tenant_configs[tenant_id] = json.load(f)
                
    def get_tenant_config(self, tenant_id: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific tenant"""
        return self.tenant_configs.get(tenant_id)
    
    def update_tenant_config(self, tenant_id: str, config: Dict[str, Any]):
        """Update configuration for a specific tenant"""
        self.tenant_configs[tenant_id] = config
        
        config_path = Path("config/tenants") / f"{tenant_id}.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
            
    def create_tenant(self, tenant_id: str, config: Dict[str, Any]):
        """Create a new tenant configuration"""
        if tenant_id in self.tenant_configs:
            raise ValueError(f"Tenant {tenant_id} already exists")
            
        # Set default limits
        config.setdefault("limits", {
            "max_agents": 10,
            "max_qubits": 16,
            "max_circuit_depth": 100,
            "concurrent_computations": 5
        })
        
        # Set default features
        config.setdefault("features", {
            "quantum_symbolic": True,
            "rag": True,
            "monitoring": True,
            "custom_circuits": False
        })
        
        # Set default integrations
        config.setdefault("integrations", {
            "quantum_provider": "local_simulator",
            "storage": "local",
            "authentication": "basic"
        })
        
        self.update_tenant_config(tenant_id, config)
        
    def delete_tenant(self, tenant_id: str):
        """Delete a tenant configuration"""
        if tenant_id not in self.tenant_configs:
            raise ValueError(f"Tenant {tenant_id} not found")
            
        config_path = Path("config/tenants") / f"{tenant_id}.json"
        config_path.unlink(missing_ok=True)
        
        del self.tenant_configs[tenant_id]

# Default SaaS tiers
SAAS_TIERS = {
    "free": {
        "max_agents": 3,
        "max_qubits": 8,
        "max_circuit_depth": 50,
        "concurrent_computations": 2,
        "features": {
            "quantum_symbolic": True,
            "rag": False,
            "monitoring": True,
            "custom_circuits": False
        }
    },
    "pro": {
        "max_agents": 10,
        "max_qubits": 16,
        "max_circuit_depth": 100,
        "concurrent_computations": 5,
        "features": {
            "quantum_symbolic": True,
            "rag": True,
            "monitoring": True,
            "custom_circuits": True
        }
    },
    "enterprise": {
        "max_agents": -1,  # Unlimited
        "max_qubits": 32,
        "max_circuit_depth": 200,
        "concurrent_computations": 20,
        "features": {
            "quantum_symbolic": True,
            "rag": True,
            "monitoring": True,
            "custom_circuits": True,
            "custom_integrations": True,
            "priority_support": True
        }
    }
}
