"""
Serialization Module
-----------------
Serialization utilities for the agent builder system.
"""

from typing import Any, Dict, List, Optional, Union, Type
import json
import yaml
import pickle
import base64
from pathlib import Path
from datetime import datetime, date
import uuid
import dataclasses
import enum

from .exceptions import SerializationError
from .logging import logger

class JsonEncoder(json.JSONEncoder):
    """Custom JSON encoder for complex types"""
    
    def default(self, obj: Any) -> Any:
        # Handle datetime
        if isinstance(obj, (datetime, date)):
            return {
                '_type': 'datetime',
                'value': obj.isoformat()
            }
            
        # Handle UUID
        if isinstance(obj, uuid.UUID):
            return {
                '_type': 'uuid',
                'value': str(obj)
            }
            
        # Handle bytes
        if isinstance(obj, bytes):
            return {
                '_type': 'bytes',
                'value': base64.b64encode(obj).decode()
            }
            
        # Handle sets
        if isinstance(obj, set):
            return {
                '_type': 'set',
                'value': list(obj)
            }
            
        # Handle Path
        if isinstance(obj, Path):
            return {
                '_type': 'path',
                'value': str(obj)
            }
            
        # Handle dataclasses
        if dataclasses.is_dataclass(obj):
            return {
                '_type': 'dataclass',
                'class': obj.__class__.__name__,
                'module': obj.__class__.__module__,
                'value': dataclasses.asdict(obj)
            }
            
        # Handle enums
        if isinstance(obj, enum.Enum):
            return {
                '_type': 'enum',
                'class': obj.__class__.__name__,
                'module': obj.__class__.__module__,
                'value': obj.name
            }
            
        return super().default(obj)

class JsonDecoder(json.JSONDecoder):
    """Custom JSON decoder for complex types"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.object_hook, *args, **kwargs)
        
    def object_hook(self, obj: Dict[str, Any]) -> Any:
        if '_type' not in obj:
            return obj
            
        try:
            type_name = obj['_type']
            
            if type_name == 'datetime':
                return datetime.fromisoformat(obj['value'])
                
            if type_name == 'uuid':
                return uuid.UUID(obj['value'])
                
            if type_name == 'bytes':
                return base64.b64decode(obj['value'])
                
            if type_name == 'set':
                return set(obj['value'])
                
            if type_name == 'path':
                return Path(obj['value'])
                
            if type_name == 'dataclass':
                # Import class dynamically
                module = __import__(obj['module'], fromlist=[obj['class']])
                cls = getattr(module, obj['class'])
                return cls(**obj['value'])
                
            if type_name == 'enum':
                # Import enum dynamically
                module = __import__(obj['module'], fromlist=[obj['class']])
                cls = getattr(module, obj['class'])
                return cls[obj['value']]
                
            return obj
            
        except Exception as e:
            logger.error(f"Failed to decode object: {str(e)}")
            return obj

class JsonSerializer:
    """JSON serialization utility"""
    
    @staticmethod
    def serialize(obj: Any,
                 pretty: bool = False) -> str:
        """
        Serialize object to JSON
        
        Args:
            obj: Object to serialize
            pretty: Whether to format output
            
        Returns:
            JSON string
        """
        try:
            return json.dumps(
                obj,
                cls=JsonEncoder,
                indent=2 if pretty else None
            )
        except Exception as e:
            raise SerializationError(f"JSON serialization failed: {str(e)}")
            
    @staticmethod
    def deserialize(data: str) -> Any:
        """
        Deserialize JSON string
        
        Args:
            data: JSON string
            
        Returns:
            Deserialized object
        """
        try:
            return json.loads(data, cls=JsonDecoder)
        except Exception as e:
            raise SerializationError(f"JSON deserialization failed: {str(e)}")

class YamlSerializer:
    """YAML serialization utility"""
    
    @staticmethod
    def serialize(obj: Any) -> str:
        """
        Serialize object to YAML
        
        Args:
            obj: Object to serialize
            
        Returns:
            YAML string
        """
        try:
            return yaml.safe_dump(obj, default_flow_style=False)
        except Exception as e:
            raise SerializationError(f"YAML serialization failed: {str(e)}")
            
    @staticmethod
    def deserialize(data: str) -> Any:
        """
        Deserialize YAML string
        
        Args:
            data: YAML string
            
        Returns:
            Deserialized object
        """
        try:
            return yaml.safe_load(data)
        except Exception as e:
            raise SerializationError(f"YAML deserialization failed: {str(e)}")

class PickleSerializer:
    """Pickle serialization utility"""
    
    @staticmethod
    def serialize(obj: Any) -> bytes:
        """
        Serialize object to bytes
        
        Args:
            obj: Object to serialize
            
        Returns:
            Serialized bytes
        """
        try:
            return pickle.dumps(obj)
        except Exception as e:
            raise SerializationError(f"Pickle serialization failed: {str(e)}")
            
    @staticmethod
    def deserialize(data: bytes) -> Any:
        """
        Deserialize bytes
        
        Args:
            data: Serialized bytes
            
        Returns:
            Deserialized object
        """
        try:
            return pickle.loads(data)
        except Exception as e:
            raise SerializationError(f"Pickle deserialization failed: {str(e)}")

class SerializedFile:
    """Utility for working with serialized files"""
    
    def __init__(self,
                 path: Union[str, Path],
                 format: str = "json"):
        self.path = Path(path)
        self.format = format.lower()
        
        if self.format not in ["json", "yaml", "pickle"]:
            raise SerializationError(f"Unsupported format: {format}")
            
    def write(self,
             obj: Any,
             pretty: bool = False) -> None:
        """Write serialized object to file"""
        try:
            # Ensure directory exists
            self.path.parent.mkdir(parents=True, exist_ok=True)
            
            if self.format == "json":
                with open(self.path, 'w') as f:
                    f.write(JsonSerializer.serialize(obj, pretty))
                    
            elif self.format == "yaml":
                with open(self.path, 'w') as f:
                    f.write(YamlSerializer.serialize(obj))
                    
            else:  # pickle
                with open(self.path, 'wb') as f:
                    f.write(PickleSerializer.serialize(obj))
                    
        except Exception as e:
            raise SerializationError(
                f"Failed to write serialized file: {str(e)}"
            )
            
    def read(self) -> Any:
        """Read serialized object from file"""
        try:
            if not self.path.exists():
                raise SerializationError(f"File not found: {self.path}")
                
            if self.format == "json":
                with open(self.path) as f:
                    return JsonSerializer.deserialize(f.read())
                    
            elif self.format == "yaml":
                with open(self.path) as f:
                    return YamlSerializer.deserialize(f.read())
                    
            else:  # pickle
                with open(self.path, 'rb') as f:
                    return PickleSerializer.deserialize(f.read())
                    
        except Exception as e:
            raise SerializationError(
                f"Failed to read serialized file: {str(e)}"
            )
