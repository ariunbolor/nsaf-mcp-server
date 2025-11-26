"""
Validation Module
-------------
Validation utilities for the agent builder system.
"""

from typing import Any, Dict, List, Optional, Union, Type, Callable
import re
import json
from pathlib import Path
from datetime import datetime
import uuid
import dataclasses

from .exceptions import ValidationError
from .logging import logger

class Validator:
    """Base validator class"""
    
    def validate(self, value: Any) -> None:
        """
        Validate value
        
        Args:
            value: Value to validate
            
        Raises:
            ValidationError: If validation fails
        """
        raise NotImplementedError

class TypeValidator(Validator):
    """Type validator"""
    
    def __init__(self, expected_type: Union[Type, tuple]):
        self.expected_type = expected_type
        
    def validate(self, value: Any) -> None:
        """Validate value type"""
        if not isinstance(value, self.expected_type):
            raise ValidationError(
                f"Expected type {self.expected_type}, got {type(value)}"
            )

class RangeValidator(Validator):
    """Range validator"""
    
    def __init__(self,
                 min_value: Optional[Union[int, float]] = None,
                 max_value: Optional[Union[int, float]] = None):
        self.min_value = min_value
        self.max_value = max_value
        
    def validate(self, value: Union[int, float]) -> None:
        """Validate value range"""
        if not isinstance(value, (int, float)):
            raise ValidationError(f"Expected number, got {type(value)}")
            
        if self.min_value is not None and value < self.min_value:
            raise ValidationError(
                f"Value {value} is less than minimum {self.min_value}"
            )
            
        if self.max_value is not None and value > self.max_value:
            raise ValidationError(
                f"Value {value} is greater than maximum {self.max_value}"
            )

class LengthValidator(Validator):
    """Length validator"""
    
    def __init__(self,
                 min_length: Optional[int] = None,
                 max_length: Optional[int] = None):
        self.min_length = min_length
        self.max_length = max_length
        
    def validate(self, value: Union[str, list, dict, set]) -> None:
        """Validate value length"""
        if not hasattr(value, '__len__'):
            raise ValidationError(
                f"Value of type {type(value)} has no length"
            )
            
        length = len(value)
        
        if self.min_length is not None and length < self.min_length:
            raise ValidationError(
                f"Length {length} is less than minimum {self.min_length}"
            )
            
        if self.max_length is not None and length > self.max_length:
            raise ValidationError(
                f"Length {length} is greater than maximum {self.max_length}"
            )

class RegexValidator(Validator):
    """Regex validator"""
    
    def __init__(self, pattern: str):
        self.pattern = pattern
        self.regex = re.compile(pattern)
        
    def validate(self, value: str) -> None:
        """Validate value against regex"""
        if not isinstance(value, str):
            raise ValidationError(f"Expected string, got {type(value)}")
            
        if not self.regex.match(value):
            raise ValidationError(
                f"Value '{value}' does not match pattern '{self.pattern}'"
            )

class EnumValidator(Validator):
    """Enum validator"""
    
    def __init__(self, allowed_values: List[Any]):
        self.allowed_values = set(allowed_values)
        
    def validate(self, value: Any) -> None:
        """Validate value is in allowed set"""
        if value not in self.allowed_values:
            raise ValidationError(
                f"Value {value} not in allowed values: {self.allowed_values}"
            )

class SchemaValidator(Validator):
    """Schema validator"""
    
    def __init__(self, schema: Dict[str, Dict[str, Any]]):
        self.schema = schema
        self.validators: Dict[str, List[Validator]] = {}
        
        # Create validators for each field
        for field, rules in schema.items():
            field_validators = []
            
            # Type validation
            if 'type' in rules:
                field_validators.append(TypeValidator(rules['type']))
                
            # Range validation
            if 'min_value' in rules or 'max_value' in rules:
                field_validators.append(RangeValidator(
                    rules.get('min_value'),
                    rules.get('max_value')
                ))
                
            # Length validation
            if 'min_length' in rules or 'max_length' in rules:
                field_validators.append(LengthValidator(
                    rules.get('min_length'),
                    rules.get('max_length')
                ))
                
            # Regex validation
            if 'pattern' in rules:
                field_validators.append(RegexValidator(rules['pattern']))
                
            # Enum validation
            if 'allowed_values' in rules:
                field_validators.append(EnumValidator(rules['allowed_values']))
                
            # Custom validation
            if 'validator' in rules:
                if callable(rules['validator']):
                    field_validators.append(rules['validator'])
                    
            self.validators[field] = field_validators
            
    def validate(self, data: Dict[str, Any]) -> None:
        """Validate data against schema"""
        if not isinstance(data, dict):
            raise ValidationError(f"Expected dict, got {type(data)}")
            
        # Check required fields
        for field, rules in self.schema.items():
            if rules.get('required', False) and field not in data:
                raise ValidationError(f"Missing required field: {field}")
                
        # Validate each field
        for field, value in data.items():
            if field not in self.schema:
                if not self.schema.get('allow_extra_fields', False):
                    raise ValidationError(f"Unknown field: {field}")
                continue
                
            # Run all validators for field
            for validator in self.validators[field]:
                if isinstance(validator, Validator):
                    validator.validate(value)
                else:
                    # Custom validator function
                    try:
                        validator(value)
                    except Exception as e:
                        raise ValidationError(
                            f"Validation failed for field {field}: {str(e)}"
                        )

class DataclassValidator(Validator):
    """Dataclass validator"""
    
    def __init__(self, dataclass: Type):
        if not dataclasses.is_dataclass(dataclass):
            raise ValidationError(f"{dataclass} is not a dataclass")
            
        self.dataclass = dataclass
        
    def validate(self, data: Dict[str, Any]) -> None:
        """Validate data against dataclass"""
        try:
            # Attempt to create instance
            self.dataclass(**data)
        except Exception as e:
            raise ValidationError(
                f"Failed to validate against {self.dataclass.__name__}: {str(e)}"
            )

class JsonValidator(Validator):
    """JSON validator"""
    
    def validate(self, data: str) -> None:
        """Validate JSON string"""
        try:
            json.loads(data)
        except Exception as e:
            raise ValidationError(f"Invalid JSON: {str(e)}")

class PathValidator(Validator):
    """Path validator"""
    
    def __init__(self,
                 must_exist: bool = False,
                 file_only: bool = False,
                 dir_only: bool = False):
        self.must_exist = must_exist
        self.file_only = file_only
        self.dir_only = dir_only
        
    def validate(self, path: Union[str, Path]) -> None:
        """Validate path"""
        try:
            path = Path(path)
            
            if self.must_exist and not path.exists():
                raise ValidationError(f"Path does not exist: {path}")
                
            if self.file_only and path.exists() and not path.is_file():
                raise ValidationError(f"Path is not a file: {path}")
                
            if self.dir_only and path.exists() and not path.is_dir():
                raise ValidationError(f"Path is not a directory: {path}")
                
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ValidationError(f"Invalid path: {str(e)}")

class DateTimeValidator(Validator):
    """DateTime validator"""
    
    def __init__(self,
                 min_date: Optional[datetime] = None,
                 max_date: Optional[datetime] = None,
                 format: Optional[str] = None):
        self.min_date = min_date
        self.max_date = max_date
        self.format = format
        
    def validate(self, value: Union[str, datetime]) -> None:
        """Validate datetime"""
        try:
            if isinstance(value, str):
                if self.format:
                    dt = datetime.strptime(value, self.format)
                else:
                    dt = datetime.fromisoformat(value)
            elif isinstance(value, datetime):
                dt = value
            else:
                raise ValidationError(
                    f"Expected string or datetime, got {type(value)}"
                )
                
            if self.min_date and dt < self.min_date:
                raise ValidationError(
                    f"Date {dt} is before minimum {self.min_date}"
                )
                
            if self.max_date and dt > self.max_date:
                raise ValidationError(
                    f"Date {dt} is after maximum {self.max_date}"
                )
                
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ValidationError(f"Invalid datetime: {str(e)}")

class UuidValidator(Validator):
    """UUID validator"""
    
    def validate(self, value: Union[str, uuid.UUID]) -> None:
        """Validate UUID"""
        try:
            if isinstance(value, str):
                uuid.UUID(value)
            elif not isinstance(value, uuid.UUID):
                raise ValidationError(
                    f"Expected string or UUID, got {type(value)}"
                )
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ValidationError(f"Invalid UUID: {str(e)}")

class EmailValidator(Validator):
    """Email validator"""
    
    def __init__(self):
        self.regex = re.compile(r"""
            ^[a-zA-Z0-9._%+-]+               # Local part
            @                                # @ symbol
            [a-zA-Z0-9.-]+                  # Domain
            \.[a-zA-Z]{2,}$                 # TLD
        """, re.VERBOSE)
        
    def validate(self, value: str) -> None:
        """Validate email"""
        if not isinstance(value, str):
            raise ValidationError(f"Expected string, got {type(value)}")
            
        if not self.regex.match(value):
            raise ValidationError(f"Invalid email address: {value}")

class UrlValidator(Validator):
    """URL validator"""
    
    def __init__(self,
                 require_protocol: bool = True,
                 allowed_protocols: Optional[List[str]] = None):
        self.require_protocol = require_protocol
        self.allowed_protocols = set(allowed_protocols or ['http', 'https'])
        
        pattern = r"""
            ^
            (?:(?P<protocol>[a-z]+)://)?    # Protocol (optional)
            (?P<host>                        # Host
                [a-z0-9]                     # First character
                [a-z0-9.-]*                  # Remaining characters
                \.[a-z]{2,}                  # TLD
            )
            (?::\d+)?                        # Port (optional)
            (?:/[^?#]*)?                     # Path (optional)
            (?:\?[^#]*)?                     # Query (optional)
            (?:#.*)?                         # Fragment (optional)
            $
        """
        self.regex = re.compile(pattern, re.VERBOSE | re.IGNORECASE)
        
    def validate(self, value: str) -> None:
        """Validate URL"""
        if not isinstance(value, str):
            raise ValidationError(f"Expected string, got {type(value)}")
            
        match = self.regex.match(value)
        if not match:
            raise ValidationError(f"Invalid URL: {value}")
            
        protocol = match.group('protocol')
        if self.require_protocol and not protocol:
            raise ValidationError("URL must include protocol")
            
        if protocol and protocol.lower() not in self.allowed_protocols:
            raise ValidationError(
                f"Protocol {protocol} not in allowed protocols: "
                f"{self.allowed_protocols}"
            )

# Common validators
type_validators = {
    'str': TypeValidator(str),
    'int': TypeValidator(int),
    'float': TypeValidator((int, float)),
    'bool': TypeValidator(bool),
    'list': TypeValidator(list),
    'dict': TypeValidator(dict),
    'set': TypeValidator(set)
}

json_validator = JsonValidator()
path_validator = PathValidator()
email_validator = EmailValidator()
url_validator = UrlValidator()
uuid_validator = UuidValidator()
