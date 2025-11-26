"""
Helper Utilities Module
--------------------
Common utility functions used across the agent builder system.
"""

import re
import json
import uuid
from typing import Any, Dict, List, Optional, Union, TypeVar, Type
from datetime import datetime, timezone
import hashlib
from pathlib import Path

from ..utils.logging import logger

T = TypeVar('T')

def generate_id(prefix: str = "") -> str:
    """
    Generate a unique identifier
    
    Args:
        prefix: Optional prefix for the ID
        
    Returns:
        Unique identifier string
    """
    unique_id = str(uuid.uuid4())
    return f"{prefix}{unique_id}" if prefix else unique_id

def validate_required_fields(data: Dict[str, Any], required_fields: List[str]) -> List[str]:
    """
    Validate that required fields are present in data
    
    Args:
        data: Dictionary containing data to validate
        required_fields: List of required field names
        
    Returns:
        List of missing field names
    """
    missing = []
    for field in required_fields:
        if field not in data or data[field] is None:
            missing.append(field)
    return missing

def safe_cast(value: Any, target_type: Type[T], default: Optional[T] = None) -> Optional[T]:
    """
    Safely cast a value to a target type
    
    Args:
        value: Value to cast
        target_type: Type to cast to
        default: Default value if cast fails
        
    Returns:
        Cast value or default
    """
    try:
        return target_type(value)
    except (ValueError, TypeError):
        return default

def sanitize_string(text: str, max_length: Optional[int] = None) -> str:
    """
    Sanitize a string by removing special characters and limiting length
    
    Args:
        text: String to sanitize
        max_length: Optional maximum length
        
    Returns:
        Sanitized string
    """
    # Remove special characters
    sanitized = re.sub(r'[^\w\s-]', '', text)
    
    # Convert to lowercase and replace spaces with hyphens
    sanitized = sanitized.lower().strip().replace(' ', '-')
    
    # Limit length if specified
    if max_length:
        sanitized = sanitized[:max_length]
        
    return sanitized

def merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any], overwrite: bool = True) -> Dict[str, Any]:
    """
    Deep merge two dictionaries
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary (takes precedence if overwrite=True)
        overwrite: Whether to overwrite existing values
        
    Returns:
        Merged dictionary
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value, overwrite)
        elif key not in result or overwrite:
            result[key] = value
            
    return result

def format_timestamp(dt: Optional[datetime] = None, format: str = "%Y-%m-%dT%H:%M:%SZ") -> str:
    """
    Format a timestamp in ISO format
    
    Args:
        dt: Datetime object (uses current time if None)
        format: Datetime format string
        
    Returns:
        Formatted timestamp string
    """
    if dt is None:
        dt = datetime.now(timezone.utc)
    return dt.strftime(format)

def calculate_hash(data: Union[str, bytes, Dict[str, Any]]) -> str:
    """
    Calculate SHA-256 hash of data
    
    Args:
        data: Data to hash (string, bytes, or dictionary)
        
    Returns:
        Hash string
    """
    if isinstance(data, dict):
        data = json.dumps(data, sort_keys=True)
    if isinstance(data, str):
        data = data.encode()
        
    return hashlib.sha256(data).hexdigest()

def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary
    
    Args:
        path: Directory path
        
    Returns:
        Path object for the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def load_json_file(path: Union[str, Path], default: Any = None) -> Any:
    """
    Load JSON from a file with error handling
    
    Args:
        path: Path to JSON file
        default: Default value if file doesn't exist or is invalid
        
    Returns:
        Loaded JSON data or default value
    """
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"Failed to load JSON file {path}: {str(e)}")
        return default

def save_json_file(data: Any, path: Union[str, Path], indent: int = 2) -> bool:
    """
    Save data to a JSON file with error handling
    
    Args:
        data: Data to save
        path: Path to save to
        indent: JSON indentation
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(path, 'w') as f:
            json.dump(data, f, indent=indent)
        return True
    except Exception as e:
        logger.error(f"Failed to save JSON file {path}: {str(e)}")
        return False

def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split a list into chunks of specified size
    
    Args:
        lst: List to chunk
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """
    Flatten a nested dictionary
    
    Args:
        d: Dictionary to flatten
        parent_key: Parent key for nested items
        sep: Separator for keys
        
    Returns:
        Flattened dictionary
    """
    items: List[tuple] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def unflatten_dict(d: Dict[str, Any], sep: str = '.') -> Dict[str, Any]:
    """
    Unflatten a dictionary with dot notation keys
    
    Args:
        d: Flattened dictionary
        sep: Separator used in keys
        
    Returns:
        Nested dictionary
    """
    result: Dict[str, Any] = {}
    
    for key, value in d.items():
        parts = key.split(sep)
        target = result
        
        for part in parts[:-1]:
            target = target.setdefault(part, {})
            
        target[parts[-1]] = value
        
    return result

def retry_operation(func: callable,
                   max_attempts: int = 3,
                   delay: float = 1.0,
                   backoff_factor: float = 2.0,
                   exceptions: tuple = (Exception,)) -> Any:
    """
    Retry an operation with exponential backoff
    
    Args:
        func: Function to retry
        max_attempts: Maximum number of attempts
        delay: Initial delay between attempts
        backoff_factor: Factor to increase delay by
        exceptions: Tuple of exceptions to catch
        
    Returns:
        Result of the function
        
    Raises:
        Last caught exception if all attempts fail
    """
    import time
    
    last_exception = None
    current_delay = delay
    
    for attempt in range(max_attempts):
        try:
            return func()
        except exceptions as e:
            last_exception = e
            if attempt < max_attempts - 1:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {current_delay}s...")
                time.sleep(current_delay)
                current_delay *= backoff_factor
            else:
                logger.error(f"All {max_attempts} attempts failed")
                
    if last_exception:
        raise last_exception

def parse_duration(duration_str: str) -> int:
    """
    Parse a duration string into seconds
    
    Args:
        duration_str: Duration string (e.g., "1h30m", "45s", "2d")
        
    Returns:
        Duration in seconds
        
    Raises:
        ValueError: If duration string is invalid
    """
    units = {
        's': 1,
        'm': 60,
        'h': 3600,
        'd': 86400,
        'w': 604800
    }
    
    pattern = re.compile(r'(\d+)([smhdw])')
    matches = pattern.findall(duration_str.lower())
    
    if not matches:
        raise ValueError(f"Invalid duration string: {duration_str}")
        
    total_seconds = 0
    for value, unit in matches:
        total_seconds += int(value) * units[unit]
        
    return total_seconds

def format_duration(seconds: int) -> str:
    """
    Format a duration in seconds to a human-readable string
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    units = [
        ('w', 604800),
        ('d', 86400),
        ('h', 3600),
        ('m', 60),
        ('s', 1)
    ]
    
    if seconds == 0:
        return "0s"
        
    parts = []
    for unit, unit_seconds in units:
        if seconds >= unit_seconds:
            value = seconds // unit_seconds
            seconds %= unit_seconds
            parts.append(f"{value}{unit}")
            
    return " ".join(parts)
