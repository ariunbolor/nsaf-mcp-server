"""
Caching Module
------------
Caching utilities for the agent builder system.
"""

from typing import Any, Dict, Optional, Union, Callable, TypeVar, Generic
import time
import json
from pathlib import Path
from functools import wraps
import threading
from datetime import datetime, timedelta
import pickle
import hashlib

from .exceptions import CacheError
from .logging import logger

T = TypeVar('T')

class CacheEntry(Generic[T]):
    """Cache entry with metadata"""
    
    def __init__(self,
                 value: T,
                 ttl: Optional[int] = None):
        self.value = value
        self.timestamp = time.time()
        self.ttl = ttl
        
    def is_expired(self) -> bool:
        """Check if entry is expired"""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl

class MemoryCache:
    """In-memory cache with TTL support"""
    
    def __init__(self,
                 default_ttl: Optional[int] = None,
                 max_size: Optional[int] = None):
        self.cache: Dict[str, CacheEntry] = {}
        self.default_ttl = default_ttl
        self.max_size = max_size
        self.lock = threading.Lock()
        
    def get(self,
            key: str,
            default: Any = None) -> Any:
        """
        Get value from cache
        
        Args:
            key: Cache key
            default: Default value if not found
            
        Returns:
            Cached value or default
        """
        with self.lock:
            if key not in self.cache:
                return default
                
            entry = self.cache[key]
            if entry.is_expired():
                del self.cache[key]
                return default
                
            return entry.value
            
    def set(self,
            key: str,
            value: Any,
            ttl: Optional[int] = None) -> None:
        """
        Set cache value
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
        """
        with self.lock:
            # Check size limit
            if self.max_size and len(self.cache) >= self.max_size:
                self._evict_oldest()
                
            self.cache[key] = CacheEntry(
                value,
                ttl or self.default_ttl
            )
            
    def delete(self, key: str) -> None:
        """Delete cache entry"""
        with self.lock:
            self.cache.pop(key, None)
            
    def clear(self) -> None:
        """Clear all cache entries"""
        with self.lock:
            self.cache.clear()
            
    def _evict_oldest(self) -> None:
        """Evict oldest cache entry"""
        oldest_key = min(
            self.cache.keys(),
            key=lambda k: self.cache[k].timestamp
        )
        del self.cache[oldest_key]

class FileCache:
    """File-based cache with TTL support"""
    
    def __init__(self,
                 cache_dir: Union[str, Path],
                 default_ttl: Optional[int] = None):
        self.cache_dir = Path(cache_dir)
        self.default_ttl = default_ttl
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_path(self, key: str) -> Path:
        """Get cache file path"""
        # Hash key to create safe filename
        filename = hashlib.sha256(key.encode()).hexdigest()
        return self.cache_dir / filename
        
    def get(self,
            key: str,
            default: Any = None) -> Any:
        """
        Get value from cache
        
        Args:
            key: Cache key
            default: Default value if not found
            
        Returns:
            Cached value or default
        """
        try:
            path = self._get_path(key)
            if not path.exists():
                return default
                
            with open(path, 'rb') as f:
                entry = pickle.load(f)
                
            if entry.is_expired():
                path.unlink()
                return default
                
            return entry.value
            
        except Exception as e:
            logger.error(f"Failed to read cache file: {str(e)}")
            return default
            
    def set(self,
            key: str,
            value: Any,
            ttl: Optional[int] = None) -> None:
        """
        Set cache value
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
        """
        try:
            path = self._get_path(key)
            entry = CacheEntry(value, ttl or self.default_ttl)
            
            with open(path, 'wb') as f:
                pickle.dump(entry, f)
                
        except Exception as e:
            logger.error(f"Failed to write cache file: {str(e)}")
            
    def delete(self, key: str) -> None:
        """Delete cache entry"""
        try:
            path = self._get_path(key)
            if path.exists():
                path.unlink()
        except Exception as e:
            logger.error(f"Failed to delete cache file: {str(e)}")
            
    def clear(self) -> None:
        """Clear all cache entries"""
        try:
            for path in self.cache_dir.glob('*'):
                path.unlink()
        except Exception as e:
            logger.error(f"Failed to clear cache directory: {str(e)}")

def cache_result(
    cache: Union[MemoryCache, FileCache],
    key_prefix: str = "",
    ttl: Optional[int] = None
):
    """
    Decorator to cache function results
    
    Args:
        cache: Cache instance to use
        key_prefix: Prefix for cache keys
        ttl: Time to live in seconds
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            key_parts = [key_prefix, func.__name__]
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(
                f"{k}={v}"
                for k, v in sorted(kwargs.items())
            )
            key = ":".join(key_parts)
            
            # Check cache
            result = cache.get(key)
            if result is not None:
                return result
                
            # Call function
            result = func(*args, **kwargs)
            
            # Cache result
            cache.set(key, result, ttl)
            return result
            
        return wrapper
    return decorator

def async_cache_result(
    cache: Union[MemoryCache, FileCache],
    key_prefix: str = "",
    ttl: Optional[int] = None
):
    """
    Decorator to cache async function results
    
    Args:
        cache: Cache instance to use
        key_prefix: Prefix for cache keys
        ttl: Time to live in seconds
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            key_parts = [key_prefix, func.__name__]
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(
                f"{k}={v}"
                for k, v in sorted(kwargs.items())
            )
            key = ":".join(key_parts)
            
            # Check cache
            result = cache.get(key)
            if result is not None:
                return result
                
            # Call function
            result = await func(*args, **kwargs)
            
            # Cache result
            cache.set(key, result, ttl)
            return result
            
        return wrapper
    return decorator

# Global cache instances
memory_cache = MemoryCache(
    default_ttl=3600,  # 1 hour
    max_size=1000
)

file_cache = FileCache(
    cache_dir=Path("cache"),
    default_ttl=86400  # 24 hours
)
