"""
Rate Limiting Module
-----------------
Rate limiting utilities for the agent builder system.
"""

from typing import Dict, Optional, List, Tuple
import time
import threading
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path

from .exceptions import RateLimitError
from .logging import logger

@dataclass
class RateLimit:
    """Rate limit configuration"""
    limit: int
    window: int  # seconds
    description: Optional[str] = None

class TokenBucket:
    """Token bucket rate limiter"""
    
    def __init__(self,
                 capacity: int,
                 refill_rate: float,
                 initial_tokens: Optional[int] = None):
        self.capacity = capacity
        self.refill_rate = refill_rate  # tokens per second
        self.tokens = initial_tokens or capacity
        self.last_refill = time.time()
        self.lock = threading.Lock()
        
    def _refill(self) -> None:
        """Refill tokens based on elapsed time"""
        now = time.time()
        elapsed = now - self.last_refill
        new_tokens = elapsed * self.refill_rate
        
        self.tokens = min(
            self.capacity,
            self.tokens + new_tokens
        )
        self.last_refill = now
        
    def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens
        
        Args:
            tokens: Number of tokens to consume
            
        Returns:
            True if tokens were consumed
        """
        with self.lock:
            self._refill()
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
                
            return False

class SlidingWindow:
    """Sliding window rate limiter"""
    
    def __init__(self,
                 limit: int,
                 window: int):
        self.limit = limit
        self.window = window  # seconds
        self.requests: List[float] = []
        self.lock = threading.Lock()
        
    def _cleanup(self) -> None:
        """Remove expired timestamps"""
        now = time.time()
        cutoff = now - self.window
        
        self.requests = [
            ts for ts in self.requests
            if ts > cutoff
        ]
        
    def check(self) -> bool:
        """
        Check if request is allowed
        
        Returns:
            True if request is allowed
        """
        with self.lock:
            self._cleanup()
            
            if len(self.requests) >= self.limit:
                return False
                
            self.requests.append(time.time())
            return True

class FixedWindow:
    """Fixed window rate limiter"""
    
    def __init__(self,
                 limit: int,
                 window: int):
        self.limit = limit
        self.window = window  # seconds
        self.count = 0
        self.window_start = time.time()
        self.lock = threading.Lock()
        
    def _check_window(self) -> None:
        """Check if window has expired"""
        now = time.time()
        if now - self.window_start >= self.window:
            self.count = 0
            self.window_start = now
            
    def check(self) -> bool:
        """
        Check if request is allowed
        
        Returns:
            True if request is allowed
        """
        with self.lock:
            self._check_window()
            
            if self.count >= self.limit:
                return False
                
            self.count += 1
            return True

class RateLimiter:
    """Rate limiter with multiple strategies"""
    
    def __init__(self,
                 limits: Dict[str, RateLimit],
                 strategy: str = "sliding_window"):
        self.limits = limits
        self.strategy = strategy
        self.limiters: Dict[str, Dict[str, object]] = {}
        
        # Create limiters for each key
        for key, limit in limits.items():
            if strategy == "token_bucket":
                self.limiters[key] = {
                    "limiter": TokenBucket(
                        capacity=limit.limit,
                        refill_rate=limit.limit / limit.window
                    )
                }
            elif strategy == "sliding_window":
                self.limiters[key] = {
                    "limiter": SlidingWindow(
                        limit=limit.limit,
                        window=limit.window
                    )
                }
            else:  # fixed_window
                self.limiters[key] = {
                    "limiter": FixedWindow(
                        limit=limit.limit,
                        window=limit.window
                    )
                }
                
    def check(self,
             key: str,
             identifier: Optional[str] = None) -> None:
        """
        Check rate limit
        
        Args:
            key: Rate limit key
            identifier: Optional identifier (e.g., user ID)
            
        Raises:
            RateLimitError: If rate limit exceeded
        """
        if key not in self.limiters:
            return
            
        limiter_key = f"{key}:{identifier}" if identifier else key
        
        if limiter_key not in self.limiters[key]:
            # Create new limiter for this identifier
            limit = self.limits[key]
            if self.strategy == "token_bucket":
                self.limiters[key][limiter_key] = TokenBucket(
                    capacity=limit.limit,
                    refill_rate=limit.limit / limit.window
                )
            elif self.strategy == "sliding_window":
                self.limiters[key][limiter_key] = SlidingWindow(
                    limit=limit.limit,
                    window=limit.window
                )
            else:  # fixed_window
                self.limiters[key][limiter_key] = FixedWindow(
                    limit=limit.limit,
                    window=limit.window
                )
                
        limiter = self.limiters[key][limiter_key]
        if not limiter.check():
            limit = self.limits[key]
            raise RateLimitError(
                limit=limit.limit,
                window=limit.window,
                details={
                    'key': key,
                    'identifier': identifier,
                    'description': limit.description
                }
            )

class RateLimitDecorator:
    """Decorator for rate limiting functions"""
    
    def __init__(self,
                 limiter: RateLimiter,
                 key: str,
                 identifier_arg: Optional[str] = None):
        self.limiter = limiter
        self.key = key
        self.identifier_arg = identifier_arg
        
    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get identifier from args/kwargs
            identifier = None
            if self.identifier_arg:
                if self.identifier_arg in kwargs:
                    identifier = str(kwargs[self.identifier_arg])
                else:
                    # Try to get from positional args using function signature
                    sig = inspect.signature(func)
                    params = list(sig.parameters.keys())
                    try:
                        idx = params.index(self.identifier_arg)
                        if idx < len(args):
                            identifier = str(args[idx])
                    except ValueError:
                        pass
                        
            # Check rate limit
            self.limiter.check(self.key, identifier)
            
            # Call function
            return func(*args, **kwargs)
            
        return wrapper

class AsyncRateLimitDecorator:
    """Decorator for rate limiting async functions"""
    
    def __init__(self,
                 limiter: RateLimiter,
                 key: str,
                 identifier_arg: Optional[str] = None):
        self.limiter = limiter
        self.key = key
        self.identifier_arg = identifier_arg
        
    def __call__(self, func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get identifier from args/kwargs
            identifier = None
            if self.identifier_arg:
                if self.identifier_arg in kwargs:
                    identifier = str(kwargs[self.identifier_arg])
                else:
                    # Try to get from positional args using function signature
                    sig = inspect.signature(func)
                    params = list(sig.parameters.keys())
                    try:
                        idx = params.index(self.identifier_arg)
                        if idx < len(args):
                            identifier = str(args[idx])
                    except ValueError:
                        pass
                        
            # Check rate limit
            self.limiter.check(self.key, identifier)
            
            # Call function
            return await func(*args, **kwargs)
            
        return wrapper

# Default rate limits
DEFAULT_LIMITS = {
    'api': RateLimit(
        limit=100,
        window=60,
        description="API requests"
    ),
    'auth': RateLimit(
        limit=5,
        window=60,
        description="Authentication attempts"
    ),
    'email': RateLimit(
        limit=10,
        window=3600,
        description="Email sending"
    )
}

# Global rate limiter instance
rate_limiter = RateLimiter(DEFAULT_LIMITS)

# Convenience decorators
def rate_limit(key: str,
              identifier_arg: Optional[str] = None):
    """Decorator for rate limiting functions"""
    return RateLimitDecorator(
        rate_limiter,
        key,
        identifier_arg
    )

def async_rate_limit(key: str,
                    identifier_arg: Optional[str] = None):
    """Decorator for rate limiting async functions"""
    return AsyncRateLimitDecorator(
        rate_limiter,
        key,
        identifier_arg
    )
