"""
Metrics Module
-----------
Metrics collection and monitoring utilities for the agent builder system.
"""

from typing import Dict, List, Optional, Union, Any
import time
from datetime import datetime, timedelta
import statistics
import threading
from collections import deque
import json
from pathlib import Path

from .exceptions import MetricsError
from .logging import logger

class Metric:
    """Base metric class"""
    
    def __init__(self,
                 name: str,
                 description: Optional[str] = None,
                 tags: Optional[Dict[str, str]] = None):
        self.name = name
        self.description = description
        self.tags = tags or {}
        self.timestamp = datetime.utcnow()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to dictionary"""
        return {
            'name': self.name,
            'description': self.description,
            'tags': self.tags,
            'timestamp': self.timestamp.isoformat()
        }

class Counter(Metric):
    """Counter metric"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.value = 0
        self.lock = threading.Lock()
        
    def increment(self, value: int = 1) -> None:
        """Increment counter"""
        with self.lock:
            self.value += value
            
    def decrement(self, value: int = 1) -> None:
        """Decrement counter"""
        with self.lock:
            self.value -= value
            
    def reset(self) -> None:
        """Reset counter"""
        with self.lock:
            self.value = 0
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert counter to dictionary"""
        data = super().to_dict()
        data['value'] = self.value
        return data

class Gauge(Metric):
    """Gauge metric"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.value = 0.0
        self.lock = threading.Lock()
        
    def set(self, value: float) -> None:
        """Set gauge value"""
        with self.lock:
            self.value = float(value)
            
    def increment(self, value: float = 1.0) -> None:
        """Increment gauge"""
        with self.lock:
            self.value += value
            
    def decrement(self, value: float = 1.0) -> None:
        """Decrement gauge"""
        with self.lock:
            self.value -= value
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert gauge to dictionary"""
        data = super().to_dict()
        data['value'] = self.value
        return data

class Timer(Metric):
    """Timer metric"""
    
    def __init__(self,
                 *args,
                 max_samples: int = 1000,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.samples = deque(maxlen=max_samples)
        self.lock = threading.Lock()
        self._start_time: Optional[float] = None
        
    def start(self) -> None:
        """Start timer"""
        self._start_time = time.time()
        
    def stop(self) -> float:
        """
        Stop timer
        
        Returns:
            Elapsed time in seconds
        """
        if self._start_time is None:
            raise MetricsError("Timer not started")
            
        elapsed = time.time() - self._start_time
        self._start_time = None
        
        with self.lock:
            self.samples.append(elapsed)
            
        return elapsed
        
    def add(self, value: float) -> None:
        """Add sample directly"""
        with self.lock:
            self.samples.append(value)
            
    def clear(self) -> None:
        """Clear samples"""
        with self.lock:
            self.samples.clear()
            
    def get_statistics(self) -> Dict[str, float]:
        """Get timer statistics"""
        with self.lock:
            if not self.samples:
                return {
                    'count': 0,
                    'min': 0,
                    'max': 0,
                    'mean': 0,
                    'median': 0,
                    'stddev': 0
                }
                
            samples = list(self.samples)
            return {
                'count': len(samples),
                'min': min(samples),
                'max': max(samples),
                'mean': statistics.mean(samples),
                'median': statistics.median(samples),
                'stddev': statistics.stdev(samples) if len(samples) > 1 else 0
            }
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert timer to dictionary"""
        data = super().to_dict()
        data.update(self.get_statistics())
        return data

class Histogram(Metric):
    """Histogram metric"""
    
    def __init__(self,
                 *args,
                 buckets: Optional[List[float]] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.buckets = sorted(buckets) if buckets else [
            0.005, 0.01, 0.025, 0.05, 0.075, 0.1,
            0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0
        ]
        self.counts = {b: 0 for b in self.buckets}
        self.lock = threading.Lock()
        
    def add(self, value: float) -> None:
        """Add sample to histogram"""
        with self.lock:
            for bucket in self.buckets:
                if value <= bucket:
                    self.counts[bucket] += 1
                    break
                    
    def reset(self) -> None:
        """Reset histogram"""
        with self.lock:
            for bucket in self.counts:
                self.counts[bucket] = 0
                
    def to_dict(self) -> Dict[str, Any]:
        """Convert histogram to dictionary"""
        data = super().to_dict()
        data['buckets'] = self.buckets
        data['counts'] = self.counts
        return data

class MetricsRegistry:
    """Metrics registry"""
    
    def __init__(self):
        self.metrics: Dict[str, Metric] = {}
        self.lock = threading.Lock()
        
    def register(self, metric: Metric) -> None:
        """Register metric"""
        with self.lock:
            if metric.name in self.metrics:
                raise MetricsError(f"Metric already exists: {metric.name}")
            self.metrics[metric.name] = metric
            
    def unregister(self, name: str) -> None:
        """Unregister metric"""
        with self.lock:
            self.metrics.pop(name, None)
            
    def get(self, name: str) -> Metric:
        """Get metric by name"""
        with self.lock:
            if name not in self.metrics:
                raise MetricsError(f"Metric not found: {name}")
            return self.metrics[name]
            
    def get_all(self) -> Dict[str, Metric]:
        """Get all metrics"""
        with self.lock:
            return self.metrics.copy()
            
    def clear(self) -> None:
        """Clear all metrics"""
        with self.lock:
            self.metrics.clear()

class MetricsExporter:
    """Metrics exporter"""
    
    def __init__(self,
                 registry: MetricsRegistry,
                 export_dir: Optional[Union[str, Path]] = None):
        self.registry = registry
        self.export_dir = Path(export_dir) if export_dir else None
        
    def export_json(self,
                   file_path: Optional[Union[str, Path]] = None) -> str:
        """
        Export metrics as JSON
        
        Args:
            file_path: Optional file to write to
            
        Returns:
            JSON string
        """
        try:
            metrics = {
                name: metric.to_dict()
                for name, metric in self.registry.get_all().items()
            }
            
            json_str = json.dumps(metrics, indent=2)
            
            if file_path:
                path = Path(file_path)
                if not path.is_absolute() and self.export_dir:
                    path = self.export_dir / path
                    
                # Ensure directory exists
                path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(path, 'w') as f:
                    f.write(json_str)
                    
            return json_str
            
        except Exception as e:
            raise MetricsError(f"Failed to export metrics: {str(e)}")

# Global registry instance
registry = MetricsRegistry()

# Common metrics
request_count = Counter(
    "request_count",
    "Total number of requests",
    {"type": "http"}
)
registry.register(request_count)

request_latency = Timer(
    "request_latency",
    "Request latency in seconds",
    {"type": "http"}
)
registry.register(request_latency)

active_connections = Gauge(
    "active_connections",
    "Number of active connections",
    {"type": "websocket"}
)
registry.register(active_connections)

response_time = Histogram(
    "response_time",
    "Response time distribution",
    {"type": "http"}
)
registry.register(response_time)
