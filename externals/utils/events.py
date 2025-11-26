"""
Events Module
----------
Event handling utilities for the agent builder system.
"""

from typing import Any, Dict, List, Optional, Set, Callable, Union
import threading
import asyncio
import inspect
from datetime import datetime
import uuid
import json
from pathlib import Path

from .exceptions import EventError
from .logging import logger

class Event:
    """Base event class"""
    
    def __init__(self,
                 event_type: str,
                 data: Optional[Dict[str, Any]] = None,
                 source: Optional[str] = None):
        self.id = str(uuid.uuid4())
        self.type = event_type
        self.data = data or {}
        self.source = source
        self.timestamp = datetime.utcnow()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary"""
        return {
            'id': self.id,
            'type': self.type,
            'data': self.data,
            'source': self.source,
            'timestamp': self.timestamp.isoformat()
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """Create event from dictionary"""
        event = cls(
            event_type=data['type'],
            data=data['data'],
            source=data['source']
        )
        event.id = data['id']
        event.timestamp = datetime.fromisoformat(data['timestamp'])
        return event

class EventHandler:
    """Event handler base class"""
    
    def __init__(self):
        self.async_mode = inspect.iscoroutinefunction(self.handle)
        
    def handle(self, event: Event) -> None:
        """Handle event"""
        raise NotImplementedError

class EventFilter:
    """Event filter"""
    
    def __init__(self,
                 event_types: Optional[Set[str]] = None,
                 sources: Optional[Set[str]] = None,
                 data_filter: Optional[Dict[str, Any]] = None):
        self.event_types = event_types
        self.sources = sources
        self.data_filter = data_filter
        
    def matches(self, event: Event) -> bool:
        """Check if event matches filter"""
        if self.event_types and event.type not in self.event_types:
            return False
            
        if self.sources and event.source not in self.sources:
            return False
            
        if self.data_filter:
            for key, value in self.data_filter.items():
                if key not in event.data or event.data[key] != value:
                    return False
                    
        return True

class EventBus:
    """Event bus for pub/sub pattern"""
    
    def __init__(self):
        self.handlers: Dict[str, List[EventHandler]] = {}
        self.filters: Dict[str, EventFilter] = {}
        self.lock = threading.Lock()
        self.async_loop: Optional[asyncio.AbstractEventLoop] = None
        
    def subscribe(self,
                 handler: EventHandler,
                 event_filter: Optional[EventFilter] = None) -> str:
        """
        Subscribe handler to events
        
        Args:
            handler: Event handler
            event_filter: Optional event filter
            
        Returns:
            Subscription ID
        """
        subscription_id = str(uuid.uuid4())
        
        with self.lock:
            self.handlers[subscription_id] = handler
            if event_filter:
                self.filters[subscription_id] = event_filter
                
        return subscription_id
        
    def unsubscribe(self, subscription_id: str) -> None:
        """Unsubscribe handler"""
        with self.lock:
            self.handlers.pop(subscription_id, None)
            self.filters.pop(subscription_id, None)
            
    def publish(self, event: Event) -> None:
        """
        Publish event
        
        Args:
            event: Event to publish
        """
        handlers_to_call = []
        
        with self.lock:
            for sub_id, handler in self.handlers.items():
                event_filter = self.filters.get(sub_id)
                if not event_filter or event_filter.matches(event):
                    handlers_to_call.append(handler)
                    
        for handler in handlers_to_call:
            try:
                if handler.async_mode:
                    if not self.async_loop:
                        self.async_loop = asyncio.new_event_loop()
                    self.async_loop.run_until_complete(handler.handle(event))
                else:
                    handler.handle(event)
            except Exception as e:
                logger.error(f"Error handling event: {str(e)}")

class EventLogger(EventHandler):
    """Event logger"""
    
    def __init__(self,
                 log_file: Optional[Union[str, Path]] = None,
                 console: bool = True):
        super().__init__()
        self.log_file = Path(log_file) if log_file else None
        self.console = console
        
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            
    def handle(self, event: Event) -> None:
        """Log event"""
        event_data = event.to_dict()
        log_entry = json.dumps(event_data)
        
        if self.console:
            print(f"Event: {log_entry}")
            
        if self.log_file:
            try:
                with open(self.log_file, 'a') as f:
                    f.write(f"{log_entry}\n")
            except Exception as e:
                logger.error(f"Failed to write event log: {str(e)}")

class AsyncEventHandler(EventHandler):
    """Async event handler base class"""
    
    async def handle(self, event: Event) -> None:
        """Handle event"""
        raise NotImplementedError

class EventBuffer:
    """Event buffer for storing and replaying events"""
    
    def __init__(self,
                 max_size: Optional[int] = None,
                 persist_file: Optional[Union[str, Path]] = None):
        self.events: List[Event] = []
        self.max_size = max_size
        self.persist_file = Path(persist_file) if persist_file else None
        self.lock = threading.Lock()
        
        if self.persist_file and self.persist_file.exists():
            self._load_events()
            
    def _load_events(self) -> None:
        """Load events from file"""
        try:
            with open(self.persist_file) as f:
                for line in f:
                    event_data = json.loads(line)
                    self.events.append(Event.from_dict(event_data))
        except Exception as e:
            logger.error(f"Failed to load events: {str(e)}")
            
    def _save_events(self) -> None:
        """Save events to file"""
        if not self.persist_file:
            return
            
        try:
            with open(self.persist_file, 'w') as f:
                for event in self.events:
                    f.write(f"{json.dumps(event.to_dict())}\n")
        except Exception as e:
            logger.error(f"Failed to save events: {str(e)}")
            
    def add(self, event: Event) -> None:
        """Add event to buffer"""
        with self.lock:
            self.events.append(event)
            
            if self.max_size and len(self.events) > self.max_size:
                self.events = self.events[-self.max_size:]
                
            if self.persist_file:
                self._save_events()
                
    def get_events(self,
                   event_filter: Optional[EventFilter] = None) -> List[Event]:
        """Get events matching filter"""
        with self.lock:
            if not event_filter:
                return self.events.copy()
                
            return [
                event for event in self.events
                if event_filter.matches(event)
            ]
            
    def clear(self) -> None:
        """Clear event buffer"""
        with self.lock:
            self.events.clear()
            if self.persist_file:
                self._save_events()

# Global event bus instance
event_bus = EventBus()

# Common event types
EVENT_SYSTEM = "system"
EVENT_USER = "user"
EVENT_AGENT = "agent"
EVENT_ERROR = "error"
EVENT_TASK = "task"
EVENT_METRIC = "metric"
