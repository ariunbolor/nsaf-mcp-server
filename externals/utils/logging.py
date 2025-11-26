"""
Logging Module
-----------
Logging utilities for the agent builder system.
"""

import logging
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, Union
import threading
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler

# Custom log levels
TRACE = 5
DEBUG = logging.DEBUG      # 10
INFO = logging.INFO       # 20
WARNING = logging.WARNING # 30
ERROR = logging.ERROR     # 40
CRITICAL = logging.CRITICAL # 50

class JsonFormatter(logging.Formatter):
    """JSON log formatter"""
    
    def __init__(self,
                 include_timestamp: bool = True,
                 include_logger: bool = True,
                 include_level: bool = True,
                 **kwargs):
        super().__init__()
        self.include_timestamp = include_timestamp
        self.include_logger = include_logger
        self.include_level = include_level
        self.extra_fields = kwargs
        
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        data = {
            'message': record.getMessage()
        }
        
        if self.include_timestamp:
            data['timestamp'] = datetime.fromtimestamp(
                record.created
            ).isoformat()
            
        if self.include_logger:
            data['logger'] = record.name
            
        if self.include_level:
            data['level'] = record.levelname
            
        if hasattr(record, 'data'):
            data['data'] = record.data
            
        if record.exc_info:
            data['exception'] = self.formatException(record.exc_info)
            
        if record.stack_info:
            data['stack_info'] = self.formatStack(record.stack_info)
            
        # Add extra fields
        data.update(self.extra_fields)
        
        # Add record attributes
        for key, value in record.__dict__.items():
            if key not in {
                'args', 'asctime', 'created', 'exc_info', 'exc_text',
                'filename', 'funcName', 'levelname', 'levelno', 'lineno',
                'module', 'msecs', 'msg', 'name', 'pathname', 'process',
                'processName', 'relativeCreated', 'stack_info', 'thread',
                'threadName', 'data'
            }:
                data[key] = value
                
        return json.dumps(data)

class StructuredLogger(logging.Logger):
    """Logger with structured logging support"""
    
    def __init__(self, name: str):
        super().__init__(name)
        
        # Add trace level
        logging.addLevelName(TRACE, 'TRACE')
        
    def trace(self,
             msg: str,
             *args,
             data: Optional[Dict[str, Any]] = None,
             **kwargs):
        """Log at trace level"""
        if data:
            kwargs['extra'] = {'data': data}
        self.log(TRACE, msg, *args, **kwargs)
        
    def debug(self,
             msg: str,
             *args,
             data: Optional[Dict[str, Any]] = None,
             **kwargs):
        """Log at debug level"""
        if data:
            kwargs['extra'] = {'data': data}
        self.log(DEBUG, msg, *args, **kwargs)
        
    def info(self,
            msg: str,
            *args,
            data: Optional[Dict[str, Any]] = None,
            **kwargs):
        """Log at info level"""
        if data:
            kwargs['extra'] = {'data': data}
        self.log(INFO, msg, *args, **kwargs)
        
    def warning(self,
               msg: str,
               *args,
               data: Optional[Dict[str, Any]] = None,
               **kwargs):
        """Log at warning level"""
        if data:
            kwargs['extra'] = {'data': data}
        self.log(WARNING, msg, *args, **kwargs)
        
    def error(self,
             msg: str,
             *args,
             data: Optional[Dict[str, Any]] = None,
             **kwargs):
        """Log at error level"""
        if data:
            kwargs['extra'] = {'data': data}
        self.log(ERROR, msg, *args, **kwargs)
        
    def critical(self,
                msg: str,
                *args,
                data: Optional[Dict[str, Any]] = None,
                **kwargs):
        """Log at critical level"""
        if data:
            kwargs['extra'] = {'data': data}
        self.log(CRITICAL, msg, *args, **kwargs)

class LogManager:
    """Log manager"""
    
    def __init__(self):
        self.loggers: Dict[str, StructuredLogger] = {}
        self.handlers: Dict[str, logging.Handler] = {}
        self.lock = threading.Lock()
        
        # Register logger class
        logging.setLoggerClass(StructuredLogger)
        
    def get_logger(self,
                  name: str,
                  level: Optional[int] = None) -> StructuredLogger:
        """
        Get or create logger
        
        Args:
            name: Logger name
            level: Optional log level
            
        Returns:
            Logger instance
        """
        with self.lock:
            if name not in self.loggers:
                logger = logging.getLogger(name)
                if level:
                    logger.setLevel(level)
                self.loggers[name] = logger
            return self.loggers[name]
            
    def add_console_handler(self,
                          name: str,
                          level: Optional[int] = None,
                          formatter: Optional[logging.Formatter] = None) -> None:
        """Add console handler"""
        handler = logging.StreamHandler(sys.stdout)
        if level:
            handler.setLevel(level)
        if formatter:
            handler.setFormatter(formatter)
            
        self.add_handler(name, handler)
        
    def add_file_handler(self,
                        name: str,
                        filename: Union[str, Path],
                        level: Optional[int] = None,
                        formatter: Optional[logging.Formatter] = None,
                        max_bytes: Optional[int] = None,
                        backup_count: Optional[int] = None,
                        when: Optional[str] = None) -> None:
        """Add file handler"""
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if max_bytes:
            handler = RotatingFileHandler(
                path,
                maxBytes=max_bytes,
                backupCount=backup_count or 5
            )
        elif when:
            handler = TimedRotatingFileHandler(
                path,
                when=when,
                backupCount=backup_count or 5
            )
        else:
            handler = logging.FileHandler(path)
            
        if level:
            handler.setLevel(level)
        if formatter:
            handler.setFormatter(formatter)
            
        self.add_handler(name, handler)
        
    def add_handler(self,
                   name: str,
                   handler: logging.Handler) -> None:
        """Add handler to logger"""
        with self.lock:
            logger = self.get_logger(name)
            logger.addHandler(handler)
            self.handlers[f"{name}_{handler.__class__.__name__}"] = handler
            
    def remove_handler(self,
                      name: str,
                      handler_id: str) -> None:
        """Remove handler from logger"""
        with self.lock:
            if handler_id in self.handlers:
                logger = self.get_logger(name)
                logger.removeHandler(self.handlers[handler_id])
                del self.handlers[handler_id]
                
    def set_level(self,
                 name: str,
                 level: int) -> None:
        """Set logger level"""
        with self.lock:
            logger = self.get_logger(name)
            logger.setLevel(level)

# Global log manager instance
log_manager = LogManager()

# Default logger
logger = log_manager.get_logger('agent_builder')
log_manager.add_console_handler(
    'agent_builder',
    formatter=JsonFormatter()
)
