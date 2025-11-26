"""
Exceptions Module
-------------
Custom exceptions for the agent builder system.
"""

class AgentBuilderError(Exception):
    """Base exception for agent builder system"""
    
    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.details = details or {}

class ValidationError(AgentBuilderError):
    """Validation error"""
    pass

class ConfigError(AgentBuilderError):
    """Configuration error"""
    pass

class SecurityError(AgentBuilderError):
    """Security error"""
    pass

class AuthenticationError(SecurityError):
    """Authentication error"""
    pass

class AuthorizationError(SecurityError):
    """Authorization error"""
    pass

class CacheError(AgentBuilderError):
    """Cache error"""
    pass

class CompressionError(AgentBuilderError):
    """Compression error"""
    pass

class SerializationError(AgentBuilderError):
    """Serialization error"""
    pass

class EventError(AgentBuilderError):
    """Event handling error"""
    pass

class MetricsError(AgentBuilderError):
    """Metrics error"""
    pass

class RateLimitError(AgentBuilderError):
    """Rate limit error"""
    
    def __init__(self,
                 limit: int,
                 window: int,
                 details: dict = None):
        message = (
            f"Rate limit exceeded: {limit} requests per {window} seconds"
        )
        super().__init__(message, details)
        self.limit = limit
        self.window = window

class AgentError(AgentBuilderError):
    """Agent error"""
    
    def __init__(self,
                 agent_id: str,
                 message: str,
                 details: dict = None):
        super().__init__(message, details)
        self.agent_id = agent_id

class SkillError(AgentBuilderError):
    """Skill error"""
    
    def __init__(self,
                 skill_id: str,
                 message: str,
                 details: dict = None):
        super().__init__(message, details)
        self.skill_id = skill_id

class TaskError(AgentBuilderError):
    """Task error"""
    
    def __init__(self,
                 task_id: str,
                 message: str,
                 details: dict = None):
        super().__init__(message, details)
        self.task_id = task_id

class ResourceError(AgentBuilderError):
    """Resource error"""
    
    def __init__(self,
                 resource_id: str,
                 message: str,
                 details: dict = None):
        super().__init__(message, details)
        self.resource_id = resource_id

class NetworkError(AgentBuilderError):
    """Network error"""
    
    def __init__(self,
                 url: str,
                 message: str,
                 details: dict = None):
        super().__init__(message, details)
        self.url = url

class DatabaseError(AgentBuilderError):
    """Database error"""
    
    def __init__(self,
                 operation: str,
                 message: str,
                 details: dict = None):
        super().__init__(message, details)
        self.operation = operation

class FileSystemError(AgentBuilderError):
    """File system error"""
    
    def __init__(self,
                 path: str,
                 message: str,
                 details: dict = None):
        super().__init__(message, details)
        self.path = path

class TimeoutError(AgentBuilderError):
    """Timeout error"""
    
    def __init__(self,
                 operation: str,
                 timeout: float,
                 details: dict = None):
        message = f"Operation '{operation}' timed out after {timeout} seconds"
        super().__init__(message, details)
        self.operation = operation
        self.timeout = timeout

class ConcurrencyError(AgentBuilderError):
    """Concurrency error"""
    pass

class MemoryError(AgentBuilderError):
    """Memory error"""
    
    def __init__(self,
                 operation: str,
                 limit: int,
                 usage: int,
                 details: dict = None):
        message = (
            f"Memory limit exceeded during '{operation}': "
            f"using {usage} bytes of {limit} bytes limit"
        )
        super().__init__(message, details)
        self.operation = operation
        self.limit = limit
        self.usage = usage

class StateError(AgentBuilderError):
    """State error"""
    
    def __init__(self,
                 component: str,
                 state: str,
                 message: str,
                 details: dict = None):
        super().__init__(message, details)
        self.component = component
        self.state = state

class PluginError(AgentBuilderError):
    """Plugin error"""
    
    def __init__(self,
                 plugin_id: str,
                 message: str,
                 details: dict = None):
        super().__init__(message, details)
        self.plugin_id = plugin_id

class IntegrationError(AgentBuilderError):
    """Integration error"""
    
    def __init__(self,
                 integration_id: str,
                 message: str,
                 details: dict = None):
        super().__init__(message, details)
        self.integration_id = integration_id

class ConfigurationError(AgentBuilderError):
    """Configuration error"""
    
    def __init__(self,
                 config_key: str,
                 message: str,
                 details: dict = None):
        super().__init__(message, details)
        self.config_key = config_key

class VersionError(AgentBuilderError):
    """Version error"""
    
    def __init__(self,
                 component: str,
                 required: str,
                 current: str,
                 details: dict = None):
        message = (
            f"Version mismatch for {component}: "
            f"required {required}, current {current}"
        )
        super().__init__(message, details)
        self.component = component
        self.required = required
        self.current = current

class DependencyError(AgentBuilderError):
    """Dependency error"""
    
    def __init__(self,
                 dependency: str,
                 message: str,
                 details: dict = None):
        super().__init__(message, details)
        self.dependency = dependency

class EnvironmentError(AgentBuilderError):
    """Environment error"""
    
    def __init__(self,
                 variable: str,
                 message: str,
                 details: dict = None):
        super().__init__(message, details)
        self.variable = variable

# Error code constants
ERROR_VALIDATION = "validation_error"
ERROR_CONFIG = "config_error"
ERROR_SECURITY = "security_error"
ERROR_AUTH = "auth_error"
ERROR_CACHE = "cache_error"
ERROR_COMPRESSION = "compression_error"
ERROR_SERIALIZATION = "serialization_error"
ERROR_EVENT = "event_error"
ERROR_METRICS = "metrics_error"
ERROR_RATE_LIMIT = "rate_limit_error"
ERROR_AGENT = "agent_error"
ERROR_SKILL = "skill_error"
ERROR_TASK = "task_error"
ERROR_RESOURCE = "resource_error"
ERROR_NETWORK = "network_error"
ERROR_DATABASE = "database_error"
ERROR_FILESYSTEM = "filesystem_error"
ERROR_TIMEOUT = "timeout_error"
ERROR_CONCURRENCY = "concurrency_error"
ERROR_MEMORY = "memory_error"
ERROR_STATE = "state_error"
ERROR_PLUGIN = "plugin_error"
ERROR_INTEGRATION = "integration_error"
ERROR_CONFIGURATION = "configuration_error"
ERROR_VERSION = "version_error"
ERROR_DEPENDENCY = "dependency_error"
ERROR_ENVIRONMENT = "environment_error"
