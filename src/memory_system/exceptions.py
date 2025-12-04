"""Custom exception classes for AI Memory System.

Provides descriptive exceptions for error handling and debugging.
"""


class MilvusConnectionError(Exception):
    """Raised when Milvus connection fails.
    
    Attributes:
        uri: The Milvus URI that failed to connect
        original_error: The underlying exception that caused the failure
    """
    
    def __init__(self, uri: str, original_error: Exception):
        """Initialize MilvusConnectionError.
        
        Args:
            uri: Milvus server URI that failed
            original_error: The original exception from the connection attempt
        """
        self.uri = uri
        self.original_error = original_error
        super().__init__(f"Failed to connect to Milvus at {uri}: {original_error}")


class OpenRouterError(Exception):
    """Raised when OpenRouter API call fails after retries.
    
    Attributes:
        model: The model ID that was being called
        attempts: Number of retry attempts made
        last_error: The last exception that occurred
    """
    
    def __init__(self, model: str, attempts: int, last_error: Exception):
        """Initialize OpenRouterError.
        
        Args:
            model: Model ID that failed
            attempts: Number of attempts made before giving up
            last_error: The final exception that caused failure
        """
        self.model = model
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(
            f"OpenRouter API failed for {model} after {attempts} attempts: {last_error}"
        )


class MemoryOperationError(Exception):
    """Raised when a memory operation fails.
    
    Attributes:
        operation: The operation that failed (add, search, update, delete, etc.)
        user_id: The user ID involved in the operation
        details: Additional details about the failure
    """
    
    def __init__(self, operation: str, user_id: str, details: str = ""):
        """Initialize MemoryOperationError.
        
        Args:
            operation: Name of the failed operation
            user_id: User ID involved
            details: Additional error details
        """
        self.operation = operation
        self.user_id = user_id
        self.details = details
        message = f"Memory operation '{operation}' failed for user '{user_id}'"
        if details:
            message += f": {details}"
        super().__init__(message)
