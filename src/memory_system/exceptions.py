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


class LLMCallError(Exception):
    """Raised when LLM API call fails after retries.
    
    Attributes:
        model: The model ID that was being called
        attempts: Number of retry attempts made
        last_error: The last exception that occurred
    """
    
    def __init__(self, model: str, attempts: int, last_error: Exception):
        """Initialize LLMCallError.
        
        Args:
            model: Model ID that failed
            attempts: Number of attempts made before giving up
            last_error: The final exception that caused failure
        """
        self.model = model
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(
            f"LLM API call failed for {model} after {attempts} attempts: {last_error}"
        )
