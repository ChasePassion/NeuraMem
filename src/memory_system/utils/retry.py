"""Retry utilities for resilient API calls.

Provides a unified retry mechanism for sync, async, and generator operations
with exponential backoff.
"""

import asyncio
import logging
import time
from typing import Callable, TypeVar, Optional, Any, Generator, AsyncGenerator, Tuple, Type

from ..exceptions import LLMCallError

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RetryExecutor:
    """Unified retry executor for API operations.
    
    Supports:
    - Synchronous operations via `execute()`
    - Asynchronous operations via `execute_async()`
    - Synchronous generators via `stream()`
    - Asynchronous generators via `stream_async()`
    
    Example:
        executor = RetryExecutor(max_retries=3, base_delay=1.0, model="gpt-4")
        result = executor.execute(lambda: client.chat.completions.create(...))
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        model: str = "",
        operation: str = "",
        retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    ):
        """Initialize retry executor.
        
        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Base delay in seconds for exponential backoff
            model: Model identifier for error reporting
            operation: Operation name for logging
            retryable_exceptions: Tuple of exception types that should trigger retry.
                                  Auth errors and similar should NOT be included.
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.model = model
        self.operation = operation
        self.retryable_exceptions = retryable_exceptions
        self._last_error: Optional[Exception] = None
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay."""
        return self.base_delay * (2 ** attempt)
    
    def _log_retry(self, attempt: int, error: Exception, is_async: bool = False) -> None:
        """Log retry attempt."""
        async_str = "async " if is_async else ""
        logger.warning(
            "%s%s attempt %s/%s for model %s failed: %s",
            async_str,
            self.operation or "API",
            attempt + 1,
            self.max_retries,
            self.model,
            error,
        )
    
    def _raise_final_error(self) -> None:
        """Raise LLMCallError after all retries exhausted."""
        raise LLMCallError(self.model, self.max_retries, self._last_error)
    
    def _should_retry(self, error: Exception) -> bool:
        """Check if the exception is retryable."""
        return isinstance(error, self.retryable_exceptions)
    
    def execute(self, operation_fn: Callable[[], T]) -> T:
        """Execute a synchronous operation with retry.
        
        Args:
            operation_fn: Zero-argument callable that performs the operation
            
        Returns:
            Result of operation_fn
            
        Raises:
            LLMCallError: If all retry attempts fail
        """
        for attempt in range(self.max_retries):
            try:
                return operation_fn()
            except Exception as e:
                if not self._should_retry(e):
                    raise
                self._last_error = e
                self._log_retry(attempt, e)
                if attempt < self.max_retries - 1:
                    delay = self._calculate_delay(attempt)
                    time.sleep(delay)
        
        self._raise_final_error()
    
    async def execute_async(self, operation_fn: Callable[[], Any]) -> Any:
        """Execute an asynchronous operation with retry.
        
        Args:
            operation_fn: Zero-argument async callable
            
        Returns:
            Result of awaited operation_fn
            
        Raises:
            LLMCallError: If all retry attempts fail
        """
        for attempt in range(self.max_retries):
            try:
                return await operation_fn()
            except Exception as e:
                if not self._should_retry(e):
                    raise
                self._last_error = e
                self._log_retry(attempt, e, is_async=True)
                if attempt < self.max_retries - 1:
                    delay = self._calculate_delay(attempt)
                    await asyncio.sleep(delay)
        
        self._raise_final_error()
    
    def stream(self, stream_fn: Callable[[], Generator[T, None, None]]) -> Generator[T, None, None]:
        """Execute a synchronous generator with retry.
        
        Retries if an exception occurs during iteration. Once streaming
        successfully completes, no more retries are attempted.
        
        Args:
            stream_fn: Zero-argument callable that returns a generator
            
        Yields:
            Items from the generator
            
        Raises:
            LLMCallError: If all retry attempts fail
        """
        for attempt in range(self.max_retries):
            try:
                yield from stream_fn()
                return  # Success, exit retry loop
            except Exception as e:
                if not self._should_retry(e):
                    raise
                self._last_error = e
                self._log_retry(attempt, e)
                if attempt < self.max_retries - 1:
                    delay = self._calculate_delay(attempt)
                    time.sleep(delay)
        
        self._raise_final_error()
    
    async def stream_async(
        self, 
        stream_fn: Callable[[], AsyncGenerator[T, None]]
    ) -> AsyncGenerator[T, None]:
        """Execute an asynchronous generator with retry.
        
        Retries if an exception occurs during iteration. Once streaming
        successfully completes, no more retries are attempted.
        
        Args:
            stream_fn: Zero-argument callable that returns an async generator
            
        Yields:
            Items from the async generator
            
        Raises:
            LLMCallError: If all retry attempts fail
        """
        for attempt in range(self.max_retries):
            try:
                async for item in stream_fn():
                    yield item
                return  # Success, exit retry loop
            except Exception as e:
                if not self._should_retry(e):
                    raise
                self._last_error = e
                self._log_retry(attempt, e, is_async=True)
                if attempt < self.max_retries - 1:
                    delay = self._calculate_delay(attempt)
                    await asyncio.sleep(delay)
        
        self._raise_final_error()
