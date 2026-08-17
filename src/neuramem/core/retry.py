"""Retry executor with HTTP-aware retry classification (architecture_target.md 6.3).

Differences from the legacy utils/retry.py (which retried every Exception):
- Only recoverable provider errors are retried: HTTP 408 / 409 / 429 / 5xx,
  an explicit ``x-should-retry: true`` response header, or exception types
  registered by adapters (connection errors carry no status attribute).
- Server-provided retry-after / retry-after-ms headers take precedence over
  exponential backoff, capped by ``max_retry_after`` — a server demand above
  the cap fails fast instead of sleeping blindly.
- Backoff: exponential 0.5 * 2^n capped at max_delay, reduced by 0-25%
  jitter (never increased — do not amplify retry storms).
- The async paths sleep via asyncio.sleep, so cancellation interrupts
  backoff waits.

This module is IO-free: HTTP shapes are duck-typed (``status_code`` /
``response.headers`` attributes) so core never imports provider SDKs.
"""

import asyncio
import logging
import random
import time
from email.utils import parsedate_to_datetime
from typing import Any, AsyncGenerator, Callable, Generator, Optional, Type, TypeVar

from neuramem.core.exceptions import LLMCallError

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Exception types without a status_code attribute that adapters register as
# retryable (e.g. openai.APIConnectionError, openai.APITimeoutError).
_EXTRA_RETRYABLE_TYPES: set[Type[Exception]] = set()

RETRYABLE_STATUS_CODES = {408, 409, 429}


def register_retryable_type(*exc_types: Type[Exception]) -> None:
    """Register connection-level exception types as retryable.

    Adapters that import provider SDKs call this at module import time;
    core itself must stay SDK-free.
    """
    _EXTRA_RETRYABLE_TYPES.update(exc_types)


def _header_value(error: Exception, name: str) -> Optional[str]:
    """Read a response header from a provider exception, duck-typed."""
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    try:
        value = headers.get(name)
    except Exception:
        return None
    return str(value) if value is not None else None


def retry_after_seconds(error: Exception) -> Optional[float]:
    """Extract the server-requested wait from retry-after headers.

    Supports retry-after-ms (milliseconds) and retry-after (integer seconds
    or an HTTP-date). Returns None when the server said nothing.
    """
    ms = _header_value(error, "retry-after-ms")
    if ms is not None:
        try:
            return float(ms) / 1000.0
        except ValueError:
            pass
    seconds = _header_value(error, "retry-after")
    if seconds is not None:
        try:
            return float(seconds)
        except ValueError:
            try:
                when = parsedate_to_datetime(seconds)
            except (TypeError, ValueError):
                return None
            return max(0.0, when.timestamp() - time.time())
    return None


def default_retryable(error: Exception, max_retry_after: float = 60.0) -> bool:
    """Decide whether a provider error is recoverable (pi-mono policy).

    Precedence: x-should-retry header wins; then status-code duck typing
    (408/409/429/5xx); then registered connection-error types. A
    server-demanded wait above max_retry_after is treated as fatal
    (fail fast instead of sleeping).
    """
    should_retry_header = _header_value(error, "x-should-retry")
    if should_retry_header is not None:
        return should_retry_header.lower() == "true"

    status_code = getattr(error, "status_code", None)
    if isinstance(status_code, int):
        retryable = status_code in RETRYABLE_STATUS_CODES or status_code >= 500
        if retryable:
            wait = retry_after_seconds(error)
            if wait is not None and max_retry_after > 0 and wait > max_retry_after:
                logger.warning(
                    "Server retry-after %ss exceeds cap %ss; failing fast",
                    wait,
                    max_retry_after,
                )
                return False
        return retryable

    if _EXTRA_RETRYABLE_TYPES:
        return isinstance(error, tuple(_EXTRA_RETRYABLE_TYPES))
    return False


class RetryExecutor:
    """Unified retry executor for sync/async operations and generators."""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 0.5,
        max_delay: float = 8.0,
        max_retry_after: float = 60.0,
        model: str = "",
        operation: str = "",
        is_retryable: Optional[Callable[[Exception], bool]] = None,
    ):
        """Initialize retry executor.

        Args:
            max_retries: Total attempts (first try included)
            base_delay: Base exponential-backoff delay in seconds
            max_delay: Upper bound for a single backoff delay
            max_retry_after: Cap for server-demanded waits; 0 disables the
                cap. Waits above the cap fail fast (6.3)
            model: Model identifier for error reporting
            operation: Operation name for logging
            is_retryable: Custom classifier; None uses default_retryable
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.max_retry_after = max_retry_after
        self.model = model
        self.operation = operation
        self._is_retryable = is_retryable
        self._last_error: Optional[Exception] = None

    def _should_retry(self, error: Exception) -> bool:
        if self._is_retryable is not None:
            return self._is_retryable(error)
        return default_retryable(error, self.max_retry_after)

    def _calculate_delay(self, attempt: int, error: Exception) -> float:
        """Server retry-after if present, else capped exponential backoff."""
        wait = retry_after_seconds(error)
        if wait is not None and wait >= 0:
            return min(wait, self.max_retry_after) if self.max_retry_after > 0 else wait
        return min(self.base_delay * (2 ** attempt), self.max_delay)

    def _jittered(self, delay: float) -> float:
        """Reduce the delay by 0-25% so concurrent workers never amplify."""
        return delay * (1 - random.uniform(0, 0.25))

    def _log_retry(self, attempt: int, error: Exception, is_async: bool = False) -> None:
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
        raise LLMCallError(self.model, self.max_retries, self._last_error)

    def execute(self, operation_fn: Callable[[], T]) -> T:
        """Execute a synchronous operation with retry.

        Raises:
            LLMCallError: If all retry attempts fail on retryable errors
            Exception: The original error immediately when non-retryable
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
                    time.sleep(self._jittered(self._calculate_delay(attempt, e)))

        self._raise_final_error()

    async def execute_async(self, operation_fn: Callable[[], Any]) -> Any:
        """Execute an asynchronous operation with retry (cancellable waits)."""
        for attempt in range(self.max_retries):
            try:
                return await operation_fn()
            except Exception as e:
                if not self._should_retry(e):
                    raise
                self._last_error = e
                self._log_retry(attempt, e, is_async=True)
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self._jittered(self._calculate_delay(attempt, e)))

        self._raise_final_error()

    def stream(self, stream_fn: Callable[[], Generator[T, None, None]]) -> Generator[T, None, None]:
        """Execute a synchronous generator with retry.

        Retries if an exception occurs during iteration; a generator that
        completes is never retried.
        """
        for attempt in range(self.max_retries):
            try:
                yield from stream_fn()
                return
            except Exception as e:
                if not self._should_retry(e):
                    raise
                self._last_error = e
                self._log_retry(attempt, e)
                if attempt < self.max_retries - 1:
                    time.sleep(self._jittered(self._calculate_delay(attempt, e)))

        self._raise_final_error()

    async def stream_async(
        self,
        stream_fn: Callable[[], AsyncGenerator[T, None]],
    ) -> AsyncGenerator[T, None]:
        """Execute an asynchronous generator with retry (cancellable waits)."""
        for attempt in range(self.max_retries):
            try:
                async for item in stream_fn():
                    yield item
                return
            except Exception as e:
                if not self._should_retry(e):
                    raise
                self._last_error = e
                self._log_retry(attempt, e, is_async=True)
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self._jittered(self._calculate_delay(attempt, e)))

        self._raise_final_error()
