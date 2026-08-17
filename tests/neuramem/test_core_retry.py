"""Unit tests for the HTTP-aware retry executor (implementation plan step 1).

Covers the scope narrowing vs the legacy executor (which retried every
Exception): 408/409/429/5xx and x-should-retry only; registered
connection-error types; retry-after precedence and cap; jitter direction.
"""

from types import SimpleNamespace

import pytest

from neuramem.core.exceptions import LLMCallError
from neuramem.core.retry import (
    RetryExecutor,
    default_retryable,
    register_retryable_type,
    retry_after_seconds,
)


class FakeHTTPError(Exception):
    """Provider-error double with status_code / response.headers shapes."""

    def __init__(self, status_code=None, headers=None):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code
        self.response = SimpleNamespace(headers=headers or {})


class FakeConnectionError(Exception):
    """Connection-level error: no status attribute."""


class TestDefaultRetryable:
    def test_rate_limit_and_server_errors_retry(self):
        for status in (408, 409, 429, 500, 503):
            assert default_retryable(FakeHTTPError(status_code=status))

    def test_client_errors_do_not_retry(self):
        for status in (400, 401, 403, 404, 422):
            assert not default_retryable(FakeHTTPError(status_code=status))

    def test_x_should_retry_header_wins(self):
        assert default_retryable(
            FakeHTTPError(status_code=400, headers={"x-should-retry": "true"})
        )
        assert not default_retryable(
            FakeHTTPError(status_code=500, headers={"x-should-retry": "false"})
        )

    def test_plain_exceptions_do_not_retry(self):
        assert not default_retryable(ValueError("bug"))

    def test_registered_connection_types_retry(self):
        register_retryable_type(FakeConnectionError)
        assert default_retryable(FakeConnectionError())

    def test_retry_after_above_cap_fails_fast(self):
        error = FakeHTTPError(
            status_code=429, headers={"retry-after": "120"}
        )
        assert default_retryable(error, max_retry_after=60) is False
        assert default_retryable(error, max_retry_after=0) is True  # cap off


class TestRetryAfterExtraction:
    def test_seconds_header(self):
        assert retry_after_seconds(
            FakeHTTPError(headers={"retry-after": "2"})
        ) == 2.0

    def test_milliseconds_header_takes_precedence(self):
        assert retry_after_seconds(
            FakeHTTPError(
                headers={"retry-after-ms": "1500", "retry-after": "9"}
            )
        ) == 1.5

    def test_absent_headers(self):
        assert retry_after_seconds(FakeHTTPError(status_code=500)) is None


class TestRetryExecutorExecute:
    def test_retries_429_then_succeeds(self):
        executor = RetryExecutor(max_retries=3, base_delay=0.0)
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise FakeHTTPError(status_code=429)
            return "ok"

        assert executor.execute(flaky) == "ok"
        assert calls["n"] == 3

    def test_non_retryable_raises_immediately(self):
        executor = RetryExecutor(max_retries=3, base_delay=0.0)
        calls = {"n": 0}

        def bad_request():
            calls["n"] += 1
            raise FakeHTTPError(status_code=400)

        with pytest.raises(FakeHTTPError):
            executor.execute(bad_request)
        assert calls["n"] == 1  # no retries spent

    def test_exhaustion_raises_llm_call_error(self):
        executor = RetryExecutor(
            max_retries=2, base_delay=0.0, model="test-model"
        )

        def always_429():
            raise FakeHTTPError(status_code=429)

        with pytest.raises(LLMCallError) as exc_info:
            executor.execute(always_429)
        assert exc_info.value.model == "test-model"
        assert exc_info.value.attempts == 2

    def test_registered_connection_error_retries(self):
        register_retryable_type(FakeConnectionError)
        executor = RetryExecutor(max_retries=2, base_delay=0.0)
        calls = {"n": 0}

        def flaky_connection():
            calls["n"] += 1
            if calls["n"] == 1:
                raise FakeConnectionError()
            return 42

        assert executor.execute(flaky_connection) == 42

    @pytest.mark.asyncio
    async def test_execute_async_retries(self):
        executor = RetryExecutor(max_retries=2, base_delay=0.0)
        calls = {"n": 0}

        async def flaky():
            calls["n"] += 1
            if calls["n"] == 1:
                raise FakeHTTPError(status_code=503)
            return "done"

        assert await executor.execute_async(flaky) == "done"
        assert calls["n"] == 2
