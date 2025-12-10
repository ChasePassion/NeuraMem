"""Unit tests for RetryExecutor utility."""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock

from src.memory_system.utils.retry import RetryExecutor
from src.memory_system.exceptions import LLMCallError


class TestRetryExecutorExecute:
    """Tests for synchronous execute() method."""
    
    def test_execute_success_first_attempt(self):
        """Test successful execution on first attempt."""
        executor = RetryExecutor(max_retries=3, model="test-model")
        result = executor.execute(lambda: "success")
        assert result == "success"
    
    def test_execute_success_after_retries(self):
        """Test successful execution after initial failures."""
        call_count = 0
        
        def operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary error")
            return "success"
        
        with patch("time.sleep"):  # Skip actual sleep
            executor = RetryExecutor(max_retries=3, model="test-model")
            result = executor.execute(operation)
        
        assert result == "success"
        assert call_count == 3
    
    def test_execute_raises_after_max_retries(self):
        """Test that LLMCallError is raised after all retries fail."""
        with patch("time.sleep"):
            executor = RetryExecutor(max_retries=3, model="test-model")
            
            with pytest.raises(LLMCallError) as exc_info:
                executor.execute(lambda: (_ for _ in ()).throw(Exception("Always fails")))
            
            assert exc_info.value.attempts == 3
            assert "test-model" in str(exc_info.value)
    
    def test_execute_non_retryable_exception(self):
        """Test that non-retryable exceptions are raised immediately."""
        executor = RetryExecutor(
            max_retries=3, 
            model="test-model",
            retryable_exceptions=(ValueError,)
        )
        
        with pytest.raises(TypeError):  # Not in retryable_exceptions
            executor.execute(lambda: (_ for _ in ()).throw(TypeError("Not retryable")))


class TestRetryExecutorExecuteAsync:
    """Tests for asynchronous execute_async() method."""
    
    def test_execute_async_success(self):
        """Test successful async execution."""
        async def run_test():
            executor = RetryExecutor(max_retries=3, model="test-model")
            
            async def async_op():
                return "async_success"
            
            return await executor.execute_async(async_op)
        
        result = asyncio.run(run_test())
        assert result == "async_success"
    
    def test_execute_async_retries(self):
        """Test async execution with retries."""
        call_count = 0
        
        async def run_test():
            nonlocal call_count
            
            async def async_op():
                nonlocal call_count
                call_count += 1
                if call_count < 2:
                    raise Exception("Temporary")
                return "success"
            
            executor = RetryExecutor(max_retries=3, model="test-model", base_delay=0.01)
            return await executor.execute_async(async_op)
        
        result = asyncio.run(run_test())
        assert result == "success"
        assert call_count == 2


class TestRetryExecutorStream:
    """Tests for synchronous stream() method."""
    
    def test_stream_success(self):
        """Test successful streaming."""
        executor = RetryExecutor(max_retries=3, model="test-model")
        
        def gen():
            yield "chunk1"
            yield "chunk2"
            yield "chunk3"
        
        chunks = list(executor.stream(gen))
        assert chunks == ["chunk1", "chunk2", "chunk3"]
    
    def test_stream_retry_on_failure(self):
        """Test that streaming retries on failure."""
        call_count = 0
        
        def gen():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Stream error")
            yield "chunk1"
            yield "chunk2"
        
        with patch("time.sleep"):
            executor = RetryExecutor(max_retries=3, model="test-model")
            chunks = list(executor.stream(gen))
        
        assert chunks == ["chunk1", "chunk2"]
        assert call_count == 2


class TestRetryExecutorStreamAsync:
    """Tests for asynchronous stream_async() method."""
    
    def test_stream_async_success(self):
        """Test successful async streaming."""
        async def run_test():
            executor = RetryExecutor(max_retries=3, model="test-model")
            
            async def async_gen():
                yield "async_chunk1"
                yield "async_chunk2"
            
            chunks = []
            async for chunk in executor.stream_async(async_gen):
                chunks.append(chunk)
            return chunks
        
        chunks = asyncio.run(run_test())
        assert chunks == ["async_chunk1", "async_chunk2"]
    
    def test_stream_async_retry(self):
        """Test async streaming with retry."""
        call_count = 0
        
        async def run_test():
            nonlocal call_count
            
            async def async_gen():
                nonlocal call_count
                call_count += 1
                if call_count < 2:
                    raise Exception("Async stream error")
                yield "chunk1"
            
            executor = RetryExecutor(max_retries=3, model="test-model", base_delay=0.01)
            chunks = []
            async for chunk in executor.stream_async(async_gen):
                chunks.append(chunk)
            return chunks
        
        chunks = asyncio.run(run_test())
        assert chunks == ["chunk1"]
        assert call_count == 2


class TestRetryExecutorBackoff:
    """Tests for exponential backoff calculation."""
    
    def test_exponential_backoff_delays(self):
        """Test that delays follow exponential backoff pattern."""
        executor = RetryExecutor(max_retries=4, base_delay=1.0, model="test-model")
        
        assert executor._calculate_delay(0) == 1.0
        assert executor._calculate_delay(1) == 2.0
        assert executor._calculate_delay(2) == 4.0
        assert executor._calculate_delay(3) == 8.0

