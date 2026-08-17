"""Unit tests for the OpenAI-compatible LLM adapter (implementation plan step 2)."""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from neuramem.config import LLMConfig
from neuramem.core.models import LLMUsage
from neuramem.llm.openai_adapter import (
    OpenAILLM,
    UsageStats,
    detect_compat,
    normalize_provider_error,
    parse_usage,
)


def _config(**overrides) -> LLMConfig:
    params = dict(base_url="https://api.test.com/v1", api_key="k", model="m")
    params.update(overrides)
    return LLMConfig(_env_file=None, **params)


def _patch_client(create_mock):
    ctor = Mock(return_value=Mock())
    ctor.return_value.chat.completions.create = create_mock
    return patch("neuramem.llm.openai_adapter.AsyncOpenAI", ctor), ctor


def _usage_mock(prompt_tokens=100, cached_tokens=30):
    usage = Mock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = 50
    details = Mock()
    details.cached_tokens = cached_tokens
    details.cache_write_tokens = None
    usage.prompt_tokens_details = details
    completion_details = Mock()
    completion_details.reasoning_tokens = 10
    usage.completion_tokens_details = completion_details
    return usage


def _response(content, usage=None):
    message = Mock()
    message.content = content
    choice = Mock()
    choice.message = message
    response = Mock()
    response.choices = [choice]
    response.usage = usage
    return response


class _AsyncIter:
    def __init__(self, items):
        self._items = list(items)

    def __aiter__(self):
        self._i = 0
        return self

    async def __anext__(self):
        if self._i >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._i]
        self._i += 1
        return item


def _chunk(content=None, usage=None, choice_usage=None, choices=None):
    chunk = Mock()
    chunk.usage = usage
    if choices is not None:
        chunk.choices = choices
    elif content is None and choice_usage is None:
        chunk.choices = []
    else:
        choice = Mock()
        delta = Mock()
        delta.content = content
        choice.delta = delta
        choice.usage = choice_usage
        chunk.choices = [choice]
    return chunk


class TestDetectCompat:
    def test_deepseek_fingerprint(self):
        compat = detect_compat("https://api.deepseek.com/v1")
        assert compat.provider == "deepseek"
        assert compat.include_stream_usage is True

    def test_default_branch(self):
        assert detect_compat("https://api.minimaxi.com/v1").provider == "openai"


class TestParseUsage:
    def test_minimax_style_details(self):
        usage = parse_usage(_usage_mock(prompt_tokens=100, cached_tokens=30))
        assert usage.input_tokens == 70
        assert usage.cache_read_tokens == 30
        assert usage.total_tokens == 70 + 50 + 30

    def test_deepseek_top_level_field(self):
        usage = Mock()
        usage.prompt_tokens = 100
        usage.completion_tokens = 10
        details = Mock()
        details.cached_tokens = None
        details.cache_write_tokens = None
        usage.prompt_tokens_details = details
        usage.completion_tokens_details = None
        usage.prompt_cache_hit_tokens = 40
        parsed = parse_usage(usage)
        assert parsed.cache_read_tokens == 40
        assert parsed.input_tokens == 60

    def test_none_passthrough(self):
        assert parse_usage(None) is None


class TestNormalizeProviderError:
    def test_status_body_message(self):
        exception = Exception("boom")
        exception.status_code = 429
        exception.body = "rate limited"
        normalized = normalize_provider_error(exception)
        assert normalized["status"] == 429
        assert normalized["body"] == "rate limited"

    def test_body_truncated(self):
        exception = Exception("boom")
        exception.status_code = 500
        exception.body = "x" * 10_000
        normalized = normalize_provider_error(exception)
        assert len(normalized["body"]) < 5_000
        assert normalized["body"].endswith("...(truncated)")


class TestComplete:
    @pytest.mark.asyncio
    async def test_complete_records_usage_with_label(self):
        create = AsyncMock(return_value=_response("hello", usage=_usage_mock()))
        with _patch_client(create)[0]:
            llm = OpenAILLM(_config())
            result = await llm.complete("sys", "user", call_label="answer")

        assert result.content == "hello"
        assert result.usage is not None
        assert result.usage.cache_read_tokens == 30
        assert llm.usage_stats.snapshot("answer")["calls"] == 1
        assert llm.usage_stats.hit_rate() is not None

    @pytest.mark.asyncio
    async def test_sdk_constructed_with_zero_retries(self):
        create = AsyncMock(return_value=_response("x"))
        patched, ctor = _patch_client(create)
        with patched:
            llm = OpenAILLM(_config())
            await llm.complete("sys", "user")
        assert ctor.call_args.kwargs["max_retries"] == 0


class TestCompleteJson:
    @pytest.mark.asyncio
    async def test_valid_json_first_attempt(self):
        create = AsyncMock(return_value=_response('{"key": "value"}'))
        with _patch_client(create)[0]:
            llm = OpenAILLM(_config())
            result = await llm.complete_json("sys", "user")

        assert result.success is True
        assert result.parsed_data == {"key": "value"}
        assert create.call_count == 1

    @pytest.mark.asyncio
    async def test_repair_retry_recovers(self):
        create = AsyncMock(
            side_effect=[_response("nope, prose"), _response('{"key": "fixed"}')]
        )
        with _patch_client(create)[0]:
            llm = OpenAILLM(_config())
            result = await llm.complete_json("sys", "user")

        assert result.success is True
        assert result.parsed_data == {"key": "fixed"}
        assert create.call_count == 2
        retry_message = create.call_args_list[1].kwargs["messages"][1]["content"]
        assert "not valid JSON" in retry_message

    @pytest.mark.asyncio
    async def test_failure_reported_after_retry(self):
        create = AsyncMock(side_effect=[_response("garbage"), _response("more garbage")])
        with _patch_client(create)[0]:
            llm = OpenAILLM(_config())
            result = await llm.complete_json("sys", "user", default={"d": 1})

        assert result.success is False
        assert result.parsed_data == {"d": 1}
        assert create.call_count == 2

    @pytest.mark.asyncio
    async def test_think_block_and_fences_stripped(self):
        think = chr(60) + "think" + chr(62) + "reasoning" + chr(60) + "/think" + chr(62)
        create = AsyncMock(
            return_value=_response(f"```json\n{think}{{\"key\": 1}}\n```")
        )
        with _patch_client(create)[0]:
            llm = OpenAILLM(_config())
            result = await llm.complete_json("sys", "user")

        assert result.success is True
        assert result.parsed_data == {"key": 1}


class TestStream:
    @pytest.mark.asyncio
    async def test_stream_requests_usage_and_records_final_chunk(self):
        final_usage = _usage_mock(prompt_tokens=200, cached_tokens=80)
        chunks = [
            _chunk(content="Hel"),
            _chunk(content="lo"),
            _chunk(usage=final_usage),
        ]
        create = AsyncMock(return_value=_AsyncIter(chunks))
        with _patch_client(create)[0]:
            llm = OpenAILLM(_config())
            collected = [part async for part in llm.stream("sys", "user", call_label="answer")]

        assert collected == ["Hel", "lo"]
        # usage must be explicitly requested (OpenAI does not send it by default)
        assert create.call_args.kwargs.get("stream_options") == {"include_usage": True}
        snapshot = llm.usage_stats.snapshot("answer")
        assert snapshot["calls"] == 1
        assert snapshot["cache_read_tokens"] == 80

    @pytest.mark.asyncio
    async def test_stream_moonshot_choice_usage_fallback(self):
        final_usage = _usage_mock(prompt_tokens=100, cached_tokens=0)
        chunks = [
            _chunk(content="x"),
            _chunk(choice_usage=final_usage),
        ]
        create = AsyncMock(return_value=_AsyncIter(chunks))
        with _patch_client(create)[0]:
            llm = OpenAILLM(_config())
            collected = [part async for part in llm.stream("sys", "user")]

        assert collected == ["x"]
        assert llm.usage_stats.snapshot()["calls"] == 1

    @pytest.mark.asyncio
    async def test_stream_chunk_with_no_choices_is_tolerated(self):
        """Mid-stream heartbeat chunks can arrive with choices=[]."""
        chunks = [_chunk(content="hi"), _chunk(), _chunk(content="bye")]
        create = AsyncMock(return_value=_AsyncIter(chunks))
        with _patch_client(create)[0]:
            llm = OpenAILLM(_config())
            collected = [p async for p in llm.stream("sys", "user")]

        assert collected == ["hi", "bye"]
        # heartbeat chunk did not crash and did not record phantom usage
        assert llm.usage_stats.snapshot()["calls"] == 0

    @pytest.mark.asyncio
    async def test_stream_multi_choice_concatenates(self):
        """n>1 streams must not silently drop the second choice."""
        def make_choice(content):
            c = Mock()
            c.delta = Mock(content=content)
            c.usage = None
            return c
        chunks = [
            _chunk(choices=[make_choice("a"), make_choice("b")]),
            _chunk(choices=[make_choice("1"), make_choice("2")]),
        ]
        create = AsyncMock(return_value=_AsyncIter(chunks))
        with _patch_client(create)[0]:
            llm = OpenAILLM(_config())
            collected = [p async for p in llm.stream("sys", "user")]

        assert collected == ["ab", "12"]


class TestUsageStatsScopes:
    def test_scope_isolation_across_tasks(self):
        stats = UsageStats()

        async def worker(n):
            with stats.scope():
                stats.record(LLMUsage(input_tokens=n), "answer")
                await asyncio.sleep(0)
                return stats.scope_snapshot("answer")["input_tokens"]

        async def main():
            return await asyncio.gather(worker(1), worker(2))

        first, second = asyncio.run(main())
        assert first == 1
        assert second == 2
        # global accumulates across scopes
        assert stats.snapshot("answer")["input_tokens"] == 3

    def test_scope_isolation_across_threads(self):
        stats = UsageStats()
        results = {}

        def worker(name, tokens):
            with stats.scope():
                stats.record(LLMUsage(input_tokens=tokens), "answer")
                results[name] = stats.scope_snapshot("answer")["input_tokens"]

        threads = [
            threading.Thread(target=worker, args=("a", 5)),
            threading.Thread(target=worker, args=("b", 7)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert results == {"a": 5, "b": 7}
        assert stats.snapshot("answer")["input_tokens"] == 12

    def test_no_scope_records_globally_only(self):
        stats = UsageStats()
        stats.record(LLMUsage(input_tokens=2))
        assert stats.scope_snapshot()["calls"] == 0
        assert stats.snapshot()["calls"] == 1
