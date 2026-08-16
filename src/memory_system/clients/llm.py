"""LLM client with provider-agnostic fallback support.

Usage accounting follows docs/architecture_target.md #18 / 6.5
(pi-mono parseChunkUsage semantics): the client is the single place that
parses provider usage (prompt / completion / cache read / cache write /
reasoning tokens) and aggregates it for monitoring and benchmark cost
statistics. Parsing happens on every successful call, independent of
Langfuse, so evaluation pipelines can consume usage without any telemetry
backend (6.5.1).
"""

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

from openai import OpenAI, AsyncOpenAI

from ..exceptions import LLMCallError
from ..utils.retry import RetryExecutor
from langfuse import observe, get_client

logger = logging.getLogger(__name__)


def _as_int(value: Any, default: int = 0) -> int:
    """Coerce a provider usage field to int, tolerating None / weird types."""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@dataclass
class LLMUsage:
    """Structured usage of a single LLM call (architecture_target.md 6.5).

    Field mapping (parseChunkUsage semantics):
    - input_tokens:      prompt_tokens - cache_read - cache_write (net new input)
    - output_tokens:     completion_tokens
    - cache_read_tokens: prompt_tokens_details.cached_tokens (or prompt_cache_hit_tokens)
    - cache_write_tokens: prompt_tokens_details.cache_write_tokens
    - reasoning_tokens:  completion_tokens_details.reasoning_tokens
                         (already inside output_tokens, not added again)
    - total_tokens:      input + output + cache_read + cache_write
    - cost:              estimated cost in USD (0.0 unless pricing supplied)
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "total_tokens": self.total_tokens,
            "cost": self.cost,
        }


def parse_usage(usage: Any) -> Optional[LLMUsage]:
    """Parse provider usage into structured LLMUsage (architecture_target.md 6.5).

    Accepts OpenAI SDK usage objects (CompletionUsage, including stream chunk
    usage) or plain dicts. Compatible with:
    - OpenAI / MiniMax: prompt_tokens_details.cached_tokens / cache_write_tokens
    - DeepSeek:         top-level prompt_cache_hit_tokens (the SDK keeps extra
                        fields because its BaseModel uses extra="allow")
    - Moonshot-style streams: usage carried on the choice; the caller passes
                        whichever object is non-None, this parses either shape

    Returns None when usage is absent (e.g. stream chunks before the final
    one, where the full usage arrives exactly once on the last chunk).
    """
    if usage is None:
        return None

    if isinstance(usage, dict):
        prompt = _as_int(usage.get("prompt_tokens"))
        completion = _as_int(usage.get("completion_tokens"))
        details = usage.get("prompt_tokens_details") or {}
        completion_details = usage.get("completion_tokens_details") or {}
        cache_read = _as_int(
            details.get("cached_tokens") if isinstance(details, dict) else None
        )
        if cache_read == 0:
            cache_read = _as_int(usage.get("prompt_cache_hit_tokens"))
        cache_write = _as_int(
            details.get("cache_write_tokens") if isinstance(details, dict) else None
        )
        reasoning = _as_int(
            completion_details.get("reasoning_tokens")
            if isinstance(completion_details, dict)
            else None
        )
    else:
        prompt = _as_int(getattr(usage, "prompt_tokens", None))
        completion = _as_int(getattr(usage, "completion_tokens", None))
        details = getattr(usage, "prompt_tokens_details", None) or {}
        completion_details = getattr(usage, "completion_tokens_details", None) or {}
        cache_read = _as_int(getattr(details, "cached_tokens", None))
        if cache_read == 0:
            # DeepSeek reports the hit count as a top-level extra field
            cache_read = _as_int(getattr(usage, "prompt_cache_hit_tokens", None))
        cache_write = _as_int(getattr(details, "cache_write_tokens", None))
        reasoning = _as_int(getattr(completion_details, "reasoning_tokens", None))

    input_tokens = max(prompt - cache_read - cache_write, 0)
    total_tokens = input_tokens + completion + cache_read + cache_write

    return LLMUsage(
        input_tokens=input_tokens,
        output_tokens=completion,
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
        reasoning_tokens=reasoning,
        total_tokens=total_tokens,
    )


def estimate_cost(usage: LLMUsage, pricing: Optional[Dict[str, float]]) -> float:
    """Estimate USD cost of one call from a per-million-token price table.

    pricing keys: "input" / "output" / "cache_read" / "cache_write"
    (USD per 1M tokens). Returns 0.0 when no pricing is supplied.
    """
    if not pricing:
        return 0.0
    per_m = 1_000_000.0
    return (
        usage.input_tokens * pricing.get("input", 0.0)
        + usage.output_tokens * pricing.get("output", 0.0)
        + usage.cache_read_tokens * pricing.get("cache_read", 0.0)
        + usage.cache_write_tokens * pricing.get("cache_write", 0.0)
    ) / per_m


@dataclass
class _UsageCounter:
    """Mutable per-scope usage totals."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0

    def add(self, usage: LLMUsage) -> None:
        self.calls += 1
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        self.cache_read_tokens += usage.cache_read_tokens
        self.cache_write_tokens += usage.cache_write_tokens
        self.reasoning_tokens += usage.reasoning_tokens
        self.total_tokens += usage.total_tokens
        self.cost += usage.cost

    def as_dict(self) -> Dict[str, Any]:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "total_tokens": self.total_tokens,
            "cost": round(self.cost, 6),
        }


class UsageStats:
    """Thread-safe usage aggregator for all LLM calls of one client.

    Maintains a global accumulator plus per-thread counters so concurrent
    consumers (e.g. benchmark worker threads) can attribute call deltas to
    their own work via thread_snapshot().
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._global = _UsageCounter()
        self._local = threading.local()

    def record(self, usage: LLMUsage) -> None:
        with self._lock:
            self._global.add(usage)
        counter = getattr(self._local, "counter", None)
        if counter is None:
            counter = self._local.counter = _UsageCounter()
        counter.add(usage)

    def snapshot(self) -> Dict[str, Any]:
        """Cumulative totals across all threads (thread-safe)."""
        with self._lock:
            return self._global.as_dict()

    def thread_snapshot(self) -> Dict[str, Any]:
        """Totals recorded on the calling thread (thread-safe)."""
        counter = getattr(self._local, "counter", None)
        return counter.as_dict() if counter else _UsageCounter().as_dict()

    @staticmethod
    def hit_rate_of(snapshot: Dict[str, Any]) -> Optional[float]:
        """Token-weighted KV/prefix cache hit rate.

        rate = cache_read / (input + cache_read + cache_write), i.e. cached
        prompt tokens over all prompt tokens. Returns None when no call
        reported cache info (provider does not expose it).
        """
        prompt = (
            snapshot["input_tokens"]
            + snapshot["cache_read_tokens"]
            + snapshot["cache_write_tokens"]
        )
        if prompt <= 0:
            return None
        return snapshot["cache_read_tokens"] / prompt

    def hit_rate(self) -> Optional[float]:
        """Overall KV/prefix cache hit rate across all recorded calls."""
        return self.hit_rate_of(self.snapshot())


class LLMClient:
    """LLM model client with configurable primary and fallback providers."""
    
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        fallback_api_key: Optional[str] = None,
        fallback_base_url: Optional[str] = None,
        fallback_model: Optional[str] = None,
        max_retries: Optional[int] = None,
        base_delay: Optional[float] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        pricing: Optional[Dict[str, float]] = None,
    ):
        """Initialize LLM client.
        
        Args:
            api_key: API key (default: DeepSeek)
            base_url: Base URL for API (default: DeepSeek)
            model: Model ID for LLM (default: deepseek-chat)
            fallback_api_key: Optional API key for fallback provider
            fallback_base_url: Optional fallback base URL
            fallback_model: Optional fallback model ID
            max_retries: Retry budget for transient API/network failures.
                Defaults to LLM_MAX_RETRIES env var, then 10. Network blips
                (proxy hiccups, TLS resets) routinely outlast 3 attempts.
            base_delay: Base exponential-backoff delay in seconds.
                Defaults to LLM_BASE_DELAY env var, then 1.0.
            extra_body: Extra request body for providers needing vendor-specific
                params (e.g. MiniMax-M3 thinking control).
            pricing: Optional per-million-token price table (see
                estimate_cost) used to compute per-call cost in usage stats.
        """
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._pricing = pricing
        
        self._fallback_client = None
        self._async_fallback_client = None
        self._fallback_model = None
        if fallback_api_key and fallback_base_url and fallback_model:
            self._fallback_client = OpenAI(
                api_key=fallback_api_key, base_url=fallback_base_url
            )
            self._async_fallback_client = AsyncOpenAI(
                api_key=fallback_api_key, base_url=fallback_base_url
            )
            self._fallback_model = fallback_model
        
        self._max_retries = (
            max_retries if max_retries is not None
            else int(os.getenv("LLM_MAX_RETRIES", "10"))
        )
        self._base_delay = (
            base_delay if base_delay is not None
            else float(os.getenv("LLM_BASE_DELAY", "1.0"))
        )
        self._extra_body = extra_body

        # Single aggregation point for every successful call's usage
        # (architecture_target.md 6.5 / 6.5.1: the adapter is the only parser).
        self._usage_stats = UsageStats()

    @property
    def usage_stats(self) -> UsageStats:
        """Aggregated token/cache/cost usage across all LLM calls."""
        return self._usage_stats

    def _extra_body_kwargs(self) -> Dict[str, Any]:
        """Extra request body for providers needing vendor-specific params.

        E.g. reasoning models: MiniMax-M3 accepts thinking: {type: "disabled"}
        to skip the think block (faster, cleaner output). Passed via the
        OpenAI SDK's extra_body escape hatch.
        """
        return {"extra_body": self._extra_body} if self._extra_body else {}
    
    @observe(as_type="generation")
    def chat(self, system_prompt: str, user_message: str) -> str:
        """Call LLM for text response.
        
        Args:
            system_prompt: System prompt for LLM
            user_message: User message to process
            
        Returns:
            LLM response text
            
        Raises:
            LLMCallError: If API call fails after retries
        """
        get_client().update_current_trace(
            tags=["llm_call", "generation"],
            metadata={
                "client": "LLMClient",
                "prompt_length": len(system_prompt) + len(user_message)
            }
        )
        content, _ = self._chat_with_usage(system_prompt, user_message)
        return content

    def _chat_with_usage(
        self,
        system_prompt: str,
        user_message: str,
    ) -> tuple[str, Optional[LLMUsage]]:
        """Chat with primary (then fallback) provider; records usage stats.

        Returns (content, usage). The usage is the parsed response usage of
        the successful call, also recorded into self._usage_stats.
        """
        try:
            content, usage = self._chat_with_retries(
                client=self._client,
                model=self._model,
                system_prompt=system_prompt,
                user_message=user_message,
            )
        except LLMCallError as primary_error:
            if not self._fallback_client:
                raise
            
            logger.warning(
                "Primary LLM failed; using fallback: %s",
                primary_error,
            )
            try:
                content, usage = self._chat_with_retries(
                    client=self._fallback_client,
                    model=self._fallback_model,
                    system_prompt=system_prompt,
                    user_message=user_message,
                )
            except LLMCallError as fallback_error:
                # Surface combined failure context
                raise LLMCallError(
                    f"{self._model} (primary + fallback {self._fallback_model})",
                    self._max_retries,
                    fallback_error.last_error,
                ) from fallback_error

        if usage is not None:
            usage.cost = estimate_cost(usage, self._pricing)
            self._usage_stats.record(usage)
        return content, usage
    
    @observe(as_type="generation")
    def chat_stream(self, system_prompt: str, user_message: str):
        """Call LLM for streaming text response.
        
        Args:
            system_prompt: System prompt for LLM
            user_message: User message to process
            
        Yields:
            Text chunks from LLM response
            
        Raises:
            LLMCallError: If API call fails after retries
        """
        get_client().update_current_trace(
            tags=["llm_call", "streaming", "generation"],
            metadata={
                "client": "LLMClient",
                "streaming": True
            }
        )
        try:
            yield from self._chat_stream_with_retries(
                client=self._client,
                model=self._model,
                system_prompt=system_prompt,
                user_message=user_message,
            )
        except LLMCallError as primary_error:
            if not self._fallback_client:
                raise
            
            logger.warning(
                "Primary LLM failed; using fallback: %s",
                primary_error,
            )
            try:
                yield from self._chat_stream_with_retries(
                    client=self._fallback_client,
                    model=self._fallback_model,
                    system_prompt=system_prompt,
                    user_message=user_message,
                )
            except LLMCallError as fallback_error:
                raise LLMCallError(
                    f"{self._model} (primary + fallback {self._fallback_model})",
                    self._max_retries,
                    fallback_error.last_error,
                ) from fallback_error
    
    @observe(as_type="generation")
    async def chat_stream_async(self, system_prompt: str, user_message: str):
        """Async streaming chat using native AsyncOpenAI client.
        
        This method provides true async streaming without blocking the event loop.
        Much more efficient than chat_stream() wrapped in asyncio.to_thread().
        
        Args:
            system_prompt: System prompt for LLM
            user_message: User message to process
            
        Yields:
            Text chunks from LLM response
            
        Raises:
            LLMCallError: If API call fails after retries
        """
        get_client().update_current_trace(
            tags=["llm_call", "async_streaming", "generation"],
            metadata={
                "client": "LLMClient",
                "streaming": True,
                "async": True
            }
        )
        try:
            async for chunk in self._chat_stream_async_with_retries(
                client=self._async_client,
                model=self._model,
                system_prompt=system_prompt,
                user_message=user_message,
            ):
                yield chunk
        except LLMCallError as primary_error:
            if not self._async_fallback_client:
                raise
            
            logger.warning(
                "Primary async LLM failed; using async fallback: %s",
                primary_error,
            )
            try:
                async for chunk in self._chat_stream_async_with_retries(
                    client=self._async_fallback_client,
                    model=self._fallback_model,
                    system_prompt=system_prompt,
                    user_message=user_message,
                ):
                    yield chunk
            except LLMCallError as fallback_error:
                raise LLMCallError(
                    f"{self._model} (primary + fallback {self._fallback_model})",
                    self._max_retries,
                    fallback_error.last_error,
                ) from fallback_error
    
    async def _chat_stream_async_with_retries(
        self,
        client: AsyncOpenAI,
        model: str,
        system_prompt: str,
        user_message: str,
    ):
        """Async streaming with retry logic using RetryExecutor."""
        executor = RetryExecutor(
            max_retries=self._max_retries,
            base_delay=self._base_delay,
            model=model,
            operation="async_stream"
        )
        
        async def do_stream():
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                stream=True,
                **self._extra_body_kwargs(),
            )
            async for chunk in response:
                self._record_stream_usage(chunk)
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        
        async for item in executor.stream_async(do_stream):
            yield item

    def _record_stream_usage(self, chunk: Any) -> None:
        """Record usage carried by a stream chunk (architecture_target.md 6.5).

        OpenAI-compatible streams put the full usage on the final chunk
        (chunk.usage); some providers (e.g. Moonshot) carry it on the choice
        instead. Only the final chunk yields a non-None usage, so a stream is
        recorded exactly once.
        """
        usage_obj = getattr(chunk, "usage", None)
        if usage_obj is None and chunk.choices and chunk.choices[0].usage is not None:
            usage_obj = chunk.choices[0].usage
        usage = parse_usage(usage_obj)
        if usage is not None:
            usage.cost = estimate_cost(usage, self._pricing)
            self._usage_stats.record(usage)

    def chat_json(
        self,
        system_prompt: str,
        user_message: str,
        default: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Call LLM and parse JSON response with safe fallback.
        
        Args:
            system_prompt: System prompt for the LLM
            user_message: User message to process
            default: Default value to return if JSON parsing fails
            
        Returns:
            Dict containing:
            - parsed_data: Parsed JSON response or default value
            - raw_response: Original response text from LLM
            - model: Model used for the request
            - usage: Structured usage of the call (LLMUsage.to_dict()), or
              None when the provider did not report usage
            - success: Whether parsing was successful
        """
        if default is None:
            default = {}
        
        try:
            response_text, usage = self._chat_with_usage(system_prompt, user_message)
            parsed_data = self._safe_parse_json(response_text, default)
            
            return {
                "parsed_data": parsed_data,
                "raw_response": response_text,
                "model": self._model,
                "usage": usage.to_dict() if usage is not None else None,
                "success": True
            }
        except LLMCallError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in chat_json: {e}")
            return {
                "parsed_data": default,
                "raw_response": "",
                "model": self._model,
                "usage": None,
                "success": False,
                "error": str(e)
            }
    
    def _safe_parse_json(
        self,
        response: str,
        default: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Parse JSON response with fallback to default.
        
        Args:
            response: Raw response text from LLM
            default: Default value if parsing fails
            
        Returns:
            Parsed JSON or default value
        """
        if not response:
            logger.error("Empty response from LLM")
            return default
        
        # Try to extract JSON from response (handle markdown code blocks and
        # reasoning-think blocks emitted by reasoning models such as MiniMax-M3)
        text = response.strip()

        # Strip the reasoning-think block if present (content before the close tag)
        think_open = chr(60) + "think" + chr(62)
        think_close = chr(60) + "/think" + chr(62)
        if think_open in text and think_close in text:
            text = text.split(think_close, 1)[-1].strip()

        # Remove markdown code block if present
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]

        if text.endswith("```"):
            text = text[:-3]

        text = text.strip()

        # Extract the JSON object wherever it appears in the remaining text
        json_start = text.find("{")
        json_end = text.rfind("}")
        if json_start != -1 and json_end > json_start:
            text = text[json_start:json_end + 1]

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response: {e}. Response: {response[:200]}")
            return default

    def _chat_with_retries(
        self,
        client: OpenAI,
        model: str,
        system_prompt: str,
        user_message: str,
    ) -> tuple[str, Optional[LLMUsage]]:
        """Call a specific client with retries using RetryExecutor.

        Returns (content, parsed usage of the successful call).
        """
        executor = RetryExecutor(
            max_retries=self._max_retries,
            base_delay=self._base_delay,
            model=model,
            operation="chat"
        )
        
        def do_chat():
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                **self._extra_body_kwargs(),
            )
            return response.choices[0].message.content, parse_usage(response.usage)
        
        return executor.execute(do_chat)
    
    def _chat_stream_with_retries(
        self,
        client: OpenAI,
        model: str,
        system_prompt: str,
        user_message: str,
    ):
        """Call a specific client with retries for streaming using RetryExecutor."""
        executor = RetryExecutor(
            max_retries=self._max_retries,
            base_delay=self._base_delay,
            model=model,
            operation="stream"
        )
        
        def do_stream():
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                stream=True,
                **self._extra_body_kwargs(),
            )
            for chunk in response:
                self._record_stream_usage(chunk)
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        
        yield from executor.stream(do_stream)
