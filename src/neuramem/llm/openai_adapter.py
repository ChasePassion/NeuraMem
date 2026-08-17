"""OpenAI-compatible LLM adapter — single provider, no fallback (ch. 6).

pi-mono patterns applied:
- official OpenAI SDK with baseURL injection, constructed with
  ``max_retries=0``: the retry budget is owned solely by RetryExecutor so
  it is predictable and observable (SDK retries would silently multiply)
- provider compat detection from base_url, overridable (detectCompat)
- streaming usage must be requested explicitly
  (``stream_options={"include_usage": True}``) unless the provider rejects
  it — never assume the final chunk carries usage on its own
- the adapter is the single usage parsing point (parseChunkUsage
  semantics, see core LLMUsage); aggregation keeps call_label buckets and
  per-context scoping via contextvars (threading.local would silently
  mis-attribute under asyncio tasks)

Errors are normalized to {status, body, message} with body truncation
(6.5.3) for logging; nothing here falls back to another provider —
multi-provider routing is consumer policy (8.5).
"""

import json
import logging
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, Iterator, Optional

from openai import AsyncOpenAI, APIConnectionError, APITimeoutError

from neuramem.core.exceptions import LLMCallError
from neuramem.core.models import LLMUsage
from neuramem.core.ports import LLMJsonResult, LLMResponse
from neuramem.core.retry import RetryExecutor, register_retryable_type
from neuramem.config import LLMConfig

logger = logging.getLogger(__name__)

# Connection-level errors carry no status_code attribute; register them so
# the core classifier (which stays SDK-free) can retry them.
register_retryable_type(APIConnectionError, APITimeoutError)

MAX_ERROR_BODY_CHARS = 4000


# ---------------------------------------------------------------------------
# Provider compat detection (pi-mono detectCompat, minimal first version)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProviderCompat:
    """Per-provider protocol quirks. Extend field by field as needed."""

    provider: str = "openai"
    include_stream_usage: bool = True  # send stream_options.include_usage


def detect_compat(base_url: str) -> ProviderCompat:
    """Fingerprint the provider from base_url (6.2).

    First version: deepseek fingerprint + default branch. Known quirk so
    far: none that changes behavior — max_tokens field naming only matters
    once we start sending token limits, and every provider we use today
    accepts the usage-in-stream request.
    """
    url = (base_url or "").lower()
    if "deepseek" in url:
        return ProviderCompat(provider="deepseek", include_stream_usage=True)
    return ProviderCompat(provider="openai", include_stream_usage=True)


# ---------------------------------------------------------------------------
# Usage parsing (single parsing point, parseChunkUsage semantics)
# ---------------------------------------------------------------------------


def _as_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_usage(usage: Any) -> Optional[LLMUsage]:
    """Parse a provider usage payload into the core LLMUsage model.

    Accepts SDK usage objects (including stream chunk usage) or plain
    dicts. Compatible with:
    - OpenAI / MiniMax: prompt_tokens_details.cached_tokens /
      cache_write_tokens
    - DeepSeek: top-level prompt_cache_hit_tokens (SDK keeps extra fields)
    - Moonshot-style streams: usage on the choice instead of the chunk;
      the caller passes whichever object is non-None

    Returns None when usage is absent (chunks before the final one).
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
            cache_read = _as_int(getattr(usage, "prompt_cache_hit_tokens", None))
        cache_write = _as_int(getattr(details, "cache_write_tokens", None))
        reasoning = _as_int(getattr(completion_details, "reasoning_tokens", None))

    input_tokens = max(prompt - cache_read - cache_write, 0)
    return LLMUsage(
        input_tokens=input_tokens,
        output_tokens=completion,
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
        reasoning_tokens=reasoning,
        total_tokens=input_tokens + completion + cache_read + cache_write,
    )


def estimate_cost(usage: LLMUsage, pricing: Optional[Dict[str, float]]) -> float:
    """USD cost from a per-million-token price table; 0.0 without pricing."""
    if not pricing:
        return 0.0
    per_m = 1_000_000.0
    return (
        usage.input_tokens * pricing.get("input", 0.0)
        + usage.output_tokens * pricing.get("output", 0.0)
        + usage.cache_read_tokens * pricing.get("cache_read", 0.0)
        + usage.cache_write_tokens * pricing.get("cache_write", 0.0)
    ) / per_m


# ---------------------------------------------------------------------------
# Usage aggregation: global + per-label + per-context scope (contextvars)
# ---------------------------------------------------------------------------


@dataclass
class _UsageCounter:
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


@dataclass
class _ScopeState:
    """Mutable counters owned by one explicit scope (thread or async task)."""

    total: _UsageCounter = field(default_factory=_UsageCounter)
    labels: Dict[str, _UsageCounter] = field(default_factory=dict)


class UsageStats:
    """Usage aggregator: global + per-call-label + per-scope attribution.

    Scoping uses contextvars so both threads and asyncio tasks attribute
    correctly (each task must enter its own scope inside the task body —
    a scope entered outside and inherited by copy would share counters).

    with stats.scope():
        ... llm calls ...
    delta = stats.scope_snapshot()
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._global = _UsageCounter()
        self._by_label: Dict[str, _UsageCounter] = {}
        self._scope_var: ContextVar[Optional[_ScopeState]] = ContextVar(
            "neuramem_usage_scope", default=None
        )

    @staticmethod
    def _label_counter(store: Dict[str, _UsageCounter], label: str) -> _UsageCounter:
        counter = store.get(label)
        if counter is None:
            counter = store[label] = _UsageCounter()
        return counter

    def record(self, usage: LLMUsage, label: Optional[str] = None) -> None:
        with self._lock:
            self._global.add(usage)
            if label:
                self._label_counter(self._by_label, label).add(usage)
        scope = self._scope_var.get()
        if scope is not None:
            scope.total.add(usage)
            if label:
                self._label_counter(scope.labels, label).add(usage)

    @contextmanager
    def scope(self) -> Iterator[_ScopeState]:
        """Attribute calls made inside this block to a fresh scope."""
        state = _ScopeState()
        token = self._scope_var.set(state)
        try:
            yield state
        finally:
            self._scope_var.reset(token)

    def snapshot(self, label: Optional[str] = None) -> Dict[str, Any]:
        """Cumulative totals across all scopes (thread-safe)."""
        with self._lock:
            if label:
                counter = self._by_label.get(label)
                return counter.as_dict() if counter else _UsageCounter().as_dict()
            return self._global.as_dict()

    def scope_snapshot(self, label: Optional[str] = None) -> Dict[str, Any]:
        """Totals recorded inside the current scope (empty dict if none)."""
        scope = self._scope_var.get()
        if scope is None:
            return _UsageCounter().as_dict()
        if label:
            counter = scope.labels.get(label)
            return counter.as_dict() if counter else _UsageCounter().as_dict()
        return scope.total.as_dict()

    def labels(self) -> list[str]:
        with self._lock:
            return sorted(self._by_label.keys())

    @staticmethod
    def hit_rate_of(snapshot: Dict[str, Any]) -> Optional[float]:
        """Token-weighted cache hit rate: cache_read / all prompt tokens."""
        prompt = (
            snapshot["input_tokens"]
            + snapshot["cache_read_tokens"]
            + snapshot["cache_write_tokens"]
        )
        if prompt <= 0:
            return None
        return snapshot["cache_read_tokens"] / prompt

    def hit_rate(self) -> Optional[float]:
        return self.hit_rate_of(self.snapshot())


# ---------------------------------------------------------------------------
# Error normalization (pi-mono normalizeProviderError shape)
# ---------------------------------------------------------------------------


def normalize_provider_error(error: Exception) -> Dict[str, Any]:
    """Normalize a provider exception to {status, body, message}.

    Body is truncated to MAX_ERROR_BODY_CHARS so logs never explode.
    """
    status = getattr(error, "status_code", None)
    body = getattr(error, "body", None)
    if body is None:
        body = getattr(error, "message", None)
    if body is None:
        # openai SDK keeps the payload on the raw response sometimes
        response = getattr(error, "response", None)
        body = getattr(response, "text", None) if response is not None else None
    body_text = str(body) if body is not None else ""
    if len(body_text) > MAX_ERROR_BODY_CHARS:
        body_text = body_text[:MAX_ERROR_BODY_CHARS] + "...(truncated)"
    return {
        "status": status if isinstance(status, int) else None,
        "body": body_text,
        "message": str(error),
    }


# ---------------------------------------------------------------------------
# The adapter
# ---------------------------------------------------------------------------


class OpenAILLM:
    """LLM port implementation for one OpenAI-compatible provider."""

    def __init__(self, config: LLMConfig, compat: Optional[ProviderCompat] = None):
        self._config = config
        self._compat = compat or detect_compat(config.base_url)
        # SDK retries off: the retry budget belongs to RetryExecutor only
        self._client = AsyncOpenAI(
            api_key=config.api_key.get_secret_value(),
            base_url=config.base_url,
            max_retries=0,
        )
        self.usage_stats = UsageStats()

    @property
    def model_id(self) -> str:
        return self._config.model

    @property
    def compat(self) -> ProviderCompat:
        return self._compat

    def _executor(self, operation: str) -> RetryExecutor:
        return RetryExecutor(
            max_retries=self._config.max_retries,
            base_delay=self._config.base_delay,
            max_delay=self._config.max_delay,
            max_retry_after=self._config.max_retry_after,
            model=self._config.model,
            operation=operation,
        )

    def _extra_body_kwargs(self) -> Dict[str, Any]:
        extra = self._config.extra_body
        return {"extra_body": extra} if extra else {}

    def _record(self, usage: Optional[LLMUsage], label: Optional[str]) -> Optional[LLMUsage]:
        if usage is None:
            return None
        usage.cost = estimate_cost(usage, self._config.pricing)
        self.usage_stats.record(usage, label)
        return usage

    # -- non-streaming -------------------------------------------------------

    async def complete(
        self,
        system_prompt: str,
        user_message: str,
        *,
        call_label: Optional[str] = None,
    ) -> LLMResponse:
        async def do_call():
            response = await self._client.chat.completions.create(
                model=self._config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                **self._extra_body_kwargs(),
            )
            return response.choices[0].message.content, parse_usage(response.usage)

        try:
            content, usage = await self._executor("chat").execute_async(do_call)
        except LLMCallError as e:
            logger.error(
                "LLM call failed (normalized: %s)", normalize_provider_error(e.last_error)
            )
            raise
        return LLMResponse(content=content, usage=self._record(usage, call_label))

    async def complete_json(
        self,
        system_prompt: str,
        user_message: str,
        default: Optional[dict] = None,
        *,
        call_label: Optional[str] = None,
    ) -> LLMJsonResult:
        if default is None:
            default = {}

        async def do_call(message: str) -> tuple[str, Optional[LLMUsage]]:
            response = await self._client.chat.completions.create(
                model=self._config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message},
                ],
                **self._extra_body_kwargs(),
            )
            return response.choices[0].message.content, parse_usage(response.usage)

        executor = self._executor("chat_json")
        response_text, usage = await executor.execute_async(
            lambda: do_call(user_message)
        )
        parsed_data, parse_ok = _parse_json_payload(response_text, default)

        if not parse_ok:
            # One corrective retry (architecture_target.md #22)
            logger.warning(
                "complete_json: response was not valid JSON; "
                "retrying once with corrective feedback"
            )
            retry_message = (
                f"{user_message}\n\n"
                "Your previous reply was not valid JSON:\n"
                f"{response_text[:500]}\n\n"
                "Reply again with ONLY a valid JSON object. "
                "No prose, no markdown fences, no extra text."
            )
            response_text, usage = await executor.execute_async(
                lambda: do_call(retry_message)
            )
            parsed_data, parse_ok = _parse_json_payload(response_text, default)

        return LLMJsonResult(
            parsed_data=parsed_data,
            raw_response=response_text,
            model=self._config.model,
            usage=self._record(usage, call_label),
            success=parse_ok,
        )

    # -- streaming ------------------------------------------------------------

    async def stream(
        self,
        system_prompt: str,
        user_message: str,
        *,
        call_label: Optional[str] = None,
    ) -> AsyncIterator[str]:
        executor = self._executor("stream")

        async def do_stream():
            kwargs: Dict[str, Any] = dict(
                model=self._config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                stream=True,
                **self._extra_body_kwargs(),
            )
            # usage never arrives unless requested on several providers
            if self._compat.include_stream_usage:
                kwargs["stream_options"] = {"include_usage": True}
            response = await self._client.chat.completions.create(**kwargs)
            async for chunk in response:
                self._record_stream_usage(chunk, call_label)
                text = _extract_chunk_text(chunk)
                if text:
                    yield text

        async for item in executor.stream_async(do_stream):
            yield item

    def _record_stream_usage(self, chunk: Any, call_label: Optional[str]) -> None:
        usage_obj = _extract_chunk_usage(chunk)
        usage = parse_usage(usage_obj)
        if usage is not None:
            self._record(usage, call_label)


def _extract_chunk_text(chunk: Any) -> Optional[str]:
    """Concatenate text content across all choices on a stream chunk.

    Multi-choice streams (n>1) would silently truncate to the first choice
    under the previous `choices[0].delta.content` read. Skip empty deltas
    defensively because providers may emit a chunk with choices=[].
    """
    choices = getattr(chunk, "choices", None) or []
    parts: list[str] = []
    for choice in choices:
        delta = getattr(choice, "delta", None)
        content = getattr(delta, "content", None)
        if content:
            parts.append(content)
    return "".join(parts) if parts else None


def _extract_chunk_usage(chunk: Any) -> Any:
    """Read the final-chunk usage object, with Moonshot-style fallback.

    OpenAI puts it on chunk.usage; some providers (e.g. Moonshot) put it
    on choice.usage instead. Both shapes are accepted; the first match
    wins so partial overlap is deterministic.
    """
    usage_obj = getattr(chunk, "usage", None)
    if usage_obj is not None:
        return usage_obj
    for choice in getattr(chunk, "choices", None) or ():
        choice_usage = getattr(choice, "usage", None)
        if choice_usage is not None:
            return choice_usage
    return None


def _parse_json_payload(response: str, default: Dict[str, Any]) -> tuple[Dict[str, Any], bool]:
    """Clean an LLM response and parse it as JSON; returns (data, ok)."""
    if not response:
        return default, False
    text = response.strip()

    # strip reasoning-think blocks (reasoning models, e.g. MiniMax-M3)
    think_open = chr(60) + "think" + chr(62)
    think_close = chr(60) + "/think" + chr(62)
    if think_open in text and think_close in text:
        text = text.split(think_close, 1)[-1].strip()

    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    json_start = text.find("{")
    json_end = text.rfind("}")
    if json_start != -1 and json_end > json_start:
        text = text[json_start:json_end + 1]

    try:
        return json.loads(text), True
    except json.JSONDecodeError:
        return default, False
