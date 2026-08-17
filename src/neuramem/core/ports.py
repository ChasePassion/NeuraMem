"""Port definitions — the only interfaces the pipeline layer may depend on.

Four protocols (architecture_target.md #3): VectorStore / LLM / Embedder /
Telemetry. Adapters (llm/ embed/ store/ telemetry/) implement them; the
pipeline consumes them; nothing here imports IO libraries.

Signatures are async-first (architecture_target.md #7): adapters that wrap
synchronous SDKs (pymilvus) bridge with to_thread internally.

The VectorStore port intentionally includes the group-query surface
(list_groups / get_group_members): the demo's narrative panel needs it, and
discovering this late forced the legacy demo to reach into private store
attributes (implementation plan step 1, risk #4).
"""

from typing import AsyncIterator, Optional, Protocol, runtime_checkable

from contextlib import AbstractAsyncContextManager

from neuramem.core.models import (
    GroupInfo,
    GroupMatch,
    LLMUsage,
    MemoryFilter,
    MemoryRecord,
    SearchHit,
    SpanStatus,
)


# --------------------------------------------------------------------------
# Telemetry span contract (pi-mono startSpan-owns-settlement model)
# --------------------------------------------------------------------------


@runtime_checkable
class TelemetrySpan(Protocol):
    """A single in-flight span. Settlement is owned by the context manager
    returned by Telemetry.start_span: on clean exit the span ends as "ok",
    on exception it ends as "error" unless an explicit status was set."""

    def add_event(self, name: str, attributes: Optional[dict] = None) -> None: ...

    def set_attributes(self, attributes: dict) -> None: ...

    def set_status(self, status: SpanStatus) -> None: ...


@runtime_checkable
class Telemetry(Protocol):
    """Vendor-neutral telemetry port (architecture_target.md ch. 7).

    Implementations: telemetry/null.py (default, zero cost),
    telemetry/memory.py (in-memory, tests/benchmark), telemetry/langfuse.py.
    Span lifecycles must follow the context-manager semantics above — there
    is deliberately no public end() to leak.
    """

    def start_span(
        self, name: str, attributes: Optional[dict] = None
    ) -> "AbstractAsyncContextManager[TelemetrySpan]": ...


# --------------------------------------------------------------------------
# LLM port (single OpenAI-compatible provider, no fallback — ch. 6)
# --------------------------------------------------------------------------


class LLMResponse:
    """Result of a non-streaming completion: content plus parsed usage."""

    __slots__ = ("content", "usage")

    def __init__(self, content: str, usage: Optional[LLMUsage] = None):
        self.content = content
        self.usage = usage


class LLMJsonResult:
    """Result of a JSON-mode completion (envelope contract of ch. 6.4).

    success reflects the parse outcome (True only when real JSON was
    parsed, after at most one corrective repair retry) — callers that must
    not proceed on the fallback default check it and raise (#22).
    """

    __slots__ = ("parsed_data", "raw_response", "model", "usage", "success", "error")

    def __init__(
        self,
        parsed_data: dict,
        raw_response: str,
        model: str,
        usage: Optional[LLMUsage] = None,
        success: bool = True,
        error: Optional[str] = None,
    ):
        self.parsed_data = parsed_data
        self.raw_response = raw_response
        self.model = model
        self.usage = usage
        self.success = success
        self.error = error


@runtime_checkable
class LLM(Protocol):
    """LLM port. One instance faces one provider; multi-provider routing is
    consumer policy implemented as another adapter (8.5)."""

    async def complete(
        self,
        system_prompt: str,
        user_message: str,
        *,
        call_label: Optional[str] = None,
    ) -> LLMResponse: ...

    async def complete_json(
        self,
        system_prompt: str,
        user_message: str,
        default: Optional[dict] = None,
        *,
        call_label: Optional[str] = None,
    ) -> LLMJsonResult: ...

    def stream(
        self,
        system_prompt: str,
        user_message: str,
        *,
        call_label: Optional[str] = None,
    ) -> AsyncIterator[str]: ...

    @property
    def model_id(self) -> str: ...


# --------------------------------------------------------------------------
# Embedder port
# --------------------------------------------------------------------------


@runtime_checkable
class Embedder(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...

    @property
    def dim(self) -> int: ...


# --------------------------------------------------------------------------
# VectorStore port (Milvus adapter + InMemory adapter implement this)
# --------------------------------------------------------------------------


@runtime_checkable
class VectorStore(Protocol):
    """Vector store port covering memories plus narrative-group bookkeeping.

    All filters are the structured MemoryFilter — never raw expressions
    (architecture_target.md #16).
    """

    # -- collection lifecycle ------------------------------------------------

    async def create_collection(self, dim: int) -> None:
        """Create the memories collection if absent (idempotent)."""
        ...

    # -- memory CRUD ----------------------------------------------------------

    async def insert(self, records: list[MemoryRecord]) -> list[int]:
        """Insert records (vector must be set); returns assigned ids."""
        ...

    async def upsert(self, records: list[MemoryRecord]) -> list[int]:
        """Update in place by id (single-step, stable ids — #17)."""
        ...

    async def search(
        self,
        vectors: list[list[float]],
        flt: Optional[MemoryFilter] = None,
        limit: int = 10,
    ) -> list[list[SearchHit]]:
        """Batched similarity search; one result list per query vector."""
        ...

    async def query(
        self,
        flt: MemoryFilter,
        limit: int = 100,
        include_vectors: bool = False,
    ) -> list[MemoryRecord]:
        """Filter query without similarity (centroid recompute uses
        include_vectors=True)."""
        ...

    async def delete(
        self,
        ids: Optional[list[int]] = None,
        flt: Optional[MemoryFilter] = None,
    ) -> int:
        """Delete by ids and/or filter; returns affected count."""
        ...

    async def count(self, flt: Optional[MemoryFilter] = None) -> int:
        """Row count (used by the benchmark's ingest-completeness check)."""
        ...

    # -- narrative groups ------------------------------------------------------

    async def search_groups(
        self, user_id: str, vector: list[float], limit: int = 1
    ) -> list[GroupMatch]:
        """Find nearest group centroids for one embedding."""
        ...

    async def insert_group(
        self, user_id: str, centroid_vector: list[float], size: int = 1
    ) -> Optional[int]: ...

    async def update_group(
        self,
        user_id: str,
        group_id: int,
        centroid_vector: Optional[list[float]] = None,
        size: Optional[int] = None,
    ) -> bool: ...

    async def delete_group(self, user_id: str, group_id: int) -> bool: ...

    async def list_groups(self, user_id: str) -> list[GroupInfo]:
        """All groups of a user (demo narrative panel)."""
        ...

    async def get_group_members(
        self, group_id: int, user_id: str, include_vectors: bool = False
    ) -> list[MemoryRecord]:
        """Members of one group (demo panel + group cleanup)."""
        ...

    async def update_memory_group_id(
        self, memory_id: int, group_id: int, user_id: str
    ) -> bool: ...
