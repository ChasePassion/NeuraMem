"""Domain models shared across the three layers (architecture_target.md #6).

Pure Pydantic value objects — no IO imports. The store port exchanges
these instead of raw dicts; the public API returns these; the server layer
serializes these.
"""

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class MemoryRecord(BaseModel):
    """A single memory row in the vector store.

    Schema fields (Milvus-compatible, unchanged from the legacy store):
    - id: record ID (store-assigned on insert)
    - user_id / memory_type ("episodic" | "semantic") / ts (unix write time)
    - chat_id: conversation identifier
    - text: main natural-language content
    - group_id: narrative group ID (-1 = ungrouped)
    - metadata: caller-owned passthrough fields (architecture_target.md 8.2);
      written to dynamic fields, filterable, never interpreted by the library
    - vector: embedding; populated only when explicitly requested from the
      store (e.g. centroid recomputation), never rendered or serialized by
      the public API paths
    """

    id: int = 0
    user_id: str = ""
    memory_type: str = ""
    ts: int = 0
    chat_id: str = ""
    text: str = ""
    group_id: int = -1
    metadata: Optional[dict[str, Any]] = None
    vector: Optional[list[float]] = None


class MemoryFilter(BaseModel):
    """Structured store filter (architecture_target.md #16).

    Adapters compile this into their native query language; the library
    never builds filter expressions by string concatenation. All fields
    are AND-combined; unset fields do not constrain.
    """

    user_id: Optional[str] = None
    memory_type: Optional[str] = None
    group_id: Optional[int] = None
    group_id_in: Optional[list[int]] = None
    id_in: Optional[list[int]] = None
    id_not: Optional[int] = None
    retired: Optional[bool] = None
    metadata: Optional[dict[str, Any]] = None

    def is_empty(self) -> bool:
        return self.model_dump(exclude_none=True) == {}


class SearchHit(BaseModel):
    """One vector-search hit: the record plus its similarity distance."""

    record: MemoryRecord
    distance: Optional[float] = None


class LLMUsage(BaseModel):
    """Structured usage of one LLM call (architecture_target.md 6.5).

    Field mapping follows pi-mono parseChunkUsage semantics:
    - input_tokens = prompt_tokens - cache_read - cache_write (net new input)
    - output_tokens = completion_tokens (already includes reasoning tokens)
    - cache_read = prompt_tokens_details.cached_tokens (or provider-specific
      top-level field); cache_read is NOT reduced by cache_write
    - total_tokens = input + output + cache_read + cache_write
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0

    @property
    def prompt_tokens(self) -> int:
        """All prompt tokens: net-new input + cache reads + cache writes."""
        return self.input_tokens + self.cache_read_tokens + self.cache_write_tokens

    def hit_rate(self) -> Optional[float]:
        """Token-weighted prefix-cache hit rate over all prompt tokens.

        Returns None when no prompt tokens were reported (provider does not
        expose cache details).
        """
        prompt = self.prompt_tokens
        if prompt <= 0:
            return None
        return self.cache_read_tokens / prompt


class SearchResult(BaseModel):
    """Correlation token of the two-phase closed loop (architecture_target.md ch. 11).

    Returned by Memory.search; passed back verbatim to
    Memory.report_usage. Pure data — serializable, holds no reference to
    the Memory instance.
    """

    query: str
    user_id: str
    episodic: list[MemoryRecord] = Field(default_factory=list)
    semantic: list[MemoryRecord] = Field(default_factory=list)

    def render(
        self,
        max_episodic: Optional[int] = None,
        max_semantic: Optional[int] = None,
    ) -> str:
        """Format the memory block for an answer prompt (pure, no IO).

        Defaults render everything search returned — the retrieval config
        owns how much is retrieved (single responsibility); consumers may
        cap explicitly (e.g. the server keeps its legacy [:5] truncation).
        Conversation history and the user message are NOT rendered here:
        they are consumer-owned session state.
        """
        parts = ["Here are the episodic memories:"]
        episodic = self.episodic if max_episodic is None else self.episodic[:max_episodic]
        if episodic:
            parts.extend(f"{i}. {m.text}" for i, m in enumerate(episodic, 1))
        else:
            parts.append("(No episodic memories)")
        parts.append("")
        parts.append("Here are the semantic memories:")
        semantic = self.semantic if max_semantic is None else self.semantic[:max_semantic]
        if semantic:
            parts.extend(f"{i}. {m.text}" for i, m in enumerate(semantic, 1))
        else:
            parts.append("(No semantic memories)")
        return "\n".join(parts)


class UsageReport(BaseModel):
    """Outcome of Memory.report_usage (the judge -> assign write-back)."""

    judged_candidates: int = 0
    used_memory_ids: list[int] = Field(default_factory=list)
    assignments: dict[int, int] = Field(default_factory=dict)


class ConsolidationStats(BaseModel):
    """Statistics from a consolidation run."""

    memories_processed: int = 0
    semantic_created: int = 0


class GroupMatch(BaseModel):
    """A narrative-group centroid match returned by the store."""

    group_id: int
    similarity: float
    size: int


class GroupInfo(BaseModel):
    """A narrative group listing entry (demo panel / observability)."""

    group_id: int
    size: int


class SpanStatus(BaseModel):
    """Terminal status of a telemetry span."""

    status: Literal["ok", "error"] = "ok"
    error_message: Optional[str] = None
