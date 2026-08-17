"""Unit tests for the neuramem core domain models (implementation plan step 1)."""

from neuramem.core.models import (
    ConsolidationStats,
    LLMUsage,
    MemoryFilter,
    MemoryRecord,
    SearchResult,
    UsageReport,
)


def _record(text: str, **kwargs) -> MemoryRecord:
    return MemoryRecord(text=text, **kwargs)


class TestSearchResultRender:
    def test_render_no_truncation_by_default(self):
        """Defaults render everything search returned (W3 semantics: full semantic injection)."""
        result = SearchResult(
            query="q",
            user_id="u1",
            episodic=[_record(f"episodic {i}") for i in range(6)],
            semantic=[_record(f"semantic {i}") for i in range(8)],
        )
        rendered = result.render()
        assert "episodic 5" in rendered  # all 6 episodic lines present
        assert "semantic 7" in rendered  # all 8 semantic lines present

    def test_render_respects_explicit_caps(self):
        result = SearchResult(
            query="q",
            user_id="u1",
            episodic=[_record(f"episodic {i}") for i in range(6)],
            semantic=[_record(f"semantic {i}") for i in range(8)],
        )
        rendered = result.render(max_episodic=2, max_semantic=3)
        assert "episodic 1" in rendered
        assert "episodic 2" not in rendered
        assert "semantic 2" in rendered
        assert "semantic 3" not in rendered

    def test_render_empty_memory_blocks(self):
        rendered = SearchResult(query="q", user_id="u1").render()
        assert "(No episodic memories)" in rendered
        assert "(No semantic memories)" in rendered

    def test_render_uses_legacy_block_headers_and_numbering(self):
        """Headers and 1-based numbering mirror the legacy server context builder."""
        result = SearchResult(
            query="q",
            user_id="u1",
            episodic=[_record("went to Hangzhou")],
            semantic=[_record("user lives in Beijing")],
        )
        rendered = result.render()
        assert "Here are the episodic memories:\n1. went to Hangzhou" in rendered
        assert "Here are the semantic memories:\n1. user lives in Beijing" in rendered


class TestLLMUsage:
    def test_hit_rate_none_without_prompt_tokens(self):
        assert LLMUsage(output_tokens=5).hit_rate() is None

    def test_hit_rate_token_weighted(self):
        usage = LLMUsage(
            input_tokens=60, cache_read_tokens=30, cache_write_tokens=10
        )
        assert usage.prompt_tokens == 100
        assert abs(usage.hit_rate() - 0.30) < 1e-9


class TestMemoryFilter:
    def test_is_empty_when_no_constraints(self):
        assert MemoryFilter().is_empty()

    def test_is_empty_false_with_constraints(self):
        assert not MemoryFilter(user_id="u1").is_empty()
        assert not MemoryFilter(group_id_in=[1, 2]).is_empty()


class TestValueObjects:
    def test_usage_report_defaults(self):
        report = UsageReport()
        assert report.judged_candidates == 0
        assert report.used_memory_ids == []
        assert report.assignments == {}

    def test_consolidation_stats_defaults(self):
        stats = ConsolidationStats()
        assert stats.memories_processed == 0
        assert stats.semantic_created == 0

    def test_memory_record_optional_fields_default(self):
        record = MemoryRecord()
        assert record.group_id == -1
        assert record.retired is False
        assert record.metadata is None
        assert record.vector is None
