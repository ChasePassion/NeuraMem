"""Unit tests for the Milvus filter compiler and telemetry conformance (step 2)."""

import pytest

from neuramem.core.models import MemoryFilter, SpanStatus
from neuramem.store.filters import compile_filter
from neuramem.telemetry.memory import InMemoryTelemetry
from neuramem.telemetry.null import NullTelemetry


class TestCompileFilter:
    def test_none_and_empty(self):
        assert compile_filter(None) == ""
        assert compile_filter(MemoryFilter()) == ""

    def test_string_literals_escaped(self):
        expr = compile_filter(MemoryFilter(user_id="u'1", memory_type="episodic"))
        assert "user_id == 'u\\'1'" in expr
        assert "memory_type == 'episodic'" in expr

    def test_in_lists_and_exclusions(self):
        expr = compile_filter(MemoryFilter(group_id_in=[1, 2], id_not=7))
        assert "group_id in [1, 2]" in expr
        assert "id != 7" in expr

    def test_empty_in_list_matches_nothing(self):
        assert "id == -2" in compile_filter(MemoryFilter(id_in=[]))
        assert "group_id == -2" in compile_filter(MemoryFilter(group_id_in=[]))

    def test_retired_flag(self):
        assert "retired == true" in compile_filter(MemoryFilter(retired=True))
        assert "retired == false" in compile_filter(MemoryFilter(retired=False))

    def test_metadata_filter(self):
        expr = compile_filter(MemoryFilter(metadata={"character_id": 5, "scene": "forest"}))
        assert "character_id == 5" in expr
        assert "scene == 'forest'" in expr

    def test_metadata_key_injection_rejected(self):
        with pytest.raises(ValueError):
            compile_filter(MemoryFilter(metadata={"a == 1 or true": 1}))

    def test_metadata_non_scalar_value_rejected(self):
        """str(dict) literals would never match — reject, don't go silent."""
        with pytest.raises(ValueError, match="scalar"):
            compile_filter(MemoryFilter(metadata={"tags": ["a", "b"]}))


# ---------------------------------------------------------------------------
# Telemetry conformance (run against every adapter)
# ---------------------------------------------------------------------------

TELEMETRY_FACTORIES = [
    ("null", NullTelemetry),
    ("memory", InMemoryTelemetry),
]


async def _run_with_telemetry(telemetry, body, fail=False):
    async with telemetry.start_span("op", attributes={"k": "v"}) as span:
        span.add_event("mid")
        if fail:
            raise ValueError("business error")
        return body()


class TestTelemetryConformance:
    @pytest.mark.parametrize("name,factory", TELEMETRY_FACTORIES)
    @pytest.mark.asyncio
    async def test_span_body_runs_exactly_once(self, name, factory):
        telemetry = factory()
        calls = {"n": 0}

        def body():
            calls["n"] += 1
            return "ok"

        assert await _run_with_telemetry(telemetry, body) == "ok"
        assert calls["n"] == 1

    @pytest.mark.parametrize("name,factory", TELEMETRY_FACTORIES)
    @pytest.mark.asyncio
    async def test_business_exception_propagates_unchanged(self, name, factory):
        telemetry = factory()
        with pytest.raises(ValueError, match="business error"):
            await _run_with_telemetry(telemetry, lambda: "x", fail=True)

    @pytest.mark.parametrize("name,factory", TELEMETRY_FACTORIES)
    @pytest.mark.asyncio
    async def test_settled_span_calls_are_inert(self, name, factory):
        telemetry = factory()
        handle_holder = {}

        async with telemetry.start_span("op") as span:
            handle_holder["span"] = span

        # post-settlement calls must not raise
        handle_holder["span"].add_event("late")
        handle_holder["span"].set_attributes({"late": True})
        handle_holder["span"].set_status(SpanStatus(status="ok"))

    @pytest.mark.asyncio
    async def test_inmemory_records_ok_error_and_nesting(self):
        telemetry = InMemoryTelemetry()

        async with telemetry.start_span("outer") as outer:
            outer.set_attributes({"phase": "1"})
            async with telemetry.start_span("inner") as inner:
                inner.add_event("deep")
            try:
                async with telemetry.start_span("bad"):
                    raise RuntimeError("x")
            except RuntimeError:
                pass
            outer.set_status(SpanStatus(status="ok"))

        spans = {s.name: s for s in telemetry.get_spans()}
        assert spans["outer"].status.status == "ok"  # explicit wins
        assert spans["outer"].attributes == {"phase": "1"}
        assert spans["inner"].parent_id == spans["outer"].id
        assert spans["inner"].events[0]["name"] == "deep"
        assert spans["bad"].final_status.status == "error"  # auto error
