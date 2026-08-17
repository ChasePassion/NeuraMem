"""Full-flow pipeline tests with InMemoryStore + scripted fakes (step 3).

These tests are the behavioral contract of the facade: manage -> search ->
report_usage closed loop, conflict elimination, update/delete semantics.
"""

import json

import pytest

from neuramem.config import (
    EmbeddingConfig,
    LLMConfig,
    MemoryConfig,
    RetrievalConfig,
    StoreConfig,
)
from neuramem.core.models import MemoryFilter
from neuramem.core.ports import LLMJsonResult, LLMResponse
from neuramem.memory import Memory
from neuramem.store.inmemory import InMemoryStore

DIM = 4

KEYWORD_VECTORS = {
    "hangzhou": [1.0, 0.0, 0.0, 0.0],
    "west lake": [0.95, 0.1, 0.0, 0.0],
    "beijing": [0.0, 1.0, 0.0, 0.0],
}
DEFAULT_VECTOR = [0.0, 0.0, 1.0, 0.0]


class FakeEmbedder:
    def __init__(self, dim: int = DIM):
        self._dim = dim

    async def embed(self, texts):
        out = []
        for text in texts:
            vector = DEFAULT_VECTOR
            for keyword, kw_vector in KEYWORD_VECTORS.items():
                if keyword in text.lower():
                    vector = kw_vector
                    break
            out.append(list(map(float, vector)))
        return out

    @property
    def dim(self) -> int:
        return self._dim


class ScriptedLLM:
    """LLM port double routing complete_json by call_label."""

    def __init__(self):
        self.scripts = {}
        self.calls = []

    def on_json(self, label, fn):
        self.scripts[label] = fn

    async def complete(self, system_prompt, user_message, *, call_label=None):
        return LLMResponse(content="ok")

    async def complete_json(
        self, system_prompt, user_message, default=None, *, call_label=None
    ):
        self.calls.append((call_label, json.loads(user_message)))
        fn = self.scripts.get(call_label, lambda payload: {})
        data = fn(json.loads(user_message))
        return LLMJsonResult(
            parsed_data=data,
            raw_response=json.dumps(data),
            model="fake",
            success=True,
        )

    async def stream(self, system_prompt, user_message, *, call_label=None):
        yield "ok"

    @property
    def model_id(self) -> str:
        return "fake"


def _make_memory(llm=None, embedder=None, store=None, **retrieval_overrides):
    retrieval = RetrievalConfig(_env_file=None, **retrieval_overrides)
    config = MemoryConfig(
        llm=LLMConfig(base_url="https://x", api_key="k", model="m", _env_file=None),
        embedding=EmbeddingConfig(api_key="k", dim=DIM, _env_file=None),
        store=StoreConfig(uri="memory://test", _env_file=None),
        retrieval=retrieval,
    )
    return Memory(
        config,
        llm=llm or ScriptedLLM(),
        embedder=embedder or FakeEmbedder(),
        store=store or InMemoryStore(),
    )


def _default_manage_script(payload):
    ops = {"add": [], "update": [], "delete": []}
    user_text = payload["current_turn"]["user"]
    if "hangzhou" in user_text.lower():
        ops["add"].append({"text": "The user visited Hangzhou on a business trip."})
        ops["add"].append({"text": "The user enjoyed the West Lake in Hangzhou."})
    if "beijing" in user_text.lower():
        ops["add"].append({"text": "The user lives in Beijing."})
    return ops


def _judge_hangzhou_script(payload):
    ids = [m["id"] for m in payload["episodic_memories"] if "Hangzhou" in m["text"]]
    return {"used_episodic_memory_ids": ids}


class TestManage:
    @pytest.mark.asyncio
    async def test_manage_adds_with_metadata_passthrough(self):
        llm = ScriptedLLM()
        llm.on_json("manage", _default_manage_script)
        store = InMemoryStore()
        memory = _make_memory(llm=llm, store=store)

        added = await memory.manage_async(
            user_text="I went to Hangzhou last week.",
            assistant_text="That sounds wonderful!",
            user_id="u1",
            chat_id="c1",
            metadata={"character_id": 7},
        )
        assert len(added) == 2
        records = await store.query(MemoryFilter(user_id="u1", memory_type="episodic"))
        assert all(r.metadata == {"character_id": 7} for r in records)
        assert all(r.group_id == -1 for r in records)

    @pytest.mark.asyncio
    async def test_manage_update_keeps_id_and_resets_group(self):
        llm = ScriptedLLM()
        llm.on_json("manage", _default_manage_script)
        store = InMemoryStore()
        memory = _make_memory(llm=llm, store=store)
        await memory.manage_async(
            "I went to Hangzhou last week.", "nice", "u1", "c1"
        )

        def update_script(payload):
            target = payload["episodic_memories"][0]
            return {
                "add": [],
                "update": [{
                    "id": target["id"],
                    "old_text": target["text"],
                    "new_text": "The user visited Hangzhou twice on business trips.",
                }],
                "delete": [],
            }

        llm.on_json("manage", update_script)
        await memory.manage_async(
            "Actually I went to Hangzhou twice.", "ok", "u1", "c1"
        )
        records = await store.query(MemoryFilter(user_id="u1"))
        assert len(records) == 2  # no duplicate rows
        texts = {r.text for r in records}
        assert "The user visited Hangzhou twice on business trips." in texts

    @pytest.mark.asyncio
    async def test_update_cleans_group_membership(self):
        """Updating a grouped memory must drop it from its group (centroid,
        size, and empty-group deletion) — not just reset its group_id."""
        llm = ScriptedLLM()
        llm.on_json("manage", _default_manage_script)
        llm.on_json("usage_judge", _judge_hangzhou_script)
        store = InMemoryStore()
        memory = _make_memory(llm=llm, store=store, k_episodic=5)
        await memory.manage_async("I went to Hangzhou last week.", "nice", "u1", "c1")
        result = await memory.search_async("hangzhou", "u1")
        await memory.report_usage_async(result, "You visited Hangzhou!")
        groups = await store.list_groups("u1")
        assert len(groups) == 1 and groups[0].size == 2

        hangzhou = [
            r for r in await store.query(MemoryFilter(user_id="u1", memory_type="episodic"))
            if "Hangzhou" in r.text
        ]

        def update_first(payload):
            target = payload["episodic_memories"][0]
            return {
                "add": [],
                "update": [{
                    "id": target["id"],
                    "old_text": target["text"],
                    "new_text": "The user visited Hangzhou twice on business trips.",
                }],
                "delete": [],
            }

        llm.on_json("manage", update_first)
        await memory.manage_async("I went twice actually.", "ok", "u1", "c1")
        # group must no longer count the updated member
        groups_after = await store.list_groups("u1")
        assert len(groups_after) == 1 and groups_after[0].size == 1


class TestSearchAndClosedLoop:
    @pytest.mark.asyncio
    async def test_two_phase_loop_forms_group_and_expands(self):
        llm = ScriptedLLM()
        llm.on_json("manage", _default_manage_script)
        llm.on_json("usage_judge", _judge_hangzhou_script)
        store = InMemoryStore()

        wide = _make_memory(llm=llm, store=store, k_episodic=5)
        await wide.manage_async("I went to Hangzhou last week.", "nice", "u1", "c1")
        await wide.manage_async("I live in Beijing.", "ok", "u1", "c1")

        # wide retrieval: all three memories are candidates; the judge marks
        # both Hangzhou memories as used -> they cluster into one group
        result = await wide.search_async("hangzhou", "u1")
        assert len(result.episodic) == 3  # 2 hangzhou + 1 beijing seeds

        report = await wide.report_usage_async(result, "You visited Hangzhou!")
        assert report.judged_candidates == 3
        assert len(report.used_memory_ids) == 2
        assert len(report.assignments) == 2
        groups = await store.list_groups("u1")
        assert len(groups) == 1 and groups[0].size == 2

        # narrow retrieval (k=1): only the top seed matches, narrative
        # expansion must supply the other group member
        narrow = _make_memory(llm=llm, store=store, k_episodic=1)
        expanded = await narrow.search_async("hangzhou", "u1")
        assert len(expanded.episodic) == 2
        assert {r.text for r in expanded.episodic} == {
            "The user visited Hangzhou on a business trip.",
            "The user enjoyed the West Lake in Hangzhou.",
        }

    @pytest.mark.asyncio
    async def test_search_result_render_reflects_content(self):
        llm = ScriptedLLM()
        llm.on_json("manage", _default_manage_script)
        memory = _make_memory(llm=llm)
        await memory.manage_async("I went to Hangzhou last week.", "nice", "u1", "c1")
        result = await memory.search_async("hangzhou", "u1")
        rendered = result.render()
        assert "The user visited Hangzhou on a business trip." in rendered

    @pytest.mark.asyncio
    async def test_report_usage_isolated_on_judge_crash(self):
        class ExplodingJudge(ScriptedLLM):
            async def complete_json(self, *args, **kwargs):
                raise RuntimeError("judge exploded")

        llm = ExplodingJudge()
        memory = _make_memory(llm=llm)
        result = await memory.search_async("hangzhou", "u1")

        report = await memory.report_usage_async(result, "answer")
        assert report.used_memory_ids == []  # swallowed, not raised

    @pytest.mark.asyncio
    async def test_report_usage_skips_llm_on_empty_candidates(self):
        llm = ScriptedLLM()
        memory = _make_memory(llm=llm)
        from neuramem.core.models import SearchResult

        report = await memory.report_usage_async(
            SearchResult(query="q", user_id="u1"), "answer"
        )
        assert report.judged_candidates == 0
        assert llm.calls == []  # no LLM call spent


class TestConsolidate:
    @pytest.mark.asyncio
    async def test_conflict_elimination_retires_stale_semantic(self):
        llm = ScriptedLLM()
        llm.on_json("manage", _default_manage_script)
        store = InMemoryStore()
        memory = _make_memory(llm=llm, store=store)
        await memory.manage_async("I went to Hangzhou last week.", "nice", "u1", "c1")

        # seed one stale semantic memory, then consolidate with retirement
        seeded = await store.insert([
            __import__("neuramem.core.models", fromlist=["MemoryRecord"]).MemoryRecord(
                user_id="u1", memory_type="semantic", ts=1, chat_id="c1",
                text="The user lives in Shanghai.", vector=[0.0, 0.0, 1.0, 0.0],
            )
        ])

        def consolidate_script(payload):
            retire = [
                e["id"] for e in payload["existing_semantic"]
                if "Shanghai" in e["text"]
            ]
            return {
                "write_semantic": True,
                "facts": ["The user often travels for work."],
                "retired_semantic_ids": retire,
            }

        llm.on_json("consolidate", consolidate_script)
        stats = await memory.consolidate_async("u1")
        assert stats.semantic_created == 1
        assert stats.memories_processed == 2

        # retired memory is filtered from retrieval but still stored
        semantic = await memory.search_async("user profile", "u1")
        texts = {r.text for r in semantic.semantic}
        assert "The user lives in Shanghai." not in texts
        assert "The user often travels for work." in texts
        all_semantic = await store.query(MemoryFilter(user_id="u1", memory_type="semantic"))
        assert len(all_semantic) == 2  # tombstoned row physically present
        assert any(r.retired and r.id == seeded[0] for r in all_semantic)

    @pytest.mark.asyncio
    async def test_consolidate_parse_failure_is_conservative_noop(self):
        llm = ScriptedLLM()
        llm.on_json("manage", _default_manage_script)
        memory = _make_memory(llm=llm)
        await memory.manage_async("I went to Hangzhou last week.", "nice", "u1", "c1")

        class BadJsonLLM(ScriptedLLM):
            async def complete_json(self, *args, **kwargs):
                return LLMJsonResult(
                    parsed_data={"write_semantic": False, "facts": [], "retired_semantic_ids": []},
                    raw_response="garbage", model="fake", success=False,
                )

        memory._semantic = __import__(
            "neuramem.pipeline.semantic", fromlist=["SemanticWriter"]
        ).SemanticWriter(BadJsonLLM())
        stats = await memory.consolidate_async("u1")
        assert stats.semantic_created == 0

    @pytest.mark.asyncio
    async def test_garbage_retire_ids_ignored_without_crash(self):
        """Model garbage in retired_semantic_ids must not crash consolidation."""
        llm = ScriptedLLM()
        llm.on_json("manage", _default_manage_script)
        store = InMemoryStore()
        memory = _make_memory(llm=llm, store=store)
        await memory.manage_async("I went to Hangzhou last week.", "nice", "u1", "c1")
        from neuramem.core.models import MemoryRecord

        seeded = await store.insert([
            MemoryRecord(user_id="u1", memory_type="semantic", ts=1, chat_id="c1",
                         text="The user lives in Shanghai.", vector=[0.0, 0.0, 1.0, 0.0]),
        ])

        def garbage_script(payload):
            return {
                "write_semantic": False,
                "facts": [],
                "retired_semantic_ids": [None, "abc", {"x": 1}, seeded[0], "not-an-int"],
            }

        llm.on_json("consolidate", garbage_script)
        stats = await memory.consolidate_async("u1")  # must not raise
        assert stats.semantic_created == 0
        all_semantic = await store.query(MemoryFilter(user_id="u1", memory_type="semantic"))
        retired = [r for r in all_semantic if r.retired]
        assert [r.id for r in retired] == [seeded[0]]  # only the valid id applied


class TestDeleteAndReset:
    @pytest.mark.asyncio
    async def test_delete_last_member_removes_group(self):
        llm = ScriptedLLM()
        llm.on_json("manage", _default_manage_script)
        llm.on_json("usage_judge", _judge_hangzhou_script)
        store = InMemoryStore()
        memory = _make_memory(llm=llm, store=store, k_episodic=5)
        await memory.manage_async("I went to Hangzhou last week.", "nice", "u1", "c1")
        result = await memory.search_async("hangzhou", "u1")
        await memory.report_usage_async(result, "You visited Hangzhou!")
        assert len(await store.list_groups("u1")) == 1  # group of 2

        records = await store.query(MemoryFilter(user_id="u1", memory_type="episodic"))
        hangzhou_records = [r for r in records if "Hangzhou" in r.text]
        assert len(hangzhou_records) == 2

        await memory.delete_async(hangzhou_records[0].id, "u1")
        assert len(await store.list_groups("u1")) == 1  # group survives with 1
        await memory.delete_async(hangzhou_records[1].id, "u1")
        assert await store.list_groups("u1") == []  # last member -> group gone

        assert await memory.delete_async(999999, "u1") is False

    @pytest.mark.asyncio
    async def test_reset_removes_memories_and_groups(self):
        llm = ScriptedLLM()
        llm.on_json("manage", _default_manage_script)
        llm.on_json("usage_judge", _judge_hangzhou_script)
        store = InMemoryStore()
        memory = _make_memory(llm=llm, store=store, k_episodic=1)
        await memory.manage_async("I went to Hangzhou last week.", "nice", "u1", "c1")
        result = await memory.search_async("hangzhou", "u1")
        await memory.report_usage_async(result, "You visited Hangzhou!")

        deleted = await memory.reset_async("u1")
        assert deleted == 2
        assert await store.list_groups("u1") == []


class TestSyncWrappers:
    def test_sync_wrappers_work_outside_event_loop(self):
        llm = ScriptedLLM()
        llm.on_json("manage", _default_manage_script)
        memory = _make_memory(llm=llm)

        added = memory.manage("I went to Hangzhou last week.", "nice", "u1", "c1")
        assert len(added) == 2
        result = memory.search("hangzhou", "u1")
        assert len(result.episodic) == 2
