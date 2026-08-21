"""Behavioral tests for same-event narrative grouping."""

import json

import pytest

from neuramem.config import RetrievalConfig
from neuramem.core.models import MemoryFilter, MemoryRecord
from neuramem.core.ports import LLMJsonResult
from neuramem.pipeline.narrative import NarrativeManager
from neuramem.store.inmemory import InMemoryStore


class ScriptedNarrativeLLM:
    def __init__(self, decision):
        self._decision = decision
        self.payloads = []

    async def complete_json(
        self, system_prompt, user_message, default=None, *, call_label=None
    ):
        payload = json.loads(user_message)
        self.payloads.append((system_prompt, payload, call_label))
        assignments = []
        for item in payload["new_memories"]:
            matched_group_id = self._decision(payload)
            assignments.append({
                "memory_id": item["memory"]["id"],
                "matched_group_id": matched_group_id,
                "new_group_key": None if matched_group_id is not None else "new-1",
            })
        data = {"assignments": assignments}
        return LLMJsonResult(
            parsed_data=data,
            raw_response=json.dumps(data),
            model="fake",
            success=True,
        )


def _record(record_id, text, vector, group_id=-1):
    return MemoryRecord(
        id=record_id,
        user_id="u1",
        memory_type="episodic",
        ts=1,
        chat_id="c1",
        text=text,
        vector=vector,
        group_id=group_id,
    )


@pytest.mark.asyncio
async def test_prompt_decides_same_group_without_similarity_threshold():
    store = InMemoryStore()
    group_id = await store.insert_group("u1", [1.0, 0.0], size=1)
    await store.insert([
        _record(1, "The user adopted Pixie.", [1.0, 0.0], group_id),
        _record(2, "The user brought Pixie home.", [0.0, 1.0]),
    ])

    llm = ScriptedNarrativeLLM(lambda payload: payload["candidate_groups"][0]["group_id"])
    manager = NarrativeManager(
        store, RetrievalConfig(_env_file=None), llm
    )

    assignments = await manager.assign_to_narrative_group([2], "u1")

    assert assignments == {2: group_id}
    stored = await store.query(
        MemoryFilter(user_id="u1", id_in=[2]),
        limit=1,
    )
    assert stored[0].group_id == group_id
    assert len(llm.payloads) == 1
    _, payload, call_label = llm.payloads[0]
    assert call_label == "narrative"
    assert "vector" not in json.dumps(payload)
    assert "score" not in json.dumps(payload)


@pytest.mark.asyncio
async def test_used_memories_are_grouped_in_one_prompt_and_one_writeback():
    store = InMemoryStore()
    group_id = await store.insert_group("u1", [1.0, 0.0], size=1)
    await store.insert([
        _record(1, "The user adopted Pixie.", [1.0, 0.0], group_id),
        _record(2, "The user brought Pixie home.", [0.0, 1.0]),
        _record(3, "The user thanked the shelter.", [0.0, 1.0]),
    ])

    llm = ScriptedNarrativeLLM(lambda payload: group_id)
    manager = NarrativeManager(
        store, RetrievalConfig(_env_file=None), llm
    )

    assignments = await manager.assign_to_narrative_group([2, 3], "u1")

    assert assignments == {2: group_id, 3: group_id}
    assert len(llm.payloads) == 1
    payload = llm.payloads[0][1]
    assert [item["memory"]["id"] for item in payload["new_memories"]] == [2, 3]


@pytest.mark.asyncio
async def test_ambiguous_prompt_result_creates_new_group():
    store = InMemoryStore()
    existing_group_id = await store.insert_group("u1", [1.0, 0.0], size=1)
    await store.insert([
        _record(
            1,
            "The user attended a dog grooming course.",
            [1.0, 0.0],
            existing_group_id,
        ),
        _record(
            2,
            "The user attended obedience training.",
            [1.0, 0.0],
        ),
    ])

    llm = ScriptedNarrativeLLM(lambda payload: None)
    manager = NarrativeManager(
        store, RetrievalConfig(_env_file=None), llm
    )

    assignments = await manager.assign_to_narrative_group([2], "u1")

    assert assignments[2] != existing_group_id
    assert len(await store.list_groups("u1")) == 2
