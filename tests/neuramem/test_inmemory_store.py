"""Unit tests for the InMemory VectorStore adapter (implementation plan step 2)."""

import pytest

from neuramem.core.models import MemoryFilter, MemoryRecord
from neuramem.store.inmemory import InMemoryStore, filter_matches


def _record(text, vector=None, **kwargs) -> MemoryRecord:
    params = dict(
        user_id="u1",
        memory_type="episodic",
        ts=1,
        chat_id="c1",
        text=text,
        vector=vector or [1.0, 0.0],
    )
    params.update(kwargs)
    return MemoryRecord(**params)


@pytest.fixture
def store():
    return InMemoryStore()


class TestFilterMatches:
    def test_none_filter_matches_all(self):
        assert filter_matches(_record("x"), None)

    def test_field_constraints(self):
        record = _record("x", user_id="u2", group_id=3, retired=True, metadata={"k": "v"})
        assert filter_matches(record, MemoryFilter(user_id="u2"))
        assert not filter_matches(record, MemoryFilter(user_id="u1"))
        assert filter_matches(record, MemoryFilter(group_id_in=[3, 4]))
        assert not filter_matches(record, MemoryFilter(group_id_in=[9]))
        assert filter_matches(record, MemoryFilter(id_not=999))
        assert filter_matches(record, MemoryFilter(retired=True))
        assert not filter_matches(record, MemoryFilter(retired=False))
        assert filter_matches(record, MemoryFilter(metadata={"k": "v"}))
        assert not filter_matches(record, MemoryFilter(metadata={"k": "other"}))


class TestCrud:
    @pytest.mark.asyncio
    async def test_insert_assigns_ids(self, store):
        ids = await store.insert([_record("a"), _record("b")])
        assert ids == [1, 2]

    @pytest.mark.asyncio
    async def test_upsert_updates_in_place_with_stable_id(self, store):
        ids = await store.insert([_record("original")])
        record_id = ids[0]
        await store.upsert([_record("updated", id=record_id)])
        records = await store.query(MemoryFilter(user_id="u1"))
        assert len(records) == 1
        assert records[0].text == "updated"
        assert records[0].id == record_id

    @pytest.mark.asyncio
    async def test_query_strips_vectors_unless_requested(self, store):
        await store.insert([_record("a", vector=[1.0, 2.0])])
        stripped = await store.query(MemoryFilter(user_id="u1"))
        assert stripped[0].vector is None
        full = await store.query(MemoryFilter(user_id="u1"), include_vectors=True)
        assert full[0].vector == [1.0, 2.0]

    @pytest.mark.asyncio
    async def test_delete_by_filter_and_count(self, store):
        await store.insert([_record("a"), _record("b", user_id="u2")])
        deleted = await store.delete(flt=MemoryFilter(user_id="u2"))
        assert deleted == 1
        assert await store.count() == 1

    @pytest.mark.asyncio
    async def test_delete_without_args_is_noop(self, store):
        """No ids and no filter must never mean 'delete everything'."""
        await store.insert([_record("a"), _record("b")])
        assert await store.delete() == 0
        assert await store.count() == 2


class TestSearch:
    @pytest.mark.asyncio
    async def test_cosine_ranking_with_filter(self, store):
        await store.insert([
            _record("same direction", vector=[1.0, 0.0]),
            _record("orthogonal", vector=[0.0, 1.0]),
            _record("other user", vector=[1.0, 0.0], user_id="u2"),
        ])
        results = await store.search(
            [[1.0, 0.0]], MemoryFilter(user_id="u1", memory_type="episodic"), limit=2
        )
        hits = results[0]
        assert hits[0].record.text == "same direction"
        assert all(h.record.user_id == "u1" for h in hits)
        assert hits[0].distance == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_retired_excluded_via_filter(self, store):
        await store.insert([
            _record("fresh", vector=[1.0, 0.0]),
            _record("stale", vector=[1.0, 0.0], retired=True),
        ])
        results = await store.search([[1.0, 0.0]], MemoryFilter(retired=False), limit=10)
        assert [h.record.text for h in results[0]] == ["fresh"]


class TestGroups:
    @pytest.mark.asyncio
    async def test_group_lifecycle(self, store):
        gid = await store.insert_group("u1", [1.0, 0.0], size=1)
        assert gid is not None
        await store.insert([_record("m1", group_id=gid), _record("m2", group_id=gid)])

        groups = await store.list_groups("u1")
        assert groups == [(gid, 1)] or [(g.group_id, g.size) for g in groups] == [(gid, 1)]

        members = await store.get_group_members(gid, "u1")
        assert {m.text for m in members} == {"m1", "m2"}

        matches = await store.search_groups("u1", [1.0, 0.1])
        assert matches[0].group_id == gid

        assert await store.update_group("u1", gid, size=3) is True
        assert (await store.list_groups("u1"))[0].size == 3

        assert await store.delete_group("u1", gid) is True
        assert await store.list_groups("u1") == []

    @pytest.mark.asyncio
    async def test_groups_isolated_by_user(self, store):
        gid_a = await store.insert_group("ua", [1.0, 0.0])
        gid_b = await store.insert_group("ub", [0.0, 1.0])
        matches = await store.search_groups("ub", [0.0, 1.0])
        assert [m.group_id for m in matches] == [gid_b]

    @pytest.mark.asyncio
    async def test_update_memory_group_id(self, store):
        ids = await store.insert([_record("m1")])
        assert await store.update_memory_group_id(ids[0], 5, "u1") is True
        members = await store.get_group_members(5, "u1")
        assert [m.id for m in members] == ids
