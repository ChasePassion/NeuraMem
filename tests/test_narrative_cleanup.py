"""Unit tests for narrative group cleanup on memory deletion.

Regression tests for the cleanup-timing fix (architecture_target.md #21):
delete_memory_from_group runs BEFORE the store delete, so the members query
must exclude the dying memory -- otherwise the recomputed centroid includes
its vector, the group size is off by one, and the empty-group deletion
branch is unreachable (orphan groups accumulate).
"""

import numpy as np

from src.memory_system.config import MemoryConfig
from src.memory_system.processors.narrative_memory_manager import (
    NarrativeMemoryManager,
)
from src.memory_system.utils import normalize


class FakeStore:
    """Minimal store double dispatching on the filter shapes this manager emits."""

    def __init__(self, rows):
        # rows: dicts with id / group_id / user_id / vector
        self.rows = rows
        self.update_group_calls = []
        self.delete_group_calls = []

    def query(self, filter_expr, output_fields=None):
        if filter_expr.startswith("id =="):
            memory_id = int(filter_expr.split("id == ")[1].split(" ")[0])
            return [r for r in self.rows if r["id"] == memory_id]
        if filter_expr.startswith("group_id =="):
            return [
                r for r in self.rows
                if f"group_id == {r['group_id']}" in filter_expr
                and f"user_id == '{r['user_id']}'" in filter_expr
                and f"id != {r['id']}" not in filter_expr
            ]
        raise AssertionError(f"unexpected filter: {filter_expr}")

    def update_group(self, user_id, group_id, centroid_vector, size):
        self.update_group_calls.append(
            {
                "user_id": user_id,
                "group_id": group_id,
                "centroid": centroid_vector,
                "size": size,
            }
        )

    def delete_group(self, user_id, group_id):
        self.delete_group_calls.append(
            {"user_id": user_id, "group_id": group_id}
        )


def test_delete_last_member_removes_group():
    """Deleting the last member must trigger group deletion (branch was dead)."""
    store = FakeStore(
        [{"id": 7, "group_id": 3, "user_id": "u1", "vector": [1.0, 0.0]}]
    )
    manager = NarrativeMemoryManager(store, MemoryConfig())

    manager.delete_memory_from_group(7, "u1")

    assert store.delete_group_calls == [{"user_id": "u1", "group_id": 3}]
    assert store.update_group_calls == []


def test_recompute_centroid_excludes_dying_memory():
    """Centroid and size must be computed over the remaining members only."""
    dying = {"id": 1, "group_id": 3, "user_id": "u1", "vector": [1.0, 0.0]}
    stay_a = {"id": 2, "group_id": 3, "user_id": "u1", "vector": [0.0, 1.0]}
    stay_b = {"id": 4, "group_id": 3, "user_id": "u1", "vector": [0.0, 1.0]}
    store = FakeStore([dying, stay_a, stay_b])
    manager = NarrativeMemoryManager(store, MemoryConfig())

    manager.delete_memory_from_group(1, "u1")

    assert store.delete_group_calls == []
    assert len(store.update_group_calls) == 1
    call = store.update_group_calls[0]
    assert call["group_id"] == 3
    assert call["size"] == 2  # dying memory must not be counted
    expected = normalize(np.mean([stay_a["vector"], stay_b["vector"]], axis=0))
    assert np.allclose(call["centroid"], expected.tolist())
    # with the bug, the dying [1.0, 0.0] vector tilts the centroid to x != 0
    assert abs(call["centroid"][0]) < 1e-9


def test_unknown_memory_is_noop():
    store = FakeStore([])
    manager = NarrativeMemoryManager(store, MemoryConfig())

    manager.delete_memory_from_group(999, "u1")

    assert store.update_group_calls == []
    assert store.delete_group_calls == []


def test_ungrouped_memory_is_noop():
    store = FakeStore(
        [{"id": 9, "group_id": -1, "user_id": "u1", "vector": [1.0, 0.0]}]
    )
    manager = NarrativeMemoryManager(store, MemoryConfig())

    manager.delete_memory_from_group(9, "u1")

    assert store.update_group_calls == []
    assert store.delete_group_calls == []
