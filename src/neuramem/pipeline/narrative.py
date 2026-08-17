"""Narrative pipeline: cluster used episodic memories into narrative groups.

Migrated from processors/narrative_memory_manager.py onto the VectorStore
port (single groups collection, #15). Behavior preserved:
- assignment is idempotent (already-grouped memories keep their group)
- threshold from RetrievalConfig.narrative_similarity_threshold
- joining an existing group recomputes the centroid from all members
  (exact, not incremental)
- group cleanup on delete excludes the dying memory from the centroid
  recomputation and deletes empty groups (#21 fix, regression-tested in
  the legacy suite)
"""

import logging
from typing import Optional

import numpy as np

from neuramem.config import RetrievalConfig
from neuramem.core.models import MemoryFilter, MemoryRecord
from neuramem.core.ports import VectorStore

logger = logging.getLogger(__name__)

MEMBERS_QUERY_LIMIT = 16_384  # Milvus hard query window cap


def _normalize(vector) -> list[float]:
    array = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(array)
    if norm == 0:
        return array.tolist()
    return (array / norm).tolist()


def _centroid(vectors: list[list[float]]) -> Optional[list[float]]:
    if not vectors:
        return None
    matrix = np.array(vectors, dtype=float)
    return _normalize(np.mean(matrix, axis=0))


class NarrativeManager:
    """Group bookkeeping over the VectorStore port."""

    def __init__(self, store: VectorStore, config: RetrievalConfig):
        self._store = store
        self._config = config

    async def assign_to_narrative_group(
        self, memory_ids: list[int], user_id: str
    ) -> dict[int, int]:
        assignments: dict[int, int] = {}
        for memory_id in memory_ids:
            try:
                assignments.update(await self._assign_one(memory_id, user_id))
            except Exception as e:  # noqa: BLE001 - one memory must not stop the rest
                logger.error(
                    "Failed to assign memory %s to narrative group: %s",
                    memory_id, e,
                )
        return assignments

    async def _assign_one(self, memory_id: int, user_id: str) -> dict[int, int]:
        records = await self._store.query(
            MemoryFilter(user_id=user_id, id_in=[memory_id]),
            limit=1,
            include_vectors=True,
        )
        if not records:
            logger.warning("Memory %s not found for grouping, skipping", memory_id)
            return {}
        record = records[0]
        if record.group_id != -1:
            return {memory_id: record.group_id}

        v_mem = _normalize(record.vector or [])
        group_hits = await self._store.search_groups(user_id, v_mem, limit=1)
        best = group_hits[0] if group_hits else None
        threshold = self._config.narrative_similarity_threshold

        if best is None or best.similarity < threshold:
            group_id = await self._store.insert_group(user_id, v_mem, size=1)
            if group_id is None:
                logger.error("Failed to create group for memory %s", memory_id)
                return {}
            await self._store.update_memory_group_id(memory_id, group_id, user_id)
            logger.info("Created new group %s for memory %s", group_id, memory_id)
            return {memory_id: group_id}

        group_id = best.group_id
        await self._store.update_memory_group_id(memory_id, group_id, user_id)
        # exact centroid recompute over all members (now including the new one)
        members = await self._store.query(
            MemoryFilter(user_id=user_id, group_id=group_id),
            limit=MEMBERS_QUERY_LIMIT,
            include_vectors=True,
        )
        centroid = _centroid([m.vector for m in members if m.vector])
        if centroid is not None:
            await self._store.update_group(
                user_id, group_id, centroid_vector=centroid, size=len(members)
            )
        logger.info(
            "Added memory %s to existing group %s (size: %d)",
            memory_id, group_id, len(members),
        )
        return {memory_id: group_id}

    async def delete_memory_from_group(self, memory_id: int, user_id: str) -> None:
        records = await self._store.query(
            MemoryFilter(user_id=user_id, id_in=[memory_id]),
            limit=1,
        )
        if not records:
            return
        group_id = records[0].group_id
        if group_id == -1:
            return
        # exclude the dying memory: cleanup runs before the store delete,
        # so it still matches the group filter (#21)
        members = await self._store.query(
            MemoryFilter(user_id=user_id, group_id=group_id, id_not=memory_id),
            limit=MEMBERS_QUERY_LIMIT,
            include_vectors=True,
        )
        if not members:
            await self._store.delete_group(user_id, group_id)
            logger.info("Deleted empty group %s", group_id)
            return
        centroid = _centroid([m.vector for m in members if m.vector])
        if centroid is not None:
            await self._store.update_group(
                user_id, group_id, centroid_vector=centroid, size=len(members)
            )
            logger.info(
                "Updated group %s centroid after removal (size: %d)",
                group_id, len(members),
            )

    async def get_group_members(
        self, group_id: int, user_id: str
    ) -> list[MemoryRecord]:
        return await self._store.get_group_members(group_id, user_id)
