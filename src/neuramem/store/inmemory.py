"""In-memory VectorStore adapter — zero external dependencies (#10).

Reference implementation of the port: brute-force cosine search, Python
filter evaluation, in-process group bookkeeping. Used by unit/property
tests and as the behavioral spec the Milvus adapter must match.
"""

import math
from typing import Optional

from neuramem.core.models import (
    GroupInfo,
    GroupMatch,
    MemoryFilter,
    MemoryRecord,
    SearchHit,
)


def filter_matches(record: MemoryRecord, flt: Optional[MemoryFilter]) -> bool:
    """Evaluate a MemoryFilter against a record (Python twin of compile_filter)."""
    if flt is None:
        return True
    if flt.user_id is not None and record.user_id != flt.user_id:
        return False
    if flt.memory_type is not None and record.memory_type != flt.memory_type:
        return False
    if flt.group_id is not None and record.group_id != flt.group_id:
        return False
    if flt.group_id_in is not None and record.group_id not in flt.group_id_in:
        return False
    if flt.id_in is not None and record.id not in flt.id_in:
        return False
    if flt.id_not is not None and record.id == flt.id_not:
        return False
    if flt.retired is not None and record.retired != flt.retired:
        return False
    if flt.metadata is not None:
        meta = record.metadata or {}
        for key, value in flt.metadata.items():
            if key not in meta or meta[key] != value:
                return False
    return True


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class InMemoryStore:
    """VectorStore over plain dicts. Not for production — data is
    process-local and dies with it."""

    def __init__(self) -> None:
        self._memories: dict[int, MemoryRecord] = {}
        self._next_id = 1
        # group_id -> {"user_id", "centroid_vector", "size"}
        self._groups: dict[int, dict] = {}
        self._next_group_id = 1
        self._dim: Optional[int] = None

    # -- lifecycle ---------------------------------------------------------

    async def create_collection(self, dim: int) -> None:
        self._dim = dim

    # -- memory CRUD ---------------------------------------------------------

    async def insert(self, records: list[MemoryRecord]) -> list[int]:
        """Insert records; returns assigned ids.

        Divergence from the Milvus adapter: an explicit record.id > 0 is
        honored here (Milvus auto_id always assigns fresh ids on insert).
        Test convenience only — production paths never pass ids to insert.
        """
        ids = []
        for record in records:
            record_id = record.id if record.id > 0 else self._next_id
            self._next_id = max(self._next_id, record_id + 1)
            stored = record.model_copy()
            stored.id = record_id
            self._memories[record_id] = stored
            ids.append(record_id)
        return ids

    async def upsert(self, records: list[MemoryRecord]) -> list[int]:
        ids = []
        for record in records:
            record_id = record.id if record.id > 0 else self._next_id
            self._next_id = max(self._next_id, record_id + 1)
            stored = record.model_copy()
            stored.id = record_id
            self._memories[record_id] = stored
            ids.append(record_id)
        return ids

    async def search(
        self,
        vectors: list[list[float]],
        flt: Optional[MemoryFilter] = None,
        limit: int = 10,
    ) -> list[list[SearchHit]]:
        results: list[list[SearchHit]] = []
        for vector in vectors:
            scored = [
                (_cosine(vector, r.vector or []), r)
                for r in self._memories.values()
                if filter_matches(r, flt) and r.vector is not None
            ]
            scored.sort(key=lambda pair: pair[0], reverse=True)
            results.append(
                [
                    SearchHit(record=record.model_copy(), distance=1.0 - similarity)
                    for similarity, record in scored[:limit]
                ]
            )
        return results

    async def query(
        self,
        flt: MemoryFilter,
        limit: int = 100,
        include_vectors: bool = False,
    ) -> list[MemoryRecord]:
        matches = [r for r in self._memories.values() if filter_matches(r, flt)]
        matches.sort(key=lambda r: r.id)
        matches = matches[:limit]
        if include_vectors:
            return [r.model_copy() for r in matches]
        return [r.model_copy(update={"vector": None}) for r in matches]

    async def delete(
        self,
        ids: Optional[list[int]] = None,
        flt: Optional[MemoryFilter] = None,
    ) -> int:
        if ids is None and flt is None:
            # no-op, matching the Milvus adapter (empty expression deletes
            # nothing) — deleting everything by accident must be impossible
            return 0
        doomed = [
            r.id
            for r in self._memories.values()
            if (ids is None or r.id in ids) and filter_matches(r, flt)
        ]
        for record_id in doomed:
            self._memories.pop(record_id, None)
        return len(doomed)

    async def count(self, flt: Optional[MemoryFilter] = None) -> int:
        return sum(1 for r in self._memories.values() if filter_matches(r, flt))

    # -- narrative groups ------------------------------------------------------

    async def search_groups(
        self, user_id: str, vector: list[float], limit: int = 1
    ) -> list[GroupMatch]:
        scored = []
        for group_id, group in self._groups.items():
            if group["user_id"] != user_id:
                continue
            similarity = _cosine(vector, group["centroid_vector"])
            scored.append(GroupMatch(group_id=group_id, similarity=similarity, size=group["size"]))
        scored.sort(key=lambda m: m.similarity, reverse=True)
        return scored[:limit]

    async def insert_group(
        self, user_id: str, centroid_vector: list[float], size: int = 1
    ) -> Optional[int]:
        group_id = self._next_group_id
        self._next_group_id += 1
        self._groups[group_id] = {
            "user_id": user_id,
            "centroid_vector": list(centroid_vector),
            "size": size,
        }
        return group_id

    async def update_group(
        self,
        user_id: str,
        group_id: int,
        centroid_vector: Optional[list[float]] = None,
        size: Optional[int] = None,
    ) -> bool:
        group = self._groups.get(group_id)
        if group is None or group["user_id"] != user_id:
            return False
        if centroid_vector is not None:
            group["centroid_vector"] = list(centroid_vector)
        if size is not None:
            group["size"] = size
        return True

    async def delete_group(self, user_id: str, group_id: int) -> bool:
        group = self._groups.get(group_id)
        if group is None or group["user_id"] != user_id:
            return False
        del self._groups[group_id]
        return True

    async def list_groups(self, user_id: str) -> list[GroupInfo]:
        return [
            GroupInfo(group_id=gid, size=g["size"])
            for gid, g in sorted(self._groups.items())
            if g["user_id"] == user_id
        ]

    async def get_group_members(
        self, group_id: int, user_id: str, include_vectors: bool = False
    ) -> list[MemoryRecord]:
        flt = MemoryFilter(user_id=user_id, group_id=group_id)
        return await self.query(flt, limit=16_384, include_vectors=include_vectors)

    async def update_memory_group_id(
        self, memory_id: int, group_id: int, user_id: str
    ) -> bool:
        record = self._memories.get(memory_id)
        if record is None or record.user_id != user_id:
            return False
        record.group_id = group_id
        return True
