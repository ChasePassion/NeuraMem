"""Narrative pipeline: assign episodic memories to same-event groups.

The episodic vector search is used only to find candidate groups. A single
LLM grouping prompt decides whether the new memory belongs to one candidate
group; group assignment never uses a centroid similarity threshold.
"""

import asyncio
import json
import logging
from typing import Optional

import numpy as np

from neuramem.config import RetrievalConfig
from neuramem.core.models import MemoryFilter, MemoryRecord
from neuramem.core.ports import LLM, VectorStore
from neuramem.prompts import NARRATIVE_MEMORY_GROUPING_PROMPT

logger = logging.getLogger(__name__)

MEMBERS_QUERY_LIMIT = 16_384  # Milvus hard query window cap
GROUP_CANDIDATE_MEMORY_LIMIT = 32
GROUP_CANDIDATE_LIMIT = 8
GROUP_PROMPT_MEMBER_LIMIT = 32


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
    """Same-event group bookkeeping over the VectorStore port."""

    def __init__(self, store: VectorStore, config: RetrievalConfig, llm: LLM):
        self._store = store
        self._config = config
        self._llm = llm
        self._write_lock = asyncio.Lock()

    async def assign_to_narrative_group(
        self,
        memory_ids: list[int],
        user_id: str,
        records: Optional[list[MemoryRecord]] = None,
    ) -> dict[int, int]:
        unique_ids = list(dict.fromkeys(memory_ids))
        if not unique_ids:
            return {}

        source_by_id = {
            record.id: record
            for record in (records or [])
            if record.user_id == user_id and record.id in unique_ids
        }
        loaded_by_id = await self._load_records(
            unique_ids, user_id, source_by_id=source_by_id
        )
        missing_ids = [memory_id for memory_id in unique_ids if memory_id not in loaded_by_id]
        if missing_ids:
            logger.warning(
                "Narrative grouping skipped missing memories in batch: ids=%s",
                missing_ids,
            )

        assignments = {
            memory_id: record.group_id
            for memory_id, record in loaded_by_id.items()
            if record.group_id != -1
        }
        pending = [
            loaded_by_id[memory_id]
            for memory_id in unique_ids
            if memory_id in loaded_by_id
            and loaded_by_id[memory_id].group_id == -1
            and loaded_by_id[memory_id].vector
        ]
        if not pending:
            return assignments

        candidate_by_memory, candidate_groups = (
            await self._find_candidate_groups_batch(pending, user_id)
        )
        decisions = await self._judge_groups(
            pending, candidate_by_memory, candidate_groups
        )
        assignments.update(
            await self._apply_batch_assignments(
                pending,
                decisions,
                user_id,
                source_by_id=source_by_id,
            )
        )
        return assignments

    async def _load_records(
        self,
        memory_ids: list[int],
        user_id: str,
        *,
        source_by_id: Optional[dict[int, MemoryRecord]] = None,
    ) -> dict[int, MemoryRecord]:
        """Load a batch and use the search snapshot for transient misses."""
        loaded: dict[int, MemoryRecord] = {}
        try:
            records = await self._store.query(
                MemoryFilter(user_id=user_id, id_in=memory_ids),
                limit=min(max(len(memory_ids), 1), MEMBERS_QUERY_LIMIT),
                include_vectors=True,
            )
            loaded = {record.id: record for record in records}
        except Exception as exc:  # noqa: BLE001 - grouping is best effort
            logger.warning("Narrative grouping batch load failed: %s", exc)

        for memory_id, source in (source_by_id or {}).items():
            loaded.setdefault(memory_id, source)
        return loaded

    async def _find_candidate_groups_batch(
        self, records: list[MemoryRecord], user_id: str
    ) -> tuple[dict[int, list[int]], list[dict]]:
        """Batch vector search and batch-load the candidate group members."""
        candidate_by_memory: dict[int, list[int]] = {
            record.id: [] for record in records
        }
        vectors = [record.vector or [] for record in records]
        try:
            search_results = await self._store.search(
                vectors,
                MemoryFilter(
                    user_id=user_id,
                    memory_type="episodic",
                    retired=False,
                ),
                limit=GROUP_CANDIDATE_MEMORY_LIMIT,
            )
        except Exception as exc:  # noqa: BLE001 - grouping is best effort
            logger.warning("Narrative candidate search batch failed: %s", exc)
            return candidate_by_memory, []

        group_ids: dict[int, None] = {}
        for record, hits in zip(records, search_results):
            for hit in hits:
                group_id = hit.record.group_id
                if group_id == -1:
                    continue
                if group_id not in candidate_by_memory[record.id]:
                    candidate_by_memory[record.id].append(group_id)
                group_ids.setdefault(group_id, None)
                if len(candidate_by_memory[record.id]) >= GROUP_CANDIDATE_LIMIT:
                    break

        if not group_ids:
            return candidate_by_memory, []

        try:
            members = await self._store.query(
                MemoryFilter(user_id=user_id, group_id_in=list(group_ids)),
                limit=MEMBERS_QUERY_LIMIT,
            )
        except Exception as exc:  # noqa: BLE001 - grouping is best effort
            logger.warning("Narrative candidate member batch load failed: %s", exc)
            return candidate_by_memory, []

        members_by_group: dict[int, list[MemoryRecord]] = {}
        for member in members:
            members_by_group.setdefault(member.group_id, []).append(member)

        candidate_groups = [
            {
                "group_id": group_id,
                "episodic_memories": [
                    self._memory_payload(member)
                    for member in members_by_group.get(group_id, [])[
                        :GROUP_PROMPT_MEMBER_LIMIT
                    ]
                ],
            }
            for group_id in group_ids
            if members_by_group.get(group_id)
        ]
        valid_group_ids = {group["group_id"] for group in candidate_groups}
        for memory_id, ids in candidate_by_memory.items():
            candidate_by_memory[memory_id] = [
                group_id for group_id in ids if group_id in valid_group_ids
            ]
        return candidate_by_memory, candidate_groups

    async def _judge_groups(
        self,
        records: list[MemoryRecord],
        candidate_by_memory: dict[int, list[int]],
        candidate_groups: list[dict],
    ) -> dict[int, tuple[Optional[int], str]]:
        """Make one LLM call for the whole usage report."""
        decisions: dict[int, tuple[Optional[int], str]] = {
            record.id: (None, f"memory-{record.id}") for record in records
        }
        if not candidate_groups and len(records) == 1:
            return decisions

        payload = {
            "new_memories": [
                {
                    "memory": self._memory_payload(record),
                    "candidate_group_ids": candidate_by_memory.get(record.id, []),
                }
                for record in records
            ],
            "candidate_groups": candidate_groups,
        }
        try:
            result = await self._llm.complete_json(
                system_prompt=NARRATIVE_MEMORY_GROUPING_PROMPT,
                user_message=json.dumps(payload, ensure_ascii=False),
                default={"assignments": []},
                call_label="narrative",
            )
        except Exception as exc:  # noqa: BLE001 - grouping is answer-isolated
            logger.warning("Narrative grouping batch prompt failed: %s", exc)
            return decisions

        if not result.success or not isinstance(result.parsed_data, dict):
            logger.warning("Narrative grouping batch returned invalid JSON")
            return decisions

        raw_assignments = result.parsed_data.get("assignments")
        if not isinstance(raw_assignments, list):
            logger.warning("Narrative grouping batch omitted assignments")
            return decisions

        known_ids = set(decisions)
        seen_ids: set[int] = set()
        for item in raw_assignments:
            if not isinstance(item, dict):
                continue
            raw_memory_id = item.get("memory_id")
            try:
                memory_id = int(raw_memory_id)
            except (TypeError, ValueError):
                continue
            if isinstance(raw_memory_id, bool) or memory_id not in known_ids:
                continue
            if memory_id in seen_ids:
                continue
            seen_ids.add(memory_id)

            raw_group_id = item.get("matched_group_id")
            if raw_group_id is not None:
                if isinstance(raw_group_id, bool):
                    continue
                try:
                    group_id = int(raw_group_id)
                except (TypeError, ValueError):
                    continue
                if group_id not in set(candidate_by_memory.get(memory_id, [])):
                    logger.warning(
                        "Narrative grouping batch returned unknown group=%s memory=%s",
                        group_id,
                        memory_id,
                    )
                    continue
                decisions[memory_id] = (group_id, "")
                continue

            new_group_key = item.get("new_group_key")
            if isinstance(new_group_key, str) and new_group_key.strip():
                decisions[memory_id] = (None, new_group_key.strip())
        return decisions

    async def _apply_batch_assignments(
        self,
        records: list[MemoryRecord],
        decisions: dict[int, tuple[Optional[int], str]],
        user_id: str,
        *,
        source_by_id: dict[int, MemoryRecord],
    ) -> dict[int, int]:
        """Recheck state, then write all changed records in one upsert."""
        assignments: dict[int, int] = {}
        async with self._write_lock:
            current_by_id = await self._query_records(
                [record.id for record in records], user_id
            )
            missing_ids = [
                record.id for record in records if record.id not in current_by_id
            ]
            if missing_ids:
                logger.warning(
                    "Narrative batch write skipped missing memories: ids=%s",
                    missing_ids,
                )

            updates: list[MemoryRecord] = []
            affected_groups: set[int] = set()
            new_groups: dict[str, int] = {}
            for record in records:
                current = current_by_id.get(record.id)
                if current is None:
                    continue
                if current.group_id != -1:
                    assignments[record.id] = current.group_id
                    continue
                if not current.vector:
                    fallback = source_by_id.get(record.id)
                    if fallback is not None and fallback.vector:
                        current.vector = fallback.vector
                if not current.vector:
                    logger.warning(
                        "Narrative batch write skipped memory without vector: id=%s",
                        record.id,
                    )
                    continue

                group_id, new_group_key = decisions.get(
                    record.id, (None, f"memory-{record.id}")
                )
                if group_id is None:
                    group_id = new_groups.get(new_group_key)
                    if group_id is None:
                        group_id = await self._store.insert_group(
                            user_id, _normalize(current.vector), size=1
                        )
                        if group_id is None:
                            logger.error(
                                "Narrative batch failed to create group for memory=%s",
                                record.id,
                            )
                            continue
                        new_groups[new_group_key] = group_id
                current.group_id = group_id
                updates.append(current)
                affected_groups.add(group_id)

            if updates:
                try:
                    await self._store.upsert(updates)
                except Exception as exc:  # noqa: BLE001 - answer-isolated
                    logger.error(
                        "Narrative batch group write failed: records=%d error=%s",
                        len(updates),
                        exc,
                    )
                    return assignments
                assignments.update({record.id: record.group_id for record in updates})

            for group_id in affected_groups:
                await self._recompute_group(user_id, group_id)

            logger.info(
                "Narrative batch grouping complete: input=%d updated=%d existing=%d",
                len(records),
                len(updates),
                len(assignments) - len(updates),
            )
        return assignments

    async def _query_records(
        self, memory_ids: list[int], user_id: str
    ) -> dict[int, MemoryRecord]:
        try:
            records = await self._store.query(
                MemoryFilter(user_id=user_id, id_in=memory_ids),
                limit=min(max(len(memory_ids), 1), MEMBERS_QUERY_LIMIT),
                include_vectors=True,
            )
        except Exception as exc:  # noqa: BLE001 - grouping is best effort
            logger.warning("Narrative batch state reload failed: %s", exc)
            return {}
        return {record.id: record for record in records}

    async def _recompute_group(self, user_id: str, group_id: int) -> None:
        members = await self._store.query(
            MemoryFilter(user_id=user_id, group_id=group_id),
            limit=MEMBERS_QUERY_LIMIT,
            include_vectors=True,
        )
        centroid = _centroid([member.vector for member in members if member.vector])
        if centroid is not None:
            await self._store.update_group(
                user_id, group_id, centroid_vector=centroid, size=len(members)
            )

    @staticmethod
    def _memory_payload(record: MemoryRecord) -> dict:
        provenance = {
            key: value
            for key, value in (record.metadata or {}).items()
            if key.startswith("provenance_")
        }
        payload = {
            "id": record.id,
            "text": record.text,
            "ts": record.ts,
            "chat_id": record.chat_id,
        }
        if provenance:
            payload["provenance"] = provenance
        return payload

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
