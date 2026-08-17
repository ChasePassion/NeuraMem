"""Memory facade — thin orchestration over the pipeline layer.

Public API (versioned contract, architecture_target.md 10.1):
- two-phase closed loop: search_async -> consumer answers -> report_usage_async
- manage/consolidate/delete/reset complete the surface
- sync wrappers via asyncio.run (must not be called from a running loop)

All components are injectable (ports); defaults are built from config.
Telemetry defaults to Null (zero cost) — spans wrap every public
operation at the facade boundary.
"""

import asyncio
import logging
import time
from typing import Optional

from neuramem.config import MemoryConfig
from neuramem.core.models import (
    ConsolidationStats,
    MemoryFilter,
    MemoryRecord,
    SearchResult,
    UsageReport,
)
from neuramem.core.ports import Embedder, LLM, Telemetry, VectorStore
from neuramem.embed.openai_adapter import OpenAIEmbedder
from neuramem.llm.openai_adapter import OpenAILLM
from neuramem.pipeline.episodic import EpisodicManager
from neuramem.pipeline.narrative import NarrativeManager
from neuramem.pipeline.retrieval import Retriever
from neuramem.pipeline.semantic import SemanticWriter
from neuramem.pipeline.usage_judge import UsageJudge
from neuramem.store.milvus import MilvusStore
from neuramem.telemetry.null import NullTelemetry

logger = logging.getLogger(__name__)

# legacy manage passed the user's whole episodic set into the decision
EPISODIC_CANDIDATE_LIMIT = 10_000
CONSOLIDATION_BATCH_LIMIT = 1_000


class Memory:
    """AI memory system facade: manage, search, report usage, consolidate."""

    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        *,
        llm: Optional[LLM] = None,
        embedder: Optional[Embedder] = None,
        store: Optional[VectorStore] = None,
        telemetry: Optional[Telemetry] = None,
    ):
        self._config = config or MemoryConfig()
        self._telemetry = telemetry or NullTelemetry()
        self._llm = llm or OpenAILLM(self._config.llm)
        self._embedder = embedder or OpenAIEmbedder(self._config.embedding)
        self._store = store or MilvusStore(self._config.store)
        self._retriever = Retriever(self._store, self._embedder, self._config.retrieval)
        self._episodic = EpisodicManager(self._llm)
        self._semantic = SemanticWriter(self._llm)
        self._usage_judge = UsageJudge(self._llm)
        self._narrative = NarrativeManager(self._store, self._config.retrieval)
        self._ready = False

        logger.info(
            "Memory initialized (store=%s, model=%s)",
            type(self._store).__name__, self._llm.model_id,
        )

    # -- lifecycle ---------------------------------------------------------

    async def _ensure_ready(self) -> None:
        """Create the collection and verify the embedding dim once (#9)."""
        if self._ready:
            return
        vectors = await self._embedder.embed(["neuramem startup dimension probe"])
        if vectors and len(vectors[0]) != self._config.embedding.dim:
            raise ValueError(
                f"embedding model serves dim {len(vectors[0])}, config says "
                f"{self._config.embedding.dim} — fix EMBEDDING_DIM before use"
            )
        await self._store.create_collection(self._config.embedding.dim)
        self._ready = True

    @property
    def store(self) -> VectorStore:
        """Store port access (tests / tooling)."""
        return self._store

    @property
    def config(self) -> MemoryConfig:
        return self._config

    # -- two-phase closed loop ------------------------------------------------

    async def search_async(self, query: str, user_id: str) -> SearchResult:
        """Phase 1 of the closed loop: retrieve + narrative expansion.

        The returned SearchResult is the correlation token: pass it back
        with the answer text to report_usage_async.
        """
        await self._ensure_ready()
        async with self._telemetry.start_span(
            "neuramem.search", {"user_id": user_id, "query_length": len(query)}
        ):
            return await self._retriever.search(query, user_id)

    async def report_usage_async(
        self, result: SearchResult, answer_text: str
    ) -> UsageReport:
        """Phase 2 of the closed loop: judge used memories -> assign groups.

        Failure-isolated by design: the write-back channel must never
        break the consumer's answer path. Safe to call repeatedly.
        """
        try:
            if not result.episodic:
                return UsageReport(judged_candidates=0)
            async with self._telemetry.start_span(
                "neuramem.report_usage",
                {"user_id": result.user_id, "candidates": len(result.episodic)},
            ) as span:
                used_ids = await self._usage_judge.judge_used_memories(
                    result.episodic, result.query, answer_text
                )
                assignments: dict[int, int] = {}
                if used_ids:
                    assignments = await self._narrative.assign_to_narrative_group(
                        used_ids, result.user_id
                    )
                span.set_attributes(
                    {"used": len(used_ids), "assigned": len(assignments)}
                )
                return UsageReport(
                    judged_candidates=len(result.episodic),
                    used_memory_ids=used_ids,
                    assignments=assignments,
                )
        except Exception as e:  # noqa: BLE001 - never break the answer path
            logger.warning("report_usage failed (isolated): %s", e)
            return UsageReport(judged_candidates=len(result.episodic))

    def search(self, query: str, user_id: str) -> SearchResult:
        return asyncio.run(self.search_async(query, user_id))

    def report_usage(self, result: SearchResult, answer_text: str) -> UsageReport:
        return asyncio.run(self.report_usage_async(result, answer_text))

    # -- manage ------------------------------------------------------------

    async def manage_async(
        self,
        user_text: str,
        assistant_text: str,
        user_id: str,
        chat_id: str,
        metadata: Optional[dict] = None,
    ) -> list[int]:
        """Run episodic CRUD for one conversation turn; returns added ids."""
        await self._ensure_ready()
        async with self._telemetry.start_span(
            "neuramem.manage",
            {"user_id": user_id, "chat_id": chat_id},
        ):
            episodic = await self._store.query(
                MemoryFilter(user_id=user_id, memory_type="episodic"),
                limit=EPISODIC_CANDIDATE_LIMIT,
            )
            plan = await self._episodic.manage_memories(
                user_text, assistant_text, episodic
            )

            for op in plan.operations:
                if op.operation_type == "delete" and op.memory_id is not None:
                    await self._delete_one(op.memory_id, user_id)

            for op in plan.operations:
                if op.operation_type == "update" and op.memory_id is not None:
                    await self._update_one(op, user_id)

            add_ops = [
                op for op in plan.operations if op.operation_type == "add" and op.text
            ]
            added_ids: list[int] = []
            if add_ops:
                embeddings = await self._embedder.embed([op.text for op in add_ops])
                records = [
                    MemoryRecord(
                        user_id=user_id,
                        memory_type="episodic",
                        ts=int(time.time()),
                        chat_id=chat_id,
                        text=op.text,
                        vector=embeddings[i],
                        group_id=-1,
                        metadata=metadata,
                    )
                    for i, op in enumerate(add_ops)
                ]
                added_ids = await self._store.insert(records)

            logger.info(
                "manage: user=%s chat=%s added=%d updated=%d deleted=%d",
                user_id, chat_id, len(added_ids),
                sum(1 for o in plan.operations if o.operation_type == "update"),
                sum(1 for o in plan.operations if o.operation_type == "delete"),
            )
            return added_ids

    def manage(
        self,
        user_text: str,
        assistant_text: str,
        user_id: str,
        chat_id: str,
        metadata: Optional[dict] = None,
    ) -> list[int]:
        return asyncio.run(
            self.manage_async(user_text, assistant_text, user_id, chat_id, metadata)
        )

    async def _update_one(self, op, user_id: str) -> None:
        records = await self._store.query(
            MemoryFilter(user_id=user_id, id_in=[op.memory_id]),
            limit=1,
        )
        if not records:
            logger.warning("update target %s not found", op.memory_id)
            return
        original = records[0]
        # group membership cleanup BEFORE resetting group_id: the record
        # leaves its group here (legacy did this via delete-then-reinsert),
        # so the centroid/size must drop it too — otherwise the group keeps
        # counting a member it no longer has, and a last-member update
        # leaves an orphan group that can absorb new memories
        if original.group_id != -1:
            try:
                await self._narrative.delete_memory_from_group(
                    original.id, user_id
                )
            except Exception as e:  # noqa: BLE001 - cleanup is best effort
                logger.warning("group cleanup for %s failed: %s", original.id, e)
        embeddings = await self._embedder.embed([op.text])
        if not embeddings:
            logger.warning("update %s failed: embedding generation failed", op.memory_id)
            return
        # stable id via upsert (#17); group reset — reassignment happens on
        # the next report_usage cycle (legacy semantics)
        await self._store.upsert([
            MemoryRecord(
                id=original.id,
                user_id=user_id,
                memory_type=original.memory_type,
                ts=int(time.time()),
                chat_id=original.chat_id,
                text=op.text,
                vector=embeddings[0],
                group_id=-1,
                metadata=original.metadata,
            )
        ])

    async def _delete_one(self, memory_id: int, user_id: str) -> int:
        """Group cleanup + delete; returns the affected row count."""
        try:
            await self._narrative.delete_memory_from_group(memory_id, user_id)
        except Exception as e:  # noqa: BLE001 - group cleanup is best effort
            logger.warning("group cleanup for %s failed: %s", memory_id, e)
        return await self._store.delete(
            ids=[memory_id], flt=MemoryFilter(user_id=user_id)
        )

    # -- consolidate ---------------------------------------------------------

    async def consolidate_async(
        self, user_id: Optional[str] = None
    ) -> ConsolidationStats:
        """Extract semantic facts; retire contradicted ones (#20 minimal)."""
        await self._ensure_ready()
        async with self._telemetry.start_span(
            "neuramem.consolidate", {"user_id": user_id or "all"}
        ):
            user_filter = MemoryFilter(user_id=user_id) if user_id else None
            episodic = await self._store.query(
                _with(user_filter, memory_type="episodic", retired=False),
                limit=CONSOLIDATION_BATCH_LIMIT,
            )
            semantic = await self._store.query(
                _with(user_filter, memory_type="semantic", retired=False),
                limit=CONSOLIDATION_BATCH_LIMIT,
            )

            extraction = await self._semantic.extract(episodic, semantic)

            # conflict elimination: tombstone contradicted semantic memories
            for retire_id in extraction.retire_ids:
                stale = await self._store.query(
                    _with(user_filter, id_in=[retire_id]),
                    limit=1,
                )
                if stale:
                    stale[0].retired = True
                    await self._store.upsert([stale[0]])  # vector backfilled

            created = 0
            if extraction.write_semantic and extraction.facts:
                source = episodic[0] if episodic else None
                embeddings = await self._embedder.embed(extraction.facts)
                records = [
                    MemoryRecord(
                        user_id=user_id or (source.user_id if source else ""),
                        memory_type="semantic",
                        ts=int(time.time()),
                        chat_id=source.chat_id if source else "",
                        text=fact,
                        vector=embeddings[i],
                        group_id=-1,
                    )
                    for i, fact in enumerate(extraction.facts)
                ]
                created = len(await self._store.insert(records))

            logger.info(
                "consolidate: user=%s processed=%d semantic_created=%d retired=%d",
                user_id or "all", len(episodic), created,
                len(extraction.retire_ids),
            )
            return ConsolidationStats(
                memories_processed=len(episodic), semantic_created=created
            )

    def consolidate(self, user_id: Optional[str] = None) -> ConsolidationStats:
        return asyncio.run(self.consolidate_async(user_id))

    # -- delete / reset ---------------------------------------------------------

    async def delete_async(self, memory_id: int, user_id: str) -> bool:
        await self._ensure_ready()
        # success = something was actually deleted (legacy semantics: an
        # unknown id reports False, not "already absent")
        return (await self._delete_one(memory_id, user_id)) > 0

    def delete(self, memory_id: int, user_id: str) -> bool:
        return asyncio.run(self.delete_async(memory_id, user_id))

    async def reset_async(self, user_id: str) -> int:
        """Delete all memories of a user (the only path that physically
        removes retired records) and their narrative groups."""
        await self._ensure_ready()
        deleted = await self._store.delete(
            flt=MemoryFilter(user_id=user_id)
        )
        for group in await self._store.list_groups(user_id):
            await self._store.delete_group(user_id, group.group_id)
        logger.info("reset: user=%s deleted=%d", user_id, deleted)
        return deleted

    def reset(self, user_id: str) -> int:
        return asyncio.run(self.reset_async(user_id))


def _with(base: Optional[MemoryFilter], **fields) -> MemoryFilter:
    """Extend an optional base filter with extra constraints."""
    data = base.model_dump(exclude_none=True) if base else {}
    data.update(fields)
    return MemoryFilter(**data)
