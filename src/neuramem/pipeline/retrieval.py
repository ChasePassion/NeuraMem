"""Retrieval pipeline: vector search + narrative group expansion (#4).

Extracted from the legacy Memory.search (~120 inline lines). Behavior
changes vs legacy, both recorded in the architecture doc:
- group expansion is ONE query with group_id in [...] instead of one
  query per group (#19, N+1 elimination)
- retired records are permanently filtered out (#20 minimal conflict
  elimination)
- returns a SearchResult (the correlation token of the two-phase closed
  loop, ch. 11) instead of a bare dict
"""

import logging
import time
from dataclasses import dataclass, field

from neuramem.config import RetrievalConfig
from neuramem.core.models import (
    MemoryFilter,
    MemoryRecord,
    RetrievalTrace,
    RetrievalTraceHit,
    SearchHit,
    SearchResult,
)
from neuramem.core.ports import Embedder, VectorStore

logger = logging.getLogger(__name__)

SEMANTIC_FETCH_LIMIT = 1000
MEMBERS_QUERY_LIMIT = 16_384  # Milvus hard query window cap


@dataclass
class _EpisodicStage:
    records: list[MemoryRecord] = field(default_factory=list)
    trace_hits: list[RetrievalTraceHit] = field(default_factory=list)
    seed_ids: list[int] = field(default_factory=list)
    expanded_ids: list[int] = field(default_factory=list)
    expanded_group_ids: list[int] = field(default_factory=list)


class Retriever:
    """Retrieves semantic + episodic memories with narrative expansion."""

    def __init__(self, store: VectorStore, embedder: Embedder, config: RetrievalConfig):
        self._store = store
        self._embedder = embedder
        self._config = config

    async def search(self, query: str, user_id: str) -> SearchResult:
        started = time.perf_counter()
        trace = RetrievalTrace(
            episodic_limit=self._config.k_episodic,
            semantic_limit=(
                SEMANTIC_FETCH_LIMIT
                if self._config.use_all_semantic
                else self._config.k_semantic
            ),
        )
        vectors = await self._embedder.embed([query])
        if not vectors:
            trace.status = "empty_embedding"
            trace.elapsed_ms = round((time.perf_counter() - started) * 1000)
            return SearchResult(
                query=query, user_id=user_id, retrieval_trace=trace
            )
        query_vector = vectors[0]

        semantic, semantic_trace_hits = await self._semantic(query_vector, user_id)
        episodic_stage = await self._episodic_with_expansion(query_vector, user_id)

        trace.seed_ids = episodic_stage.seed_ids
        trace.expanded_ids = episodic_stage.expanded_ids
        trace.semantic_ids = [record.id for record in semantic]
        trace.expanded_group_ids = episodic_stage.expanded_group_ids
        trace.hits = episodic_stage.trace_hits + semantic_trace_hits
        trace.elapsed_ms = round((time.perf_counter() - started) * 1000)

        logger.info(
            "retrieval: user=%s episodic=%d semantic=%d seeds=%d expanded=%d groups=%d",
            user_id,
            len(episodic_stage.records),
            len(semantic),
            len(trace.seed_ids),
            len(trace.expanded_ids),
            len(trace.expanded_group_ids),
        )
        return SearchResult(
            query=query,
            user_id=user_id,
            episodic=episodic_stage.records,
            semantic=semantic,
            retrieval_trace=trace,
        )

    async def _semantic(
        self, query_vector: list[float], user_id: str
    ) -> tuple[list[MemoryRecord], list[RetrievalTraceHit]]:
        base_filter = MemoryFilter(
            user_id=user_id, memory_type="semantic", retired=False
        )
        if self._config.use_all_semantic:
            records = await self._store.query(base_filter, limit=SEMANTIC_FETCH_LIMIT)
            return records, [
                self._record_trace(record, source="semantic_query")
                for record in records
            ]
        hits = await self._store.search(
            [query_vector], base_filter, limit=self._config.k_semantic
        )
        search_hits = hits[0] if hits else []
        return [hit.record for hit in search_hits], [
            self._search_hit_trace(hit, source="semantic_search")
            for hit in search_hits
        ]

    async def _episodic_with_expansion(
        self, query_vector: list[float], user_id: str
    ) -> _EpisodicStage:
        seeds_hits = await self._store.search(
            [query_vector],
            MemoryFilter(user_id=user_id, memory_type="episodic", retired=False),
            limit=self._config.k_episodic,
        )
        search_hits = seeds_hits[0] if seeds_hits else []
        seeds = [hit.record for hit in search_hits]
        if not seeds:
            return _EpisodicStage()

        seed_ids = [record.id for record in seeds]
        seed_id_set = set(seed_ids)
        trace_hits = [
            self._search_hit_trace(hit, source="episodic_seed", is_seed=True)
            for hit in search_hits
        ]

        expansion_group_ids = {s.group_id for s in seeds if s.group_id != -1}
        members: list[MemoryRecord] = []
        if expansion_group_ids:
            # single batched query instead of one per group (#19)
            members = await self._store.query(
                MemoryFilter(
                    user_id=user_id,
                    memory_type="episodic",
                    retired=False,
                    group_id_in=sorted(expansion_group_ids),
                ),
                limit=MEMBERS_QUERY_LIMIT,
            )

        # seeds first (similarity order), then expanded members, deduped
        expanded: list[MemoryRecord] = []
        expanded_ids: list[int] = []
        seen_expanded_ids: set[int] = set()
        for member in members:
            if member.id in seed_id_set or member.id in seen_expanded_ids:
                continue
            expanded.append(member)
            expanded_ids.append(member.id)
            seen_expanded_ids.add(member.id)

        trace_hits.extend(
            self._record_trace(member, source="episodic_group_expansion")
            for member in expanded
        )
        episodic = list(seeds)
        episodic.extend(expanded)
        return _EpisodicStage(
            records=episodic,
            trace_hits=trace_hits,
            seed_ids=seed_ids,
            expanded_ids=expanded_ids,
            expanded_group_ids=sorted(expansion_group_ids),
        )

    @staticmethod
    def _search_hit_trace(
        hit: SearchHit, source: str, is_seed: bool = False
    ) -> RetrievalTraceHit:
        return RetrievalTraceHit(
            memory_id=hit.record.id,
            memory_type=hit.record.memory_type,
            group_id=hit.record.group_id,
            distance=hit.distance,
            score=hit.score if hit.score is not None else hit.distance,
            is_seed=is_seed,
            source=source,
        )

    @staticmethod
    def _record_trace(record: MemoryRecord, source: str) -> RetrievalTraceHit:
        return RetrievalTraceHit(
            memory_id=record.id,
            memory_type=record.memory_type,
            group_id=record.group_id,
            source=source,
        )
