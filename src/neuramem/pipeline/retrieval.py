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

from neuramem.config import RetrievalConfig
from neuramem.core.models import MemoryFilter, MemoryRecord, SearchResult
from neuramem.core.ports import Embedder, VectorStore

logger = logging.getLogger(__name__)

SEMANTIC_FETCH_LIMIT = 1000
MEMBERS_QUERY_LIMIT = 16_384  # Milvus hard query window cap


class Retriever:
    """Retrieves semantic + episodic memories with narrative expansion."""

    def __init__(self, store: VectorStore, embedder: Embedder, config: RetrievalConfig):
        self._store = store
        self._embedder = embedder
        self._config = config

    async def search(self, query: str, user_id: str) -> SearchResult:
        vectors = await self._embedder.embed([query])
        if not vectors:
            return SearchResult(query=query, user_id=user_id)
        query_vector = vectors[0]

        semantic = await self._semantic(query_vector, user_id)
        episodic = await self._episodic_with_expansion(query_vector, user_id)

        logger.info(
            "retrieval: user=%s episodic=%d semantic=%d",
            user_id, len(episodic), len(semantic),
        )
        return SearchResult(query=query, user_id=user_id, episodic=episodic, semantic=semantic)

    async def _semantic(self, query_vector: list[float], user_id: str) -> list[MemoryRecord]:
        base_filter = MemoryFilter(
            user_id=user_id, memory_type="semantic", retired=False
        )
        if self._config.use_all_semantic:
            return await self._store.query(base_filter, limit=SEMANTIC_FETCH_LIMIT)
        hits = await self._store.search(
            [query_vector], base_filter, limit=self._config.k_semantic
        )
        return [hit.record for hit in hits[0]] if hits else []

    async def _episodic_with_expansion(
        self, query_vector: list[float], user_id: str
    ) -> list[MemoryRecord]:
        seeds_hits = await self._store.search(
            [query_vector],
            MemoryFilter(user_id=user_id, memory_type="episodic", retired=False),
            limit=self._config.k_episodic,
        )
        seeds = [hit.record for hit in seeds_hits[0]] if seeds_hits else []
        if not seeds:
            return []

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
        seed_ids = {s.id for s in seeds}
        episodic = list(seeds)
        episodic.extend(m for m in members if m.id not in seed_ids)
        return episodic
