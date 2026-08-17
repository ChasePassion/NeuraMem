"""Milvus VectorStore adapter (schema-compatible with the legacy store).

Differences from legacy clients/milvus_store.py:
- structured MemoryFilter compiled via store/filters.py — no f-string
  filter expressions built from caller input (#16)
- ONE groups collection with a user_id field instead of per-user
  groups_{user_id} collections (#15) — collection count no longer scales
  with tenants
- upsert for in-place updates with stable ids (#17); delete+add is gone
- all calls bridge to a dedicated thread pool (pymilvus has no native
  async), pool size explicit in StoreConfig (#7)
- dim is a parameter everywhere (no 2560 literals)

The memories collection schema is field-for-field the legacy one, so
existing data keeps working; the retired flag (#20) and metadata
passthrough (8.2) ride on enable_dynamic_field, which legacy already set.
"""

import asyncio
import logging
import time
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient

from neuramem.config import StoreConfig
from neuramem.core.exceptions import MilvusConnectionError
from neuramem.core.models import (
    GroupInfo,
    GroupMatch,
    MemoryFilter,
    MemoryRecord,
    SearchHit,
)
from neuramem.store.filters import compile_filter, escape_string

logger = logging.getLogger(__name__)

# pymilvus query has a hard limit of 16384 rows per call
QUERY_PAGE_LIMIT = 16384

_KNOWN_FIELDS = {
    "id", "user_id", "memory_type", "ts", "chat_id", "text", "vector",
    "group_id", "retired",
}


class MilvusStore:
    """VectorStore implementation over pymilvus MilvusClient."""

    def __init__(self, config: StoreConfig):
        self._config = config
        self._client = self._connect()
        self._executor = ThreadPoolExecutor(
            max_workers=config.thread_pool_size, thread_name_prefix="neuramem-store"
        )

    # -- connection ---------------------------------------------------------

    def _connect(self) -> MilvusClient:
        last_error: Optional[Exception] = None
        for _ in range(self._config.connect_retries):
            try:
                return MilvusClient(uri=self._config.uri, timeout=self._config.connect_timeout)
            except Exception as e:  # noqa: BLE001 - connection retries are best effort
                last_error = e
                logger.warning(
                    "Milvus connect to %s failed (%s); retrying", self._config.uri, e
                )
                time.sleep(3.0)
        raise MilvusConnectionError(self._config.uri, last_error)

    async def _run(self, fn, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, partial(fn, *args, **kwargs))

    # -- collection lifecycle -------------------------------------------------

    async def create_collection(self, dim: int) -> None:
        await self._run(self._create_collection_sync, dim)

    def _create_collection_sync(self, dim: int) -> None:
        name = self._config.collection_name
        if self._client.has_collection(name):
            self._client.load_collection(name)
            return
        fields = [
            FieldSchema("id", DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema("user_id", DataType.VARCHAR, max_length=128),
            FieldSchema("memory_type", DataType.VARCHAR, max_length=32),
            FieldSchema("ts", DataType.INT64),
            FieldSchema("chat_id", DataType.VARCHAR, max_length=128),
            FieldSchema("text", DataType.VARCHAR, max_length=65535),
            FieldSchema("vector", DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema("group_id", DataType.INT64, default_value=-1),
        ]
        schema = CollectionSchema(fields, enable_dynamic_field=True)
        index_params = self._client.prepare_index_params()
        index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")
        # Strong consistency: default Session/Bounded levels leave freshly
        # written rows invisible to single-clause queries for a short window
        # (observed on this server), which breaks read-after-write flows
        # (upsert -> query, insert -> count). Query latency is irrelevant
        # next to the LLM calls that dominate every flow.
        self._client.create_collection(
            name, schema=schema, index_params=index_params, consistency_level="Strong"
        )
        self._client.load_collection(name)
        logger.info("Created memories collection '%s' (dim=%d, Strong)", name, dim)

    def _ensure_groups_collection_sync(self, dim: int) -> str:
        name = self._config.groups_collection_name
        if self._client.has_collection(name):
            self._client.load_collection(name)
            return name
        fields = [
            FieldSchema("group_id", DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema("user_id", DataType.VARCHAR, max_length=128),
            FieldSchema("centroid_vector", DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema("size", DataType.INT64),
        ]
        schema = CollectionSchema(fields, enable_dynamic_field=True)
        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name="centroid_vector", index_type="AUTOINDEX", metric_type="IP"
        )
        self._client.create_collection(
            name, schema=schema, index_params=index_params, consistency_level="Strong"
        )
        self._client.load_collection(name)
        logger.info("Created groups collection '%s' (dim=%d, Strong)", name, dim)
        return name

    # -- entity mapping ---------------------------------------------------------

    @staticmethod
    def _record_to_entity(record: MemoryRecord, include_id: bool) -> dict:
        entity: dict[str, Any] = {
            "user_id": record.user_id,
            "memory_type": record.memory_type,
            "ts": record.ts,
            "chat_id": record.chat_id,
            "text": record.text,
            "vector": record.vector,
            "group_id": record.group_id,
            "retired": record.retired,
        }
        if record.metadata:
            entity.update(record.metadata)
        if include_id:
            entity["id"] = record.id
        return entity

    @staticmethod
    def _entity_to_record(entity: dict) -> MemoryRecord:
        metadata = {
            key: value
            for key, value in entity.items()
            if key not in _KNOWN_FIELDS
        }
        return MemoryRecord(
            id=entity.get("id", 0),
            user_id=entity.get("user_id", ""),
            memory_type=entity.get("memory_type", ""),
            ts=entity.get("ts", 0),
            chat_id=entity.get("chat_id", ""),
            text=entity.get("text", ""),
            group_id=entity.get("group_id", -1),
            retired=bool(entity.get("retired", False)),
            metadata=metadata or None,
            vector=entity.get("vector"),
        )

    # -- memory CRUD -------------------------------------------------------------

    async def insert(self, records: list[MemoryRecord]) -> list[int]:
        entities = [self._record_to_entity(r, include_id=False) for r in records]
        result = await self._run(
            self._client.insert, collection_name=self._config.collection_name, data=entities
        )
        return list(result["ids"])

    async def upsert(self, records: list[MemoryRecord]) -> list[int]:
        records = [r.model_copy() for r in records]
        missing_vector = [r for r in records if r.vector is None]
        if missing_vector:
            # Milvus upsert requires the full row including the vector.
            # Partial updates (e.g. flipping the retired flag) arrive here
            # without one — fetch the stored vectors and merge (#17).
            ids = [r.id for r in missing_vector]
            if any(i <= 0 for i in ids):
                raise ValueError("upsert needs a vector or an existing id per record")
            stored = {
                r.id: r
                for r in await self.query(
                    MemoryFilter(id_in=ids), limit=len(ids), include_vectors=True
                )
            }
            for record in missing_vector:
                existing = stored.get(record.id)
                if existing is None:
                    raise ValueError(
                        f"cannot upsert unknown id {record.id} without a vector"
                    )
                record.vector = existing.vector
        entities = [self._record_to_entity(r, include_id=True) for r in records]
        result = await self._run(
            self._client.upsert, collection_name=self._config.collection_name, data=entities
        )
        return list(result.get("ids", []))

    async def search(
        self,
        vectors: list[list[float]],
        flt: Optional[MemoryFilter] = None,
        limit: int = 10,
    ) -> list[list[SearchHit]]:
        expr = compile_filter(flt)
        raw = await self._run(
            self._client.search,
            collection_name=self._config.collection_name,
            data=vectors,
            filter=expr,
            limit=limit,
            output_fields=["*"],
        )
        results: list[list[SearchHit]] = []
        for hits in raw:
            page = []
            for hit in hits:
                entity = dict(hit.get("entity", {}))
                entity["id"] = hit.get("id")
                page.append(
                    SearchHit(
                        record=self._entity_to_record(entity),
                        distance=hit.get("distance"),
                    )
                )
            results.append(page)
        return results

    async def query(
        self,
        flt: MemoryFilter,
        limit: int = 100,
        include_vectors: bool = False,
    ) -> list[MemoryRecord]:
        expr = compile_filter(flt)
        # "*" is required to also return dynamic fields (metadata, retired);
        # the vector is stripped afterwards unless explicitly requested
        rows = await self._run(
            self._client.query,
            collection_name=self._config.collection_name,
            filter=expr,
            output_fields=["*"],
            limit=limit,
        )
        records = [self._entity_to_record(dict(row)) for row in rows]
        if not include_vectors:
            for record in records:
                record.vector = None
        return records

    async def delete(
        self,
        ids: Optional[list[int]] = None,
        flt: Optional[MemoryFilter] = None,
    ) -> int:
        if ids is not None and flt is not None:
            # combine: id in [...] and (filter)
            combined = MemoryFilter(**flt.model_dump(exclude_none=True))
            combined.id_in = ids if combined.id_in is None else [
                i for i in combined.id_in if i in ids
            ]
            expr = compile_filter(combined)
        elif ids is not None:
            expr = "id in [" + ", ".join(str(i) for i in ids) + "]"
        else:
            expr = compile_filter(flt)
        if not expr:
            return 0
        # count first (delete returns no count on MilvusClient)
        rows = await self._run(
            self._client.query,
            collection_name=self._config.collection_name,
            filter=expr,
            output_fields=["id"],
            limit=QUERY_PAGE_LIMIT,
        )
        await self._run(
            self._client.delete, collection_name=self._config.collection_name, filter=expr
        )
        return len(rows)

    async def count(self, flt: Optional[MemoryFilter] = None) -> int:
        # Always query-based: get_collection_stats row_count lags unflushed
        # growing segments (observed returning 0 right after insert), and the
        # benchmark's ingest-completeness check needs accurate counts. Capped
        # at QUERY_PAGE_LIMIT rows.
        rows = await self._run(
            self._client.query,
            collection_name=self._config.collection_name,
            filter=compile_filter(flt),
            output_fields=["id"],
            limit=QUERY_PAGE_LIMIT,
        )
        return len(rows)

    # -- narrative groups -----------------------------------------------------------

    async def search_groups(
        self, user_id: str, vector: list[float], limit: int = 1
    ) -> list[GroupMatch]:
        name = self._config.groups_collection_name
        if not await self._run(self._client.has_collection, collection_name=name):
            return []
        raw = await self._run(
            self._client.search,
            collection_name=name,
            data=[vector],
            filter=f"user_id == {escape_string(user_id)}",
            limit=limit,
            output_fields=["group_id", "size"],
            search_params={"metric_type": "IP"},
        )
        matches = []
        for hit in raw[0] if raw else []:
            entity = hit.get("entity", {})
            matches.append(
                GroupMatch(
                    group_id=entity.get("group_id") or hit.get("id"),
                    similarity=hit.get("distance"),
                    size=entity.get("size", 0),
                )
            )
        return matches

    async def insert_group(
        self, user_id: str, centroid_vector: list[float], size: int = 1
    ) -> Optional[int]:
        def _insert():
            name = self._ensure_groups_collection_sync(len(centroid_vector))
            result = self._client.insert(
                collection_name=name,
                data=[{
                    "user_id": user_id,
                    "centroid_vector": centroid_vector,
                    "size": size,
                }],
            )
            ids = result.get("ids") or result.get("primary_keys")
            return list(ids)[0] if ids else None

        return await self._run(_insert)

    async def update_group(
        self,
        user_id: str,
        group_id: int,
        centroid_vector: Optional[list[float]] = None,
        size: Optional[int] = None,
    ) -> bool:
        def _update() -> bool:
            name = self._config.groups_collection_name
            if not self._client.has_collection(name):
                return False
            rows = self._client.query(
                collection_name=name,
                filter=f"group_id == {group_id} and user_id == {escape_string(user_id)}",
                output_fields=["*"],
                limit=1,
            )
            if not rows:
                return False
            record = dict(rows[0])
            if centroid_vector is not None:
                record["centroid_vector"] = centroid_vector
            if size is not None:
                record["size"] = size
            self._client.upsert(collection_name=name, data=[record])
            return True

        return await self._run(_update)

    async def delete_group(self, user_id: str, group_id: int) -> bool:
        name = self._config.groups_collection_name
        if not await self._run(self._client.has_collection, collection_name=name):
            return False
        expr = f"group_id == {group_id} and user_id == {escape_string(user_id)}"
        await self._run(
            self._client.delete, collection_name=name, filter=expr
        )
        return True

    async def list_groups(self, user_id: str) -> list[GroupInfo]:
        name = self._config.groups_collection_name
        if not await self._run(self._client.has_collection, collection_name=name):
            return []
        rows = await self._run(
            self._client.query,
            collection_name=name,
            filter=f"user_id == {escape_string(user_id)}",
            output_fields=["group_id", "size"],
            limit=QUERY_PAGE_LIMIT,
        )
        return [
            GroupInfo(group_id=row.get("group_id"), size=row.get("size", 0))
            for row in rows
        ]

    async def get_group_members(
        self, group_id: int, user_id: str, include_vectors: bool = False
    ) -> list[MemoryRecord]:
        return await self.query(
            MemoryFilter(user_id=user_id, group_id=group_id),
            limit=QUERY_PAGE_LIMIT,
            include_vectors=include_vectors,
        )

    async def update_memory_group_id(
        self, memory_id: int, group_id: int, user_id: str
    ) -> bool:
        records = await self.query(
            MemoryFilter(user_id=user_id, id_in=[memory_id]),
            limit=1,
            include_vectors=True,
        )
        if not records:
            return False
        record = records[0]
        record.group_id = group_id
        await self.upsert([record])
        return True
