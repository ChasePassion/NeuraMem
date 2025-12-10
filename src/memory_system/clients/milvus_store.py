"""Milvus vector store client for memory storage."""

import logging
from typing import List, Dict, Any, Optional

from pymilvus import (
    MilvusClient,
    DataType,
    FieldSchema,
    CollectionSchema,
)

from ..exceptions import MilvusConnectionError


logger = logging.getLogger(__name__)


class MilvusStore:
    """Milvus vector store wrapper for memory operations.
    
    Provides CRUD operations for the memories collection with support for
    vector similarity search and narrative grouping.
    
    **Main Collection Schema (v2)**:
    - id: INT64, primary key, auto_id
    - user_id: VARCHAR(128), user identifier
    - memory_type: VARCHAR(32), "episodic" or "semantic"
    - ts: INT64, Unix timestamp of write time
    - chat_id: VARCHAR(128), conversation/thread identifier
    - text: VARCHAR(65535), natural-language content for embedding/search
    - vector: FLOAT_VECTOR(2560), embedding vector
    - group_id: INT64, narrative group ID (-1 = ungrouped)
    
    **Groups Collection Schema** (per-user: ``groups_{user_id}``):
    - group_id: INT64, primary key, auto_id
    - user_id: VARCHAR(128)
    - centroid_vector: FLOAT_VECTOR(2560)
    - size: INT64, member count
    """
    
    # Schema field definitions (simplified v2)
    SCHEMA_FIELDS = [
        ("id", DataType.INT64, {"is_primary": True, "auto_id": True}),
        ("user_id", DataType.VARCHAR, {"max_length": 128}),
        ("memory_type", DataType.VARCHAR, {"max_length": 32}),
        ("ts", DataType.INT64, {}),
        ("chat_id", DataType.VARCHAR, {"max_length": 128}),
        ("text", DataType.VARCHAR, {"max_length": 65535}),
        ("vector", DataType.FLOAT_VECTOR, {"dim": 2560}),
        # 新增，用于叙事分组，-1表示未分组；显式提供默认值以防遗漏
        ("group_id", DataType.INT64, {"default_value": -1}),
    ]
    
    # 叙事组表schema
    GROUP_SCHEMA_FIELDS = [
        ("group_id", DataType.INT64, {"is_primary": True, "auto_id": True}),
        ("user_id", DataType.VARCHAR, {"max_length": 128}),
        ("centroid_vector", DataType.FLOAT_VECTOR, {"dim": 2560}),
        ("size", DataType.INT64, {}),   # 当前组内成员数量
    ]
    
    def __init__(
        self,
        uri: str,
        collection_name: str
    ):
        """Initialize Milvus connection.
        
        Args:
            uri: Milvus server URI
            collection_name: Name of the collection to use
            
        Raises:
            MilvusConnectionError: If connection fails
        """
        self._uri = uri
        self._collection_name = collection_name
        
        try:
            self._client = MilvusClient(uri=uri)
        except Exception as e:
            raise MilvusConnectionError(uri, e)
    
    def create_collection(self, dim: int = 2560) -> None:
        """Create the memories collection with full schema.
        
        Args:
            dim: Vector dimension (default 2560 for qwen3-embedding-4b)
        """
        # Check if collection exists
        if self._client.has_collection(self._collection_name):
            logger.info(f"Collection '{self._collection_name}' already exists")
            return
        
        # Build schema
        fields = []
        for name, dtype, params in self.SCHEMA_FIELDS:
            if name == "vector":
                params = {"dim": dim}
            field = FieldSchema(name=name, dtype=dtype, **params)
            fields.append(field)
        
        schema = CollectionSchema(
            fields=fields,
            description="AI Memory System storage",
            enable_dynamic_field=True
        )
        
        # Create collection with index
        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="AUTOINDEX",
            metric_type="COSINE"
        )
        
        self._client.create_collection(
            collection_name=self._collection_name,
            schema=schema,
            index_params=index_params
        )
        
        logger.info(f"Created collection '{self._collection_name}' with dim={dim}")
    
    def insert(self, entities: List[Dict[str, Any]]) -> List[int]:
        """Insert memory records.
        
        Args:
            entities: List of memory record dicts
            
        Returns:
            List of inserted record IDs
        """
        if not entities:
            return []
        
        # Ensure group_id is always present to satisfy collection schema
        for ent in entities:
            ent.setdefault("group_id", -1)
        
        result = self._client.insert(
            collection_name=self._collection_name,
            data=entities
        )
        
        ids = result.get("ids", [])
        # Convert to regular Python list if needed
        ids = list(ids) if ids else []
        logger.info(f"Inserted {len(ids)} records into '{self._collection_name}'")
        return ids
    
    def search(
        self,
        vectors: List[List[float]],
        filter_expr: str = "",
        limit: int = 10,
        output_fields: Optional[List[str]] = None
    ) -> List[List[Dict[str, Any]]]:
        """Vector similarity search.
        
        Args:
            vectors: Query vectors
            filter_expr: Filter expression (e.g., "user_id == 'u123'")
            limit: Maximum results per query
            output_fields: Fields to return (None for all)
            
        Returns:
            List of search results per query vector
        """
        if not vectors:
            return []
        
        if output_fields is None:
            output_fields = ["*"]
        
        results = self._client.search(
            collection_name=self._collection_name,
            data=vectors,
            filter=filter_expr,
            limit=limit,
            output_fields=output_fields
        )
        
        # Convert to list of dicts
        formatted_results = []
        for hits in results:
            formatted_hits = []
            for hit in hits:
                record = hit.get("entity", {})
                record["id"] = hit.get("id")
                record["distance"] = hit.get("distance")
                formatted_hits.append(record)
            formatted_results.append(formatted_hits)
        
        return formatted_results
    
    def query(
        self,
        filter_expr: str,
        output_fields: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Query records by filter expression.
        
        Args:
            filter_expr: Filter expression
            output_fields: Fields to return (None for all)
            limit: Maximum results
            
        Returns:
            List of matching records
        """
        if output_fields is None:
            output_fields = ["*"]
        
        results = self._client.query(
            collection_name=self._collection_name,
            filter=filter_expr,
            output_fields=output_fields,
            limit=limit
        )
        
        return results
    
    # ========== Groups Collection Operations ==========
    
    def _get_groups_collection_name(self, user_id: str) -> str:
        """Get the groups collection name for a user."""
        return f"groups_{user_id}"
    
    def create_groups_collection(self, user_id: str, dim: int = 2560) -> str:
        """Create groups collection for a user if it doesn't exist.
        
        Args:
            user_id: User identifier
            dim: Vector dimension
            
        Returns:
            Groups collection name
        """
        groups_collection_name = self._get_groups_collection_name(user_id)
        
        if self._client.has_collection(groups_collection_name):
            logger.info(f"Groups collection '{groups_collection_name}' already exists")
            return groups_collection_name
        
        # Build schema
        fields = []
        for name, dtype, params in self.GROUP_SCHEMA_FIELDS:
            if name == "centroid_vector":
                params = {"dim": dim}
            field = FieldSchema(name=name, dtype=dtype, **params)
            fields.append(field)
        
        schema = CollectionSchema(
            fields=fields,
            description=f"Narrative memory groups for user {user_id}",
            enable_dynamic_field=True
        )
        
        # Create collection with index
        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name="centroid_vector",
            index_type="AUTOINDEX",
            metric_type="IP"  # Inner product for normalized vectors
        )
        
        self._client.create_collection(
            collection_name=groups_collection_name,
            schema=schema,
            index_params=index_params
        )
        
        logger.info(f"Created groups collection '{groups_collection_name}' with dim={dim}")
        return groups_collection_name
    
    def search_groups(
        self,
        user_id: str,
        vector: List[float],
        limit: int = 1
    ) -> List[Dict[str, Any]]:
        """Search for similar groups by centroid vector.
        
        Args:
            user_id: User identifier
            vector: Query vector (normalized)
            limit: Maximum results
            
        Returns:
            List of matching groups with group_id, sim (distance), and size
        """
        groups_collection = self._get_groups_collection_name(user_id)
        
        if not self._client.has_collection(groups_collection):
            return []
        
        results = self._client.search(
            collection_name=groups_collection,
            data=[vector],
            anns_field="centroid_vector",
            limit=limit,
            search_params={"metric_type": "IP", "params": {"nprobe": 10}},
            filter=f"user_id == '{user_id}'",
            output_fields=["group_id", "size"],  # 显式请求主键字段
        )
        
        hits = results[0] if results else []
        groups = []
        for hit in hits:
            # MilvusClient search 返回的 hit 结构：
            # - hit["id"] 或 hit.get("id") 用于访问主键（但主键字段名决定实际key）
            # - 对于 groups 表，主键字段名是 "group_id"
            # - entity 中包含 output_fields 请求的字段
            entity = hit.get("entity", {})
            # 优先从 entity 获取 group_id（显式请求的字段），否则尝试 hit["id"]
            group_id = entity.get("group_id") or hit.get("id")
            groups.append({
                "group_id": group_id,
                "sim": hit.get("distance", 0),
                "size": entity.get("size", 0),
            })
        
        return groups
    
    def insert_group(
        self,
        user_id: str,
        centroid_vector: List[float],
        size: int = 1
    ) -> Optional[int]:
        """Insert a new group.
        
        Args:
            user_id: User identifier
            centroid_vector: Initial centroid vector
            size: Initial group size
            
        Returns:
            group_id of inserted group, or None on failure
        """
        groups_collection = self._get_groups_collection_name(user_id)
        
        # Ensure collection exists
        self.create_groups_collection(user_id, dim=len(centroid_vector))
        
        result = self._client.insert(
            collection_name=groups_collection,
            data=[{
                "user_id": user_id,
                "centroid_vector": centroid_vector,
                "size": size,
            }]
        )
        
        primary_keys = result.get("ids", [])
        if not primary_keys:
            primary_keys = result.get("primary_keys", [])
        
        group_id = primary_keys[0] if primary_keys else None
        
        if group_id:
            logger.info(f"Inserted group {group_id} for user {user_id}")
        else:
            logger.error(f"Failed to insert group for user {user_id}")
        
        return group_id
    
    def update_group(
        self,
        user_id: str,
        group_id: int,
        centroid_vector: Optional[List[float]] = None,
        size: Optional[int] = None
    ) -> bool:
        """Update a group's centroid and/or size.
        
        Args:
            user_id: User identifier
            group_id: Group ID to update
            centroid_vector: New centroid vector (optional)
            size: New size (optional)
            
        Returns:
            True if update succeeded
        """
        groups_collection = self._get_groups_collection_name(user_id)
        
        if not self._client.has_collection(groups_collection):
            return False
        
        try:
            # Fetch existing record
            existing = self._client.query(
                collection_name=groups_collection,
                filter=f"group_id == {group_id}",
                output_fields=["*"]
            )
            
            if not existing:
                logger.warning(f"Group {group_id} not found for update")
                return False
            
            record = existing[0].copy()
            
            if centroid_vector is not None:
                record["centroid_vector"] = centroid_vector
            if size is not None:
                record["size"] = size
            
            self._client.upsert(
                collection_name=groups_collection,
                data=[record]
            )
            
            logger.info(f"Updated group {group_id}")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to update group {group_id}: {e}")
            return False
    
    def delete_group(self, user_id: str, group_id: int) -> bool:
        """Delete a group.
        
        Args:
            user_id: User identifier
            group_id: Group ID to delete
            
        Returns:
            True if delete succeeded
        """
        groups_collection = self._get_groups_collection_name(user_id)
        
        if not self._client.has_collection(groups_collection):
            return False
        
        try:
            self._client.delete(
                collection_name=groups_collection,
                filter=f"group_id == {group_id} and user_id == '{user_id}'"
            )
            logger.info(f"Deleted group {group_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to delete group {group_id}: {e}")
            return False
    
    def update_memory_group_id(self, memory_id: int, group_id: int, user_id: str) -> bool:
        """Update a memory's group_id field.
        
        Args:
            memory_id: Memory ID to update
            group_id: New group ID
            user_id: User identifier
            
        Returns:
            True if update succeeded
        """
        try:
            # Fetch existing record
            existing = self._client.query(
                collection_name=self._collection_name,
                filter=f"id == {memory_id} and user_id == '{user_id}'",
                output_fields=["*"]
            )
            
            if not existing:
                logger.warning(f"Memory {memory_id} not found for group_id update")
                return False
            
            record = existing[0].copy()
            record["group_id"] = group_id
            
            self._client.upsert(
                collection_name=self._collection_name,
                data=[record]
            )
            
            logger.info(f"Updated memory {memory_id} group_id to {group_id}")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to update memory {memory_id} group_id: {e}")
            return False
    
    def delete(
        self,
        ids: Optional[List[int]] = None,
        filter_expr: Optional[str] = None
    ) -> int:
        """Delete memory records.
        
        Args:
            ids: List of record IDs to delete
            filter_expr: Filter expression for deletion
            
        Returns:
            Number of deleted records
        """
        if ids is not None:
            # Delete by IDs
            id_list = ",".join(str(i) for i in ids)
            filter_expr = f"id in [{id_list}]"
        
        if not filter_expr:
            logger.warning("No filter provided for delete operation")
            return 0
        
        # Count before delete
        before = self._client.query(
            collection_name=self._collection_name,
            filter=filter_expr,
            output_fields=["id"]
        )
        count = len(before)
        
        self._client.delete(
            collection_name=self._collection_name,
            filter=filter_expr
        )
        
        logger.info(f"Deleted {count} records from '{self._collection_name}'")
        return count
    
    def flush(self, timeout: Optional[float] = None) -> None:
        """Flush pending writes so they are immediately queryable."""
        self._client.flush(collection_name=self._collection_name, timeout=timeout)
    
    def count(self, filter_expr: str = "") -> int:
        """Count records matching filter.
        
        Args:
            filter_expr: Filter expression (empty for all)
            
        Returns:
            Number of matching records
        """
        if filter_expr:
            results = self._client.query(
                collection_name=self._collection_name,
                filter=filter_expr,
                output_fields=["id"]
            )
            return len(results)
        else:
            # Get collection stats
            stats = self._client.get_collection_stats(self._collection_name)
            return stats.get("row_count", 0)
    
    def drop_collection(self) -> None:
        """Drop the collection (for testing/cleanup)."""
        if self._client.has_collection(self._collection_name):
            self._client.drop_collection(self._collection_name)
            logger.info(f"Dropped collection '{self._collection_name}'")
