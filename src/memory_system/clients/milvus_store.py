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
    
    Provides CRUD operations for the memories collection with
    support for vector similarity search.
    
    Simplified schema (v2):
    - id: INT64, primary key, auto_id
    - user_id: VARCHAR(128), user identifier
    - memory_type: VARCHAR(32), "episodic" or "semantic"
    - ts: INT64, Unix timestamp of write time
    - chat_id: VARCHAR(128), conversation/thread identifier
    - text: VARCHAR(65535), main natural-language content used for embedding and search
            (includes time, where, who, thing, reason)
    - vector: FLOAT_VECTOR(2560), embedding vector for similarity search
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
    
    def update(self, id: int, data: Dict[str, Any], base_record: Optional[Dict[str, Any]] = None) -> bool:
        """Update a memory record.
        
        Args:
            id: Record ID to update
            data: Fields to update
            base_record: Optional record payload already fetched (avoids a second query)
            
        Returns:
            True if update succeeded
        """
        try:
            record = None
            
            # Prefer caller-provided base record (e.g., from search hit)
            if base_record:
                record = base_record.copy()
            else:
                # Fetch current record to preserve required fields
                existing = self._client.query(
                    collection_name=self._collection_name,
                    filter=f"id in [{int(id)}]",
                    output_fields=["*"]
                )
                if existing:
                    record = existing[0].copy()
            
            if record is None:
                logger.warning(f"Record {id} not found for update")
                return False
            
            # Merge fields and keep the same primary key
            record.update(data)
            record["id"] = id
            
            # Upsert keeps the primary key stable and avoids delete/reinsert races
            self._client.upsert(
                collection_name=self._collection_name,
                data=[record]
            )
            
            # Make the change immediately visible
            self._client.flush(collection_name=self._collection_name)
            
            logger.info(f"Updated record {id}")
            return True
        
        except Exception as e:
            logger.warning(f"Failed to update record {id}: {e}")
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
