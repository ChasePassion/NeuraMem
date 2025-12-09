"""Narrative Memory Manager - Handles narrative memory grouping operations.

This module implements the logic to group episodic memories into narrative
groups based on vector similarity and usage patterns.
"""

import logging
import time
import numpy as np
from typing import List, Dict, Any, Optional

from ..clients import MilvusStore
from ..config import MemoryConfig
from langfuse import observe, get_client

logger = logging.getLogger(__name__)


def normalize(vec: np.ndarray) -> np.ndarray:
    """向量归一化"""
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


class NarrativeMemoryManager:
    """Manager for narrative memory grouping operations.
    
    This class handles logic of grouping episodic memories into
    narrative groups based on vector similarity.
    """
    
    def __init__(self, milvus_store: MilvusStore, config: MemoryConfig):
        """Initialize narrative memory manager.
        
        Args:
            milvus_store: Milvus store instance
            config: Memory system configuration
        """
        self._store = milvus_store
        self._config = config
        self._client = milvus_store._client
        self._user_collections = set()  # Track created user collections

    def _build_session_id(self, user_id: str, operation: str) -> str:
        """Build a stable session id for Langfuse traces."""
        return f"{operation}_{user_id}_{int(time.time())}"
    
    def _update_record(self, collection_name: str, record_id: int, data: Dict[str, Any], id_field: str = "id") -> bool:
        """更新记录（使用upsert实现）
        
        Args:
            collection_name: collection名称
            record_id: 记录ID
            data: 要更新的字段
            id_field: 主键字段名
            
        Returns:
            是否更新成功
        """
        try:
            # 先查询现有记录
            existing = self._client.query(
                collection_name=collection_name,
                filter=f"{id_field} == {record_id}",
                output_fields=["*"]
            )
            
            if not existing:
                logger.warning(f"Record {record_id} not found in {collection_name}")
                return False
            
            # 合并数据
            record = existing[0].copy()
            record.update(data)
            
            # 使用upsert更新
            self._client.upsert(
                collection_name=collection_name,
                data=[record]
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update record {record_id} in {collection_name}: {e}")
            return False
    
    def _ensure_groups_collection(self, user_id: str) -> str:
        """确保用户特定的groups collection存在
        
        Args:
            user_id: 用户标识
            
        Returns:
            groups collection名称
        """
        groups_collection_name = f"groups_{user_id}"
        
        if groups_collection_name in self._user_collections:
            return groups_collection_name
        
        if self._client.has_collection(groups_collection_name):
            logger.info(f"Groups collection '{groups_collection_name}' already exists")
            self._user_collections.add(groups_collection_name)
            return groups_collection_name
        
        # 创建groups collection schema
        fields = []
        for name, dtype, params in MilvusStore.GROUP_SCHEMA_FIELDS:
            if name == "centroid_vector":
                params = {"dim": self._config.embedding_dim}
            from pymilvus import FieldSchema
            field = FieldSchema(name=name, dtype=dtype, **params)
            fields.append(field)
        
        from pymilvus import CollectionSchema
        schema = CollectionSchema(
            fields=fields,
            description=f"Narrative memory groups for user {user_id}",
            enable_dynamic_field=True
        )
        
        # 创建collection with index
        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name="centroid_vector",
            index_type="AUTOINDEX",
            metric_type="IP"  # 使用内积，因为向量已归一化
        )
        
        self._client.create_collection(
            collection_name=groups_collection_name,
            schema=schema,
            index_params=index_params
        )
        
        self._user_collections.add(groups_collection_name)
        logger.info(f"Created groups collection '{groups_collection_name}' with dim={self._config.embedding_dim}")
        return groups_collection_name
    
    @observe(as_type="chain", name="narrative_assign_to_group")
    def assign_to_narrative_group(self, memory_ids: List[int], user_id: str) -> Dict[int, int]:
        """将被使用的情景记忆分配到叙事组。
        
        Args:
            memory_ids: 被MemoryUsageJudge判断为实际使用的情景记忆ID列表
            user_id: 用户标识
            
        Returns:
            Dict[int, int] - memory_id到group_id的映射
        """
        session_id = self._build_session_id(user_id, "narrative_assign")
        get_client().update_current_trace(
            session_id=session_id,
            user_id=user_id,
            tags=["narrative_memory", "group_assignment"],
            metadata={
                "requested_memory_ids": memory_ids,
                "memory_ids_count": len(memory_ids),
                "collection": self._store._collection_name,
                "similarity_threshold": self._config.narrative_similarity_threshold,
            }
        )

        if not memory_ids:
            get_client().update_current_trace(
                session_id=session_id,
                output={"assigned_groups": {}, "success": True},
                metadata={"reason": "empty_input"}
            )
            return {}
        
        results = {}
        memories_collection = self._store._collection_name
        groups_collection = self._ensure_groups_collection(user_id)
        created_groups = 0
        reused_groups = 0
        failed_ids = []
        
        for memory_id in memory_ids:
            try:
                # 步骤1：检查是否已分组
                mem_res = self._client.query(
                    collection_name=memories_collection,
                    filter=f"id == {memory_id} and user_id == '{user_id}'",
                    output_fields=["id", "group_id", "vector"],
                )
                
                if not mem_res:
                    logger.warning(f"Memory {memory_id} not found, skipping")
                    continue
                
                current_group_id = mem_res[0]["group_id"]
                v_mem = np.array(mem_res[0]["vector"])
                v_mem = normalize(v_mem)
                
                # 如果已经分组，跳过
                if current_group_id != -1:
                    logger.debug(f"Memory {memory_id} already in group {current_group_id}")
                    results[memory_id] = current_group_id
                    continue
                
                # 步骤2：在groups上做ANN搜索，找最相似组
                group_search_res = self._client.search(
                    collection_name=groups_collection,
                    data=[v_mem.tolist()],
                    anns_field="centroid_vector",
                    limit=1,  # 只要top-1
                    search_params={"metric_type": "IP", "params": {"nprobe": 10}},
                    filter=f"user_id == '{user_id}'",
                    output_fields=["size"],
                )
                
                hits = group_search_res[0] if group_search_res else []
                best_group = None
                
                if hits:
                    hit0 = hits[0]
                    best_group = {
                        "group_id": hit0.get("id"),
                        "sim": hit0.get("distance", 0),  # 单位向量 + IP => cos 相似度
                        "size": hit0.get("entity", {}).get("size", 0),
                    }
                
                # 步骤3：阈值判断：新建组 or 加入已有组
                threshold = self._config.narrative_similarity_threshold
                
                if best_group is None or best_group["sim"] < threshold:
                    # 步骤3.1：新建组
                    # 使用列表格式插入单条记录
                    group_insert_res = self._client.insert(
                        collection_name=groups_collection,
                        data=[{
                            "user_id": user_id,
                            "centroid_vector": v_mem.tolist(),
                            "size": 1,
                        }]
                    )
                    # 获取插入的主键ID
                    primary_keys = group_insert_res.get("ids", [])
                    if not primary_keys:
                        primary_keys = group_insert_res.get("primary_keys", [])
                    group_id = primary_keys[0] if primary_keys else None
                    
                    if group_id is None:
                        logger.error(f"Failed to get group_id from insert result: {group_insert_res}")
                        continue
                    
                    # 更新memory的group_id
                    self._update_record(
                        collection_name=memories_collection,
                        record_id=memory_id,
                        data={"group_id": group_id}
                    )
                    
                    logger.info(f"Created new group {group_id} for memory {memory_id}")
                    results[memory_id] = group_id
                    created_groups += 1
                    
                else:
                    # 步骤3.2：加入已有组（重算中心）
                    group_id = best_group["group_id"]
                    
                    # 1) 更新memories.group_id
                    self._update_record(
                        collection_name=memories_collection,
                        record_id=memory_id,
                        data={"group_id": group_id}
                    )
                    
                    # 2) 重算这个组的centroid_vector & size（精确版）
                    members_res = self._client.query(
                        collection_name=memories_collection,
                        filter=f"group_id == {group_id} and user_id == '{user_id}'",
                        output_fields=["id", "vector"],
                    )
                    vectors = [row["vector"] for row in members_res]
                    size = len(vectors)
                    
                    if vectors:
                        new_centroid = normalize(np.mean(np.array(vectors), axis=0))
                        
                        self._update_record(
                            collection_name=groups_collection,
                            record_id=group_id,
                            data={
                                "centroid_vector": new_centroid.tolist(),
                                "size": size,
                            },
                            id_field="group_id"
                        )
                    
                    logger.info(f"Added memory {memory_id} to existing group {group_id} (size: {size})")
                    results[memory_id] = group_id
                    reused_groups += 1
                    
            except Exception as e:
                logger.error(f"Failed to assign memory {memory_id} to narrative group: {e}")
                failed_ids.append(memory_id)
                continue

        get_client().update_current_trace(
            session_id=session_id,
            output={
                "assigned_groups": results,
                "created_groups": created_groups,
                "reused_groups": reused_groups,
                "failed_ids": failed_ids,
                "success": True
            },
            metadata={
                "completed_memory_ids": list(results.keys()),
                "missing_memory_ids": [mid for mid in memory_ids if mid not in results and mid not in failed_ids],
                "threshold": self._config.narrative_similarity_threshold
            }
        )
        
        return results
    
    @observe(as_type="chain", name="narrative_delete_from_group")
    def delete_memory_from_group(self, memory_id: int, user_id: str) -> None:
        """删除记忆时同步更新叙事组。
        
        Args:
            memory_id: 要删除的记忆ID
            user_id: 用户标识
        """
        session_id = self._build_session_id(user_id, "narrative_delete")
        get_client().update_current_trace(
            session_id=session_id,
            user_id=user_id,
            tags=["narrative_memory", "group_cleanup"],
            metadata={"memory_id": memory_id}
        )
        try:
            memories_collection = self._store._collection_name
            groups_collection = self._ensure_groups_collection(user_id)
            
            # 步骤1：查出group_id
            res = self._client.query(
                collection_name=memories_collection,
                filter=f"id == {memory_id} and user_id == '{user_id}'",
                output_fields=["group_id"],
            )
            
            if not res:
                logger.warning(f"Memory {memory_id} not found for group cleanup")
                get_client().update_current_trace(
                    session_id=session_id,
                    output={"found": False, "cleanup_performed": False}
                )
                return
            
            group_id = res[0]["group_id"]
            group_deleted = False
            group_updated = False
            
            # 步骤2：删除memories中该条记录（这个在Memory.delete中会处理）
            
            # 步骤3：如有必要，更新或删除组
            if group_id != -1:
                members_res = self._client.query(
                    collection_name=memories_collection,
                    filter=f"group_id == {group_id} and user_id == '{user_id}'",
                    output_fields=["id", "vector"],
                )
                n = len(members_res)
                
                if n == 0:
                    # 该组已经空了，删除组
                    self._client.delete(
                        collection_name=groups_collection,
                        filter=f"group_id == {group_id} and user_id == '{user_id}'"
                    )
                    logger.info(f"Deleted empty group {group_id}")
                    group_deleted = True
                else:
                    vectors = [row["vector"] for row in members_res]
                    if vectors:
                        new_centroid = normalize(np.mean(np.array(vectors), axis=0))
                        self._update_record(
                            collection_name=groups_collection,
                            record_id=group_id,
                            data={
                                "centroid_vector": new_centroid.tolist(),
                                "size": n,
                            },
                            id_field="group_id"
                        )
                        logger.info(f"Updated group {group_id} centroid (size: {n})")
                        group_updated = True
            
            get_client().update_current_trace(
                session_id=session_id,
                output={
                    "found": True,
                    "group_id": group_id,
                    "group_deleted": group_deleted,
                    "group_updated": group_updated
                }
            )
                    
        except Exception as e:
            logger.error(f"Failed to cleanup group for memory {memory_id}: {e}")
            get_client().update_current_trace(
                session_id=session_id,
                output={"found": False, "error": str(e)}
            )
    
    @observe(as_type="chain", name="narrative_get_group_members")
    def get_group_members(self, group_id: int, user_id: str) -> List[Dict[str, Any]]:
        """获取叙事组中的所有成员。
        
        Args:
            group_id: 叙事组ID
            user_id: 用户标识
            
        Returns:
            组内所有记忆列表
        """
        session_id = self._build_session_id(user_id, "narrative_members")
        get_client().update_current_trace(
            session_id=session_id,
            user_id=user_id,
            tags=["narrative_memory", "group_members"],
            metadata={"group_id": group_id}
        )
        try:
            memories_collection = self._store._collection_name
            
            members_res = self._client.query(
                collection_name=memories_collection,
                filter=f"group_id == {group_id} and user_id == '{user_id}'",
                output_fields=["id", "user_id", "memory_type", "ts", "chat_id", "text", "group_id"],
            )
            
            get_client().update_current_trace(
                session_id=session_id,
                output={
                    "group_id": group_id,
                    "members_count": len(members_res),
                    "success": True
                }
            )
            return members_res
            
        except Exception as e:
            logger.error(f"Failed to get members for group {group_id}: {e}")
            get_client().update_current_trace(
                session_id=session_id,
                output={"group_id": group_id, "success": False, "error": str(e)}
            )
            return []
