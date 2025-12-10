"""Narrative Memory Manager - Handles narrative memory grouping operations.

This module implements the logic to group episodic memories into narrative
groups based on vector similarity and usage patterns.
"""

import logging
import time
import numpy as np
from typing import List, Dict, Any

from ..clients import MilvusStore
from ..config import MemoryConfig
from ..utils import normalize
from langfuse import observe, get_client

logger = logging.getLogger(__name__)


class NarrativeMemoryManager:
    """Manager for narrative memory grouping operations.
    
    This class handles logic of grouping episodic memories into
    narrative groups based on vector similarity.
    
    Uses MilvusStore's public API for all database operations.
    """
    
    def __init__(self, milvus_store: MilvusStore, config: MemoryConfig):
        """Initialize narrative memory manager.
        
        Args:
            milvus_store: Milvus store instance
            config: Memory system configuration
        """
        self._store = milvus_store
        self._config = config

    def _build_session_id(self, user_id: str, operation: str) -> str:
        """Build a stable session id for Langfuse traces."""
        return f"{operation}_{user_id}_{int(time.time())}"
    
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
        created_groups = 0
        reused_groups = 0
        failed_ids = []
        
        # Ensure groups collection exists
        self._store.create_groups_collection(user_id, dim=self._config.embedding_dim)
        
        for memory_id in memory_ids:
            try:
                # 步骤1：检查是否已分组
                mem_res = self._store.query(
                    filter_expr=f"id == {memory_id} and user_id == '{user_id}'",
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
                group_hits = self._store.search_groups(
                    user_id=user_id,
                    vector=v_mem.tolist(),
                    limit=1
                )
                
                best_group = group_hits[0] if group_hits else None
                
                # 步骤3：阈值判断：新建组 or 加入已有组
                threshold = self._config.narrative_similarity_threshold
                
                if best_group is None or best_group["sim"] < threshold:
                    # 步骤3.1：新建组
                    group_id = self._store.insert_group(
                        user_id=user_id,
                        centroid_vector=v_mem.tolist(),
                        size=1
                    )
                    
                    if group_id is None:
                        logger.error(f"Failed to create group for memory {memory_id}")
                        failed_ids.append(memory_id)
                        continue
                    
                    # 更新memory的group_id
                    self._store.update_memory_group_id(memory_id, group_id, user_id)
                    
                    logger.info(f"Created new group {group_id} for memory {memory_id}")
                    results[memory_id] = group_id
                    created_groups += 1
                    
                else:
                    # 步骤3.2：加入已有组（重算中心）
                    group_id = best_group["group_id"]
                    
                    # 防止 group_id 为 None 导致无效查询表达式
                    if group_id is None:
                        logger.error(f"Invalid group_id (None) from search_groups for memory {memory_id}")
                        failed_ids.append(memory_id)
                        continue
                    
                    # 1) 更新memories.group_id
                    self._store.update_memory_group_id(memory_id, group_id, user_id)
                    
                    # 2) 重算这个组的centroid_vector & size（精确版）
                    members_res = self._store.query(
                        filter_expr=f"group_id == {group_id} and user_id == '{user_id}'",
                        output_fields=["id", "vector"],
                    )
                    vectors = [row["vector"] for row in members_res]
                    size = len(vectors)
                    
                    if vectors:
                        new_centroid = normalize(np.mean(np.array(vectors), axis=0))
                        self._store.update_group(
                            user_id=user_id,
                            group_id=group_id,
                            centroid_vector=new_centroid.tolist(),
                            size=size
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
            # 步骤1：查出group_id
            res = self._store.query(
                filter_expr=f"id == {memory_id} and user_id == '{user_id}'",
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
            
            # 步骤2：如有必要，更新或删除组
            if group_id != -1:
                members_res = self._store.query(
                    filter_expr=f"group_id == {group_id} and user_id == '{user_id}'",
                    output_fields=["id", "vector"],
                )
                n = len(members_res)
                
                if n == 0:
                    # 该组已经空了，删除组
                    self._store.delete_group(user_id, group_id)
                    logger.info(f"Deleted empty group {group_id}")
                    group_deleted = True
                else:
                    vectors = [row["vector"] for row in members_res]
                    if vectors:
                        new_centroid = normalize(np.mean(np.array(vectors), axis=0))
                        self._store.update_group(
                            user_id=user_id,
                            group_id=group_id,
                            centroid_vector=new_centroid.tolist(),
                            size=n
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
            members_res = self._store.query(
                filter_expr=f"group_id == {group_id} and user_id == '{user_id}'",
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
