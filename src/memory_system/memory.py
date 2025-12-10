"""Memory class - Main API for AI Memory System.

Provides a mem0-style interface for memory operations including
add, search, update, delete, reset, and consolidate.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np
from langfuse import observe, get_client
from .config import MemoryConfig
from .clients import EmbeddingClient, LLMClient, MilvusStore
from .processors import (
    EpisodicMemoryManager,
    SemanticWriter,
    MemoryUsageJudge,
    NarrativeMemoryManager,
)
from .utils import normalize

logger = logging.getLogger(__name__)


@dataclass
class MemoryRecord:
    """A memory record returned from search operations.
    
    Schema fields:
    - id: Record ID
    - user_id: User identifier
    - memory_type: "episodic" or "semantic"
    - ts: Unix timestamp of write time
    - chat_id: Conversation/thread identifier
    - text: Main natural-language content
    - group_id: Narrative group ID (-1 = ungrouped)
    """
    id: int
    user_id: str
    memory_type: str
    ts: int
    chat_id: str
    text: str
    group_id: int = -1


@dataclass
class ConsolidationStats:
    """Statistics from a consolidation run."""
    memories_processed: int = 0
    semantic_created: int = 0


class Memory:
    """AI Memory System main class.
    
    Provides a mem0-style API interface for memory operations.
    Uses factory pattern to instantiate infrastructure clients and processors.
    """
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        """Initialize the memory system.
        
        Args:
            config: Memory system configuration. Uses defaults if not provided.
            
        Raises:
            MilvusConnectionError: If connection to Milvus fails
        """
        self._config = config or MemoryConfig()
        
        # Initialize infrastructure clients using factory pattern
        self._embedding_client = self._create_embedding_client()
        self._llm_client = self._create_llm_client()
        self._store = self._create_milvus_store()
        
        # Initialize processor modules
        self._memory_manager = EpisodicMemoryManager(self._llm_client)
        self._semantic_writer = SemanticWriter(self._llm_client)
        self._memory_usage_judge = MemoryUsageJudge(self._llm_client)
        self._narrative_manager = NarrativeMemoryManager(self._store, self._config)
        
        # Create collection if not exists
        self._store.create_collection(dim=self._config.embedding_dim)
        
        # Initialize Langfuse client if available
        self._langfuse_client = self._create_langfuse_client()
        
        logger.info(
            f"Memory system initialized with collection '{self._config.collection_name}'"
        )
    
    def _create_embedding_client(self) -> EmbeddingClient:
        """Factory method to create EmbeddingClient."""
        return EmbeddingClient(
            api_key=self._config.embedding_api_key,
            base_url=self._config.embedding_base_url,
            model=self._config.embedding_model
        )
    
    def _create_llm_client(self) -> LLMClient:
        """Factory method to create LLMClient."""
        return LLMClient(
            api_key=self._config.llm_primary_api_key,
            base_url=self._config.llm_primary_base_url,
            model=self._config.llm_primary_model,
            fallback_api_key=self._config.llm_fallback_api_key,
            fallback_base_url=self._config.llm_fallback_base_url,
            fallback_model=None
        )
    
    def _create_milvus_store(self) -> MilvusStore:
        """Factory method to create MilvusStore."""
        return MilvusStore(
            uri=self._config.milvus_uri,
            collection_name=self._config.collection_name
        )
    
    def _create_langfuse_client(self):
        """Create Langfuse client if configuration is available."""

        secret_key = self._config.langfuse_secret_key
        public_key = self._config.langfuse_public_key
        host = self._config.langfuse_base_url

        if not (secret_key and public_key):
            logger.info("Langfuse keys not configured; skipping Langfuse client initialization")
            return None

        try:
            from langfuse import Langfuse
            client = Langfuse(
                secret_key=secret_key,
                public_key=public_key,
                host=host
            )
            logger.info(f"Langfuse client initialized with host '{host}'")
            return client
        except Exception as e:
            logger.warning(f"Failed to create Langfuse client: {e}")
            return None
    
    def _generate_session_id(self, user_id: str, chat_id: str) -> str:
        """Generate consistent session ID for Langfuse tracking.
        
        Args:
            user_id: User identifier
            chat_id: Chat identifier
            
        Returns:
            Consistent session ID string
        """
        return f"chat_{user_id}_{chat_id}"

    @observe(as_type="agent", name="memory_manage_operation")
    def manage(
        self,
        user_text: str,
        assistant_text: str,
        user_id: str,
        chat_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[int]:
        """Manage memories with CRUD operations based on conversation.
        
        Processes conversation through EpisodicMemoryManager to determine
        add, update, or delete operations on episodic memories.
        
        Args:
            user_text: User input text
            assistant_text: Assistant response text
            user_id: User identifier
            chat_id: Conversation/thread identifier
            metadata: Optional additional metadata (ignored in v2 schema)
            
        Returns:
            List of newly added memory IDs
        """
        get_client().update_current_trace(
            session_id=self._generate_session_id(user_id, chat_id),
            user_id=user_id,
            tags=["memory_manage", "episodic"],
            metadata={
                "operation": "manage_memory",
                "chat_id": chat_id,
                "user_text_length": len(user_text),
                "assistant_text_length": len(assistant_text)
            }
        )
        
        # 1. Query user all episodic memories (不限制数量)
        episodic_filter = f'user_id == "{user_id}" and memory_type == "episodic"'
        episodic_memories = self._store.query(filter_expr=episodic_filter, limit=10000)
        
        # 2. 调用记忆管理器
        result = self._memory_manager.manage_memories(
            user_text=user_text,
            assistant_text=assistant_text,
            episodic_memories=episodic_memories
        )
        
        # 3. 执行CRUD操作
        added_ids = []
        
        # 处理删除操作
        for op in result.operations:
            if op.operation_type == "delete":
                self.delete(op.memory_id, user_id)
        
        # 处理更新操作
        for op in result.operations:
            if op.operation_type == "update":
                self.update(op.memory_id, {"text": op.text}, user_id)
        
        # 处理添加操作
        add_operations = [op for op in result.operations if op.operation_type == "add"]
        if add_operations:
            add_texts = [op.text for op in add_operations]
            embeddings = self._embedding_client.encode(add_texts)
            
            current_ts = int(time.time())
            entities = []
            
            for i, text in enumerate(add_texts):
                entity = {
                    "user_id": user_id,
                    "memory_type": "episodic",
                    "ts": current_ts,
                    "chat_id": chat_id,
                    "text": text,
                    "vector": embeddings[i],
                    "group_id": -1,
                }
                entities.append(entity)
            
            added_ids = self._store.insert(entities)
        
        # 记录最终的操作结果到Langfuse
        operation_summary = {
            "added_count": len(added_ids),
            "updated_count": len([op for op in result.operations if op.operation_type == 'update']),
            "deleted_count": len([op for op in result.operations if op.operation_type == 'delete']),
            "total_operations": len(result.operations),
            "decision_trace_available": True
        }
        
        get_client().update_current_trace(
            output={
                "operation_summary": operation_summary,
                "added_memory_ids": added_ids,
                "execution_success": True
            },
            metadata={
                "episodic_memories_processed": len(episodic_memories),
                "final_operation_counts": operation_summary
            }
        )
        
        logger.info(
            f"Memory operation 'manage': type=episodic, user_id={user_id}, "
            f"chat_id={chat_id}, added={len(added_ids)}, "
            f"updated={len([op for op in result.operations if op.operation_type == 'update'])}, "
            f"deleted={len([op for op in result.operations if op.operation_type == 'delete'])}"
        )
        
        return added_ids

    @observe(as_type="agent")
    async def manage_async(
        self,
        user_text: str,
        assistant_text: str,
        user_id: str,
        chat_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[int]:
        """Async variant of manage that offloads blocking steps to a thread pool."""
        get_client().update_current_trace(
            session_id=self._generate_session_id(user_id, chat_id),
            user_id=user_id,
            tags=["memory_manage_async", "episodic"],
            metadata={
                "operation": "manage_memory_async",
                "chat_id": chat_id,
                "user_text_length": len(user_text),
                "assistant_text_length": len(assistant_text)
            }
        )

        # Query user all episodic memories (不限制数量)
        episodic_filter = f'user_id == "{user_id}" and memory_type == "episodic"'
        episodic_memories = await asyncio.to_thread(
            self._store.query, filter_expr=episodic_filter, limit=10000
        )
        
        # 调用记忆管理器 (LLM call in background thread)
        result = await asyncio.to_thread(
            self._memory_manager.manage_memories,
            user_text=user_text,
            assistant_text=assistant_text,
            episodic_memories=episodic_memories
        )
        
        # 执行CRUD操作
        added_ids = []
        
        # 处理删除操作
        for op in result.operations:
            if op.operation_type == "delete":
                await asyncio.to_thread(self.delete, op.memory_id, user_id)
        
        # 处理更新操作
        for op in result.operations:
            if op.operation_type == "update":
                await asyncio.to_thread(self.update, op.memory_id, {"text": op.text}, user_id)
        
        # 处理添加操作
        add_operations = [op for op in result.operations if op.operation_type == "add"]
        if add_operations:
            add_texts = [op.text for op in add_operations]
            
            # Embedding + Milvus insert are synchronous; run them in threads to avoid blocking event loop
            embeddings = await asyncio.to_thread(self._embedding_client.encode, add_texts)
            
            current_ts = int(time.time())
            entities = []
            
            for i, text in enumerate(add_texts):
                entity = {
                    "user_id": user_id,
                    "memory_type": "episodic",
                    "ts": current_ts,
                    "chat_id": chat_id,
                    "text": text,
                    "vector": embeddings[i],
                    "group_id": -1,
                }
                entities.append(entity)
            
            added_ids = await asyncio.to_thread(self._store.insert, entities)

        logger.info(
            f"Memory operation 'manage_async': type=episodic, user_id={user_id}, "
            f"chat_id={chat_id}, added={len(added_ids)}, "
            f"updated={len([op for op in result.operations if op.operation_type == 'update'])}, "
            f"deleted={len([op for op in result.operations if op.operation_type == 'delete'])}"
        )

        return added_ids

    @observe(as_type="agent")
    def search(
        self,
        query: str,
        user_id: str
    ) -> Dict[str, List[MemoryRecord]]:
        """Search memories for a user with narrative group expansion.
        
        Retrieves both episodic and semantic memories. Episodic memories are
        expanded to include all members of their narrative groups.
        
        Args:
            query: Search query text
            user_id: User identifier
            
        Returns:
            Dict with separated episodic and semantic memories
        """
        import numpy as np
        
        get_client().update_current_trace(
            session_id=f"search_{user_id}_{int(time.time())}",
            user_id=user_id,
            tags=["memory_search", "retrieval", "narrative_expansion"],
            metadata={
                "operation": "search_memory",
                "query_length": len(query)
            }
        )
        
        # Generate embedding for query
        query_vectors = self._embedding_client.encode([query])
        
        if not query_vectors:
            return {"episodic": [], "semantic": []}
        
        query_vector = query_vectors[0]
        q = normalize(np.array(query_vector))
        
        # 根据配置选择不同的语义记忆获取方式
        if self._config.use_all_semantic:
            # 直接查询所有语义记忆，跳过向量检索
            semantic_filter = f'user_id == "{user_id}" and memory_type == "semantic"'
            semantic_records = self._store.query(filter_expr=semantic_filter, limit=1000)
            semantic_memories = [self._hit_to_memory_record(hit) for hit in semantic_records]
        else:
            # 使用向量检索获取前k条最相关的语义记忆
            semantic_filter = f'user_id == "{user_id}" and memory_type == "semantic"'
            semantic_results = self._store.search(
                vectors=[query_vector],
                filter_expr=semantic_filter,
                limit=self._config.k_semantic
            )
            semantic_memories = []
            if semantic_results and semantic_results[0]:
                semantic_memories = [self._hit_to_memory_record(hit) for hit in semantic_results[0]]
        
        # 步骤1：向量检索情景记忆种子
        episodic_filter = f'user_id == "{user_id}" and memory_type == "episodic"'
        episodic_results = self._store.search(
            vectors=[q.tolist()],
            filter_expr=episodic_filter,
            limit=self._config.k_episodic,
            output_fields=["id", "group_id", "user_id", "memory_type", "ts", "chat_id", "text"],
        )
        
        seeds = episodic_results[0] if episodic_results and episodic_results[0] else []
        if not seeds:
            # 无种子，直接返回空情景记忆 + 语义记忆
            result = {
                "episodic": [],
                "semantic": semantic_memories
            }
            logger.info(
                f"Memory operation 'search': user_id={user_id}, "
                f"episodic_results=0, semantic_results={len(semantic_memories)}"
            )
            return result
        
        # 步骤2：根据种子的group_id决定扩展哪些组
        expansion_group_ids = set()
        
        for hit in seeds:
            g_id = hit.get("group_id")
            # 只有group_id >= 0的才扩展，-1表示未分组，不扩展
            if g_id is None or g_id == -1:
                continue
            expansion_group_ids.add(g_id)
        
        # 步骤3：拉出这些扩展组的所有成员
        expanded_member_ids = set()
        
        for g_id in expansion_group_ids:
            members_res = self._store.query(
                filter_expr=f"group_id == {g_id} and user_id == '{user_id}'",
                output_fields=["id"],
            )
            member_ids = [row["id"] for row in members_res]
            # 不限制每组的记忆数
            expanded_member_ids.update(member_ids)
        
        # 步骤4：合并种子 + 扩展成员 → 去重 → 拉完整内容
        seed_ids = {hit["id"] for hit in seeds}
        all_ids = seed_ids | expanded_member_ids
        
        if not all_ids:
            final_memories = []
        else:
            id_list = list(all_ids)
            
            mem_res = self._store.query(
                filter_expr=f"id in {id_list} and user_id == '{user_id}'",
                output_fields=["id", "user_id", "memory_type", "ts", "chat_id", "text", "group_id"],
            )
            
            id2row = {row["id"]: row for row in mem_res}
            
            final_memories = []
            
            # 先放种子，保证它们在prompt里靠前（按相似度排序）
            for hit in seeds:
                row = id2row.get(hit["id"])
                if row:
                    final_memories.append(self._hit_to_memory_record(row))
            
            # 再放扩展成员（去掉已经是种子的）
            for mid in expanded_member_ids:
                if mid in seed_ids:
                    continue
                row = id2row.get(mid)
                if row:
                    final_memories.append(self._hit_to_memory_record(row))
        
        result = {
            "episodic": final_memories,  # 种子 + 叙事组扩展
            "semantic": semantic_memories
        }
        
        # Count by type for logging
        logger.info(
            f"Memory operation 'search': user_id={user_id}, "
            f"episodic_results={len(final_memories)} (seeds={len(seeds)}, expanded={len(expanded_member_ids)}), "
            f"semantic_results={len(semantic_memories)}"
        )
        
        return result

    
    def _hit_to_memory_record(self, hit: Dict[str, Any]) -> MemoryRecord:
        """Convert a search hit to MemoryRecord."""
        return MemoryRecord(
            id=hit.get("id", 0),
            user_id=hit.get("user_id", ""),
            memory_type=hit.get("memory_type", ""),
            ts=hit.get("ts", 0),
            chat_id=hit.get("chat_id", ""),
            text=hit.get("text", ""),
            group_id=hit.get("group_id", -1)
        )
    
    @observe(as_type="agent", name="memory_assign_to_narrative_group")
    def assign_to_narrative_group(self, memory_ids: List[int], user_id: str) -> Dict[int, int]:
        """将被使用的情景记忆分配到叙事组。
        
        Args:
            memory_ids: 被MemoryUsageJudge判断为实际使用的情景记忆ID列表
            user_id: 用户标识
            
        Returns:
            Dict[int, int] - memory_id到group_id的映射
        """
        session_id = f"narrative_assign_{user_id}_{int(time.time())}"
        get_client().update_current_trace(
            session_id=session_id,
            user_id=user_id,
            tags=["narrative_memory", "group_assignment"],
            metadata={
                "memory_ids": memory_ids,
                "memory_ids_count": len(memory_ids)
            }
        )

        assignments = self._narrative_manager.assign_to_narrative_group(memory_ids, user_id)

        get_client().update_current_trace(
            session_id=session_id,
            output={
                "assigned_groups": assignments,
                "requested_ids_count": len(memory_ids),
                "assigned_ids_count": len(assignments),
                "success": True
            },
            metadata={
                "missing_ids": [mid for mid in memory_ids if mid not in assignments]
            }
        )

        return assignments
    
    


    def update(self, memory_id: int, data: Dict[str, Any], user_id: str = None) -> bool:
        """Update a memory record using delete + add strategy.
        
        Uses delete + add strategy to handle narrative group properly.
        Updated memory will have group_id reset to -1 and will be
        reassigned to narrative group only when used in future conversations.
        
        Args:
            memory_id: ID of memory to update
            data: Fields to update (must include 'text')
            user_id: User ID (required for narrative group cleanup)
            
        Returns:
            True if update succeeded
        """
        if "text" not in data:
            logger.warning(f"Memory operation 'update' failed: memory_id={memory_id}, no 'text' field provided")
            return False
        
        if user_id is None:
            logger.warning(f"Memory operation 'update' failed: memory_id={memory_id}, user_id is required")
            return False
        
        try:
            # 1. 查询原记忆信息
            original_memories = self._store.query(
                filter_expr=f"id == {memory_id} and user_id == '{user_id}'",
                output_fields=["id", "user_id", "memory_type", "ts", "chat_id", "text"]
            )
            
            if not original_memories:
                logger.warning(f"Memory operation 'update' failed: memory_id={memory_id}, memory not found")
                return False
            
            original = original_memories[0]
            
            # 2. 删除原记忆（会自动处理叙事组清理）
            self.delete(memory_id, user_id)
            
            # 3. 创建新记忆（group_id默认为-1）
            new_text = data["text"]
            embeddings = self._embedding_client.encode([new_text])
            
            if not embeddings:
                logger.warning(f"Memory operation 'update' failed: memory_id={memory_id}, embedding generation failed")
                return False
            
            entity = {
                "user_id": user_id,
                "memory_type": original["memory_type"],
                "ts": int(time.time()),  # 更新时间戳
                "chat_id": original["chat_id"],
                "text": new_text,
                "vector": embeddings[0],
                "group_id": -1
            }
            
            new_ids = self._store.insert([entity])
            
            if new_ids:
                logger.info(
                    f"Memory operation 'update': memory_id={memory_id} -> new_id={new_ids[0]}, "
                    f"text_updated=True, group_reset=True, affected_count=1"
                )
                return True
            else:
                logger.warning(f"Memory operation 'update' failed: memory_id={memory_id}, insert failed")
                return False
                
        except Exception as e:
            logger.warning(f"Memory operation 'update' failed: memory_id={memory_id}, error: {e}")
            return False
    
    def delete(self, memory_id: int, user_id: str = None) -> bool:
        """Delete a memory record with narrative group cleanup.
        
        Args:
            memory_id: ID of memory to delete
            user_id: User ID (required for narrative group cleanup)
            
        Returns:
            True if deletion succeeded
        """
        # 如果提供了user_id，先进行叙事组清理
        if user_id is not None:
            try:
                self._narrative_manager.delete_memory_from_group(memory_id, user_id)
            except Exception as e:
                logger.warning(f"Failed to cleanup narrative group for memory {memory_id}: {e}")
        
        count = self._store.delete(ids=[memory_id])
        success = count > 0
        
        if success:
            logger.info(
                f"Memory operation 'delete': memory_id={memory_id}, affected_count=1"
            )
        else:
            logger.warning(
                f"Memory operation 'delete' failed: memory_id={memory_id}, "
                f"affected_count=0 (not found)"
            )
        
        return success
    
    def reset(self, user_id: str) -> int:
        """Delete all memories for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Number of deleted memories
        """
        filter_expr = f'user_id == "{user_id}"'
        count = self._store.delete(filter_expr=filter_expr)
        
        logger.info(
            f"Memory operation 'reset': user_id={user_id}, affected_count={count}"
        )
        
        return count
    


    @observe(as_type="generation")
    def consolidate(self, user_id: Optional[str] = None) -> ConsolidationStats:
        """Run consolidation process for memories.
        
        Performs batch pattern merging: analyzes multiple episodic memories together
        to extract stable, long-term semantic facts.
        
        Args:
            user_id: Optional user to consolidate. If None, consolidates all.
            
        Returns:
            ConsolidationStats with operation counts
        """
        get_client().update_current_trace(
            session_id=f"consolidate_{user_id or 'all'}_{int(time.time())}",
            user_id=user_id,
            tags=["consolidation", "semantic_extraction"],
            metadata={"operation": "batch_consolidation"}
        )
        stats = ConsolidationStats()
        
        # 1. Query episodic memories to process
        if user_id:
            episodic_filter = f'user_id == "{user_id}" and memory_type == "episodic"'
            semantic_filter = f'user_id == "{user_id}" and memory_type == "semantic"'
        else:
            episodic_filter = 'memory_type == "episodic"'
            semantic_filter = 'memory_type == "semantic"'
        
        episodic_memories = self._store.query(filter_expr=episodic_filter, limit=1000)
        semantic_memories = self._store.query(filter_expr=semantic_filter, limit=1000)
        
        stats.memories_processed = len(episodic_memories)
        
        logger.info(
            f"Consolidation started for user_id={user_id or 'all'}: "
            f"processing {len(episodic_memories)} episodic memories, "
            f"{len(semantic_memories)} existing semantic memories"
        )
        
        # 2. Prepare batch processing data
        episodic_texts = [mem.get("text", "") for mem in episodic_memories]
        existing_semantic_texts = [mem.get("text", "") for mem in semantic_memories]
        
        consolidation_data = {
            "episodic_texts": episodic_texts,
            "existing_semantic_texts": existing_semantic_texts
        }
        
        # 3. Call batch pattern merging
        extraction = self._semantic_writer.extract(consolidation_data)
        
        # 4. Create new semantic memories
        if extraction.write_semantic and extraction.facts:
            # Use first episodic memory as source for metadata (user_id, chat_id)
            source_memory = episodic_memories[0] if episodic_memories else {}
            self._create_semantic_memories(source_memory, extraction.facts)
            stats.semantic_created += len(extraction.facts)
        
        # Log consolidation statistics
        logger.info(
            f"Consolidation complete for user_id={user_id or 'all'}: "
            f"processed={stats.memories_processed}, semantic_created={stats.semantic_created}"
        )
        
        return stats


    
    def _create_semantic_memories(
        self,
        source_memory: Dict[str, Any],
        facts: List[str]
    ) -> List[int]:
        """Create semantic memories from extracted facts.
        
        In v2 schema, all information is stored in the text field.
        """
        user_id = source_memory.get("user_id", "")
        source_chat_id = source_memory.get("chat_id", "")
        
        # Generate embeddings for facts
        embeddings = self._embedding_client.encode(facts)
        
        if len(embeddings) != len(facts):
            return []
        
        entities = []
        current_ts = int(time.time())
        
        for i, fact in enumerate(facts):
            # In v2 schema, the fact is stored directly in text field
            entity = {
                "user_id": user_id,
                "memory_type": "semantic",
                "ts": current_ts,
                "chat_id": source_chat_id,
                "text": fact,
                "vector": embeddings[i],
                "group_id": -1,
            }
            entities.append(entity)
        
        ids = self._store.insert(entities)
        
        logger.info(
            f"Memory operation 'create_semantic': type=semantic, user_id={user_id}, "
            f"source_chat_id={source_chat_id}, affected_count={len(ids)}"
        )
        
        return ids
    
    @property
    def store(self) -> MilvusStore:
        """Access the underlying MilvusStore (for testing)."""
        return self._store
    
    @property
    def config(self) -> MemoryConfig:
        """Access the configuration."""
        return self._config
