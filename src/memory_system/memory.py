"""Memory class - Main API for AI Memory System.

Provides a mem0-style interface for memory operations including
add, search, update, delete, reset, and consolidate.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from langfuse import observe, get_client
from .config import MemoryConfig
from .clients import EmbeddingClient, LLMClient, MilvusStore
from .processors import (
    EpisodicMemoryManager,
    SemanticWriter,
    MemoryUsageJudge,
)

logger = logging.getLogger(__name__)


@dataclass
class MemoryRecord:
    """A memory record returned from search operations.
    
    Simplified schema (v2):
    - id: Record ID
    - user_id: User identifier
    - memory_type: "episodic" or "semantic"
    - ts: Unix timestamp of write time
    - chat_id: Conversation/thread identifier
    - text: Main natural-language content (includes time, where, who, thing, reason)
    - distance: Similarity score from search (0.0 = identical)
    """
    id: int
    user_id: str
    memory_type: str
    ts: int
    chat_id: str
    text: str
    distance: float = 0.0  # Similarity score from search


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
        
        # Create collection if not exists
        self._store.create_collection(dim=self._config.embedding_dim)
        
        # Initialize Langfuse client if available
        self._langfuse_client = self._create_langfuse_client()
        # Cache availability flag to avoid global lookups in threads
        self._langfuse_available = bool(globals().get("LANGFUSE_AVAILABLE", False))
        
        logger.info(
            f"Memory system initialized with collection '{self._config.collection_name}'"
        )
    
    def _create_embedding_client(self) -> EmbeddingClient:
        """Factory method to create EmbeddingClient."""
        return EmbeddingClient(
            api_key=self._config.siliconflow_api_key,
            base_url=self._config.embedding_base_url,
            model=self._config.embedding_model
        )
    
    def _create_llm_client(self) -> LLMClient:
        """Factory method to create LLMClient."""
        return LLMClient(
            api_key=self._config.deepseek_api_key,
            base_url=self._config.llm_base_url,
            model=self._config.llm_model
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
            
        Requirements: 2.1, 2.2, 2.3, 8.1
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
                self.delete(op.memory_id)
        
        # 处理更新操作
        for op in result.operations:
            if op.operation_type == "update":
                self.update(op.memory_id, {"text": op.text})
        
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
                await asyncio.to_thread(self.delete, op.memory_id)
        
        # 处理更新操作
        for op in result.operations:
            if op.operation_type == "update":
                await asyncio.to_thread(self.update, op.memory_id, {"text": op.text})
        
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
        """Search memories for a user.
        
        Retrieves both episodic and semantic memories without ranking.
        
        Args:
            query: Search query text
            user_id: User identifier
            
        Returns:
            Dict with separated episodic and semantic memories
        """
        get_client().update_current_trace(
            session_id=f"search_{user_id}_{int(time.time())}",
            user_id=user_id,
            tags=["memory_search", "retrieval"],
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
        
        # Search episodic memories
        episodic_filter = f'user_id == "{user_id}" and memory_type == "episodic"'
        episodic_results = self._store.search(
            vectors=[query_vector],
            filter_expr=episodic_filter,
            limit=self._config.k_episodic
        )
        episodic_memories = []
        if episodic_results and episodic_results[0]:
            episodic_memories = [self._hit_to_memory_record(hit) for hit in episodic_results[0]]
        
        # 直接返回分离的结果，不排序
        result = {
            "episodic": episodic_memories,
            "semantic": semantic_memories
        }
        
        # Count by type for logging
        logger.info(
            f"Memory operation 'search': user_id={user_id}, "
            f"episodic_results={len(episodic_memories)}, semantic_results={len(semantic_memories)}"
        )
        
        return result

    
    def _hit_to_memory_record(self, hit: Dict[str, Any]) -> MemoryRecord:
        """Convert a search hit to MemoryRecord (v2 schema)."""
        return MemoryRecord(
            id=hit.get("id", 0),
            user_id=hit.get("user_id", ""),
            memory_type=hit.get("memory_type", ""),
            ts=hit.get("ts", 0),
            chat_id=hit.get("chat_id", ""),
            text=hit.get("text", ""),
            distance=hit.get("distance", 0.0)
        )
    
    


    def update(self, memory_id: int, data: Dict[str, Any]) -> bool:
        """Update a memory record.
        
        If text is changed, regenerates the embedding vector.
        
        Args:
            memory_id: ID of memory to update
            data: Fields to update
            
        Returns:
            True if update succeeded
            
        Requirements: 8.3
        """
        # Check if text is being updated
        if "text" in data:
            # Regenerate embedding for new text
            embeddings = self._embedding_client.encode([data["text"]])
            if embeddings:
                data["vector"] = embeddings[0]
        
        success = self._store.update(memory_id, data)
        
        if success:
            logger.info(
                f"Memory operation 'update': memory_id={memory_id}, "
                f"fields_updated={list(data.keys())}, affected_count=1"
            )
        else:
            logger.warning(
                f"Memory operation 'update' failed: memory_id={memory_id}, "
                f"affected_count=0"
            )
        
        return success
    
    def delete(self, memory_id: int) -> bool:
        """Delete a memory record.
        
        Args:
            memory_id: ID of memory to delete
            
        Returns:
            True if deletion succeeded
            
        Requirements: 8.4
        """
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
            
        Requirements: 8.5
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
            
        Requirements: 6.1, 6.6
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
        
        Requirements: 6.6
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
