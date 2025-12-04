"""Memory class - Main API for AI Memory System.

Provides a mem0-style interface for memory operations including
add, search, update, delete, reset, and consolidate.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from .config import MemoryConfig
from .clients import EmbeddingClient, LLMClient, MilvusStore
from .processors import (
    EpisodicWriteDecider,
    SemanticWriter,
    EpisodicReconsolidator,
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
        self._write_decider = EpisodicWriteDecider(self._llm_client)
        self._semantic_writer = SemanticWriter(self._llm_client)
        self._reconsolidator = EpisodicReconsolidator(self._llm_client)
        
        # Create collection if not exists
        self._store.create_collection(dim=self._config.embedding_dim)
        
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

    def add(
        self,
        text: str,
        user_id: str,
        chat_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[int]:
        """Add memories from conversation text.
        
        Processes text through EpisodicWriteDecider to determine if it should
        be stored as episodic memory. All information is integrated into the
        text field for unified vector search.
        
        Args:
            text: Conversation text to process
            user_id: User identifier
            chat_id: Conversation/thread identifier
            metadata: Optional additional metadata (ignored in v2 schema)
            
        Returns:
            List of created memory IDs
            
        Requirements: 2.1, 2.2, 2.3, 8.1
        """
        # Prepare conversation turns for write decider
        turns = [{"role": "user", "content": text}]
        
        # Call EpisodicWriteDecider to determine if content should be stored
        decision = self._write_decider.decide(chat_id, turns)
        
        if not decision.write_episodic or not decision.records:
            logger.info(f"No episodic memory to write for user={user_id}, chat={chat_id}")
            return []
        
        # Generate embeddings for qualifying records
        texts_to_embed = [record.text for record in decision.records]
        embeddings = self._embedding_client.encode(texts_to_embed)
        
        # Prepare entities for insertion (simplified v2 schema)
        current_ts = int(time.time())
        entities = []
        
        for i, record in enumerate(decision.records):
            # In v2 schema, all information is in the text field
            entity = {
                "user_id": user_id,
                "memory_type": "episodic",
                "ts": current_ts,
                "chat_id": chat_id,
                "text": record.text,
                "vector": embeddings[i],
            }
            entities.append(entity)
        
        # Insert into Milvus
        ids = self._store.insert(entities)
        
        logger.info(
            f"Memory operation 'add': type=episodic, user_id={user_id}, "
            f"chat_id={chat_id}, affected_count={len(ids)}"
        )
        
        return ids

    async def add_async(
        self,
        text: str,
        user_id: str,
        chat_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[int]:
        """Async variant of add that offloads blocking steps to a thread pool."""
        turns = [{"role": "user", "content": text}]

        # Decide whether to write episodic memory (LLM call in background thread)
        decision = await asyncio.to_thread(self._write_decider.decide, chat_id, turns)

        if not decision.write_episodic or not decision.records:
            logger.info(f"No episodic memory to write for user={user_id}, chat={chat_id}")
            return []

        texts_to_embed = [record.text for record in decision.records]

        # Embedding + Milvus insert are synchronous; run them in threads to avoid blocking event loop
        embeddings = await asyncio.to_thread(self._embedding_client.encode, texts_to_embed)

        current_ts = int(time.time())
        entities = []

        for i, record in enumerate(decision.records):
            entity = {
                "user_id": user_id,
                "memory_type": "episodic",
                "ts": current_ts,
                "chat_id": chat_id,
                "text": record.text,
                "vector": embeddings[i],
            }
            entities.append(entity)

        ids = await asyncio.to_thread(self._store.insert, entities)

        logger.info(
            f"Memory operation 'add_async': type=episodic, user_id={user_id}, "
            f"chat_id={chat_id}, affected_count={len(ids)}"
        )

        return ids

    def search(
        self,
        query: str,
        user_id: str,
        limit: int = 10,
        reconsolidate: bool = True
    ) -> List[MemoryRecord]:
        """Search memories for a user.
        
        Retrieves both episodic and semantic memories, ranks them by
        similarity, type, and time decay. Optionally reconsolidates
        episodic memories with the current query context.
        
        Args:
            query: Search query text
            user_id: User identifier
            limit: Maximum total results to return
            reconsolidate: Whether to reconsolidate episodic memories (default True)
            
        Returns:
            Ranked list of MemoryRecord objects
            
        Requirements: 3.1, 3.2, 3.3, 3.4, 4.1, 4.5
        """
        # Generate embedding for query
        query_vectors = self._embedding_client.encode([query])
        
        if not query_vectors:
            return []
        
        query_vector = query_vectors[0]
        
        # Search semantic memories
        semantic_filter = f'user_id == "{user_id}" and memory_type == "semantic"'
        semantic_results = self._store.search(
            vectors=[query_vector],
            filter_expr=semantic_filter,
            limit=self._config.k_semantic
        )
        
        # Search episodic memories
        episodic_filter = f'user_id == "{user_id}" and memory_type == "episodic"'
        episodic_results = self._store.search(
            vectors=[query_vector],
            filter_expr=episodic_filter,
            limit=self._config.k_episodic
        )
        
        # Combine results
        all_results = []
        
        # Process semantic results
        if semantic_results and semantic_results[0]:
            for hit in semantic_results[0]:
                record = self._hit_to_memory_record(hit)
                all_results.append(record)
        
        # Process episodic results and collect for reconsolidation
        episodic_hits = []
        if episodic_results and episodic_results[0]:
            for hit in episodic_results[0]:
                record = self._hit_to_memory_record(hit)
                all_results.append(record)
                episodic_hits.append(hit)
        
        # Rank results by similarity, type, and time decay
        ranked_results = self._rank_results(all_results)
        
        # Limit total results
        ranked_results = ranked_results[:limit]
        
        # Reconsolidate episodic memories with current context (Requirements 4.1, 4.5)
        if reconsolidate and episodic_hits:
            self._reconsolidate_episodic_memories(episodic_hits, query)
        
        # Count by type for logging
        semantic_count = sum(1 for r in ranked_results if r.memory_type == "semantic")
        episodic_count = sum(1 for r in ranked_results if r.memory_type == "episodic")
        
        logger.info(
            f"Memory operation 'search': user_id={user_id}, "
            f"total_results={len(ranked_results)}, semantic={semantic_count}, "
            f"episodic={episodic_count}"
        )
        
        return ranked_results

    async def reconsolidate_async(
        self,
        query: str,
        user_id: str,
        limit: int = 10
    ) -> List[MemoryRecord]:
        """Async helper to reconsolidate episodic memories without blocking event loop."""
        return await asyncio.to_thread(self.search, query, user_id, limit, True)
    
    def _reconsolidate_episodic_memories(
        self,
        episodic_hits: List[Dict[str, Any]],
        current_context: str
    ) -> None:
        """Reconsolidate episodic memories with current context.
        
        After search returns results, identifies used episodic memories,
        calls EpisodicReconsolidator with current context, updates memory
        records with reconsolidated content, and regenerates embeddings.
        
        In v2 schema, all information is stored in the text field.
        
        Args:
            episodic_hits: List of episodic memory hits from search
            current_context: Current query/context text
            
        Requirements: 4.1, 4.5
        """
        for hit in episodic_hits:
            memory_id = hit.get("id")
            if memory_id is None:
                continue
            
            try:
                # Call EpisodicReconsolidator with old memory and current context
                updated_memory = self._reconsolidator.reconsolidate(
                    old_memory=hit,
                    current_context=current_context
                )
                
                if not updated_memory:
                    continue
                
                # Check if text was changed (Requirement 4.5)
                old_text = hit.get("text", "")
                new_text = updated_memory.get("text", "")
                
                # Prepare update data (v2 schema: only text and vector)
                update_data = {}
                
                # Update text if changed
                if new_text and new_text != old_text:
                    update_data["text"] = new_text
                    # Regenerate embedding for new text
                    embeddings = self._embedding_client.encode([new_text])
                    if embeddings:
                        update_data["vector"] = embeddings[0]
                
                # Apply updates to Milvus if there are changes
                if update_data:
                    self._store.update(memory_id, update_data)
                    logger.debug(
                        f"Reconsolidated memory {memory_id} with context"
                    )
                    
            except Exception as e:
                logger.warning(
                    f"Failed to reconsolidate memory {memory_id}: {e}"
                )
    
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
    
    def _rank_results(self, results: List[MemoryRecord]) -> List[MemoryRecord]:
        """Rank search results by multiple factors.
        
        Factors (v2 schema - simplified):
        - Similarity score (distance, lower is better for COSINE)
        - Memory type (semantic weighted higher)
        - Time decay (recent episodic more important)
        """
        current_time = int(time.time())
        
        def score(record: MemoryRecord) -> float:
            # Base score from similarity (1 - distance for COSINE)
            similarity = 1.0 - record.distance
            
            # Type weight: semantic memories get a boost
            type_weight = 1.2 if record.memory_type == "semantic" else 1.0
            
            # Time decay for episodic (recent is better)
            time_weight = 1.0
            if record.memory_type == "episodic" and record.ts > 0:
                age_days = (current_time - record.ts) / 86400
                time_weight = 1.0 / (1.0 + age_days * 0.1)  # Decay factor
            
            return similarity * type_weight * time_weight
        
        return sorted(results, key=score, reverse=True)
    


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
    


    def consolidate(self, user_id: Optional[str] = None) -> ConsolidationStats:
        """Run consolidation process for memories.
        
        Performs semantic fact extraction from episodic memories.
        
        Args:
            user_id: Optional user to consolidate. If None, consolidates all.
            
        Returns:
            ConsolidationStats with operation counts
            
        Requirements: 6.1, 6.6
        """
        stats = ConsolidationStats()
        
        # Query episodic memories to process
        if user_id:
            filter_expr = f'user_id == "{user_id}" and memory_type == "episodic"'
        else:
            filter_expr = 'memory_type == "episodic"'
        
        memories = self._store.query(filter_expr=filter_expr, limit=1000)
        stats.memories_processed = len(memories)
        
        logger.info(
            f"Consolidation started for user_id={user_id or 'all'}: "
            f"processing {len(memories)} episodic memories"
        )
        
        # Semantic extraction from episodic memories
        for memory in memories:
            # Extract semantic facts
            extraction = self._semantic_writer.extract(memory)
            if extraction.write_semantic and extraction.facts:
                self._create_semantic_memories(memory, extraction.facts)
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
