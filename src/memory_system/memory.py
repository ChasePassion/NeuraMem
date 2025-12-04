"""Memory class - Main API for AI Memory System.

Provides a mem0-style interface for memory operations including
add, search, update, delete, reset, and consolidate.
"""

import logging
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from .config import MemoryConfig
from .clients import EmbeddingClient, LLMClient, MilvusStore
from .processors import (
    EpisodicWriteDecider,
    SemanticWriter,
    EpisodicMerger,
    EpisodicSeparator,
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
    memories_merged: int = 0
    memories_separated: int = 0
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
        self._merger = EpisodicMerger(self._llm_client)
        self._separator = EpisodicSeparator(self._llm_client)
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
    
    def find_similar_candidates(
        self,
        memory: Dict[str, Any],
        top_n: int = 10,
        exclude_ids: Optional[set] = None
    ) -> List[Dict[str, Any]]:
        """Find Top-N similar episodic memory candidates for consolidation.
        
        Performs vector similarity search and calculates cosine similarity scores.
        
        Args:
            memory: Source memory to find similar candidates for
            top_n: Maximum number of candidates to return
            exclude_ids: Set of memory IDs to exclude from results
            
        Returns:
            List of candidate memories with similarity scores added
            
        Requirements: 5.1
        """
        vector = memory.get("vector")
        if not vector:
            return []
        
        memory_id = memory.get("id")
        user_id = memory.get("user_id", "")
        
        # Build filter to exclude self and already processed memories
        filter_parts = [
            f'user_id == "{user_id}"',
            'memory_type == "episodic"'
        ]
        if memory_id is not None:
            filter_parts.append(f'id != {memory_id}')
        
        filter_expr = " and ".join(filter_parts)
        
        # Search for similar candidates
        results = self._store.search(
            vectors=[vector],
            filter_expr=filter_expr,
            limit=top_n
        )
        
        if not results or not results[0]:
            return []
        
        # Process results and add similarity scores
        candidates = []
        for hit in results[0]:
            candidate_id = hit.get("id")
            
            # Skip excluded IDs
            if exclude_ids and candidate_id in exclude_ids:
                continue
            
            # Calculate cosine similarity from distance
            # Milvus COSINE metric returns distance = 1 - similarity
            distance = hit.get("distance", 1.0)
            similarity = 1.0 - distance
            
            # Add similarity to candidate
            hit["similarity"] = similarity
            candidates.append(hit)
        
        return candidates
    
    def categorize_by_similarity(
        self,
        candidates: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Categorize candidates by similarity thresholds.
        
        Categories:
        - 'merge': similarity >= T_merge_high (0.85)
        - 'separate': T_amb_low (0.65) <= similarity < T_merge_high (0.85)
        - 'distinct': similarity < T_amb_low (0.65)
        
        Args:
            candidates: List of candidates with similarity scores
            
        Returns:
            Dict with 'merge', 'separate', 'distinct' lists
        """
        result = {
            'merge': [],
            'separate': [],
            'distinct': []
        }
        
        for candidate in candidates:
            similarity = candidate.get("similarity", 0.0)
            
            if similarity >= self._config.t_merge_high:
                result['merge'].append(candidate)
            elif similarity >= self._config.t_amb_low:
                result['separate'].append(candidate)
            else:
                result['distinct'].append(candidate)
        
        return result

    def consolidate(self, user_id: Optional[str] = None) -> ConsolidationStats:
        """Run consolidation process for memories.
        
        Performs:
        - Merge highly similar episodic memories (similarity >= 0.85)
        - Separate ambiguously similar memories (0.65 <= similarity < 0.85)
        - Extract semantic facts from episodic memories
        
        Args:
            user_id: Optional user to consolidate. If None, consolidates all.
            
        Returns:
            ConsolidationStats with operation counts
            
        Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 6.1, 6.6
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
        
        # Track processed memory IDs to avoid double-processing
        processed_ids = set()
        # Track separated pairs to avoid re-separating
        separated_pairs = set()
        
        # Phase 1: Merge and Separation
        for memory in memories:
            memory_id = memory.get("id")
            if memory_id in processed_ids:
                continue
            
            # Find similar candidates using dedicated method
            candidates = self.find_similar_candidates(
                memory=memory,
                top_n=10,
                exclude_ids=processed_ids
            )
            
            if not candidates:
                continue
            
            # Categorize candidates by similarity thresholds
            categorized = self.categorize_by_similarity(candidates)
            
            # Process merge candidates (similarity >= T_merge_high)
            for candidate in categorized['merge']:
                candidate_id = candidate.get("id")
                if candidate_id in processed_ids:
                    continue
                
                # Check merge constraints (who, time, chat_id)
                if self._can_merge(memory, candidate):
                    merged_id = self._perform_merge(memory, candidate)
                    if merged_id is not None:
                        processed_ids.add(memory_id)
                        processed_ids.add(candidate_id)
                        stats.memories_merged += 1
                        break  # Memory was merged, move to next
            
            # If memory was merged, skip separation
            if memory_id in processed_ids:
                continue
            
            # Process separation candidates (T_amb_low <= similarity < T_merge_high)
            for candidate in categorized['separate']:
                candidate_id = candidate.get("id")
                if candidate_id in processed_ids:
                    continue
                
                # Create pair key to avoid re-separating
                pair_key = tuple(sorted([memory_id, candidate_id]))
                if pair_key in separated_pairs:
                    continue
                
                self._perform_separation(memory, candidate)
                separated_pairs.add(pair_key)
                stats.memories_separated += 1
        
        # Phase 2: Semantic extraction for non-merged memories
        # Re-query to get updated memories after merge/separation
        memories = self._store.query(filter_expr=filter_expr, limit=1000)
        
        for memory in memories:
            memory_id = memory.get("id")
            
            # Extract semantic facts
            extraction = self._semantic_writer.extract(memory)
            if extraction.write_semantic and extraction.facts:
                self._create_semantic_memories(memory, extraction.facts)
                stats.semantic_created += len(extraction.facts)
        
        # Log detailed consolidation statistics (Requirements 10.5)
        logger.info(
            f"Consolidation complete for user_id={user_id or 'all'}: "
            f"processed={stats.memories_processed}, merged={stats.memories_merged}, "
            f"separated={stats.memories_separated}, semantic_created={stats.semantic_created}"
        )
        
        return stats

    def check_merge_constraints(
        self,
        memory_a: Dict[str, Any],
        memory_b: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check merge constraints between two memories.
        
        Validates (v2 schema - simplified):
        - Time constraints based on chat_id (Requirements 9.1, 9.2)
        
        Args:
            memory_a: First memory record
            memory_b: Second memory record
            
        Returns:
            Dict with:
                - can_merge: bool indicating if merge is allowed
                - reason: str explaining why merge is/isn't allowed
                - same_chat: bool indicating if chat_ids match
                - time_diff: int time difference in seconds
                - time_constraint: int applicable time window in seconds
        """
        result = {
            "can_merge": True,
            "reason": "All constraints satisfied",
            "same_chat": False,
            "time_diff": 0,
            "time_constraint": 0
        }
        
        # Check time constraints
        ts_a = memory_a.get("ts", 0)
        ts_b = memory_b.get("ts", 0)
        time_diff = abs(ts_a - ts_b)
        result["time_diff"] = time_diff
        
        chat_a = memory_a.get("chat_id", "")
        chat_b = memory_b.get("chat_id", "")
        result["same_chat"] = (chat_a == chat_b)
        
        if result["same_chat"]:
            # Same chat: 30 minute window (Requirement 9.1)
            result["time_constraint"] = self._config.merge_time_window_same_chat
            if time_diff > self._config.merge_time_window_same_chat:
                result["can_merge"] = False
                result["reason"] = (
                    f"Same chat time constraint violated: "
                    f"{time_diff}s > {self._config.merge_time_window_same_chat}s (30 min)"
                )
        else:
            # Different chats: 7 day window (Requirement 9.2)
            result["time_constraint"] = self._config.merge_time_window_diff_chat
            if time_diff > self._config.merge_time_window_diff_chat:
                result["can_merge"] = False
                result["reason"] = (
                    f"Different chat time constraint violated: "
                    f"{time_diff}s > {self._config.merge_time_window_diff_chat}s (7 days)"
                )
        
        return result
    
    def _can_merge(self, memory_a: Dict[str, Any], memory_b: Dict[str, Any]) -> bool:
        """Check if two memories can be merged based on constraints.
        
        This is a convenience wrapper around check_merge_constraints().
        
        Requirements: 9.1, 9.2
        """
        result = self.check_merge_constraints(memory_a, memory_b)
        return result["can_merge"]
    
    def _perform_merge(
        self,
        memory_a: Dict[str, Any],
        memory_b: Dict[str, Any]
    ) -> Optional[int]:
        """Merge two memories into one.
        
        In v2 schema, all information is stored in the text field.
        
        Requirements: 5.2, 5.4, 9.4, 9.5
        """
        # Call merger to create merged content
        merged = self._merger.merge(memory_a, memory_b)
        
        if not merged:
            return None
        
        # Generate embedding for merged text
        embeddings = self._embedding_client.encode([merged.get("text", "")])
        if not embeddings:
            return None
        
        # Build v2 schema record
        merged_record = {
            "user_id": memory_a.get("user_id", ""),
            "memory_type": "episodic",
            "ts": min(memory_a.get("ts", 0), memory_b.get("ts", 0)),
            "chat_id": memory_a.get("chat_id", ""),
            "text": merged.get("text", ""),
            "vector": embeddings[0],
        }
        
        # Delete original records
        self._store.delete(ids=[memory_a.get("id"), memory_b.get("id")])
        
        # Insert merged record
        ids = self._store.insert([merged_record])
        
        return ids[0] if ids else None
    
    def _perform_separation(
        self,
        memory_a: Dict[str, Any],
        memory_b: Dict[str, Any]
    ) -> None:
        """Separate two similar memories to make them more distinct.
        
        Requirements: 5.3, 5.5
        """
        # Call separator to rewrite memories
        separated = self._separator.separate(memory_a, memory_b)
        
        if not separated:
            return
        
        updated_a, updated_b = separated
        
        # Regenerate embeddings
        texts = [updated_a.get("text", ""), updated_b.get("text", "")]
        embeddings = self._embedding_client.encode(texts)
        
        if len(embeddings) >= 2:
            # Update memory A
            updated_a["vector"] = embeddings[0]
            self._store.update(memory_a.get("id"), updated_a)
            
            # Update memory B
            updated_b["vector"] = embeddings[1]
            self._store.update(memory_b.get("id"), updated_b)
    
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
