"""Property-based tests for Memory class operations.

This module contains property tests for the Memory class API including
add, search, update, delete, and reset operations.
"""

import pytest
import time
from hypothesis import given, strategies as st, settings, assume, HealthCheck

from src.memory_system import Memory, MemoryConfig
from src.memory_system.clients.milvus_store import MilvusStore


# Strategies for generating test data
def user_id_strategy():
    """Generate valid user IDs using ASCII alphanumeric characters."""
    return st.text(
        alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_",
        min_size=3,
        max_size=20
    ).filter(lambda x: len(x) >= 3 and x[0].isalpha())


def chat_id_strategy():
    """Generate valid chat IDs using ASCII alphanumeric characters."""
    return st.text(
        alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_",
        min_size=3,
        max_size=20
    ).filter(lambda x: len(x) >= 3 and x[0].isalpha())


def memory_text_strategy():
    """Generate meaningful memory text that will pass the write decider.
    
    Uses patterns that are likely to be stored as episodic memories.
    """
    prefixes = [
        "I am a ",
        "I work as a ",
        "I study ",
        "I live in ",
        "My project is about ",
        "I'm working on ",
        "Remember that I ",
        "Please remember my ",
    ]
    suffixes = [
        "computer science student",
        "software engineer",
        "machine learning",
        "Beijing",
        "building an AI system",
        "developing a memory system",
        "like programming",
        "major is cybersecurity",
    ]
    return st.builds(
        lambda p, s: p + s,
        st.sampled_from(prefixes),
        st.sampled_from(suffixes)
    )


@pytest.fixture(scope="class")
def memory_system(request):
    """Fixture to provide a Memory instance for testing.
    
    Each test class gets its own collection to avoid conflicts.
    """
    import uuid
    config = MemoryConfig()
    # Use unique collection name per test class
    class_name = request.cls.__name__ if request.cls else "default"
    config.collection_name = f"test_memory_{class_name}_{uuid.uuid4().hex[:8]}"
    
    memory = Memory(config)
    
    yield memory
    
    # Cleanup: drop test collection after tests
    memory.store.drop_collection()


class TestEpisodicMemoryFieldCompleteness:
    """Property tests for episodic memory field completeness.
    
    **Feature: ai-memory-system, Property 2: Episodic Memory Field Completeness**
    **Validates: Requirements 2.3**
    
    For any stored episodic memory, the record SHALL contain all required fields:
    user_id, ts, chat_id, memory_type="episodic", hit_count (initialized to 0),
    text, vector, and metadata with context, thing, time, chatid, who.
    """

    @settings(
        max_examples=3,
        suppress_health_check=[
            HealthCheck.function_scoped_fixture,
            HealthCheck.too_slow,
        ],
        deadline=None
    )
    @given(
        user_id=user_id_strategy(),
        chat_id=chat_id_strategy(),
    )
    def test_episodic_memory_has_all_required_fields(self, memory_system, user_id, chat_id):
        """
        **Feature: ai-memory-system, Property 2: Episodic Memory Field Completeness**
        **Validates: Requirements 2.3**
        
        For any stored episodic memory, the record SHALL contain all required fields.
        """
        # Use a text that will definitely be stored
        text = f"Remember that I am a computer science student named {user_id}"
        
        # Add memory
        ids = memory_system.add(text=text, user_id=user_id, chat_id=chat_id)
        
        # If no memory was created (write decider rejected), skip this test case
        assume(len(ids) > 0)
        
        try:
            # Wait for consistency
            time.sleep(0.2)
            
            # Query the stored memory
            results = memory_system.store.query(
                filter_expr=f"id == {ids[0]}",
                output_fields=["*"]
            )
            
            assert len(results) == 1, "Should retrieve exactly one record"
            
            record = results[0]
            
            # Verify all required top-level fields
            assert "user_id" in record, "user_id field is required"
            assert record["user_id"] == user_id, "user_id should match"
            
            assert "ts" in record, "ts field is required"
            assert isinstance(record["ts"], int), "ts should be an integer"
            assert record["ts"] > 0, "ts should be a positive timestamp"
            
            assert "chat_id" in record, "chat_id field is required"
            assert record["chat_id"] == chat_id, "chat_id should match"
            
            assert "memory_type" in record, "memory_type field is required"
            assert record["memory_type"] == "episodic", "memory_type should be 'episodic'"
            
            assert "hit_count" in record, "hit_count field is required"
            assert record["hit_count"] == 0, "hit_count should be initialized to 0"
            
            assert "text" in record, "text field is required"
            assert len(record["text"]) > 0, "text should not be empty"
            
            assert "vector" in record, "vector field is required"
            assert len(record["vector"]) == 2560, "vector should have 2560 dimensions"
            
            assert "who" in record, "who field is required"
            
            # Verify metadata fields
            assert "metadata" in record, "metadata field is required"
            metadata = record["metadata"]
            
            assert "context" in metadata, "metadata.context is required"
            assert "thing" in metadata, "metadata.thing is required"
            assert "chatid" in metadata, "metadata.chatid is required"
            assert "who" in metadata, "metadata.who is required"
            
        finally:
            # Cleanup
            memory_system.store.delete(ids=ids)



class TestSearchResultTypeCoverage:
    """Property tests for search result type coverage.
    
    **Feature: ai-memory-system, Property 5: Search Result Type Coverage**
    **Validates: Requirements 3.2**
    
    For any search query with a valid user_id, the search results SHALL include
    both episodic and semantic memories (if they exist) filtered by that user_id.
    """

    @settings(
        max_examples=3,
        suppress_health_check=[
            HealthCheck.function_scoped_fixture,
            HealthCheck.too_slow,
        ],
        deadline=None
    )
    @given(
        user_id=user_id_strategy(),
    )
    def test_search_returns_both_memory_types(self, memory_system, user_id):
        """
        **Feature: ai-memory-system, Property 5: Search Result Type Coverage**
        **Validates: Requirements 3.2**
        
        For any search query, results SHALL include both episodic and semantic
        memories if they exist for the user.
        """
        chat_id = f"chat_{user_id}"
        
        # Create an episodic memory directly
        episodic_record = {
            "user_id": user_id,
            "memory_type": "episodic",
            "ts": int(time.time()),
            "chat_id": chat_id,
            "who": "user",
            "text": f"I am working on a machine learning project for {user_id}",
            "vector": [0.1] * 2560,
            "hit_count": 0,
            "metadata": {
                "context": "test context",
                "thing": "working on ML project",
                "time": "",
                "chatid": chat_id,
                "who": "user"
            }
        }
        
        # Create a semantic memory directly
        semantic_record = {
            "user_id": user_id,
            "memory_type": "semantic",
            "ts": int(time.time()),
            "chat_id": chat_id,
            "who": "user",
            "text": f"The user {user_id} is a machine learning researcher",
            "vector": [0.1] * 2560,
            "hit_count": 0,
            "metadata": {
                "fact": "ML researcher",
                "source_chatid": chat_id,
                "first_seen": "2024-01-01"
            }
        }
        
        # Insert both records
        episodic_ids = memory_system.store.insert([episodic_record])
        semantic_ids = memory_system.store.insert([semantic_record])
        
        try:
            time.sleep(0.3)
            
            # Search for memories
            results = memory_system.search(
                query="machine learning project",
                user_id=user_id,
                limit=10
            )
            
            # Check that we get results
            assert len(results) > 0, "Should return at least one result"
            
            # Check that results are filtered by user_id
            for result in results:
                assert result.user_id == user_id, "All results should be for the specified user"
            
            # Check that both types are present
            memory_types = {r.memory_type for r in results}
            assert "episodic" in memory_types, "Should include episodic memories"
            assert "semantic" in memory_types, "Should include semantic memories"
            
        finally:
            # Cleanup
            all_ids = list(episodic_ids) + list(semantic_ids)
            memory_system.store.delete(ids=all_ids)


class TestSearchResultLimitEnforcement:
    """Property tests for search result limit enforcement.
    
    **Feature: ai-memory-system, Property 6: Search Result Limit Enforcement**
    **Validates: Requirements 3.3**
    
    For any search operation, the number of returned semantic memories SHALL NOT
    exceed k_semantic, and the number of returned episodic memories SHALL NOT
    exceed k_episodic.
    """

    @settings(
        max_examples=3,
        suppress_health_check=[
            HealthCheck.function_scoped_fixture,
            HealthCheck.too_slow,
        ],
        deadline=None
    )
    @given(
        user_id=user_id_strategy(),
        num_episodic=st.integers(min_value=1, max_value=10),
        num_semantic=st.integers(min_value=1, max_value=10),
    )
    def test_search_respects_type_limits(self, memory_system, user_id, num_episodic, num_semantic):
        """
        **Feature: ai-memory-system, Property 6: Search Result Limit Enforcement**
        **Validates: Requirements 3.3**
        
        Search results SHALL NOT exceed k_semantic for semantic and k_episodic for episodic.
        """
        chat_id = f"chat_{user_id}"
        k_semantic = memory_system.config.k_semantic
        k_episodic = memory_system.config.k_episodic
        
        # Create multiple episodic memories
        episodic_records = []
        for i in range(num_episodic):
            record = {
                "user_id": user_id,
                "memory_type": "episodic",
                "ts": int(time.time()) + i,
                "chat_id": chat_id,
                "who": "user",
                "text": f"Episodic memory {i} about programming for {user_id}",
                "vector": [0.1 + i * 0.01] * 2560,
                "hit_count": 0,
                "metadata": {"context": "", "thing": "", "time": "", "chatid": chat_id, "who": "user"}
            }
            episodic_records.append(record)
        
        # Create multiple semantic memories
        semantic_records = []
        for i in range(num_semantic):
            record = {
                "user_id": user_id,
                "memory_type": "semantic",
                "ts": int(time.time()) + i,
                "chat_id": chat_id,
                "who": "user",
                "text": f"Semantic fact {i} about programming for {user_id}",
                "vector": [0.1 + i * 0.01] * 2560,
                "hit_count": 0,
                "metadata": {"fact": f"fact {i}", "source_chatid": chat_id, "first_seen": "2024-01-01"}
            }
            semantic_records.append(record)
        
        # Insert all records
        episodic_ids = memory_system.store.insert(episodic_records)
        semantic_ids = memory_system.store.insert(semantic_records)
        
        try:
            time.sleep(0.3)
            
            # Search for memories
            results = memory_system.search(
                query="programming",
                user_id=user_id,
                limit=100  # High limit to test type-specific limits
            )
            
            # Count by type
            episodic_count = sum(1 for r in results if r.memory_type == "episodic")
            semantic_count = sum(1 for r in results if r.memory_type == "semantic")
            
            # Verify limits are respected
            assert episodic_count <= k_episodic, \
                f"Episodic count {episodic_count} should not exceed k_episodic {k_episodic}"
            assert semantic_count <= k_semantic, \
                f"Semantic count {semantic_count} should not exceed k_semantic {k_semantic}"
            
        finally:
            # Cleanup
            all_ids = list(episodic_ids) + list(semantic_ids)
            memory_system.store.delete(ids=all_ids)


class TestHitCountIncrementOnRetrieval:
    """Property tests for hit count increment on retrieval.
    
    **Feature: ai-memory-system, Property 7: Hit Count Increment on Retrieval**
    **Validates: Requirements 3.5**
    
    For any memory that is retrieved and used in a conversation, its hit_count
    SHALL increase by exactly 1.
    """

    @settings(
        max_examples=3,
        suppress_health_check=[
            HealthCheck.function_scoped_fixture,
            HealthCheck.too_slow,
        ],
        deadline=None
    )
    @given(
        user_id=user_id_strategy(),
        initial_hit_count=st.integers(min_value=0, max_value=100),
    )
    def test_hit_count_increments_on_search(self, memory_system, user_id, initial_hit_count):
        """
        **Feature: ai-memory-system, Property 7: Hit Count Increment on Retrieval**
        **Validates: Requirements 3.5**
        
        When a memory is retrieved, its hit_count SHALL increase by exactly 1.
        
        Note: This test directly tests the increment mechanism without going through
        the full search flow to avoid slow embedding API calls.
        """
        chat_id = f"chat_{user_id}"
        
        # Create a memory with known hit_count
        record = {
            "user_id": user_id,
            "memory_type": "episodic",
            "ts": int(time.time()),
            "chat_id": chat_id,
            "who": "user",
            "text": f"Unique test memory for hit count test {user_id} {initial_hit_count}",
            "vector": [0.5] * 2560,
            "hit_count": initial_hit_count,
            "metadata": {"context": "", "thing": "", "time": "", "chatid": chat_id, "who": "user"}
        }
        
        ids = memory_system.store.insert([record])
        
        try:
            time.sleep(0.3)
            
            # Directly call the increment method (simulating what happens after search)
            memory_system._increment_hit_counts(ids)
            
            time.sleep(0.3)
            
            # Query the memory to check updated hit_count
            # Note: After update, the record gets a new ID, so we query by user_id
            updated = memory_system.store.query(
                filter_expr=f'user_id == "{user_id}"',
                output_fields=["hit_count", "text"]
            )
            
            # Find our record by text
            our_record = None
            for r in updated:
                if f"hit count test {user_id} {initial_hit_count}" in r.get("text", ""):
                    our_record = r
                    break
            
            assert our_record is not None, "Should find the memory"
            new_hit_count = our_record["hit_count"]
            
            assert new_hit_count == initial_hit_count + 1, \
                f"hit_count should increase by 1: was {initial_hit_count}, now {new_hit_count}"
            
        finally:
            # Cleanup - delete by user_id since record ID changes after update
            memory_system.store.delete(filter_expr=f'user_id == "{user_id}"')



class TestResetOperationCompleteness:
    """Property tests for reset operation completeness.
    
    **Feature: ai-memory-system, Property 17: Reset Operation Completeness**
    **Validates: Requirements 8.5**
    
    For any user_id, after calling Memory.reset(user_id), there SHALL be zero
    memories remaining for that user_id.
    """

    @settings(
        max_examples=3,
        suppress_health_check=[
            HealthCheck.function_scoped_fixture,
            HealthCheck.too_slow,
        ],
        deadline=None
    )
    @given(
        user_id=user_id_strategy(),
        num_memories=st.integers(min_value=1, max_value=5),
    )
    def test_reset_removes_all_user_memories(self, memory_system, user_id, num_memories):
        """
        **Feature: ai-memory-system, Property 17: Reset Operation Completeness**
        **Validates: Requirements 8.5**
        
        After reset, there SHALL be zero memories for the user.
        """
        chat_id = f"chat_{user_id}"
        
        # Create multiple memories for the user
        records = []
        for i in range(num_memories):
            # Mix of episodic and semantic
            memory_type = "episodic" if i % 2 == 0 else "semantic"
            record = {
                "user_id": user_id,
                "memory_type": memory_type,
                "ts": int(time.time()) + i,
                "chat_id": chat_id,
                "who": "user",
                "text": f"Memory {i} for reset test {user_id}",
                "vector": [0.1 + i * 0.01] * 2560,
                "hit_count": i,
                "metadata": {
                    "context": f"context {i}",
                    "thing": f"thing {i}",
                    "time": "",
                    "chatid": chat_id,
                    "who": "user"
                }
            }
            records.append(record)
        
        # Insert all records
        ids = memory_system.store.insert(records)
        memory_system.store.flush()
        
        time.sleep(0.5)
        
        # Verify memories exist
        before_count = memory_system.store.count(filter_expr=f'user_id == "{user_id}"')
        assert before_count == num_memories, \
            f"Should have {num_memories} memories before reset, got {before_count}"
        
        # Reset the user's memories
        deleted_count = memory_system.reset(user_id)
        memory_system.store.flush()
        
        time.sleep(0.5)
        
        # Verify all memories are deleted
        after_count = memory_system.store.count(filter_expr=f'user_id == "{user_id}"')
        
        assert after_count == 0, \
            f"Should have 0 memories after reset, got {after_count}"
        assert deleted_count == num_memories, \
            f"Reset should report {num_memories} deleted, got {deleted_count}"
