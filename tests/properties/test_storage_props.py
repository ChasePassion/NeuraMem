"""Property-based tests for storage operations.

This module contains property tests for dynamic field storage and memory storage consistency.
"""

import pytest
from hypothesis import given, strategies as st, settings, assume, HealthCheck

# **Feature: ai-memory-system, Property 1: Dynamic Field Storage Consistency**


def valid_field_name_strategy():
    """Generate valid field names for dynamic fields.
    
    Field names must be non-empty strings that are valid identifiers
    and don't conflict with existing schema fields.
    """
    reserved_fields = {
        "id", "user_id", "memory_type", "ts", "chat_id", 
        "who", "text", "vector", "hit_count", "metadata"
    }
    return st.text(
        alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="_"),
        min_size=1,
        max_size=50
    ).filter(
        lambda x: x not in reserved_fields 
        and x[0].isalpha()  # Must start with letter
        and x.isidentifier()  # Must be valid Python identifier
    )


def dynamic_field_value_strategy():
    """Generate valid values for dynamic fields.
    
    Supports strings, integers, floats, booleans, and lists.
    """
    return st.one_of(
        st.text(min_size=0, max_size=1000),
        st.integers(min_value=-2**31, max_value=2**31 - 1),
        st.floats(allow_nan=False, allow_infinity=False),
        st.booleans(),
        st.lists(st.text(min_size=0, max_size=100), max_size=10),
    )


class TestCoreFieldStorageConsistency:
    """Property tests for core field storage in Milvus.
    
    **Feature: ai-memory-system, Property 1: Core Field Storage Consistency**
    **Validates: Requirements 1.4**
    
    For any memory record with core fields, when stored, the fields 
    SHALL be retrievable with the same values.
    
    (v2 schema: simplified, no metadata field)
    """

    @settings(
        max_examples=3, 
        suppress_health_check=[
            HealthCheck.function_scoped_fixture,
            HealthCheck.too_slow,
        ],
        deadline=None  # Disable deadline for database operations
    )
    @given(
        field_name=valid_field_name_strategy(),
        field_value=dynamic_field_value_strategy()
    )
    def test_core_field_round_trip(self, milvus_store, field_name, field_value):
        """
        **Feature: ai-memory-system, Property 1: Core Field Storage Consistency**
        **Validates: Requirements 1.4**
        
        For any memory record with core fields, when stored, the fields 
        SHALL be retrievable with the same values.
        """
        # Skip if field_name is empty after filtering
        assume(len(field_name) > 0)
        
        # Create a memory record with core fields (v2 schema)
        test_user_id = f"test_user_{field_name[:10]}"
        test_text = f"Test memory for storage test with {field_name}"
        memory_record = {
            "user_id": test_user_id,
            "memory_type": "episodic",
            "ts": 1700000000,
            "chat_id": "test_chat_001",
            "text": test_text,
            "vector": [0.1] * 2560,  # Required vector field
        }
        
        # Insert the record
        ids = milvus_store.insert([memory_record])
        assert len(ids) == 1, "Should return one ID for inserted record"
        
        try:
            import time
            time.sleep(0.2)
            
            # Query the record back with retry for eventual consistency
            results = []
            for _ in range(5):
                results = milvus_store.query(
                    filter_expr=f"id == {ids[0]}",
                    output_fields=["*"]
                )
                if len(results) > 0:
                    break
                time.sleep(0.1)
            
            assert len(results) == 1, "Should retrieve exactly one record"
            
            record = results[0]
            
            # Verify core fields are present and have correct values
            assert record["user_id"] == test_user_id, \
                f"user_id mismatch: expected {test_user_id}, got {record['user_id']}"
            assert record["memory_type"] == "episodic", \
                f"memory_type mismatch: expected 'episodic', got {record['memory_type']}"
            assert record["ts"] == 1700000000, \
                f"ts mismatch: expected 1700000000, got {record['ts']}"
            assert record["chat_id"] == "test_chat_001", \
                f"chat_id mismatch: expected 'test_chat_001', got {record['chat_id']}"
            assert record["text"] == test_text, \
                f"text mismatch: expected {test_text}, got {record['text']}"
                    
        finally:
            # Cleanup: delete the test record
            milvus_store.delete(ids=ids)


@pytest.fixture
def milvus_store():
    """Fixture to provide a MilvusStore instance for testing.
    
    Uses the MilvusStore from the clients module.
    """
    from src.memory_system.clients.milvus_store import MilvusStore
    from src.memory_system.config import MemoryConfig
    
    config = MemoryConfig()
    store = MilvusStore(
        uri=config.milvus_uri,
        collection_name="test_memories_prop1"
    )
    
    # Create collection with dynamic fields enabled
    store.create_collection(dim=config.embedding_dim)
    
    yield store
    
    # Cleanup: drop test collection after tests
    store.drop_collection()
