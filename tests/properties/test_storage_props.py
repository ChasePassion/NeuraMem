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


class TestDynamicFieldStorageConsistency:
    """Property tests for dynamic field storage in Milvus.
    
    **Feature: ai-memory-system, Property 1: Dynamic Field Storage Consistency**
    **Validates: Requirements 1.4**
    
    For any additional metadata field not in the schema, when stored in a 
    memory record, the field SHALL be retrievable with the same value.
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
    def test_dynamic_field_round_trip(self, milvus_store, field_name, field_value):
        """
        **Feature: ai-memory-system, Property 1: Dynamic Field Storage Consistency**
        **Validates: Requirements 1.4**
        
        For any additional metadata field not in the schema, when stored in a 
        memory record, the field SHALL be retrievable with the same value.
        """
        # Skip if field_name is empty after filtering
        assume(len(field_name) > 0)
        
        # Create a memory record with a dynamic field in metadata
        test_user_id = f"test_user_{field_name[:10]}"
        memory_record = {
            "user_id": test_user_id,
            "memory_type": "episodic",
            "ts": 1700000000,
            "chat_id": "test_chat_001",
            "who": "user",
            "text": "Test memory for dynamic field storage",
            "vector": [0.1] * 2560,  # Required vector field
            "hit_count": 0,
            "metadata": {
                "context": "test context",
                "thing": "test thing",
                "time": "2024-01-01T00:00:00Z",
                "chatid": "test_chat_001",
                "who": "user",
                field_name: field_value  # Dynamic field
            }
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
                    output_fields=["metadata"]
                )
                if len(results) > 0:
                    break
                time.sleep(0.1)
            
            assert len(results) == 1, "Should retrieve exactly one record"
            
            retrieved_metadata = results[0]["metadata"]
            
            # Verify the dynamic field is present and has the same value
            assert field_name in retrieved_metadata, \
                f"Dynamic field '{field_name}' should be present in retrieved metadata"
            
            retrieved_value = retrieved_metadata[field_name]
            
            # Handle float comparison with tolerance
            if isinstance(field_value, float):
                assert abs(retrieved_value - field_value) < 1e-6, \
                    f"Dynamic field value mismatch: expected {field_value}, got {retrieved_value}"
            else:
                assert retrieved_value == field_value, \
                    f"Dynamic field value mismatch: expected {field_value}, got {retrieved_value}"
                    
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
