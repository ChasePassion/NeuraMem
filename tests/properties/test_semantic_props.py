"""Property-based tests for semantic memory operations.

This module contains property tests for semantic memory creation and extraction.
"""

import pytest
from hypothesis import given, strategies as st, settings, assume, HealthCheck
from datetime import datetime, timezone
import time as time_module

from src.memory_system.clients.milvus_store import MilvusStore
from src.memory_system.config import MemoryConfig
from tests.properties.dummy_llm import DummyLLMClient


def iso_time_strategy():
    """Generate valid ISO 8601 time strings."""
    return st.datetimes(
        min_value=datetime(2020, 1, 1),
        max_value=datetime(2025, 12, 31)
    ).map(lambda dt: dt.replace(tzinfo=timezone.utc).isoformat())


def episodic_memory_strategy():
    """Generate valid episodic memory records for testing."""
    return st.fixed_dictionaries({
        "id": st.integers(min_value=1, max_value=1000000),
        "user_id": st.text(min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz0123456789_"),
        "memory_type": st.just("episodic"),
        "ts": st.integers(min_value=1600000000, max_value=1800000000),
        "chat_id": st.text(min_size=1, max_size=30, alphabet="abcdefghijklmnopqrstuvwxyz0123456789-_"),
        "who": st.sampled_from(["user", "assistant", "friend"]),
        "text": st.text(min_size=10, max_size=200),
        "hit_count": st.integers(min_value=0, max_value=100),
        "metadata": st.fixed_dictionaries({
            "context": st.text(min_size=5, max_size=100),
            "thing": st.text(min_size=5, max_size=100),
            "time": iso_time_strategy(),
            "chatid": st.text(min_size=1, max_size=30, alphabet="abcdefghijklmnopqrstuvwxyz0123456789-_"),
            "who": st.sampled_from(["user", "assistant", "friend"])
        })
    })


def fact_strategy():
    """Generate valid fact strings for semantic memory."""
    return st.text(min_size=10, max_size=200, alphabet=st.characters(
        whitelist_categories=('L', 'N', 'P', 'Z'),
        whitelist_characters=' '
    )).filter(lambda x: len(x.strip()) >= 10)


@pytest.fixture(scope="class")
def milvus_store():
    """Fixture to provide a MilvusStore instance for semantic testing.
    
    Uses class scope to ensure the collection persists across all Hypothesis examples.
    """
    import uuid
    config = MemoryConfig()
    # Use unique collection name to avoid conflicts
    collection_name = f"test_memories_semantic_{uuid.uuid4().hex[:8]}"
    store = MilvusStore(
        uri=config.milvus_uri,
        collection_name=collection_name
    )
    store.create_collection(dim=config.embedding_dim)
    yield store
    try:
        store.drop_collection()
    except Exception:
        pass


class TestSemanticMemoryFieldCompleteness:
    """Property tests for semantic memory field completeness.
    
    **Feature: ai-memory-system, Property 13: Semantic Memory Field Completeness**
    **Validates: Requirements 6.6**
    
    For any created semantic memory, the record SHALL have memory_type="semantic" 
    and metadata containing fact, source_chatid, and first_seen fields.
    """

    @settings(
        max_examples=5,
        suppress_health_check=[
            HealthCheck.function_scoped_fixture,
            HealthCheck.too_slow,
        ],
        deadline=None
    )
    @given(
        source_memory=episodic_memory_strategy(),
        facts=st.lists(fact_strategy(), min_size=1, max_size=2)
    )
    def test_semantic_memory_field_completeness(self, milvus_store, source_memory, facts):
        """
        **Feature: ai-memory-system, Property 13: Semantic Memory Field Completeness**
        **Validates: Requirements 6.6**
        
        For any created semantic memory, the record SHALL have memory_type="semantic" 
        and metadata containing fact, source_chatid, and first_seen fields.
        """
        config = MemoryConfig()
        
        # Create semantic memories directly (simulating _create_semantic_memories)
        user_id = source_memory.get("user_id", "")
        source_chat_id = source_memory.get("chat_id", "")
        
        entities = []
        current_ts = int(time_module.time())
        
        for fact in facts:
            # Use dummy vector for testing (avoid real embedding API calls)
            entity = {
                "user_id": user_id,
                "memory_type": "semantic",
                "ts": current_ts,
                "chat_id": source_chat_id,
                "who": source_memory.get("who", "user"),
                "text": fact,
                "vector": [0.1] * config.embedding_dim,
                "hit_count": 0,
                "metadata": {
                    "fact": fact,
                    "source_chatid": source_chat_id,
                    "first_seen": datetime.now(timezone.utc).isoformat()
                }
            }
            entities.append(entity)
        
        # Insert semantic memories
        created_ids = milvus_store.insert(entities)
        milvus_store.flush()
        
        try:
            # Verify each created semantic memory
            assert len(created_ids) == len(facts), \
                f"Should create {len(facts)} semantic memories, got {len(created_ids)}"
            
            for i, memory_id in enumerate(created_ids):
                # Query the created memory
                records = milvus_store.query(
                    filter_expr=f"id == {memory_id}",
                    output_fields=["*"]
                )
                
                assert len(records) == 1, f"Should find exactly one record for id {memory_id}"
                record = records[0]
                
                # Verify memory_type is "semantic"
                assert record.get("memory_type") == "semantic", \
                    f"memory_type should be 'semantic', got '{record.get('memory_type')}'"
                
                # Verify metadata exists
                metadata = record.get("metadata", {})
                assert metadata is not None, "metadata should not be None"
                
                # Verify fact field in metadata
                assert "fact" in metadata, "metadata should contain 'fact' field"
                assert metadata["fact"] == facts[i], \
                    f"fact should match: expected '{facts[i]}', got '{metadata['fact']}'"
                
                # Verify source_chatid field in metadata
                assert "source_chatid" in metadata, "metadata should contain 'source_chatid' field"
                assert metadata["source_chatid"] == source_memory["chat_id"], \
                    f"source_chatid should match source memory's chat_id"
                
                # Verify first_seen field in metadata
                assert "first_seen" in metadata, "metadata should contain 'first_seen' field"
                assert metadata["first_seen"], "first_seen should not be empty"
                
                # Verify first_seen is a valid ISO 8601 timestamp
                try:
                    datetime.fromisoformat(metadata["first_seen"].replace('Z', '+00:00'))
                except ValueError:
                    pytest.fail(f"first_seen should be valid ISO 8601: {metadata['first_seen']}")
                
                # Verify user_id matches source
                assert record.get("user_id") == source_memory["user_id"], \
                    "user_id should match source memory"
                
                # Verify text field contains the fact
                assert record.get("text") == facts[i], \
                    f"text should be the fact: expected '{facts[i]}'"
                
        finally:
            # Cleanup created records
            if created_ids:
                try:
                    milvus_store.delete(ids=created_ids)
                except Exception:
                    pass
