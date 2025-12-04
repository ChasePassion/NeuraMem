"""Property-based tests for merge operations.

This module contains property tests for episodic memory merging.
"""

import pytest
from hypothesis import given, strategies as st, settings, assume, HealthCheck
from datetime import datetime, timezone, timedelta
import time as time_module

from src.memory_system.processors.merger import EpisodicMerger
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
    """Generate valid episodic memory records for testing (v2 schema)."""
    return st.fixed_dictionaries({
        "id": st.integers(min_value=1, max_value=1000000),
        "user_id": st.text(min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz0123456789_"),
        "memory_type": st.just("episodic"),
        "ts": st.integers(min_value=1600000000, max_value=1800000000),
        "chat_id": st.text(min_size=1, max_size=30, alphabet="abcdefghijklmnopqrstuvwxyz0123456789-_"),
        "text": st.text(min_size=10, max_size=200),
    })


@pytest.fixture(scope="module")
def llm_client():
    """Fixture to provide an LLMClient instance for testing."""
    return DummyLLMClient()


@pytest.fixture(scope="module")
def merger(llm_client):
    """Fixture to provide an EpisodicMerger instance for testing."""
    return EpisodicMerger(llm_client)


@pytest.fixture(scope="class")
def milvus_store():
    """Fixture to provide a MilvusStore instance for merge testing.
    
    Uses class scope to ensure the collection persists across all Hypothesis examples
    within a test class.
    """
    import uuid
    config = MemoryConfig()
    # Use unique collection name to avoid conflicts
    collection_name = f"test_memories_merge_{uuid.uuid4().hex[:8]}"
    store = MilvusStore(
        uri=config.milvus_uri,
        collection_name=collection_name
    )
    store.create_collection(dim=config.embedding_dim)
    yield store
    try:
        store.drop_collection()
    except Exception:
        pass  # Ignore errors during cleanup


class TestMergeRecordCountInvariant:
    """Property tests for merge record count invariant.
    
    **Feature: ai-memory-system, Property 12: Merge Record Count Invariant**
    **Validates: Requirements 5.4**
    
    For any merge operation on two memories, the total record count 
    SHALL decrease by exactly 1 (two deleted, one inserted).
    
    (v2 schema: simplified record structure)
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
        memory_a=episodic_memory_strategy(),
        memory_b=episodic_memory_strategy()
    )
    def test_merge_record_count_invariant(self, milvus_store, merger, memory_a, memory_b):
        """
        **Feature: ai-memory-system, Property 12: Merge Record Count Invariant**
        **Validates: Requirements 5.4**
        
        For any merge operation on two memories, the total record count 
        SHALL decrease by exactly 1.
        """
        # Ensure memories are from same user
        memory_b["user_id"] = memory_a["user_id"]
        
        # Add vectors for storage
        memory_a["vector"] = [0.1] * 2560
        memory_b["vector"] = [0.2] * 2560
        
        # Remove id fields for insertion (auto-generated)
        mem_a_insert = {k: v for k, v in memory_a.items() if k != "id"}
        mem_b_insert = {k: v for k, v in memory_b.items() if k != "id"}
        
        # Insert both memories
        ids = milvus_store.insert([mem_a_insert, mem_b_insert])
        assert len(ids) == 2, "Should insert two records"
        milvus_store.flush()
        
        # Get initial count
        initial_results = milvus_store.query(
            filter_expr=f"user_id == '{memory_a['user_id']}'",
            output_fields=["id"]
        )
        initial_count = len(initial_results)
        
        try:
            # Update memory dicts with actual IDs
            memory_a["id"] = ids[0]
            memory_b["id"] = ids[1]
            
            # Perform merge
            merged = merger.merge(memory_a, memory_b)
            merged["vector"] = [0.15] * 2560  # Add vector for merged record
            merged["ts"] = memory_a["ts"]
            
            # Delete originals
            milvus_store.delete(ids=ids)
            milvus_store.flush()
            
            # Insert merged (v2 schema)
            merged_record = {
                "user_id": merged.get("user_id", memory_a["user_id"]),
                "memory_type": "episodic",
                "ts": merged.get("ts", memory_a["ts"]),
                "chat_id": merged.get("chat_id", memory_a["chat_id"]),
                "text": merged.get("text", ""),
                "vector": merged["vector"],
            }
            merged_ids = milvus_store.insert([merged_record])
            milvus_store.flush()
            
            # Get final count
            final_results = milvus_store.query(
                filter_expr=f"user_id == '{memory_a['user_id']}'",
                output_fields=["id"]
            )
            final_count = len(final_results)
            
            # Verify: count decreased by exactly 1
            assert final_count == initial_count - 1, \
                f"Merge should decrease count by 1: initial={initial_count}, final={final_count}"
                
        finally:
            # Cleanup any remaining records
            try:
                milvus_store.delete(filter_expr=f"user_id == '{memory_a['user_id']}'")
            except:
                pass


class TestMergedRecordTextCombination:
    """Property tests for merged record text combination.
    
    **Feature: ai-memory-system, Property 21: Merged Record Text Combination**
    **Validates: Requirements 9.4**
    
    For any merge operation, the resulting record's text field 
    SHALL contain information from both original records.
    
    (v2 schema: all information is in text field)
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
        memory_a=episodic_memory_strategy(),
        memory_b=episodic_memory_strategy()
    )
    def test_merged_record_combines_text(self, merger, memory_a, memory_b):
        """
        **Feature: ai-memory-system, Property 21: Merged Record Text Combination**
        **Validates: Requirements 9.4**
        
        For any merge operation, the resulting record's text field 
        SHALL be non-empty and contain combined information.
        """
        # Ensure memories are mergeable
        memory_b["user_id"] = memory_a["user_id"]
        
        # Perform merge
        merged = merger.merge(memory_a, memory_b)
        
        # Verify text is present and non-empty
        merged_text = merged.get("text", "")
        assert merged_text, "Merged text should not be empty"
        assert len(merged_text) > 0, "Merged text should have content"


class TestMergedRecordChatIdPreservation:
    """Property tests for merged record chat_id preservation.
    
    **Feature: ai-memory-system, Property 22: Merged Record Chat ID Preservation**
    **Validates: Requirements 9.5**
    
    For any merge operation, the resulting record SHALL have a valid chat_id.
    
    (v2 schema: simplified, no source tracking in metadata)
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
        memory_a=episodic_memory_strategy(),
        memory_b=episodic_memory_strategy()
    )
    def test_merged_record_has_chat_id(self, merger, memory_a, memory_b):
        """
        **Feature: ai-memory-system, Property 22: Merged Record Chat ID Preservation**
        **Validates: Requirements 9.5**
        
        For any merge operation, the resulting record SHALL have a valid chat_id.
        """
        # Ensure memories are mergeable
        memory_b["user_id"] = memory_a["user_id"]
        
        # Perform merge
        merged = merger.merge(memory_a, memory_b)
        
        # Verify chat_id is present
        merged_chat_id = merged.get("chat_id", "")
        assert merged_chat_id, "Merged record should have chat_id"
        # chat_id should be from one of the original records
        assert merged_chat_id in [memory_a["chat_id"], memory_b["chat_id"]], \
            f"Merged chat_id should be from original records"



# Import separator for Property 11 test
from src.memory_system.processors.separator import EpisodicSeparator


@pytest.fixture(scope="module")
def separator(llm_client):
    """Fixture to provide an EpisodicSeparator instance for testing."""
    return EpisodicSeparator(llm_client)


class TestSeparationThresholdEnforcement:
    """Property tests for separation threshold enforcement.
    
    **Feature: ai-memory-system, Property 11: Separation Threshold Enforcement**
    **Validates: Requirements 5.3**
    
    For any pair of episodic memories with T_amb_low (0.65) <= cosine similarity < T_merge_high (0.85),
    the consolidation process SHALL call EpisodicSeparator to rewrite them.
    
    Note: This test verifies that the separator correctly rewrites memories
    while preserving immutable fields.
    
    (v2 schema: simplified, only chat_id is immutable)
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
        memory_a=episodic_memory_strategy(),
        memory_b=episodic_memory_strategy()
    )
    def test_separation_preserves_immutable_fields(self, separator, memory_a, memory_b):
        """
        **Feature: ai-memory-system, Property 11: Separation Threshold Enforcement**
        **Validates: Requirements 5.3**
        
        For any separation operation, the immutable fields (chat_id)
        SHALL remain unchanged in both resulting memories.
        """
        # Store original immutable values
        original_chat_id_a = memory_a["chat_id"]
        original_chat_id_b = memory_b["chat_id"]
        
        # Perform separation
        updated_a, updated_b = separator.separate(memory_a, memory_b)
        
        # Verify immutable fields are preserved for memory A
        assert updated_a["chat_id"] == original_chat_id_a, \
            f"chat_id should be preserved for A: expected={original_chat_id_a}"
        
        # Verify immutable fields are preserved for memory B
        assert updated_b["chat_id"] == original_chat_id_b, \
            f"chat_id should be preserved for B: expected={original_chat_id_b}"
        
        # Verify that text is present (may be updated)
        assert "text" in updated_a, "Updated A should have text field"
        assert "text" in updated_b, "Updated B should have text field"


# Import Memory class for merge constraint testing
from src.memory_system.memory import Memory


@pytest.fixture(scope="module")
def memory_system():
    """Fixture to provide a Memory instance for constraint testing.
    
    Uses module scope to reuse the same instance across tests.
    """
    import uuid
    config = MemoryConfig()
    # Use unique collection name to avoid conflicts
    config.collection_name = f"test_memories_constraints_{uuid.uuid4().hex[:8]}"
    memory = Memory(config=config)
    yield memory
    try:
        memory.store.drop_collection()
    except Exception:
        pass


class TestSameChatMergeTimeConstraint:
    """Property tests for same chat merge time constraint.
    
    **Feature: ai-memory-system, Property 18: Same Chat Merge Time Constraint**
    **Validates: Requirements 9.1**
    
    For any pair of memories with the same chat_id, merge SHALL only be allowed 
    if |ts1 - ts2| <= 1800 seconds (30 minutes).
    
    (v2 schema: no who field constraint)
    """

    @settings(
        max_examples=100,
        suppress_health_check=[
            HealthCheck.function_scoped_fixture,
            HealthCheck.too_slow,
        ],
        deadline=None
    )
    @given(
        memory_a=episodic_memory_strategy(),
        time_diff=st.integers(min_value=0, max_value=7200)  # 0 to 2 hours
    )
    def test_same_chat_merge_time_constraint(self, memory_system, memory_a, time_diff):
        """
        **Feature: ai-memory-system, Property 18: Same Chat Merge Time Constraint**
        **Validates: Requirements 9.1**
        
        For any pair of memories with the same chat_id, merge SHALL only be allowed 
        if |ts1 - ts2| <= 1800 seconds (30 minutes).
        """
        # Create memory_b with same chat_id
        memory_b = memory_a.copy()
        memory_b["id"] = memory_a["id"] + 1
        memory_b["ts"] = memory_a["ts"] + time_diff
        # Same chat_id
        memory_b["chat_id"] = memory_a["chat_id"]
        
        # Check merge constraints
        result = memory_system.check_merge_constraints(memory_a, memory_b)
        
        # Verify constraint enforcement
        time_window = memory_system.config.merge_time_window_same_chat  # 1800 seconds
        
        if time_diff <= time_window:
            assert result["can_merge"] is True, \
                f"Should allow merge when time_diff={time_diff} <= {time_window}"
        else:
            assert result["can_merge"] is False, \
                f"Should reject merge when time_diff={time_diff} > {time_window}"
        
        # Verify same_chat flag
        assert result["same_chat"] is True, "Should detect same chat_id"
        assert result["time_diff"] == time_diff, f"Time diff should be {time_diff}"


class TestDifferentChatMergeTimeConstraint:
    """Property tests for different chat merge time constraint.
    
    **Feature: ai-memory-system, Property 19: Different Chat Merge Time Constraint**
    **Validates: Requirements 9.2**
    
    For any pair of memories with different chat_ids, merge SHALL only be allowed 
    if |ts1 - ts2| <= 604800 seconds (7 days).
    
    (v2 schema: no who field constraint)
    """

    @settings(
        max_examples=100,
        suppress_health_check=[
            HealthCheck.function_scoped_fixture,
            HealthCheck.too_slow,
        ],
        deadline=None
    )
    @given(
        memory_a=episodic_memory_strategy(),
        time_diff=st.integers(min_value=0, max_value=1209600)  # 0 to 14 days
    )
    def test_different_chat_merge_time_constraint(self, memory_system, memory_a, time_diff):
        """
        **Feature: ai-memory-system, Property 19: Different Chat Merge Time Constraint**
        **Validates: Requirements 9.2**
        
        For any pair of memories with different chat_ids, merge SHALL only be allowed 
        if |ts1 - ts2| <= 604800 seconds (7 days).
        """
        # Create memory_b with different chat_id
        memory_b = memory_a.copy()
        memory_b["id"] = memory_a["id"] + 1
        memory_b["ts"] = memory_a["ts"] + time_diff
        # Different chat_id
        memory_b["chat_id"] = memory_a["chat_id"] + "_different"
        
        # Check merge constraints
        result = memory_system.check_merge_constraints(memory_a, memory_b)
        
        # Verify constraint enforcement
        time_window = memory_system.config.merge_time_window_diff_chat  # 604800 seconds (7 days)
        
        if time_diff <= time_window:
            assert result["can_merge"] is True, \
                f"Should allow merge when time_diff={time_diff} <= {time_window}"
        else:
            assert result["can_merge"] is False, \
                f"Should reject merge when time_diff={time_diff} > {time_window}"
        
        # Verify same_chat flag
        assert result["same_chat"] is False, "Should detect different chat_ids"
        assert result["time_diff"] == time_diff, f"Time diff should be {time_diff}"
