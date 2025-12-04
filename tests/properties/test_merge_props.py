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
        # Ensure memories are from same user and mergeable
        memory_b["user_id"] = memory_a["user_id"]
        memory_b["who"] = memory_a["who"]
        memory_b["metadata"]["who"] = memory_a["metadata"]["who"]
        
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
            merged["hit_count"] = 0
            merged["ts"] = memory_a["ts"]
            
            # Delete originals
            milvus_store.delete(ids=ids)
            milvus_store.flush()
            
            # Insert merged
            merged_ids = milvus_store.insert([merged])
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


class TestMergedRecordTimeSelection:
    """Property tests for merged record time selection.
    
    **Feature: ai-memory-system, Property 21: Merged Record Time Selection**
    **Validates: Requirements 9.4**
    
    For any merge operation, the resulting record's metadata.time 
    SHALL equal the earlier of the two original metadata.time values.
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
    def test_merged_record_uses_earliest_time(self, merger, memory_a, memory_b):
        """
        **Feature: ai-memory-system, Property 21: Merged Record Time Selection**
        **Validates: Requirements 9.4**
        
        For any merge operation, the resulting record's metadata.time 
        SHALL equal the earlier of the two original metadata.time values.
        """
        # Ensure memories are mergeable
        memory_b["user_id"] = memory_a["user_id"]
        memory_b["who"] = memory_a["who"]
        
        time_a = memory_a["metadata"]["time"]
        time_b = memory_b["metadata"]["time"]
        expected_earliest = min(time_a, time_b)
        
        # Perform merge
        merged = merger.merge(memory_a, memory_b)
        
        # Verify time selection
        merged_time = merged.get("metadata", {}).get("time", "")
        assert merged_time == expected_earliest, \
            f"Merged time should be earliest: expected={expected_earliest}, got={merged_time}"


class TestMergedRecordSourceTracking:
    """Property tests for merged record source tracking.
    
    **Feature: ai-memory-system, Property 22: Merged Record Source Tracking**
    **Validates: Requirements 9.5**
    
    For any merge operation, the resulting record's metadata.source_chat_ids 
    SHALL contain both original chat_ids.
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
    def test_merged_record_tracks_sources(self, merger, memory_a, memory_b):
        """
        **Feature: ai-memory-system, Property 22: Merged Record Source Tracking**
        **Validates: Requirements 9.5**
        
        For any merge operation, the resulting record's metadata.source_chat_ids 
        SHALL contain both original chat_ids.
        """
        # Ensure memories are mergeable
        memory_b["user_id"] = memory_a["user_id"]
        memory_b["who"] = memory_a["who"]
        
        chat_id_a = memory_a["chat_id"]
        chat_id_b = memory_b["chat_id"]
        
        # Perform merge
        merged = merger.merge(memory_a, memory_b)
        
        # Verify source tracking
        source_chat_ids = merged.get("metadata", {}).get("source_chat_ids", [])
        
        assert chat_id_a in source_chat_ids, \
            f"source_chat_ids should contain chat_id_a: {chat_id_a}"
        assert chat_id_b in source_chat_ids, \
            f"source_chat_ids should contain chat_id_b: {chat_id_b}"
        
        # Also verify merged_from_ids contains both original IDs
        merged_from_ids = merged.get("metadata", {}).get("merged_from_ids", [])
        assert memory_a["id"] in merged_from_ids, \
            f"merged_from_ids should contain id_a: {memory_a['id']}"
        assert memory_b["id"] in merged_from_ids, \
            f"merged_from_ids should contain id_b: {memory_b['id']}"



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
        
        For any separation operation, the immutable fields (time, chatid, who in metadata)
        SHALL remain unchanged in both resulting memories.
        """
        # Store original immutable values
        original_time_a = memory_a["metadata"]["time"]
        original_chatid_a = memory_a["metadata"]["chatid"]
        original_who_a = memory_a["metadata"]["who"]
        
        original_time_b = memory_b["metadata"]["time"]
        original_chatid_b = memory_b["metadata"]["chatid"]
        original_who_b = memory_b["metadata"]["who"]
        
        # Perform separation
        updated_a, updated_b = separator.separate(memory_a, memory_b)
        
        # Verify immutable fields are preserved for memory A
        assert updated_a["metadata"]["time"] == original_time_a, \
            f"metadata.time should be preserved for A: expected={original_time_a}"
        assert updated_a["metadata"]["chatid"] == original_chatid_a, \
            f"metadata.chatid should be preserved for A: expected={original_chatid_a}"
        assert updated_a["metadata"]["who"] == original_who_a, \
            f"metadata.who should be preserved for A: expected={original_who_a}"
        
        # Verify immutable fields are preserved for memory B
        assert updated_b["metadata"]["time"] == original_time_b, \
            f"metadata.time should be preserved for B: expected={original_time_b}"
        assert updated_b["metadata"]["chatid"] == original_chatid_b, \
            f"metadata.chatid should be preserved for B: expected={original_chatid_b}"
        assert updated_b["metadata"]["who"] == original_who_b, \
            f"metadata.who should be preserved for B: expected={original_who_b}"
        
        # Verify that text and metadata.context/thing are present (may be updated)
        assert "text" in updated_a, "Updated A should have text field"
        assert "text" in updated_b, "Updated B should have text field"
        assert "context" in updated_a["metadata"], "Updated A should have context"
        assert "thing" in updated_a["metadata"], "Updated A should have thing"
        assert "context" in updated_b["metadata"], "Updated B should have context"
        assert "thing" in updated_b["metadata"], "Updated B should have thing"


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
        # Create memory_b with same chat_id and who
        memory_b = memory_a.copy()
        memory_b["id"] = memory_a["id"] + 1
        memory_b["metadata"] = memory_a["metadata"].copy()
        memory_b["ts"] = memory_a["ts"] + time_diff
        # Same chat_id and who
        memory_b["chat_id"] = memory_a["chat_id"]
        memory_b["who"] = memory_a["who"]
        
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
        # Create memory_b with different chat_id but same who
        memory_b = memory_a.copy()
        memory_b["id"] = memory_a["id"] + 1
        memory_b["metadata"] = memory_a["metadata"].copy()
        memory_b["ts"] = memory_a["ts"] + time_diff
        # Different chat_id, same who
        memory_b["chat_id"] = memory_a["chat_id"] + "_different"
        memory_b["who"] = memory_a["who"]
        
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


class TestWhoFieldMergeConstraint:
    """Property tests for who field merge constraint.
    
    **Feature: ai-memory-system, Property 20: Who Field Merge Constraint**
    **Validates: Requirements 9.3**
    
    For any pair of memories where who1 != who2, the system SHALL NOT merge them.
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
        who_b=st.sampled_from(["user", "assistant", "friend", "other"])
    )
    def test_who_field_merge_constraint(self, memory_system, memory_a, who_b):
        """
        **Feature: ai-memory-system, Property 20: Who Field Merge Constraint**
        **Validates: Requirements 9.3**
        
        For any pair of memories where who1 != who2, the system SHALL NOT merge them.
        """
        # Create memory_b with potentially different who
        memory_b = memory_a.copy()
        memory_b["id"] = memory_a["id"] + 1
        memory_b["metadata"] = memory_a["metadata"].copy()
        memory_b["who"] = who_b
        # Same chat_id and within time window to isolate who constraint
        memory_b["chat_id"] = memory_a["chat_id"]
        memory_b["ts"] = memory_a["ts"] + 100  # Within 30 min window
        
        # Check merge constraints
        result = memory_system.check_merge_constraints(memory_a, memory_b)
        
        who_a = memory_a["who"]
        
        if who_a == who_b:
            # Same who - should allow merge (assuming time constraint passes)
            assert result["who_match"] is True, "Should detect matching who fields"
            assert result["can_merge"] is True, \
                f"Should allow merge when who fields match: '{who_a}' == '{who_b}'"
        else:
            # Different who - should reject merge
            assert result["who_match"] is False, "Should detect different who fields"
            assert result["can_merge"] is False, \
                f"Should reject merge when who fields differ: '{who_a}' != '{who_b}'"
            assert "who" in result["reason"].lower(), \
                "Rejection reason should mention 'who' field"
