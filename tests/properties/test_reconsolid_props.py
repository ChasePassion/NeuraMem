"""Property-based tests for reconsolidation operations.

This module contains property tests for episodic memory reconsolidation.
"""

import pytest
from hypothesis import given, strategies as st, settings, assume, HealthCheck
from datetime import datetime, timezone

from src.memory_system.processors.reconsolidator import EpisodicReconsolidator
from src.memory_system.config import MemoryConfig
from tests.properties.dummy_llm import DummyLLMClient


def iso_time_strategy():
    """Generate valid ISO 8601 time strings."""
    return st.datetimes(
        min_value=datetime(2020, 1, 1),
        max_value=datetime(2025, 12, 31)
    ).map(lambda dt: dt.replace(tzinfo=timezone.utc).isoformat())


def update_entry_strategy():
    """Generate valid update entries for metadata.updates array."""
    return st.fixed_dictionaries({
        "time": iso_time_strategy(),
        "desc": st.text(min_size=5, max_size=100)
    })


def episodic_memory_with_updates_strategy():
    """Generate episodic memory records with optional updates array."""
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
            "who": st.sampled_from(["user", "assistant", "friend"]),
            "updates": st.lists(update_entry_strategy(), min_size=0, max_size=5)
        })
    })


def context_text_strategy():
    """Generate context text for reconsolidation."""
    return st.text(min_size=10, max_size=300)


@pytest.fixture(scope="module")
def llm_client():
    """Fixture to provide an LLMClient instance for testing."""
    return DummyLLMClient()


@pytest.fixture(scope="module")
def reconsolidator(llm_client):
    """Fixture to provide an EpisodicReconsolidator instance for testing."""
    return EpisodicReconsolidator(llm_client)


class TestReconsolidationFieldPreservation:
    """Property tests for reconsolidation field preservation.
    
    **Feature: ai-memory-system, Property 8: Reconsolidation Field Preservation**
    **Validates: Requirements 4.2**
    
    For any episodic memory undergoing reconsolidation, the fields metadata.time, 
    chat_id, who, and metadata.who SHALL remain unchanged from the original values.
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
        old_memory=episodic_memory_with_updates_strategy(),
        current_context=context_text_strategy()
    )
    def test_reconsolidation_preserves_immutable_fields(
        self, reconsolidator, old_memory, current_context
    ):
        """
        **Feature: ai-memory-system, Property 8: Reconsolidation Field Preservation**
        **Validates: Requirements 4.2**
        
        For any episodic memory undergoing reconsolidation, the fields metadata.time, 
        chat_id, who, and metadata.who SHALL remain unchanged from the original values.
        """
        # Store original immutable values
        original_time = old_memory["metadata"]["time"]
        original_chat_id = old_memory["chat_id"]
        original_who = old_memory["who"]
        original_metadata_chatid = old_memory["metadata"]["chatid"]
        original_metadata_who = old_memory["metadata"]["who"]
        
        # Perform reconsolidation
        updated = reconsolidator.reconsolidate(old_memory, current_context)
        
        # Verify immutable fields are preserved
        assert updated["metadata"]["time"] == original_time, \
            f"metadata.time should be preserved: expected={original_time}, got={updated['metadata']['time']}"
        
        assert updated["chat_id"] == original_chat_id, \
            f"chat_id should be preserved: expected={original_chat_id}, got={updated['chat_id']}"
        
        assert updated["who"] == original_who, \
            f"who should be preserved: expected={original_who}, got={updated['who']}"
        
        assert updated["metadata"]["chatid"] == original_metadata_chatid, \
            f"metadata.chatid should be preserved: expected={original_metadata_chatid}"
        
        assert updated["metadata"]["who"] == original_metadata_who, \
            f"metadata.who should be preserved: expected={original_metadata_who}"


class TestReconsolidationUpdatesArrayGrowth:
    """Property tests for reconsolidation updates array growth.
    
    **Feature: ai-memory-system, Property 9: Reconsolidation Updates Array Growth**
    **Validates: Requirements 4.4**
    
    For any reconsolidation operation, the metadata.updates array length 
    SHALL increase by at least 1.
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
        old_memory=episodic_memory_with_updates_strategy(),
        current_context=context_text_strategy()
    )
    def test_reconsolidation_grows_updates_array(
        self, reconsolidator, old_memory, current_context
    ):
        """
        **Feature: ai-memory-system, Property 9: Reconsolidation Updates Array Growth**
        **Validates: Requirements 4.4**
        
        For any reconsolidation operation, the metadata.updates array length 
        SHALL increase by at least 1.
        """
        # Get original updates count
        original_updates = old_memory.get("metadata", {}).get("updates", [])
        original_count = len(original_updates)
        
        # Perform reconsolidation
        updated = reconsolidator.reconsolidate(old_memory, current_context)
        
        # Get new updates count
        new_updates = updated.get("metadata", {}).get("updates", [])
        new_count = len(new_updates)
        
        # Verify updates array grew by at least 1
        assert new_count >= original_count + 1, \
            f"Updates array should grow by at least 1: original={original_count}, new={new_count}"
        
        # Verify the new update has required fields
        if new_count > 0:
            latest_update = new_updates[-1]
            assert "time" in latest_update or "desc" in latest_update, \
                "New update entry should have time or desc field"
