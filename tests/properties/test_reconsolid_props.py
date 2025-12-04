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


def episodic_memory_strategy():
    """Generate episodic memory records for testing (v2 schema)."""
    return st.fixed_dictionaries({
        "id": st.integers(min_value=1, max_value=1000000),
        "user_id": st.text(min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz0123456789_"),
        "memory_type": st.just("episodic"),
        "ts": st.integers(min_value=1600000000, max_value=1800000000),
        "chat_id": st.text(min_size=1, max_size=30, alphabet="abcdefghijklmnopqrstuvwxyz0123456789-_"),
        "text": st.text(min_size=10, max_size=200),
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
    
    For any episodic memory undergoing reconsolidation, the chat_id field
    SHALL remain unchanged from the original value.
    
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
        old_memory=episodic_memory_strategy(),
        current_context=context_text_strategy()
    )
    def test_reconsolidation_preserves_immutable_fields(
        self, reconsolidator, old_memory, current_context
    ):
        """
        **Feature: ai-memory-system, Property 8: Reconsolidation Field Preservation**
        **Validates: Requirements 4.2**
        
        For any episodic memory undergoing reconsolidation, the chat_id field
        SHALL remain unchanged from the original value.
        """
        # Store original immutable values
        original_chat_id = old_memory["chat_id"]
        
        # Perform reconsolidation
        updated = reconsolidator.reconsolidate(old_memory, current_context)
        
        # Verify immutable fields are preserved
        assert updated["chat_id"] == original_chat_id, \
            f"chat_id should be preserved: expected={original_chat_id}, got={updated['chat_id']}"


class TestReconsolidationTextUpdate:
    """Property tests for reconsolidation text update.
    
    **Feature: ai-memory-system, Property 9: Reconsolidation Text Update**
    **Validates: Requirements 4.4**
    
    For any reconsolidation operation, the text field SHALL be updated
    to include new context information.
    
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
        old_memory=episodic_memory_strategy(),
        current_context=context_text_strategy()
    )
    def test_reconsolidation_updates_text(
        self, reconsolidator, old_memory, current_context
    ):
        """
        **Feature: ai-memory-system, Property 9: Reconsolidation Text Update**
        **Validates: Requirements 4.4**
        
        For any reconsolidation operation, the text field SHALL be present
        and non-empty.
        """
        # Perform reconsolidation
        updated = reconsolidator.reconsolidate(old_memory, current_context)
        
        # Verify text field is present and non-empty
        assert "text" in updated, "Updated memory should have text field"
        assert updated["text"], "Updated text should not be empty"
        assert len(updated["text"]) > 0, "Updated text should have content"
