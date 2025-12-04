"""Unit tests for processor functionality.

v2 schema: simplified, only core fields (no who, hit_count, metadata)
"""

from src.memory_system.processors.reconsolidator import EpisodicReconsolidator
from src.memory_system.processors.separator import EpisodicSeparator


class MockLLM:
    def chat_json(self, system_prompt, user_message, default):
        return default
    
    def chat(self, system_prompt, user_message):
        return ""  # Empty means keep


def test_reconsolidation_preserves_chat_id():
    """Test that reconsolidation preserves chat_id field (v2 schema)."""
    reconsolidator = EpisodicReconsolidator(MockLLM())
    
    old_memory = {
        'id': 1,
        'user_id': 'test_user',
        'chat_id': 'chat-original',
        'text': 'Original text',
    }
    
    updated = reconsolidator.reconsolidate(old_memory, "New context information")
    
    # Verify chat_id preserved (v2 schema)
    assert updated['chat_id'] == 'chat-original'


def test_reconsolidation_updates_text():
    """Test that reconsolidation updates text field (v2 schema)."""
    reconsolidator = EpisodicReconsolidator(MockLLM())
    
    old_memory = {
        'id': 1,
        'user_id': 'test_user',
        'chat_id': 'chat-1',
        'text': 'Test',
    }
    
    updated = reconsolidator.reconsolidate(old_memory, "New info")
    
    # Verify text field is present and non-empty
    assert 'text' in updated
    assert updated['text']


def test_separator_preserves_chat_id():
    """Test that separator preserves chat_id field (v2 schema)."""
    separator = EpisodicSeparator(MockLLM())
    
    memory_a = {
        'id': 1,
        'user_id': 'test_user',
        'chat_id': 'chat-a',
        'text': 'Memory A',
    }
    
    memory_b = {
        'id': 2,
        'user_id': 'test_user',
        'chat_id': 'chat-b',
        'text': 'Memory B',
    }
    
    updated_a, updated_b = separator.separate(memory_a, memory_b)
    
    # Verify chat_id preserved for A (v2 schema)
    assert updated_a['chat_id'] == 'chat-a'
    
    # Verify chat_id preserved for B (v2 schema)
    assert updated_b['chat_id'] == 'chat-b'


if __name__ == '__main__':
    test_reconsolidation_preserves_chat_id()
    test_reconsolidation_updates_text()
    test_separator_preserves_chat_id()
    print('All processor unit tests passed!')
