"""Unit tests for processor functionality."""

from src.memory_system.processors.reconsolidator import EpisodicReconsolidator
from src.memory_system.processors.separator import EpisodicSeparator


class MockLLM:
    def chat_json(self, system_prompt, user_message, default):
        return default
    
    def chat(self, system_prompt, user_message):
        return ""  # Empty means keep


def test_reconsolidation_preserves_fields():
    """Test that reconsolidation preserves immutable fields."""
    reconsolidator = EpisodicReconsolidator(MockLLM())
    
    old_memory = {
        'id': 1,
        'user_id': 'test_user',
        'chat_id': 'chat-original',
        'who': 'user',
        'text': 'Original text',
        'metadata': {
            'time': '2025-01-01T10:00:00Z',
            'chatid': 'chat-original',
            'who': 'user',
            'context': 'original context',
            'thing': 'original thing',
            'updates': []
        }
    }
    
    updated = reconsolidator.reconsolidate(old_memory, "New context information")
    
    # Verify immutable fields preserved
    assert updated['metadata']['time'] == '2025-01-01T10:00:00Z'
    assert updated['chat_id'] == 'chat-original'
    assert updated['who'] == 'user'
    assert updated['metadata']['chatid'] == 'chat-original'
    assert updated['metadata']['who'] == 'user'


def test_reconsolidation_grows_updates():
    """Test that reconsolidation grows updates array."""
    reconsolidator = EpisodicReconsolidator(MockLLM())
    
    old_memory = {
        'id': 1,
        'user_id': 'test_user',
        'chat_id': 'chat-1',
        'who': 'user',
        'text': 'Test',
        'metadata': {
            'time': '2025-01-01T10:00:00Z',
            'chatid': 'chat-1',
            'who': 'user',
            'context': 'test',
            'thing': 'test',
            'updates': [
                {'time': '2025-01-02T10:00:00Z', 'desc': 'First update'}
            ]
        }
    }
    
    updated = reconsolidator.reconsolidate(old_memory, "New info")
    
    # Verify updates array grew
    assert len(updated['metadata']['updates']) >= 2


def test_separator_preserves_immutable_fields():
    """Test that separator preserves immutable fields."""
    separator = EpisodicSeparator(MockLLM())
    
    memory_a = {
        'id': 1,
        'user_id': 'test_user',
        'chat_id': 'chat-a',
        'who': 'user',
        'text': 'Memory A',
        'metadata': {
            'time': '2025-01-01T10:00:00Z',
            'chatid': 'chat-a',
            'who': 'user',
            'context': 'context a',
            'thing': 'thing a'
        }
    }
    
    memory_b = {
        'id': 2,
        'user_id': 'test_user',
        'chat_id': 'chat-b',
        'who': 'friend',
        'text': 'Memory B',
        'metadata': {
            'time': '2025-01-02T10:00:00Z',
            'chatid': 'chat-b',
            'who': 'friend',
            'context': 'context b',
            'thing': 'thing b'
        }
    }
    
    updated_a, updated_b = separator.separate(memory_a, memory_b)
    
    # Verify immutable fields preserved for A
    assert updated_a['metadata']['time'] == '2025-01-01T10:00:00Z'
    assert updated_a['metadata']['chatid'] == 'chat-a'
    assert updated_a['metadata']['who'] == 'user'
    
    # Verify immutable fields preserved for B
    assert updated_b['metadata']['time'] == '2025-01-02T10:00:00Z'
    assert updated_b['metadata']['chatid'] == 'chat-b'
    assert updated_b['metadata']['who'] == 'friend'


if __name__ == '__main__':
    test_reconsolidation_preserves_fields()
    test_reconsolidation_grows_updates()
    test_separator_preserves_immutable_fields()
    print('All processor unit tests passed!')
