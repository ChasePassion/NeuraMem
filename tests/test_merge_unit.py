"""Unit test for merge functionality."""

from src.memory_system.processors.merger import EpisodicMerger


class MockLLM:
    def chat_json(self, system_prompt, user_message, default):
        return default


def test_merge_time_selection():
    """Test that merge uses earliest time."""
    merger = EpisodicMerger(MockLLM())
    
    memory_a = {
        'id': 1,
        'user_id': 'test_user',
        'chat_id': 'chat_a',
        'who': 'user',
        'metadata': {
            'time': '2025-01-01T10:00:00Z',
            'context': 'context a',
            'thing': 'thing a'
        }
    }
    
    memory_b = {
        'id': 2,
        'user_id': 'test_user',
        'chat_id': 'chat_b',
        'who': 'user',
        'metadata': {
            'time': '2025-01-02T10:00:00Z',
            'context': 'context b',
            'thing': 'thing b'
        }
    }
    
    merged = merger.merge(memory_a, memory_b)
    
    # Verify time selection (should be earliest)
    assert merged['metadata']['time'] == '2025-01-01T10:00:00Z'
    
    # Verify source tracking
    assert 'chat_a' in merged['metadata']['source_chat_ids']
    assert 'chat_b' in merged['metadata']['source_chat_ids']
    assert 1 in merged['metadata']['merged_from_ids']
    assert 2 in merged['metadata']['merged_from_ids']


def test_merge_source_tracking():
    """Test that merge tracks source chat_ids and merged_from_ids."""
    merger = EpisodicMerger(MockLLM())
    
    memory_a = {
        'id': 100,
        'user_id': 'user1',
        'chat_id': 'chat-100',
        'who': 'user',
        'metadata': {
            'time': '2025-06-01T12:00:00Z',
            'context': 'test',
            'thing': 'test'
        }
    }
    
    memory_b = {
        'id': 200,
        'user_id': 'user1',
        'chat_id': 'chat-200',
        'who': 'user',
        'metadata': {
            'time': '2025-06-02T12:00:00Z',
            'context': 'test',
            'thing': 'test'
        }
    }
    
    merged = merger.merge(memory_a, memory_b)
    
    # Verify source_chat_ids contains both
    source_ids = merged['metadata']['source_chat_ids']
    assert 'chat-100' in source_ids
    assert 'chat-200' in source_ids
    
    # Verify merged_from_ids contains both
    merged_ids = merged['metadata']['merged_from_ids']
    assert 100 in merged_ids
    assert 200 in merged_ids


if __name__ == '__main__':
    test_merge_time_selection()
    test_merge_source_tracking()
    print('All merge unit tests passed!')
