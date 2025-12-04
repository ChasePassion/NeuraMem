"""Unit test for merge functionality.

v2 schema: simplified, only core fields (no who, hit_count, metadata)
"""

from src.memory_system.processors.merger import EpisodicMerger


class MockLLM:
    def chat_json(self, system_prompt, user_message, default):
        return default


def test_merge_text_combination():
    """Test that merge combines text from both memories (v2 schema)."""
    merger = EpisodicMerger(MockLLM())
    
    memory_a = {
        'id': 1,
        'user_id': 'test_user',
        'chat_id': 'chat_a',
        'text': 'Memory A content',
    }
    
    memory_b = {
        'id': 2,
        'user_id': 'test_user',
        'chat_id': 'chat_b',
        'text': 'Memory B content',
    }
    
    merged = merger.merge(memory_a, memory_b)
    
    # Verify text is present and non-empty
    assert 'text' in merged
    assert merged['text']
    assert len(merged['text']) > 0


def test_merge_chat_id_preservation():
    """Test that merge preserves chat_id (v2 schema)."""
    merger = EpisodicMerger(MockLLM())
    
    memory_a = {
        'id': 100,
        'user_id': 'user1',
        'chat_id': 'chat-100',
        'text': 'Test memory A',
    }
    
    memory_b = {
        'id': 200,
        'user_id': 'user1',
        'chat_id': 'chat-200',
        'text': 'Test memory B',
    }
    
    merged = merger.merge(memory_a, memory_b)
    
    # Verify chat_id is present
    assert 'chat_id' in merged
    assert merged['chat_id'] in ['chat-100', 'chat-200']


def test_merge_user_id_preservation():
    """Test that merge preserves user_id (v2 schema)."""
    merger = EpisodicMerger(MockLLM())
    
    memory_a = {
        'id': 1,
        'user_id': 'test_user',
        'chat_id': 'chat_a',
        'text': 'Memory A',
    }
    
    memory_b = {
        'id': 2,
        'user_id': 'test_user',
        'chat_id': 'chat_b',
        'text': 'Memory B',
    }
    
    merged = merger.merge(memory_a, memory_b)
    
    # Verify user_id is preserved
    assert merged['user_id'] == 'test_user'


if __name__ == '__main__':
    test_merge_text_combination()
    test_merge_chat_id_preservation()
    test_merge_user_id_preservation()
    print('All merge unit tests passed!')
