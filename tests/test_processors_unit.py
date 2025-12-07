"""Unit tests for processor functionality.

v2 schema: simplified, only core fields (no who, hit_count, metadata)
Updated for batch pattern merging consolidation logic.
"""

from src.memory_system.processors.semantic_writer import SemanticWriter


class MockLLM:
    def chat_json(self, system_prompt, user_message, default):
        return default
    
    def chat(self, system_prompt, user_message):
        return ""  # Empty means keep


class MockLLMWithFacts:
    """Mock LLM that returns specific facts for testing."""
    
    def __init__(self, facts):
        self.facts = facts
    
    def chat_json(self, system_prompt, user_message, default):
        if self.facts:
            return {
                "write_semantic": True,
                "facts": self.facts
            }
        return {
            "write_semantic": False,
            "facts": []
        }


def test_semantic_writer_batch_processing():
    """Test that SemanticWriter accepts batch consolidation data."""
    facts = ["User is a software engineer.", "User likes Python programming."]
    writer = SemanticWriter(MockLLMWithFacts(facts))
    
    consolidation_data = {
        "episodic_texts": [
            "Today I worked on a Python project.",
            "I enjoy coding in Python every day."
        ],
        "existing_semantic_texts": []
    }
    
    extraction = writer.extract(consolidation_data)
    
    assert extraction.write_semantic == True
    assert len(extraction.facts) == 2
    assert extraction.facts[0] == "User is a software engineer."
    assert extraction.facts[1] == "User likes Python programming."


def test_semantic_writer_no_facts():
    """Test that SemanticWriter handles no-write case correctly."""
    writer = SemanticWriter(MockLLM())
    
    consolidation_data = {
        "episodic_texts": ["Some random text."],
        "existing_semantic_texts": []
    }
    
    extraction = writer.extract(consolidation_data)
    
    assert extraction.write_semantic == False
    assert len(extraction.facts) == 0


if __name__ == '__main__':
    test_semantic_writer_batch_processing()
    test_semantic_writer_no_facts()
    print('All processor unit tests passed!')
