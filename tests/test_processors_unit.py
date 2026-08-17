"""Unit tests for processor functionality.

v2 schema: simplified, only core fields (no who, hit_count, metadata)
Updated for batch pattern merging consolidation logic.
"""

import pytest

from src.memory_system.exceptions import LLMParseError
from src.memory_system.processors.memory_manager import EpisodicMemoryManager
from src.memory_system.processors.semantic_writer import SemanticWriter


class MockLLM:
    def chat_json(self, system_prompt, user_message, default, call_label=None):
        return {
            "parsed_data": default,
            "raw_response": "",
            "model": "mock-model",
            "success": True
        }

    def chat(self, system_prompt, user_message):
        return ""  # Empty means keep


class MockLLMBadParse:
    """Mock LLM whose responses never parse as JSON (after repair retry)."""

    def chat_json(self, system_prompt, user_message, default, call_label=None):
        return {
            "parsed_data": default,
            "raw_response": "definitely not json",
            "model": "mock-model",
            "success": False,
        }


class MockLLMWithFacts:
    """Mock LLM that returns specific facts for testing."""
    
    def __init__(self, facts):
        self.facts = facts
    
    def chat_json(self, system_prompt, user_message, default, call_label=None):
        if self.facts:
            return {
                "parsed_data": {
                    "write_semantic": True,
                    "facts": self.facts
                },
                "raw_response": "",
                "model": "mock-model",
                "success": True
            }
        return {
            "parsed_data": {
                "write_semantic": False,
                "facts": []
            },
            "raw_response": "",
            "model": "mock-model",
            "success": True
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


def test_episodic_manager_raises_on_parse_failure():
    """Parse failure must raise, not masquerade as an empty operation set.

    A silent fallback would drop the turn's CRUD decisions with only a log
    line (architecture_target.md #22).
    """
    manager = EpisodicMemoryManager(MockLLMBadParse())

    with pytest.raises(LLMParseError):
        manager.manage_memories(
            user_text="I visited Paris last week",
            assistant_text="That sounds wonderful!",
            episodic_memories=[],
        )


if __name__ == '__main__':
    test_semantic_writer_batch_processing()
    test_semantic_writer_no_facts()
    test_episodic_manager_raises_on_parse_failure()
    print('All processor unit tests passed!')
