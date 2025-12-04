"""Property-based tests for EpisodicWriteDecider processor.

This module contains property tests for the write decision logic.
"""

import pytest
from hypothesis import given, strategies as st, settings, assume, HealthCheck

from src.memory_system.processors.write_decider import EpisodicWriteDecider, WriteDecision
from tests.properties.dummy_llm import DummyLLMClient


# Strategies for generating test data

def chitchat_message_strategy():
    """Generate pure chitchat messages that should NOT be stored.
    
    These are greetings, single tokens, or meaningless fragments.
    """
    greetings = [
        "你好", "Hi", "Hello", "嗨", "Hey", "早上好", "Good morning",
        "晚上好", "Good evening", "在吗", "Are you there?", "嗯", "啊",
        "哈哈", "哈哈哈", "ok", "OK", "好的", "嗯嗯", "哦", "呵呵",
        "😊", "👍", "谢谢", "Thanks", "好", "行", "可以"
    ]
    return st.sampled_from(greetings)


def knowledge_query_strategy():
    """Generate pure objective knowledge questions without personal info.
    
    These should NOT be stored as episodic memory.
    """
    queries = [
        "What is the GDP of the United States?",
        "How is a hash table implemented?",
        "什么是机器学习?",
        "Python的列表和元组有什么区别?",
        "How does TCP/IP work?",
        "What is the capital of France?",
        "解释一下什么是递归",
        "What is the time complexity of quicksort?",
        "HTTP和HTTPS有什么区别?",
        "What is a binary search tree?",
    ]
    return st.sampled_from(queries)


def remember_request_strategy():
    """Generate explicit remember requests with personal information.
    
    These SHOULD be stored as episodic memory.
    """
    requests = [
        "请记住我是北京大学的学生",
        "Remember that my major is computer science",
        "帮我记住我住在上海",
        "Please remember I'm working on a machine learning project",
        "记住我的研究方向是联邦学习",
        "Remember that I'm a software engineer at Google",
        "请记住我喜欢喝茶",
        "Remember I have an exam next week",
        "帮我记住我的导师是张教授",
        "Please remember my name is John and I'm from New York",
    ]
    return st.sampled_from(requests)


def personal_info_strategy():
    """Generate messages with personal information that should be stored.
    
    These contain identity, background, projects, or self-reflection.
    """
    messages = [
        "我是一名大三的计算机专业学生",
        "I'm currently working on my thesis about federated learning",
        "我最近在开发一个预算管理应用",
        "I've been struggling with time management lately",
        "我的研究方向是网络安全",
        "I'm a PhD student at MIT",
        "我每天早上都会跑步锻炼",
        "I'm planning to apply for jobs in AI next year",
        "我和我的导师正在合作一个项目",
        "I usually study at the library until 10pm",
    ]
    return st.sampled_from(messages)


@pytest.fixture(scope="module")
def llm_client():
    """Fixture to provide an LLMClient instance for testing."""
    return DummyLLMClient()


@pytest.fixture(scope="module")
def write_decider(llm_client):
    """Fixture to provide an EpisodicWriteDecider instance for testing."""
    return EpisodicWriteDecider(llm_client)


class TestChitchatFiltering:
    """Property tests for chitchat and knowledge query filtering.
    
    **Feature: ai-memory-system, Property 3: Chitchat and Knowledge Query Filtering**
    **Validates: Requirements 2.4**
    
    For any input that is pure chitchat (greetings, single tokens), objective 
    knowledge questions without personal information, or meaningless fragments, 
    the EpisodicWriteDecider SHALL return write_episodic=false.
    """

    @settings(
        max_examples=3,
        suppress_health_check=[
            HealthCheck.function_scoped_fixture,
            HealthCheck.too_slow,
        ],
        deadline=None  # Disable deadline for LLM API calls
    )
    @given(message=chitchat_message_strategy())
    def test_chitchat_not_stored(self, write_decider, message):
        """
        **Feature: ai-memory-system, Property 3: Chitchat and Knowledge Query Filtering**
        **Validates: Requirements 2.4**
        
        For any pure chitchat message (greetings, single tokens, meaningless fragments),
        the EpisodicWriteDecider SHALL return write_episodic=false.
        """
        turns = [{"role": "user", "content": message}]
        
        result = write_decider.decide(chat_id="test_chat", turns=turns)
        
        assert isinstance(result, WriteDecision)
        assert result.write_episodic is False, \
            f"Chitchat message '{message}' should NOT be stored as episodic memory"
        assert len(result.records) == 0, \
            f"Chitchat message should produce no records, got {len(result.records)}"

    @settings(
        max_examples=3,
        suppress_health_check=[
            HealthCheck.function_scoped_fixture,
            HealthCheck.too_slow,
        ],
        deadline=None
    )
    @given(message=knowledge_query_strategy())
    def test_knowledge_query_not_stored(self, write_decider, message):
        """
        **Feature: ai-memory-system, Property 3: Chitchat and Knowledge Query Filtering**
        **Validates: Requirements 2.4**
        
        For any pure objective knowledge question without personal information,
        the EpisodicWriteDecider SHALL return write_episodic=false.
        """
        turns = [{"role": "user", "content": message}]
        
        result = write_decider.decide(chat_id="test_chat", turns=turns)
        
        assert isinstance(result, WriteDecision)
        assert result.write_episodic is False, \
            f"Knowledge query '{message}' should NOT be stored as episodic memory"
        assert len(result.records) == 0, \
            f"Knowledge query should produce no records, got {len(result.records)}"


class TestExplicitRememberRequest:
    """Property tests for explicit remember request storage.
    
    **Feature: ai-memory-system, Property 4: Explicit Remember Request Storage**
    **Validates: Requirements 2.5**
    
    For any user message containing explicit "remember this" or similar phrases 
    with personal information, the EpisodicWriteDecider SHALL return write_episodic=true.
    """

    @settings(
        max_examples=3,
        suppress_health_check=[
            HealthCheck.function_scoped_fixture,
            HealthCheck.too_slow,
        ],
        deadline=None
    )
    @given(message=remember_request_strategy())
    def test_remember_request_stored(self, write_decider, message):
        """
        **Feature: ai-memory-system, Property 4: Explicit Remember Request Storage**
        **Validates: Requirements 2.5**
        
        For any user message containing explicit "remember this" or similar phrases
        with personal information, the EpisodicWriteDecider SHALL return write_episodic=true.
        """
        turns = [{"role": "user", "content": message}]
        
        result = write_decider.decide(chat_id="test_chat", turns=turns)
        
        assert isinstance(result, WriteDecision)
        assert result.write_episodic is True, \
            f"Remember request '{message}' SHOULD be stored as episodic memory"
        assert len(result.records) >= 1, \
            f"Remember request should produce at least one record, got {len(result.records)}"
        
        # Verify record structure (v2 schema: only text field)
        for record in result.records:
            assert record.text, "Record should have 'text' field"

    @settings(
        max_examples=3,
        suppress_health_check=[
            HealthCheck.function_scoped_fixture,
            HealthCheck.too_slow,
        ],
        deadline=None
    )
    @given(message=personal_info_strategy())
    def test_personal_info_stored(self, write_decider, message):
        """
        **Feature: ai-memory-system, Property 4: Explicit Remember Request Storage**
        **Validates: Requirements 2.5**
        
        For any user message containing personal information (identity, background,
        projects, self-reflection), the EpisodicWriteDecider SHALL return write_episodic=true.
        """
        turns = [{"role": "user", "content": message}]
        
        result = write_decider.decide(chat_id="test_chat", turns=turns)
        
        assert isinstance(result, WriteDecision)
        assert result.write_episodic is True, \
            f"Personal info message '{message}' SHOULD be stored as episodic memory"
        assert len(result.records) >= 1, \
            f"Personal info should produce at least one record, got {len(result.records)}"
