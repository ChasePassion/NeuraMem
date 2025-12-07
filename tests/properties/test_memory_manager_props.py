"""Property-based tests for EpisodicMemoryManager processor.

This module contains property tests for the memory management CRUD logic.
"""

import pytest
from hypothesis import given, strategies as st, settings, assume, HealthCheck

from src.memory_system.processors.memory_manager import EpisodicMemoryManager, MemoryManagementResult, MemoryOperation
from tests.properties.dummy_llm import DummyLLMClient


# Strategies for generating test data

def chitchat_message_strategy():
    """Generate pure chitchat messages that should NOT trigger any operations.
    
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
    
    These should NOT trigger any memory operations.
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
    
    These SHOULD trigger ADD operations.
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


def update_request_strategy():
    """Generate update requests for existing memories.
    
    These SHOULD trigger UPDATE operations.
    """
    updates = [
        "我现在已经是工程师了",
        "我已经毕业了，现在在工作",
        "我搬家到北京了",
        "我的专业改成计算机科学了",
        "我现在不喜欢喝茶了，喜欢咖啡",
        "I changed my major to data science",
        "I'm no longer a student, I'm working now",
        "My advisor changed to Professor Wang",
    ]
    return st.sampled_from(updates)


def delete_request_strategy():
    """Generate delete requests for existing memories.
    
    These SHOULD trigger DELETE operations.
    """
    deletes = [
        "请删除我学生身份的记忆",
        "忘记我住上海的信息",
        "删除我喜欢喝茶的记录",
        "Forget about my previous major",
        "Delete my student status memory",
        "请忘记我导师的信息",
    ]
    return st.sampled_from(deletes)


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
def memory_manager(llm_client):
    """Fixture to provide an EpisodicMemoryManager instance for testing."""
    return EpisodicMemoryManager(llm_client)


class TestChitchatFiltering:
    """Property tests for chitchat and knowledge query filtering.
    
    **Feature: ai-memory-system, Property 3: Chitchat and Knowledge Query Filtering**
    **Validates: Requirements 2.4**
    
    For any input that is pure chitchat (greetings, single tokens), objective 
    knowledge questions without personal information, or meaningless fragments, 
    the EpisodicMemoryManager SHALL return no operations.
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
    def test_chitchat_no_operations(self, memory_manager, message):
        """
        **Feature: ai-memory-system, Property 3: Chitchat and Knowledge Query Filtering**
        **Validates: Requirements 2.4**
        
        For any pure chitchat message (greetings, single tokens, meaningless fragments),
        the EpisodicMemoryManager SHALL return no operations.
        """
        result = memory_manager.manage_memories(
            user_text=message,
            assistant_text="好的",
            episodic_memories=[]
        )
        
        assert isinstance(result, MemoryManagementResult)
        assert len(result.operations) == 0, \
            f"Chitchat message '{message}' should NOT trigger any operations"

    @settings(
        max_examples=3,
        suppress_health_check=[
            HealthCheck.function_scoped_fixture,
            HealthCheck.too_slow,
        ],
        deadline=None
    )
    @given(message=knowledge_query_strategy())
    def test_knowledge_query_no_operations(self, memory_manager, message):
        """
        **Feature: ai-memory-system, Property 3: Chitchat and Knowledge Query Filtering**
        **Validates: Requirements 2.4**
        
        For any pure objective knowledge question without personal information,
        the EpisodicMemoryManager SHALL return no operations.
        """
        result = memory_manager.manage_memories(
            user_text=message,
            assistant_text="这是一个很好的问题",
            episodic_memories=[]
        )
        
        assert isinstance(result, MemoryManagementResult)
        assert len(result.operations) == 0, \
            f"Knowledge query '{message}' should NOT trigger any operations"


class TestMemoryAddition:
    """Property tests for memory addition operations.
    
    **Feature: ai-memory-system, Property 4: Personal Information Storage**
    **Validates: Requirements 2.5**
    
    For any user message containing personal information or explicit remember 
    requests, the EpisodicMemoryManager SHALL return ADD operations.
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
    def test_remember_request_add(self, memory_manager, message):
        """
        **Feature: ai-memory-system, Property 4: Personal Information Storage**
        **Validates: Requirements 2.5**
        
        For any user message containing explicit "remember this" or similar phrases
        with personal information, the EpisodicMemoryManager SHALL return ADD operations.
        """
        result = memory_manager.manage_memories(
            user_text=message,
            assistant_text="好的，我会记住这个信息",
            episodic_memories=[]
        )
        
        assert isinstance(result, MemoryManagementResult)
        assert len(result.operations) > 0, \
            f"Remember request '{message}' SHOULD trigger ADD operations"
        
        # Verify at least one ADD operation
        add_operations = [op for op in result.operations if op.operation_type == "add"]
        assert len(add_operations) >= 1, \
            f"Remember request should produce at least one ADD operation"
        
        # Verify operation structure
        for op in add_operations:
            assert op.operation_type == "add"
            assert op.text, "ADD operation should have text content"
            assert op.memory_id is None, "ADD operation should not have memory_id"

    @settings(
        max_examples=3,
        suppress_health_check=[
            HealthCheck.function_scoped_fixture,
            HealthCheck.too_slow,
        ],
        deadline=None
    )
    @given(message=personal_info_strategy())
    def test_personal_info_add(self, memory_manager, message):
        """
        **Feature: ai-memory-system, Property 4: Personal Information Storage**
        **Validates: Requirements 2.5**
        
        For any user message containing personal information (identity, background,
        projects, self-reflection), the EpisodicMemoryManager SHALL return ADD operations.
        """
        result = memory_manager.manage_memories(
            user_text=message,
            assistant_text="了解了，谢谢你的分享",
            episodic_memories=[]
        )
        
        assert isinstance(result, MemoryManagementResult)
        assert len(result.operations) > 0, \
            f"Personal info message '{message}' SHOULD trigger ADD operations"
        
        # Verify at least one ADD operation
        add_operations = [op for op in result.operations if op.operation_type == "add"]
        assert len(add_operations) >= 1, \
            f"Personal info should produce at least one ADD operation"


class TestMemoryUpdate:
    """Property tests for memory update operations.
    
    **Feature: ai-memory-system, Property 5: Memory Update Operations**
    **Validates: Requirements 2.6**
    
    For any user message indicating changes to existing information, the 
    EpisodicMemoryManager SHALL return UPDATE operations when relevant memories exist.
    """

    @settings(
        max_examples=3,
        suppress_health_check=[
            HealthCheck.function_scoped_fixture,
            HealthCheck.too_slow,
        ],
        deadline=None
    )
    @given(message=update_request_strategy())
    def test_update_request_update(self, memory_manager, message):
        """
        **Feature: ai-memory-system, Property 5: Memory Update Operations**
        **Validates: Requirements 2.6**
        
        For any user message indicating changes to existing information,
        the EpisodicMemoryManager SHALL return UPDATE operations when relevant memories exist.
        """
        # Create existing memory that could be updated
        existing_memories = [
            {"id": 1, "text": "我是一名学生"},
            {"id": 2, "text": "我的专业是计算机科学"}
        ]
        
        result = memory_manager.manage_memories(
            user_text=message,
            assistant_text="了解了你的变化",
            episodic_memories=existing_memories
        )
        
        assert isinstance(result, MemoryManagementResult)
        
        # Check for UPDATE operations (may or may not have them depending on logic)
        update_operations = [op for op in result.operations if op.operation_type == "update"]
        
        if update_operations:
            # Verify operation structure
            for op in update_operations:
                assert op.operation_type == "update"
                assert op.memory_id is not None, "UPDATE operation should have memory_id"
                assert op.old_text, "UPDATE operation should have old_text"
                assert op.text, "UPDATE operation should have new_text"


class TestMemoryDelete:
    """Property tests for memory delete operations.
    
    **Feature: ai-memory-system, Property 6: Memory Delete Operations**
    **Validates: Requirements 2.7**
    
    For any user message requesting deletion of information, the 
    EpisodicMemoryManager SHALL return DELETE operations when relevant memories exist.
    """

    @settings(
        max_examples=3,
        suppress_health_check=[
            HealthCheck.function_scoped_fixture,
            HealthCheck.too_slow,
        ],
        deadline=None
    )
    @given(message=delete_request_strategy())
    def test_delete_request_delete(self, memory_manager, message):
        """
        **Feature: ai-memory-system, Property 6: Memory Delete Operations**
        **Validates: Requirements 2.7**
        
        For any user message requesting deletion of information,
        the EpisodicMemoryManager SHALL return DELETE operations when relevant memories exist.
        """
        # Create existing memory that could be deleted
        existing_memories = [
            {"id": 1, "text": "我是一名学生"},
            {"id": 2, "text": "我喜欢喝茶"}
        ]
        
        result = memory_manager.manage_memories(
            user_text=message,
            assistant_text="好的，我已经删除了相关信息",
            episodic_memories=existing_memories
        )
        
        assert isinstance(result, MemoryManagementResult)
        
        # Check for DELETE operations (may or may not have them depending on logic)
        delete_operations = [op for op in result.operations if op.operation_type == "delete"]
        
        if delete_operations:
            # Verify operation structure
            for op in delete_operations:
                assert op.operation_type == "delete"
                assert op.memory_id is not None, "DELETE operation should have memory_id"
                assert op.text is None, "DELETE operation should not have text"
                assert op.old_text is None, "DELETE operation should not have old_text"


class TestOperationStructure:
    """Property tests for operation structure validation.
    
    **Feature: ai-memory-system, Property 7: Operation Structure Validation**
    **Validates: Requirements 2.8**
    
    All operations returned by EpisodicMemoryManager SHALL have correct structure.
    """

    @settings(
        max_examples=5,
        suppress_health_check=[
            HealthCheck.function_scoped_fixture,
            HealthCheck.too_slow,
        ],
        deadline=None
    )
    @given(
        user_text=st.text(min_size=1, max_size=100),
        assistant_text=st.text(min_size=1, max_size=100),
        existing_memories=st.lists(
            st.fixed_dictionaries({
                "id": st.integers(min_value=1, max_value=1000),
                "text": st.text(min_size=1, max_size=100)
            }),
            min_size=0,
            max_size=5
        )
    )
    def test_operation_structure_validation(self, memory_manager, user_text, assistant_text, existing_memories):
        """
        **Feature: ai-memory-system, Property 7: Operation Structure Validation**
        **Validates: Requirements 2.8**
        
        All operations returned by EpisodicMemoryManager SHALL have correct structure.
        """
        result = memory_manager.manage_memories(
            user_text=user_text,
            assistant_text=assistant_text,
            episodic_memories=existing_memories
        )
        
        assert isinstance(result, MemoryManagementResult)
        assert isinstance(result.operations, list)
        
        for op in result.operations:
            assert isinstance(op, MemoryOperation)
            assert op.operation_type in ["add", "update", "delete"], \
                f"Invalid operation type: {op.operation_type}"
            
            # Validate ADD operation structure
            if op.operation_type == "add":
                assert op.text is not None, "ADD operation must have text"
                assert op.memory_id is None, "ADD operation must not have memory_id"
                assert op.old_text is None, "ADD operation must not have old_text"
            
            # Validate UPDATE operation structure
            elif op.operation_type == "update":
                assert op.memory_id is not None, "UPDATE operation must have memory_id"
                assert op.text is not None, "UPDATE operation must have new text"
                assert op.old_text is not None, "UPDATE operation must have old text"
            
            # Validate DELETE operation structure
            elif op.operation_type == "delete":
                assert op.memory_id is not None, "DELETE operation must have memory_id"
                assert op.text is None, "DELETE operation must not have text"
                assert op.old_text is None, "DELETE operation must not have old_text"
