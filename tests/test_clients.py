"""Unit tests for infrastructure layer clients.

Tests for EmbeddingClient, LLMClient, and MilvusStore.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock

from src.memory_system.clients.embedding import EmbeddingClient, OpenRouterError
from src.memory_system.clients.llm import LLMClient
from src.memory_system.clients.milvus_store import MilvusStore, MilvusConnectionError
from src.memory_system.config import MemoryConfig


class TestEmbeddingClient:
    """Unit tests for EmbeddingClient."""
    
    def test_dim_property_returns_2560(self):
        """Test that dim property returns correct dimension (2560)."""
        with patch("src.memory_system.clients.embedding.OpenAI"):
            client = EmbeddingClient(api_key="test_key")
            assert client.dim == 2560
    
    def test_encode_empty_list_returns_empty(self):
        """Test that encoding empty list returns empty list."""
        with patch("src.memory_system.clients.embedding.OpenAI"):
            client = EmbeddingClient(api_key="test_key")
            result = client.encode([])
            assert result == []
    
    def test_encode_returns_correct_dimension_vectors(self):
        """Test that encode() returns vectors with correct dimensions."""
        mock_openai = Mock()
        mock_embedding = Mock()
        mock_embedding.embedding = [0.1] * 2560
        mock_response = Mock()
        mock_response.data = [mock_embedding]
        mock_openai.return_value.embeddings.create.return_value = mock_response
        
        with patch("src.memory_system.clients.embedding.OpenAI", mock_openai):
            client = EmbeddingClient(api_key="test_key")
            result = client.encode(["test text"])
            
            assert len(result) == 1
            assert len(result[0]) == 2560
    
    def test_encode_retries_on_failure(self):
        """Test that encode retries with exponential backoff."""
        mock_openai = Mock()
        mock_embedding = Mock()
        mock_embedding.embedding = [0.1] * 2560
        mock_response = Mock()
        mock_response.data = [mock_embedding]
        
        # Fail twice, succeed on third attempt
        mock_openai.return_value.embeddings.create.side_effect = [
            Exception("API Error 1"),
            Exception("API Error 2"),
            mock_response
        ]
        
        with patch("src.memory_system.clients.embedding.OpenAI", mock_openai):
            with patch("time.sleep"):  # Skip actual sleep
                client = EmbeddingClient(api_key="test_key")
                result = client.encode(["test text"])
                
                assert len(result) == 1
                assert mock_openai.return_value.embeddings.create.call_count == 3
    
    def test_encode_raises_after_max_retries(self):
        """Test that encode raises OpenRouterError after max retries."""
        mock_openai = Mock()
        mock_openai.return_value.embeddings.create.side_effect = Exception("API Error")
        
        with patch("src.memory_system.clients.embedding.OpenAI", mock_openai):
            with patch("time.sleep"):
                client = EmbeddingClient(api_key="test_key")
                
                with pytest.raises(OpenRouterError) as exc_info:
                    client.encode(["test text"])
                
                assert exc_info.value.attempts == 3
                assert "qwen/qwen3-embedding-4b" in str(exc_info.value)


class TestLLMClient:
    """Unit tests for LLMClient."""
    
    def test_chat_returns_response_content(self):
        """Test that chat() returns LLM response content."""
        mock_openai = Mock()
        mock_message = Mock()
        mock_message.content = "Test response"
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        mock_openai.return_value.chat.completions.create.return_value = mock_response
        
        with patch("src.memory_system.clients.llm.OpenAI", mock_openai):
            client = LLMClient(api_key="test_key")
            result = client.chat("system prompt", "user message")
            
            assert result == "Test response"
    
    def test_chat_json_parses_valid_json(self):
        """Test that chat_json() correctly parses valid JSON response."""
        mock_openai = Mock()
        mock_message = Mock()
        mock_message.content = '{"key": "value", "number": 42}'
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        mock_openai.return_value.chat.completions.create.return_value = mock_response
        
        with patch("src.memory_system.clients.llm.OpenAI", mock_openai):
            client = LLMClient(api_key="test_key")
            result = client.chat_json("system prompt", "user message")
            
            assert result == {"key": "value", "number": 42}
    
    def test_chat_json_handles_markdown_code_block(self):
        """Test that chat_json() handles JSON wrapped in markdown code blocks."""
        mock_openai = Mock()
        mock_message = Mock()
        mock_message.content = '```json\n{"key": "value"}\n```'
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        mock_openai.return_value.chat.completions.create.return_value = mock_response
        
        with patch("src.memory_system.clients.llm.OpenAI", mock_openai):
            client = LLMClient(api_key="test_key")
            result = client.chat_json("system prompt", "user message")
            
            assert result == {"key": "value"}
    
    def test_chat_json_returns_default_on_invalid_json(self):
        """Test that chat_json() returns default value on invalid JSON."""
        mock_openai = Mock()
        mock_message = Mock()
        mock_message.content = "This is not valid JSON"
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        mock_openai.return_value.chat.completions.create.return_value = mock_response
        
        with patch("src.memory_system.clients.llm.OpenAI", mock_openai):
            client = LLMClient(api_key="test_key")
            default = {"default": True}
            result = client.chat_json("system prompt", "user message", default=default)
            
            assert result == default
    
    def test_chat_json_returns_empty_dict_on_invalid_json_no_default(self):
        """Test that chat_json() returns empty dict when no default provided."""
        mock_openai = Mock()
        mock_message = Mock()
        mock_message.content = "Invalid JSON"
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        mock_openai.return_value.chat.completions.create.return_value = mock_response
        
        with patch("src.memory_system.clients.llm.OpenAI", mock_openai):
            client = LLMClient(api_key="test_key")
            result = client.chat_json("system prompt", "user message")
            
            assert result == {}
    
    def test_chat_falls_back_to_deepseek_on_failure(self):
        """Test that chat falls back to DeepSeek when OpenRouter fails."""
        # Primary client fails all retries
        primary_client = MagicMock()
        primary_client.chat.completions.create.side_effect = [Exception("rate limit")] * 3
        
        # Fallback client succeeds
        fallback_client = MagicMock()
        fallback_message = MagicMock()
        fallback_message.content = "fallback response"
        fallback_choice = MagicMock()
        fallback_choice.message = fallback_message
        fallback_response = MagicMock()
        fallback_response.choices = [fallback_choice]
        fallback_client.chat.completions.create.return_value = fallback_response
        
        # OpenAI constructor returns primary then fallback client
        with patch(
            "src.memory_system.clients.llm.OpenAI",
            side_effect=[primary_client, fallback_client],
        ) as mock_openai_ctor:
            client = LLMClient(
                api_key="primary",
                fallback_api_key="deepseek_key",
                fallback_base_url="https://api.deepseek.com",
                fallback_model="deepseek-chat",
            )
            
            result = client.chat("system prompt", "user message")
            
            # Verify primary exhausted retries, then fallback succeeded
            assert primary_client.chat.completions.create.call_count == 3
            fallback_client.chat.completions.create.assert_called_once()
            assert result == "fallback response"
            assert mock_openai_ctor.call_count == 2


class TestMilvusStore:
    """Unit tests for MilvusStore CRUD operations."""
    
    @pytest.fixture
    def milvus_store(self):
        """Create a MilvusStore instance for testing."""
        config = MemoryConfig()
        store = MilvusStore(
            uri=config.milvus_uri,
            collection_name="test_unit_memories"
        )
        store.create_collection(dim=config.embedding_dim)
        yield store
        store.drop_collection()
    
    def test_create_collection_creates_with_correct_schema(self, milvus_store):
        """Test that collection is created with correct schema."""
        # Collection should exist after fixture setup
        assert milvus_store._client.has_collection("test_unit_memories")
    
    def test_insert_returns_ids(self, milvus_store):
        """Test that insert returns list of IDs."""
        record = {
            "user_id": "test_user",
            "memory_type": "episodic",
            "ts": 1700000000,
            "chat_id": "chat_001",
            "who": "user",
            "text": "Test memory",
            "vector": [0.1] * 2560,
            "hit_count": 0,
            "metadata": {"context": "test", "thing": "test", "time": "2024-01-01", "chatid": "chat_001", "who": "user"}
        }
        
        ids = milvus_store.insert([record])
        
        assert len(ids) == 1
        assert isinstance(ids[0], int)
        
        # Cleanup
        milvus_store.delete(ids=ids)
    
    def test_query_returns_matching_records(self, milvus_store):
        """Test that query returns records matching filter."""
        record = {
            "user_id": "query_test_user",
            "memory_type": "episodic",
            "ts": 1700000000,
            "chat_id": "chat_query",
            "who": "user",
            "text": "Query test memory",
            "vector": [0.2] * 2560,
            "hit_count": 5,
            "metadata": {"context": "query test", "thing": "test", "time": "2024-01-01", "chatid": "chat_query", "who": "user"}
        }
        
        ids = milvus_store.insert([record])
        milvus_store.flush()
        
        # Query by ID
        results = milvus_store.query(
            filter_expr=f"id == {ids[0]}",
            output_fields=["user_id", "text", "hit_count"]
        )
        
        # Convert to list if needed
        results_list = list(results) if results else []
        
        assert len(results_list) >= 1
        assert results_list[0]["user_id"] == "query_test_user"
        assert results_list[0]["text"] == "Query test memory"
        assert results_list[0]["hit_count"] == 5
        
        # Cleanup
        milvus_store.delete(ids=ids)
    
    def test_delete_removes_records(self, milvus_store):
        """Test that delete removes specified records."""
        record = {
            "user_id": "delete_test_user",
            "memory_type": "episodic",
            "ts": 1700000000,
            "chat_id": "chat_delete",
            "who": "user",
            "text": "Delete test memory",
            "vector": [0.3] * 2560,
            "hit_count": 0,
            "metadata": {"context": "delete test", "thing": "test", "time": "2024-01-01", "chatid": "chat_delete", "who": "user"}
        }
        
        ids = milvus_store.insert([record])
        milvus_store.flush()
        
        # Verify record exists before delete
        results_before = milvus_store.query(
            filter_expr=f"id == {ids[0]}",
            output_fields=["id"]
        )
        results_before_list = list(results_before) if results_before else []
        assert len(results_before_list) == 1
        
        # Delete by IDs
        deleted_count = milvus_store.delete(ids=ids)
        assert deleted_count == 1
        
        milvus_store.flush()
        
        # Verify record is gone
        results_after = milvus_store.query(
            filter_expr=f"id == {ids[0]}",
            output_fields=["id"]
        )
        results_after_list = list(results_after) if results_after else []
        assert len(results_after_list) == 0
    
    def test_search_returns_similar_vectors(self, milvus_store):
        """Test that search returns records with similar vectors."""
        # Insert a record with known vector
        base_vector = [0.5] * 2560
        record = {
            "user_id": "search_test_user",
            "memory_type": "episodic",
            "ts": 1700000000,
            "chat_id": "chat_search",
            "who": "user",
            "text": "Search test memory",
            "vector": base_vector,
            "hit_count": 0,
            "metadata": {"context": "search test", "thing": "test", "time": "2024-01-01", "chatid": "chat_search", "who": "user"}
        }
        
        ids = milvus_store.insert([record])
        milvus_store.flush()
        
        # Search with similar vector
        results = milvus_store.search(
            vectors=[base_vector],
            filter_expr="user_id == 'search_test_user'",
            limit=5,
            output_fields=["user_id", "text"]
        )
        
        assert len(results) == 1  # One query vector
        assert len(results[0]) >= 1  # At least one result
        assert results[0][0]["user_id"] == "search_test_user"
        
        # Cleanup
        milvus_store.delete(ids=ids)
