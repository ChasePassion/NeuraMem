"""Unit tests for infrastructure layer clients.

Tests for EmbeddingClient, LLMClient, and MilvusStore.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock

from src.memory_system.clients.embedding import EmbeddingClient
from src.memory_system.exceptions import LLMCallError
from src.memory_system.clients.llm import LLMClient
from src.memory_system.clients.milvus_store import MilvusStore, MilvusConnectionError
from src.memory_system.config import MemoryConfig


class TestEmbeddingClient:
    """Unit tests for EmbeddingClient."""
    
    def test_dim_property_returns_2560(self):
        """Test that dim property returns correct dimension (2560)."""
        with patch("src.memory_system.clients.embedding.OpenAI"):
            client = EmbeddingClient(api_key="test_key", base_url="https://api.test.com", model="test-model")
            assert client.dim == 2560
    
    def test_encode_empty_list_returns_empty(self):
        """Test that encoding empty list returns empty list."""
        with patch("src.memory_system.clients.embedding.OpenAI"):
            client = EmbeddingClient(api_key="test_key", base_url="https://api.test.com", model="test-model")
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
            client = EmbeddingClient(api_key="test_key", base_url="https://api.test.com", model="test-model")
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
                client = EmbeddingClient(api_key="test_key", base_url="https://api.test.com", model="test-model")
                result = client.encode(["test text"])
                
                assert len(result) == 1
                assert mock_openai.return_value.embeddings.create.call_count == 3
    
    def test_encode_raises_after_max_retries(self):
        """Test that encode raises LLMCallError after max retries."""
        mock_openai = Mock()
        mock_openai.return_value.embeddings.create.side_effect = Exception("API Error")
        
        with patch("src.memory_system.clients.embedding.OpenAI", mock_openai):
            with patch("time.sleep"):
                client = EmbeddingClient(api_key="test_key", base_url="https://api.test.com", model="test-model")
                
                with pytest.raises(LLMCallError) as exc_info:
                    client.encode(["test text"])
                
                assert exc_info.value.attempts == 3
                assert "test-model" in str(exc_info.value)


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
            client = LLMClient(api_key="test_key", base_url="https://api.test.com", model="test-model")
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
            client = LLMClient(api_key="test_key", base_url="https://api.test.com", model="test-model")
            result = client.chat_json("system prompt", "user message")
            assert result["success"] == True
            assert result["parsed_data"] == {"key": "value", "number": 42}
    
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
            client = LLMClient(api_key="test_key", base_url="https://api.test.com", model="test-model")
            result = client.chat_json("system prompt", "user message")
            assert result["success"] == True
            assert result["parsed_data"] == {"key": "value"}
    
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
            client = LLMClient(api_key="test_key", base_url="https://api.test.com", model="test-model")
            default = {"default": True}
            result = client.chat_json("system prompt", "user message", default=default)
            assert result["parsed_data"] == default
    
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
            client = LLMClient(api_key="test_key", base_url="https://api.test.com", model="test-model")
            result = client.chat_json("system prompt", "user message")
            assert result["parsed_data"] == {}
    
    def test_chat_falls_back_to_deepseek_on_failure(self):
        """Test that chat falls back to secondary provider when primary fails."""
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
            with patch("src.memory_system.clients.llm.AsyncOpenAI"):
                client = LLMClient(
                    api_key="primary",
                    base_url="https://api.primary.com",
                    model="primary-model",
                    fallback_api_key="deepseek_key",
                    fallback_base_url="https://api.deepseek.com",
                    fallback_model="deepseek-chat",
                    max_retries=3,
                )
                
                result = client.chat("system prompt", "user message")
                
                # Verify primary exhausted retries, then fallback succeeded
                assert primary_client.chat.completions.create.call_count == 3
                fallback_client.chat.completions.create.assert_called_once()
                assert result == "fallback response"
                assert mock_openai_ctor.call_count == 2

    # --- usage / KV cache accounting (architecture_target.md 6.5) ---

    @staticmethod
    def _make_usage_mock(
        prompt_tokens: int = 100,
        completion_tokens: int = 50,
        cached_tokens=None,
        deepseek_style: bool = False,
    ):
        """Build a response usage mock in MiniMax/OpenAI or DeepSeek shape."""
        usage = Mock()
        usage.prompt_tokens = prompt_tokens
        usage.completion_tokens = completion_tokens
        if deepseek_style:
            usage.prompt_tokens_details = Mock()
            usage.prompt_tokens_details.cached_tokens = None
            usage.prompt_tokens_details.cache_write_tokens = None
            usage.prompt_cache_hit_tokens = cached_tokens
        else:
            details = Mock()
            details.cached_tokens = cached_tokens
            details.cache_write_tokens = None
            usage.prompt_tokens_details = details
        usage.completion_tokens_details = Mock()
        usage.completion_tokens_details.reasoning_tokens = 10
        return usage

    def _make_response_mock(self, usage):
        message = Mock()
        message.content = "Test response"
        choice = Mock()
        choice.message = message
        response = Mock()
        response.choices = [choice]
        response.usage = usage
        return response

    def test_chat_records_minimax_style_cache_usage(self):
        """chat() records prompt/cache tokens from prompt_tokens_details.cached_tokens."""
        usage = self._make_usage_mock(prompt_tokens=200, cached_tokens=80)
        response = self._make_response_mock(usage)
        mock_openai = Mock()
        mock_openai.return_value.chat.completions.create.return_value = response

        with patch("src.memory_system.clients.llm.OpenAI", mock_openai):
            client = LLMClient(api_key="test_key", base_url="https://api.test.com", model="test-model")
            client.chat("system prompt", "user message")

        stats = client.usage_stats.snapshot()
        assert stats["calls"] == 1
        assert stats["cache_read_tokens"] == 80
        # prompt = input + cache_read + cache_write = 120 + 80 + 0
        assert stats["input_tokens"] == 120
        assert stats["total_tokens"] == 120 + 50 + 80
        assert client.usage_stats.hit_rate() == 80 / 200

    def test_chat_records_deepseek_style_cache_fields(self):
        """chat() also picks up top-level prompt_cache_hit_tokens (DeepSeek)."""
        usage = self._make_usage_mock(prompt_tokens=200, cached_tokens=60, deepseek_style=True)
        response = self._make_response_mock(usage)
        mock_openai = Mock()
        mock_openai.return_value.chat.completions.create.return_value = response

        with patch("src.memory_system.clients.llm.OpenAI", mock_openai):
            client = LLMClient(api_key="test_key", base_url="https://api.test.com", model="test-model")
            client.chat("system prompt", "user message")

        stats = client.usage_stats.snapshot()
        assert stats["cache_read_tokens"] == 60
        assert client.usage_stats.hit_rate() == 60 / 200

    def test_chat_without_usage_keeps_stats_empty(self):
        """chat() with no usage in the response records nothing (hit_rate None)."""
        response = self._make_response_mock(None)
        mock_openai = Mock()
        mock_openai.return_value.chat.completions.create.return_value = response

        with patch("src.memory_system.clients.llm.OpenAI", mock_openai):
            client = LLMClient(api_key="test_key", base_url="https://api.test.com", model="test-model")
            client.chat("system prompt", "user message")

        stats = client.usage_stats.snapshot()
        assert stats["calls"] == 0
        assert client.usage_stats.hit_rate() is None

    def test_chat_json_includes_usage(self):
        """chat_json() returns the parsed usage alongside parsed_data."""
        usage = self._make_usage_mock(prompt_tokens=100, cached_tokens=30)
        response = self._make_response_mock(usage)
        response.choices[0].message.content = '{"key": "value"}'
        mock_openai = Mock()
        mock_openai.return_value.chat.completions.create.return_value = response

        with patch("src.memory_system.clients.llm.OpenAI", mock_openai):
            client = LLMClient(api_key="test_key", base_url="https://api.test.com", model="test-model")
            result = client.chat_json("system prompt", "user message")

        assert result["success"] is True
        assert result["parsed_data"] == {"key": "value"}
        assert result["usage"] == {
            "input_tokens": 70,
            "output_tokens": 50,
            "cache_read_tokens": 30,
            "cache_write_tokens": 0,
            "reasoning_tokens": 10,
            "total_tokens": 70 + 50 + 30,
            "cost": 0.0,
        }

    def test_chat_json_includes_usage_on_parse_failure(self):
        """chat_json() reports success=False when parsing fails even after retry.

        The mock always returns unparseable text, so the corrective retry
        (second create call) also fails. usage reflects the last attempt and
        parsed_data falls back to the default (architecture_target.md #22).
        """
        usage = self._make_usage_mock(prompt_tokens=100, cached_tokens=0)
        response = self._make_response_mock(usage)
        response.choices[0].message.content = "not json"
        mock_openai = Mock()
        mock_openai.return_value.chat.completions.create.return_value = response

        with patch("src.memory_system.clients.llm.OpenAI", mock_openai):
            client = LLMClient(api_key="test_key", base_url="https://api.test.com", model="test-model")
            result = client.chat_json("system prompt", "user message", default={"d": 1})

        assert result["success"] is False
        assert result["parsed_data"] == {"d": 1}
        assert result["usage"] is not None
        assert result["usage"]["input_tokens"] == 100
        create = mock_openai.return_value.chat.completions.create
        assert create.call_count == 2  # initial call + one repair retry

    def test_chat_json_repairs_invalid_json_first_attempt(self):
        """chat_json() retries once with corrective feedback and succeeds."""
        usage = self._make_usage_mock(prompt_tokens=100, cached_tokens=0)
        bad_response = self._make_response_mock(usage)
        bad_response.choices[0].message.content = "Sure! Here is my answer..."
        good_response = self._make_response_mock(usage)
        good_response.choices[0].message.content = '{"key": "fixed"}'
        mock_openai = Mock()
        mock_openai.return_value.chat.completions.create.side_effect = [
            bad_response,
            good_response,
        ]

        with patch("src.memory_system.clients.llm.OpenAI", mock_openai):
            client = LLMClient(api_key="test_key", base_url="https://api.test.com", model="test-model")
            result = client.chat_json("system prompt", "user message")

        assert result["success"] is True
        assert result["parsed_data"] == {"key": "fixed"}
        assert result["raw_response"] == '{"key": "fixed"}'
        create = mock_openai.return_value.chat.completions.create
        assert create.call_count == 2
        # the retry message carries the corrective feedback
        retry_message = create.call_args_list[1].kwargs["messages"][1]["content"]
        assert "not valid JSON" in retry_message

    def test_stream_records_usage_from_final_chunk(self):
        """chat_stream() records usage carried by the final stream chunk once."""
        usage = self._make_usage_mock(prompt_tokens=100, cached_tokens=40)
        chunk_mid = Mock()
        chunk_mid.usage = None
        chunk_mid.choices = [Mock()]
        chunk_mid.choices[0].usage = None  # real SDK chunks have None usage
        chunk_mid.choices[0].delta.content = "Hello "
        chunk_last = Mock()
        chunk_last.usage = usage
        chunk_last.choices = [Mock()]
        chunk_last.choices[0].usage = None
        chunk_last.choices[0].delta.content = "world"

        mock_openai = Mock()
        mock_openai.return_value.chat.completions.create.return_value = iter([chunk_mid, chunk_last])

        with patch("src.memory_system.clients.llm.OpenAI", mock_openai):
            client = LLMClient(api_key="test_key", base_url="https://api.test.com", model="test-model")
            text = "".join(client.chat_stream("system prompt", "user message"))

        assert text == "Hello world"
        stats = client.usage_stats.snapshot()
        assert stats["calls"] == 1
        assert stats["cache_read_tokens"] == 40
        assert client.usage_stats.hit_rate() == 40 / 100

    def test_stream_records_usage_from_choice(self):
        """chat_stream() also reads usage from choice.usage (Moonshot style)."""
        usage = self._make_usage_mock(prompt_tokens=100, cached_tokens=25)
        choice = Mock()
        choice.usage = usage
        choice.delta.content = "payload"
        chunk = Mock()
        chunk.usage = None
        chunk.choices = [choice]

        mock_openai = Mock()
        mock_openai.return_value.chat.completions.create.return_value = iter([chunk])

        with patch("src.memory_system.clients.llm.OpenAI", mock_openai):
            client = LLMClient(api_key="test_key", base_url="https://api.test.com", model="test-model")
            text = "".join(client.chat_stream("system prompt", "user message"))

        assert text == "payload"
        assert client.usage_stats.snapshot()["cache_read_tokens"] == 25

    def test_usage_stats_thread_snapshot_isolation(self):
        """thread_snapshot() attributes calls to the recording thread only."""
        usage = self._make_usage_mock(prompt_tokens=100, cached_tokens=10)
        response = self._make_response_mock(usage)
        mock_openai = Mock()
        mock_openai.return_value.chat.completions.create.return_value = response

        with patch("src.memory_system.clients.llm.OpenAI", mock_openai):
            client = LLMClient(api_key="test_key", base_url="https://api.test.com", model="test-model")
            client.chat("system prompt", "user message")

            seen = {}

            def worker():
                client.chat("system prompt", "user message")
                seen["worker"] = client.usage_stats.thread_snapshot()

            import threading
            t = threading.Thread(target=worker)
            t.start()
            t.join()

        # Main thread saw only its own call; worker saw only its own call
        assert client.usage_stats.thread_snapshot()["calls"] == 1
        assert seen["worker"]["calls"] == 1
        # Global aggregation covers both
        assert client.usage_stats.snapshot()["calls"] == 2

    def test_usage_stats_groups_by_label(self):
        """UsageStats separates calls by label (benchmark per-type hit rates)."""
        usage = self._make_usage_mock(prompt_tokens=200, cached_tokens=80)
        response = self._make_response_mock(usage)
        mock_openai = Mock()
        mock_openai.return_value.chat.completions.create.return_value = response

        with patch("src.memory_system.clients.llm.OpenAI", mock_openai):
            client = LLMClient(api_key="test_key", base_url="https://api.test.com", model="test-model")
            client.chat("system prompt", "user message", call_label="answer")
            client.chat("system prompt", "user message", call_label="judge")
            client.chat("system prompt", "user message")  # unlabeled

        assert client.usage_stats.snapshot()["calls"] == 3
        answer = client.usage_stats.snapshot("answer")
        assert answer["calls"] == 1
        assert answer["cache_read_tokens"] == 80
        assert client.usage_stats.hit_rate_of(answer) == 80 / 200
        judge = client.usage_stats.snapshot("judge")
        assert judge["calls"] == 1
        assert client.usage_stats.snapshot("usage_judge")["calls"] == 0
        assert client.usage_stats.labels() == ["answer", "judge"]

    def test_usage_stats_label_thread_snapshot(self):
        """thread_snapshot(label) tracks per-label calls on the calling thread."""
        usage = self._make_usage_mock(prompt_tokens=100, cached_tokens=10)
        response = self._make_response_mock(usage)
        mock_openai = Mock()
        mock_openai.return_value.chat.completions.create.return_value = response

        with patch("src.memory_system.clients.llm.OpenAI", mock_openai):
            client = LLMClient(api_key="test_key", base_url="https://api.test.com", model="test-model")
            client.chat("system prompt", "user message", call_label="answer")

            seen = {}

            def worker():
                client.chat("system prompt", "user message", call_label="answer")
                client.chat("system prompt", "user message", call_label="judge")
                seen["worker_answer"] = client.usage_stats.thread_snapshot("answer")
                seen["worker_total"] = client.usage_stats.thread_snapshot()

            import threading
            t = threading.Thread(target=worker)
            t.start()
            t.join()

        # Main thread sees only its own answer call
        assert client.usage_stats.thread_snapshot("answer")["calls"] == 1
        assert seen["worker_answer"]["calls"] == 1
        assert seen["worker_total"]["calls"] == 2
        # Global per-label aggregation covers both threads
        assert client.usage_stats.snapshot("answer")["calls"] == 2
        assert client.usage_stats.snapshot("judge")["calls"] == 1

    def test_chat_json_accepts_call_label(self):
        """chat_json() forwards the label into usage stats."""
        usage = self._make_usage_mock(prompt_tokens=100, cached_tokens=20)
        response = self._make_response_mock(usage)
        response.choices[0].message.content = '{"ok": true}'
        mock_openai = Mock()
        mock_openai.return_value.chat.completions.create.return_value = response

        with patch("src.memory_system.clients.llm.OpenAI", mock_openai):
            client = LLMClient(api_key="test_key", base_url="https://api.test.com", model="test-model")
            result = client.chat_json("system prompt", "user message", call_label="usage_judge")

        assert result["usage"]["cache_read_tokens"] == 20
        assert client.usage_stats.snapshot("usage_judge")["calls"] == 1
        assert client.usage_stats.snapshot("answer")["calls"] == 0

    def test_estimate_cost_uses_pricing_table(self):
        """estimate_cost() applies per-million-token prices per component."""
        from src.memory_system.clients.llm import LLMUsage, estimate_cost

        usage = LLMUsage(
            input_tokens=1000,
            output_tokens=500,
            cache_read_tokens=2000,
            cache_write_tokens=1000,
        )
        pricing = {
            "input": 2.0,
            "output": 8.0,
            "cache_read": 0.4,
            "cache_write": 2.5,
        }
        cost = estimate_cost(usage, pricing)
        expected = (1000 * 2.0 + 500 * 8.0 + 2000 * 0.4 + 1000 * 2.5) / 1_000_000
        assert cost == pytest.approx(expected)
        assert estimate_cost(usage, None) == 0.0


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
        """Test that insert returns list of IDs (v2 schema)."""
        record = {
            "user_id": "test_user",
            "memory_type": "episodic",
            "ts": 1700000000,
            "chat_id": "chat_001",
            "text": "Test memory",
            "vector": [0.1] * 2560,
        }
        
        ids = milvus_store.insert([record])
        
        assert len(ids) == 1
        assert isinstance(ids[0], int)
        
        # Cleanup
        milvus_store.delete(ids=ids)
    
    def test_query_returns_matching_records(self, milvus_store):
        """Test that query returns records matching filter (v2 schema)."""
        record = {
            "user_id": "query_test_user",
            "memory_type": "episodic",
            "ts": 1700000000,
            "chat_id": "chat_query",
            "text": "Query test memory",
            "vector": [0.2] * 2560,
        }
        
        ids = milvus_store.insert([record])
        milvus_store.flush()
        
        # Query by ID
        results = milvus_store.query(
            filter_expr=f"id == {ids[0]}",
            output_fields=["user_id", "text"]
        )
        
        # Convert to list if needed
        results_list = list(results) if results else []
        
        assert len(results_list) >= 1
        assert results_list[0]["user_id"] == "query_test_user"
        assert results_list[0]["text"] == "Query test memory"
        
        # Cleanup
        milvus_store.delete(ids=ids)
    
    def test_delete_removes_records(self, milvus_store):
        """Test that delete removes specified records (v2 schema)."""
        import time
        
        record = {
            "user_id": "delete_test_user",
            "memory_type": "episodic",
            "ts": 1700000000,
            "chat_id": "chat_delete",
            "text": "Delete test memory",
            "vector": [0.3] * 2560,
        }
        
        ids = milvus_store.insert([record])
        milvus_store.flush()
        time.sleep(0.5)  # Wait for data sync
        
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
        time.sleep(0.5)  # Wait for data sync
        
        # Verify record is gone
        results_after = milvus_store.query(
            filter_expr=f"id == {ids[0]}",
            output_fields=["id"]
        )
        results_after_list = list(results_after) if results_after else []
        assert len(results_after_list) == 0
    
    def test_search_returns_similar_vectors(self, milvus_store):
        """Test that search returns records with similar vectors (v2 schema)."""
        # Insert a record with known vector
        base_vector = [0.5] * 2560
        record = {
            "user_id": "search_test_user",
            "memory_type": "episodic",
            "ts": 1700000000,
            "chat_id": "chat_search",
            "text": "Search test memory",
            "vector": base_vector,
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
