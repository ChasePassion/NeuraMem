"""Server contract tests: /v1/* request/response shapes (implementation plan step 4).

The legacy REST contract must survive the rewrite unchanged: same routes,
same schemas, same SSE event format, same error codes. Memory and the
chat LLM are fakes injected via dependency overrides.
"""

import json
import threading

from fastapi.testclient import TestClient

import pytest

from neuramem.core.exceptions import LLMCallError, MilvusConnectionError
from neuramem.core.models import MemoryRecord, SearchResult
from neuramem_server import app as app_module
from neuramem_server.app import app
from neuramem_server.deps import get_chat_llm, get_memory_system


class FakeMemory:
    def __init__(self):
        self.calls = []
        self.report_event = threading.Event()
        self.manage_event = threading.Event()
        self.fail_with = None

    async def manage_async(self, user_text, assistant_text, user_id, chat_id, metadata=None):
        self.calls.append(("manage", user_id, chat_id))
        self.manage_event.set()
        return [11, 12]

    async def search_async(self, query, user_id):
        if self.fail_with:
            raise self.fail_with
        self.calls.append(("search", query, user_id))
        return SearchResult(
            query=query,
            user_id=user_id,
            episodic=[
                MemoryRecord(id=1, user_id=user_id, memory_type="episodic", ts=1,
                             chat_id="c", text="visited Hangzhou")
            ],
            semantic=[
                MemoryRecord(id=2, user_id=user_id, memory_type="semantic", ts=2,
                             chat_id="c", text="user lives in Beijing")
            ],
        )

    async def report_usage_async(self, result, answer_text):
        self.calls.append(("report_usage", result.query, answer_text))
        self.report_event.set()
        from neuramem.core.models import UsageReport

        return UsageReport(judged_candidates=1, used_memory_ids=[1], assignments={1: 5})

    async def delete_async(self, memory_id, user_id):
        self.calls.append(("delete", memory_id, user_id))
        return memory_id == 123

    async def reset_async(self, user_id):
        self.calls.append(("reset", user_id))
        return 7

    async def consolidate_async(self, user_id=None):
        self.calls.append(("consolidate", user_id))
        from neuramem.core.models import ConsolidationStats

        return ConsolidationStats(memories_processed=9, semantic_created=2)


class FakeChatLLM:
    async def complete(self, system_prompt, user_message, *, call_label=None):
        return None

    async def stream(self, system_prompt, user_message, *, call_label=None):
        yield "Hello"
        yield " there"


@pytest.fixture
def client():
    memory = FakeMemory()
    app.dependency_overrides[get_memory_system] = lambda: memory
    app.dependency_overrides[get_chat_llm] = lambda: FakeChatLLM()
    original = app_module.get_memory_system
    app_module.get_memory_system = lambda: memory  # lifespan fail-fast probe

    class _Bundle:
        def __init__(self):
            self.memory = memory

        def __getattr__(self, name):
            return getattr(test_client, name)

    with TestClient(app) as test_client:
        yield _Bundle()
    app.dependency_overrides.clear()
    app_module.get_memory_system = original
    memory.report_event.clear()
    memory.manage_event.clear()


class TestHealthAndRoot:
    def test_health_contract(self, client):
        response = client.get("/v1/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "version": "1.0.0"}

    def test_root_info(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["name"] == "NeuraMem API"


class TestMemoriesContract:
    def test_manage_response_shape(self, client):
        response = client.post(
            "/v1/memories/manage",
            json={
                "user_id": "u1", "chat_id": "c1",
                "user_text": "hi", "assistant_text": "hello",
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body == {"added_ids": [11, 12], "success": True}
        assert ("manage", "u1", "c1") in client.memory.calls

    def test_search_response_shape(self, client):
        response = client.post(
            "/v1/memories/search", json={"user_id": "u1", "query": "trip"}
        )
        assert response.status_code == 200
        body = response.json()
        assert set(body.keys()) == {"episodic", "semantic"}
        record = body["episodic"][0]
        assert set(record.keys()) == {
            "id", "user_id", "memory_type", "ts", "chat_id", "text", "group_id"
        }
        assert record["text"] == "visited Hangzhou"
        assert body["semantic"][0]["memory_type"] == "semantic"

    def test_delete_known_and_unknown(self, client):
        response = client.delete("/v1/memories/123", params={"user_id": "u1"})
        assert response.status_code == 200
        assert response.json() == {"success": True, "deleted_count": 1}

        response = client.delete("/v1/memories/999", params={"user_id": "u1"})
        assert response.status_code == 404
        assert response.json()["error_code"] == "MEMORY_NOT_FOUND"

    def test_reset_contract(self, client):
        response = client.request(
            "DELETE", "/v1/memories/reset", json={"user_id": "u1"}
        )
        assert response.status_code == 200
        assert response.json() == {"success": True, "deleted_count": 7}

    def test_consolidate_contract(self, client):
        response = client.post(
            "/v1/memories/consolidate", json={"user_id": "u1"}
        )
        assert response.status_code == 200
        assert response.json() == {"memories_processed": 9, "semantic_created": 2}

    def test_invalid_user_id_rejected(self, client):
        response = client.post(
            "/v1/memories/search",
            json={"user_id": "u'; drop table --", "query": "x"},
        )
        assert response.status_code == 422

        response = client.post(
            "/v1/memories/search",
            json={"user_id": "x" * 65, "query": "x"},
        )
        assert response.status_code == 422


class TestChatSse:
    def test_sse_events_and_closed_loop(self, client):
        response = client.post(
            "/v1/chat",
            json={
                "user_id": "u1", "chat_id": "c1", "message": "Where did I go?",
                "history": [{"role": "user", "content": "earlier"}],
            },
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")

        events = [
            json.loads(line[len("data: "):])
            for line in response.text.splitlines()
            if line.startswith("data: ")
        ]
        chunks = [e for e in events if e["type"] == "chunk"]
        done = [e for e in events if e["type"] == "done"]
        assert [c["content"] for c in chunks] == ["Hello", " there"]
        assert done and done[0]["full_content"] == "Hello there"

        # closed loop: report_usage ran with the accumulated answer,
        # then manage — the legacy server never did the first part (#14)
        assert client.memory.report_event.wait(timeout=5)
        assert client.memory.manage_event.wait(timeout=5)
        report_calls = [c for c in client.memory.calls if c[0] == "report_usage"]
        assert report_calls and report_calls[0][2] == "Hello there"

    def test_error_event_on_failure(self, client):
        memory = client.memory
        client.memory.fail_with = LLMCallError("m", 3, RuntimeError("boom"))
        response = client.post(
            "/v1/chat",
            json={"user_id": "u1", "chat_id": "c1", "message": "hi", "history": []},
        )
        events = [
            json.loads(line[len("data: "):])
            for line in response.text.splitlines()
            if line.startswith("data: ")
        ]
        assert any(e["type"] == "error" for e in events)


class TestErrorMapping:
    def test_llm_call_error_maps_to_502(self, client):
        memory = client.memory
        client.memory.fail_with = LLMCallError("model-x", 5, RuntimeError("upstream"))
        response = client.post(
            "/v1/memories/search", json={"user_id": "u1", "query": "q"}
        )
        assert response.status_code == 502
        body = response.json()
        assert body["error_code"] == "LLM_SERVICE_ERROR"
        assert body["model"] == "model-x"

    def test_milvus_connection_error_maps_to_503(self, client):
        client.memory.fail_with = MilvusConnectionError(
            "http://milvus:19530", RuntimeError("down")
        )
        response = client.post(
            "/v1/memories/search", json={"user_id": "u1", "query": "q"}
        )
        assert response.status_code == 503
        assert response.json()["error_code"] == "DB_CONNECTION_ERROR"
