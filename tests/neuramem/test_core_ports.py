"""Structural tests for the port protocols (implementation plan step 1).

A minimal fake satisfying each protocol must pass the runtime-checkable
isinstance check — this pins the protocol shapes adapters will implement
in step 2 and fails loudly if a port signature drifts.
"""

from typing import AsyncIterator, Optional

from neuramem.core.models import MemoryFilter, MemoryRecord
from neuramem.core.ports import (
    Embedder,
    LLM,
    LLMJsonResult,
    LLMResponse,
    Telemetry,
    VectorStore,
)


class FakeVectorStore:
    async def create_collection(self, dim: int) -> None: ...

    async def insert(self, records: list[MemoryRecord]) -> list[int]:
        return [0] * len(records)

    async def upsert(self, records: list[MemoryRecord]) -> list[int]:
        return [r.id for r in records]

    async def search(
        self,
        vectors: list[list[float]],
        flt: Optional[MemoryFilter] = None,
        limit: int = 10,
    ):
        return [[] for _ in vectors]

    async def query(
        self,
        flt: MemoryFilter,
        limit: int = 100,
        include_vectors: bool = False,
    ) -> list[MemoryRecord]:
        return []

    async def delete(self, ids=None, flt=None) -> int:
        return 0

    async def count(self, flt: Optional[MemoryFilter] = None) -> int:
        return 0

    async def search_groups(
        self, user_id: str, vector: list[float], limit: int = 1
    ):
        return []

    async def insert_group(
        self, user_id: str, centroid_vector: list[float], size: int = 1
    ) -> Optional[int]:
        return 1

    async def update_group(
        self,
        user_id: str,
        group_id: int,
        centroid_vector: Optional[list[float]] = None,
        size: Optional[int] = None,
    ) -> bool:
        return True

    async def delete_group(self, user_id: str, group_id: int) -> bool:
        return True

    async def list_groups(self, user_id: str):
        return []

    async def get_group_members(
        self, group_id: int, user_id: str, include_vectors: bool = False
    ) -> list[MemoryRecord]:
        return []

    async def update_memory_group_id(
        self, memory_id: int, group_id: int, user_id: str
    ) -> bool:
        return True


class FakeEmbedder:
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    @property
    def dim(self) -> int:
        return 1


class FakeLLM:
    async def complete(
        self,
        system_prompt: str,
        user_message: str,
        *,
        call_label: Optional[str] = None,
    ) -> LLMResponse:
        return LLMResponse(content="")

    async def complete_json(
        self,
        system_prompt: str,
        user_message: str,
        default: Optional[dict] = None,
        *,
        call_label: Optional[str] = None,
    ) -> LLMJsonResult:
        return LLMJsonResult(parsed_data={}, raw_response="", model="fake")

    async def stream(
        self,
        system_prompt: str,
        user_message: str,
        *,
        call_label: Optional[str] = None,
    ) -> AsyncIterator[str]:
        yield ""

    @property
    def model_id(self) -> str:
        return "fake-model"


class _NullSpan:
    async def __aenter__(self) -> "FakeSpan":
        return FakeSpan()

    async def __aexit__(self, *exc_info) -> bool:
        return False


class FakeSpan:
    def add_event(self, name: str, attributes: Optional[dict] = None) -> None: ...

    def set_attributes(self, attributes: dict) -> None: ...

    def set_status(self, status) -> None: ...


class FakeTelemetry:
    def start_span(self, name: str, attributes: Optional[dict] = None) -> _NullSpan:
        return _NullSpan()


def test_fakes_satisfy_port_protocols():
    assert isinstance(FakeVectorStore(), VectorStore)
    assert isinstance(FakeEmbedder(), Embedder)
    assert isinstance(FakeLLM(), LLM)
    assert isinstance(FakeTelemetry(), Telemetry)
