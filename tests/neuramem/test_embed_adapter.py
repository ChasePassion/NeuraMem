"""Unit tests for the embedding adapter (implementation plan step 2)."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from neuramem.config import EmbeddingConfig
from neuramem.embed.openai_adapter import OpenAIEmbedder


def _patch_client(create_mock):
    ctor = Mock(return_value=Mock())
    ctor.return_value.embeddings.create = create_mock
    return patch("neuramem.embed.openai_adapter.AsyncOpenAI", ctor), ctor


def _embed_response(vectors):
    response = Mock()
    response.data = [
        Mock(embedding=vector) for vector in vectors
    ]
    return response


def _config(**overrides) -> EmbeddingConfig:
    params = dict(api_key="k", dim=4)
    params.update(overrides)
    return EmbeddingConfig(_env_file=None, **params)


class TestOpenAIEmbedder:
    @pytest.mark.asyncio
    async def test_embed_returns_vectors(self):
        create = AsyncMock(return_value=_embed_response([[1.0, 0.0, 0.0, 0.0]]))
        with _patch_client(create)[0]:
            embedder = OpenAIEmbedder(_config())
            vectors = await embedder.embed(["hello"])

        assert vectors == [[1.0, 0.0, 0.0, 0.0]]
        assert embedder.dim == 4  # dim from config, not a 2560 literal

    @pytest.mark.asyncio
    async def test_empty_input_short_circuits(self):
        create = AsyncMock()
        with _patch_client(create)[0]:
            embedder = OpenAIEmbedder(_config())
            assert await embedder.embed([]) == []
        create.assert_not_called()

    @pytest.mark.asyncio
    async def test_sdk_constructed_with_zero_retries(self):
        create = AsyncMock(return_value=_embed_response([[0.0] * 4]))
        patched, ctor = _patch_client(create)
        with patched:
            embedder = OpenAIEmbedder(_config())
            await embedder.embed(["x"])
        assert ctor.call_args.kwargs["max_retries"] == 0
