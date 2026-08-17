"""Unit tests for the embedder adapter (step 2 self-review)."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from neuramem.config import EmbeddingConfig
from neuramem.embed.openai_adapter import OpenAIEmbedder


def _config() -> EmbeddingConfig:
    return EmbeddingConfig(_env_file=None, api_key="k")


def _patch_client(create_mock):
    ctor = Mock(return_value=Mock())
    ctor.return_value.embeddings.create = create_mock
    return patch("neuramem.embed.openai_adapter.AsyncOpenAI", ctor), ctor


def _embed_resp(values):
    """Build a fake embeddings.create response with the given vectors."""
    response = Mock()
    items = []
    for vector in values:
        item = Mock()
        item.embedding = vector
        items.append(item)
    response.data = items
    return response


class TestEmbedder:
    @pytest.mark.asyncio
    async def test_empty_input_returns_empty(self):
        embedder = OpenAIEmbedder(_config())
        assert await embedder.embed([]) == []

    @pytest.mark.asyncio
    async def test_dim_property(self):
        embedder = OpenAIEmbedder(_config())
        assert embedder.dim == 2560

    @pytest.mark.asyncio
    async def test_blank_text_is_replaced_with_empty_vector(self):
        """Single empty string would otherwise be sent to the provider.

        Provider is only called with the non-empty slice; the result
        returned to the caller keeps blank entries as empty vectors in
        their original positions.
        """
        create = AsyncMock(return_value=_embed_resp([[1.0, 2.0]]))
        with _patch_client(create)[0]:
            embedder = OpenAIEmbedder(_config())
            result = await embedder.embed(["", "real", ""])

        assert result == [[], [1.0, 2.0], []]
        # only non-empty texts were forwarded
        assert create.call_args.kwargs["input"] == ["real"]

    @pytest.mark.asyncio
    async def test_whitespace_only_is_treated_as_blank(self):
        create = AsyncMock(return_value=_embed_resp([[1.0, 2.0]]))
        with _patch_client(create)[0]:
            embedder = OpenAIEmbedder(_config())
            result = await embedder.embed(["   ", "real"])

        assert result == [[], [1.0, 2.0]]

    @pytest.mark.asyncio
    async def test_sdk_constructed_with_zero_retries(self):
        create = AsyncMock(return_value=_embed_resp([[1.0, 2.0]]))
        patched, ctor = _patch_client(create)
        with patched:
            embedder = OpenAIEmbedder(_config())
            await embedder.embed(["real"])
        assert ctor.call_args.kwargs["max_retries"] == 0