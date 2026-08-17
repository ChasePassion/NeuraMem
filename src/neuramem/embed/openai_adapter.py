"""OpenAI-compatible embedding adapter — native async, single provider."""

import logging
from typing import Optional

from openai import AsyncOpenAI, APIConnectionError, APITimeoutError

from neuramem.config import EmbeddingConfig
from neuramem.core.exceptions import LLMCallError
from neuramem.core.retry import RetryExecutor, register_retryable_type

logger = logging.getLogger(__name__)

register_retryable_type(APIConnectionError, APITimeoutError)


class OpenAIEmbedder:
    """Embedder port implementation over an OpenAI-compatible endpoint.

    dim comes from configuration (single source, #9); a startup probe that
    verifies the served model's dim against the config lands with the
    facade (step 3).
    """

    def __init__(self, config: EmbeddingConfig):
        self._config = config
        # SDK retries off: RetryExecutor owns the retry budget (6.3)
        self._client = AsyncOpenAI(
            api_key=config.api_key.get_secret_value(),
            base_url=config.base_url,
            max_retries=0,
        )

    @property
    def dim(self) -> int:
        return self._config.dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        executor = RetryExecutor(
            max_retries=self._config.max_retries,
            base_delay=self._config.base_delay,
            max_delay=8.0,
            model=self._config.model,
            operation="embedding",
        )

        async def do_embed():
            response = await self._client.embeddings.create(
                model=self._config.model, input=texts
            )
            return [item.embedding for item in response.data]

        try:
            return await executor.execute_async(do_embed)
        except LLMCallError:
            logger.error("embedding call failed after retries for %d texts", len(texts))
            raise
