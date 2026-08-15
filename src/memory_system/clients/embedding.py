"""Embedding client for OpenAI-compatible embedding APIs."""

import logging
import os
from typing import List, Optional

from openai import OpenAI

from ..exceptions import LLMCallError
from ..utils.retry import RetryExecutor

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """Embedding model client using OpenAI-compatible API.
    
    Default configuration uses SiliconFlow/Qwen with 2560 dimensions.
    """
    
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        max_retries: Optional[int] = None,
        base_delay: Optional[float] = None,
    ):
        """Initialize embedding client.
        
        Args:
            api_key: API key for the embedding service
            base_url: Base URL for the embedding API
            model: Model ID for embeddings
            max_retries: Retry budget for transient API/network failures.
                Defaults to LLM_MAX_RETRIES env var, then 10 (same knob as
                LLMClient so a single env var tunes the whole pipeline).
            base_delay: Base exponential-backoff delay in seconds.
                Defaults to LLM_BASE_DELAY env var, then 1.0.
        """
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._dim = 2560
        self._max_retries = (
            max_retries if max_retries is not None
            else int(os.getenv("LLM_MAX_RETRIES", "10"))
        )
        self._base_delay = (
            base_delay if base_delay is not None
            else float(os.getenv("LLM_BASE_DELAY", "1.0"))
        )
    
    @property
    def dim(self) -> int:
        """Return embedding vector dimension (2560)."""
        return self._dim
    
    def encode(self, texts: List[str]) -> List[List[float]]:
        """Batch encode texts to embedding vectors.
        
        Args:
            texts: List of texts to encode
            
        Returns:
            List of embedding vectors (each with 2560 dimensions)
            
        Raises:
            LLMCallError: If API call fails after retries
        """
        if not texts:
            return []
        
        executor = RetryExecutor(
            max_retries=self._max_retries,
            base_delay=self._base_delay,
            model=self._model,
            operation="embedding"
        )
        
        def do_encode():
            response = self._client.embeddings.create(
                model=self._model,
                input=texts
            )
            return [item.embedding for item in response.data]
        
        return executor.execute(do_encode)

