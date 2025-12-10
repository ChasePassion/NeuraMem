"""Embedding client for OpenAI-compatible embedding APIs."""

import time
import logging
from typing import List

from openai import OpenAI

from ..exceptions import LLMCallError

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """Embedding model client using OpenAI-compatible API.
    
    Default configuration uses SiliconFlow/Qwen with 2560 dimensions.
    """
    
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str
    ):
        """Initialize embedding client.
        
        Args:
            api_key: API key for the embedding service
            base_url: Base URL for the embedding API
            model: Model ID for embeddings
        """
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._dim = 2560
        self._max_retries = 3
        self._base_delay = 1.0
    
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
        
        last_error = None
        
        for attempt in range(self._max_retries):
            try:
                response = self._client.embeddings.create(
                    model=self._model,
                    input=texts
                )
                # Extract embeddings in order
                embeddings = [item.embedding for item in response.data]
                return embeddings
                
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Embedding API attempt {attempt + 1}/{self._max_retries} failed: {e}"
                )
                
                if attempt < self._max_retries - 1:
                    delay = self._base_delay * (2 ** attempt)
                    time.sleep(delay)
        
        raise LLMCallError(self._model, self._max_retries, last_error)
