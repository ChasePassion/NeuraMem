"""Component assembly / dependency injection.

The server is just another consumer of the library (ch. 9 dependency
direction): it builds its OWN LLM adapter instance for answer generation
— the Memory facade is never pried open for private attributes (the
legacy chat router reached into memory._llm_client).
"""

import logging
from functools import lru_cache

from neuramem.config import MemoryConfig
from neuramem.llm.openai_adapter import OpenAILLM
from neuramem.memory import Memory
from neuramem.core.ports import LLM

logger = logging.getLogger(__name__)


@lru_cache
def get_config() -> MemoryConfig:
    return MemoryConfig()


@lru_cache
def get_memory_system() -> Memory:
    config = get_config()
    logger.info("Initializing Memory system (collection '%s')", config.store.collection_name)
    return Memory(config)


@lru_cache
def get_chat_llm() -> LLM:
    """Server-owned LLM used for answer generation (same provider config).

    Cached: one AsyncOpenAI connection pool and one UsageStats aggregate
    for the process — per-request instances would leak pools and lose all
    usage visibility.
    """
    return OpenAILLM(get_config().llm)
