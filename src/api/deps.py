"""Dependency injection for FastAPI endpoints."""

from functools import lru_cache
import logging

from src.memory_system import Memory, MemoryConfig

logger = logging.getLogger(__name__)


@lru_cache()
def get_memory_system() -> Memory:
    """Get singleton Memory instance.
    
    Uses lru_cache to ensure only one Memory instance is created
    across all requests, maintaining connection pools and state.
    
    Returns:
        Memory: Singleton memory system instance
    """
    logger.info("Initializing Memory system singleton...")
    config = MemoryConfig()
    memory = Memory(config)
    logger.info(f"Memory system initialized with collection: {config.collection_name}")
    return memory
