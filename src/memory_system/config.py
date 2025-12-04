"""Configuration module for AI Memory System."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MemoryConfig:
    """Configuration for the AI Memory System.
    
    Attributes:
        milvus_uri: URI for Milvus vector database connection
        openrouter_api_key: API key for OpenRouter services
        openrouter_base_url: Base URL for OpenRouter API
        embedding_model: Model ID for text embedding (qwen/qwen3-embedding-4b)
        llm_model: Model ID for LLM operations (x-ai/grok-4.1-fast:free)
        embedding_dim: Dimension of embedding vectors (2560 for qwen3-embedding-4b)
        collection_name: Name of the Milvus collection for memories
        k_semantic: Maximum number of semantic memories to retrieve
        k_episodic: Maximum number of episodic memories to retrieve
    """
    
    # Milvus configuration
    milvus_uri: str = "http://115.190.109.17:19530"
    collection_name: str = "memories"
    
    # SiliconFlow configuration (for embeddings)
    siliconflow_api_key: str = "sk-polezsgvrdxmqxfwfwxgonkpzphiamsxojsmokmjirrvbzdp"
    siliconflow_base_url: str = "https://api.siliconflow.cn/v1"
    
    # OpenRouter configuration (kept for fallback)
    openrouter_api_key: str = "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"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    
    # DeepSeek configuration (primary LLM)
    deepseek_api_key: str = "sk-d99c433f066744e3b9489b3ce80ac943"
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    
    # Model configuration
    embedding_model: str = "Qwen/Qwen3-Embedding-4B"  # SiliconFlow model ID
    embedding_base_url: str = "https://api.siliconflow.cn/v1"  # SiliconFlow API
    llm_model: str = "deepseek-chat"  # Use DeepSeek as primary LLM
    llm_base_url: str = "https://api.deepseek.com"  # DeepSeek API base URL
    embedding_dim: int = 2560
    
    # Retrieval configuration
    k_semantic: int = 5
    k_episodic: int = 5
    
    # Note: Merge/separate functionality has been removed.
    # The following fields are no longer used:
    # - t_merge_high (merge threshold)
    # - t_amb_low (ambiguous similarity threshold)
    # - merge_time_window_same_chat (time window for same chat merge)
    # - merge_time_window_diff_chat (time window for different chat merge)
