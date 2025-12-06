"""Configuration module for AI Memory System."""

from dataclasses import dataclass, field
from typing import Optional
import os
import dotenv

dotenv.load_dotenv()

@dataclass
class MemoryConfig:
    """Configuration for the AI Memory System.
    
    Attributes:
        milvus_uri: URI for Milvus vector database connection
        openrouter_api_key: API key for OpenRouter services
        openrouter_base_url: Base URL for OpenRouter API
        embedding_model: Model ID for text embedding (qwen/qwen3-embedding-4b)
        llm_model: Model ID for LLM operations
        embedding_dim: Dimension of embedding vectors (2560 for qwen3-embedding-4b)
        collection_name: Name of the Milvus collection for memories
        k_semantic: Maximum number of semantic memories to retrieve
        k_episodic: Maximum number of episodic memories to retrieve
    """
    
    # Milvus configuration
    milvus_uri: str = os.getenv("MILVUS_URL")
    collection_name: str = "memories"
    
    # SiliconFlow configuration (for embeddings)
    siliconflow_api_key: str = os.getenv("SILICONFLOW_API_KEY")
    siliconflow_base_url: str = "https://api.siliconflow.cn/v1"
    
    # OpenRouter configuration (kept for fallback)
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY")
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    
    # DeepSeek configuration (primary LLM)
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY")
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    
    # Model configuration
    embedding_model: str = "Qwen/Qwen3-Embedding-4B"  # SiliconFlow model ID
    embedding_base_url: str = "https://api.siliconflow.cn/v1"  # SiliconFlow API
    llm_model: str = "deepseek-chat"  # Use DeepSeek as primary LLM
    llm_base_url: str = "https://api.deepseek.com"  # DeepSeek API base URL
    embedding_dim: int = 2560
    
    # Retrieval configuration
    k_semantic: int = field(default_factory=lambda: int(os.getenv("K_SEMANTIC", "5")))
    k_episodic: int = field(default_factory=lambda: int(os.getenv("K_EPISODIC", "5")))
    use_all_semantic: bool = field(default_factory=lambda: os.getenv("USE_ALL_SEMANTIC", "true").lower() == "true")
    
    # Langfuse configuration
    langfuse_secret_key: str = os.getenv("LANGFUSE_SECRET_KEY")
    langfuse_public_key: str = os.getenv("LANGFUSE_PUBLIC_KEY")
    langfuse_base_url: str = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")
