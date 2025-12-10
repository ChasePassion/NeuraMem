"""Configuration module for AI Memory System."""

from dataclasses import dataclass, field
from typing import Optional
import os
import dotenv

dotenv.load_dotenv()

@dataclass
class MemoryConfig:
    """Configuration for the AI Memory System."""
    
    # Milvus 向量数据库配置
    milvus_uri: str = field(default_factory=lambda: os.getenv("MILVUS_URL"))
    collection_name: str = "memories"
    
    # Embedding 模型配置
    embedding_api_key: str = field(default_factory=lambda: os.getenv("SILICONFLOW_API_KEY"))
    embedding_base_url: str = field(default_factory=lambda: os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"))
    embedding_model: str = field(default_factory=lambda: os.getenv("SILICONFLOW_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-4B"))
    embedding_dim: int = 2560
    
    # 主要 LLM 配置 (DeepSeek)
    llm_primary_api_key: str = field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY"))
    llm_primary_base_url: str = field(default_factory=lambda: os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    llm_primary_model: str = field(default_factory=lambda: os.getenv("DEEPSEEK_MODEL", "deepseek-chat"))
    
    # 备用 LLM 配置 (OpenRouter)
    llm_fallback_api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY"))
    llm_fallback_base_url: str = field(default_factory=lambda: os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"))
    
    # GLM（备用）
    glm_api_key: str = field(default_factory=lambda: os.getenv("GLM_API_KEY"))
    glm_base_url: str = field(default_factory=lambda: os.getenv("GLM_BASE_URL", "https://open.bigmodel.cn/api/coding/paas/v4"))
    glm_model: str = field(default_factory=lambda: os.getenv("GLM_MODEL", "glm-4.6v"))
    
    # 检索配置
    k_semantic: int = field(default_factory=lambda: int(os.getenv("K_SEMANTIC", "5")))
    k_episodic: int = field(default_factory=lambda: int(os.getenv("K_EPISODIC", "5")))
    use_all_semantic: bool = field(default_factory=lambda: os.getenv("USE_ALL_SEMANTIC", "true").lower() == "true")
    
    # Langfuse 监控配置
    langfuse_secret_key: str = field(default_factory=lambda: os.getenv("LANGFUSE_SECRET_KEY"))
    langfuse_public_key: str = field(default_factory=lambda: os.getenv("LANGFUSE_PUBLIC_KEY"))
    langfuse_base_url: str = field(
        default_factory=lambda: os.getenv("LANGFUSE_BASE_URL")
        or os.getenv("LANGFUSE_HOST")
        or "https://cloud.langfuse.com"
    )
    
    # 叙事记忆配置
    narrative_similarity_threshold: float = field(
        default_factory=lambda: float(os.getenv("NARRATIVE_SIMILARITY_THRESHOLD", "0.8"))
    )
