"""Layered configuration (architecture_target.md #8 / 8.3).

Each sub-config is its own BaseSettings with an env prefix: LLM_BASE_URL
binds llm.base_url, EMBEDDING_API_KEY binds embedding.api_key, and so on.
(A single root with env_nested_delimiter would split LLM_API_KEY into
llm -> api -> key, because the delimiter is applied at every underscore —
prefix-per-section avoids that without renaming fields to camelCase.)

Misconfiguration fails at construction time, not at first use: building
MemoryConfig() without the required env/args raises ValidationError.

The legacy dataclass config (src/memory_system/config.py) and its env
names (DEEPSEEK_*, SILICONFLOW_*, ...) stay untouched until the legacy
package is removed in implementation plan step 4.
"""

from typing import Optional

from pydantic import BaseModel, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMConfig(BaseSettings):
    """Single OpenAI-compatible provider; no fallback channel (6.1 / 8.5).

    extra_body is the vendor-parameter escape hatch — user policy such as
    disabling thinking on reasoning models
    (``{"thinking": {"type": "disabled"}}``), passed through verbatim and
    never interpreted here.
    """

    model_config = SettingsConfigDict(
        env_prefix="LLM_", env_file=".env", env_file_encoding="utf-8",
        extra="ignore",
    )

    base_url: str
    api_key: SecretStr
    model: str
    max_retries: int = 3
    base_delay: float = 0.5
    max_delay: float = 8.0
    max_retry_after: float = 60.0
    extra_body: Optional[dict] = None
    # per-million-token price table for cost estimation (6.5); repo JSON +
    # override lands with the adapter layer
    pricing: Optional[dict[str, float]] = None


class EmbeddingConfig(BaseSettings):
    """Embedding provider (OpenAI-compatible) — single provider, like LLM."""

    model_config = SettingsConfigDict(
        env_prefix="EMBEDDING_", env_file=".env", env_file_encoding="utf-8",
        extra="ignore",
    )

    api_key: SecretStr
    base_url: str = "https://api.siliconflow.cn/v1"
    model: str = "Qwen/Qwen3-Embedding-4B"
    dim: int = 2560
    max_retries: int = 3
    base_delay: float = 0.5

    @field_validator("dim")
    @classmethod
    def _dim_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("embedding dim must be positive")
        return v


class StoreConfig(BaseSettings):
    """Vector store connection (provider-neutral; Milvus adapter consumes)."""

    model_config = SettingsConfigDict(
        env_prefix="STORE_", env_file=".env", env_file_encoding="utf-8",
        extra="ignore",
    )

    uri: str
    collection_name: str = "memories"
    groups_collection_name: str = "groups"
    connect_timeout: float = 30.0
    connect_retries: int = 5
    # pymilvus has no native async; the adapter bridges with a thread pool
    # whose size is explicit here instead of a hidden default (#7)
    thread_pool_size: int = 8


class RetrievalConfig(BaseSettings):
    """Retrieval behavior knobs (all configurable, no code changes needed)."""

    model_config = SettingsConfigDict(
        env_prefix="RETRIEVAL_", env_file=".env", env_file_encoding="utf-8",
        extra="ignore",
    )

    k_episodic: int = 5
    k_semantic: int = 5
    use_all_semantic: bool = True
    narrative_similarity_threshold: float = 0.8


class LangfuseConfig(BaseSettings):
    """Telemetry backend settings; disabled by default (Null telemetry).

    Note the legacy enable flag was LANGFUSE_TRACING_ENABLED; the new one
    is LANGFUSE_ENABLED (and the default is off).
    """

    model_config = SettingsConfigDict(
        env_prefix="LANGFUSE_", env_file=".env", env_file_encoding="utf-8",
        extra="ignore",
    )

    enabled: bool = False
    secret_key: Optional[SecretStr] = None
    public_key: Optional[SecretStr] = None
    host: str = "https://cloud.langfuse.com"


class MemoryConfig(BaseModel):
    """Root configuration aggregating all sub-configs.

    Defaults are env-backed: MemoryConfig() reads each section from its
    prefixed env vars (and .env). Sections can also be passed explicitly:
    MemoryConfig(llm=LLMConfig(base_url=..., api_key=..., model=...), ...).

    Env example:
        LLM_BASE_URL=https://api.minimaxi.com/v1
        LLM_API_KEY=...
        LLM_MODEL=MiniMax-M3
        LLM_EXTRA_BODY={"thinking": {"type": "disabled"}}
        EMBEDDING_API_KEY=...
        STORE_URI=http://localhost:19530
    """

    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    store: StoreConfig = Field(default_factory=StoreConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    langfuse: LangfuseConfig = Field(default_factory=LangfuseConfig)
