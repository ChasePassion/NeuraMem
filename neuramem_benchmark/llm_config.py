"""Benchmark LLM configuration (single provider, W3 profile).

build_benchmark_config() assembles a full MemoryConfig from the legacy
env names (DEEPSEEK_*/SILICONFLOW_*/MILVUS_URL — the repo .env keeps
working), then apply_minimax_primary() overrides the LLM section with
MiniMax when MINIMAX_API_KEY is set, using the W3-stable retry profile
(10 attempts, base 1.0s, cap 30s — RUN_RECORD §8) and the thinking-off
escape hatch. W4 must lock the same base_url as W3 (api.minimaxi.com)
for comparability; MINIMAX_BASE_URL / MINIMAX_MODEL allow overrides
(e.g. the domestic api.minimax.chat endpoint for local runs).
"""

import logging
import os

from neuramem.config import (
    EmbeddingConfig,
    LLMConfig,
    MemoryConfig,
    StoreConfig,
)

logger = logging.getLogger(__name__)

MINIMAX_DEFAULT_BASE_URL = "https://api.minimaxi.com/v1"
MINIMAX_DEFAULT_MODEL = "MiniMax-M3"

DEFAULT_MILVUS_URI = "http://117.72.161.187:19530"


def apply_minimax_primary(config: MemoryConfig) -> bool:
    """Override the LLM section with MiniMax (W3 profile). Returns applied?"""
    api_key = os.getenv("MINIMAX_API_KEY")
    if not api_key:
        return False
    config.llm = LLMConfig(
        _env_file=None,
        base_url=os.getenv("MINIMAX_BASE_URL", MINIMAX_DEFAULT_BASE_URL),
        api_key=api_key,
        model=os.getenv("MINIMAX_MODEL", MINIMAX_DEFAULT_MODEL),
        max_retries=10,
        base_delay=1.0,
        max_delay=30.0,
        extra_body={"thinking": {"type": "disabled"}},
    )
    logger.info(
        "MiniMax primary applied: %s @ %s (W3 retry profile)",
        config.llm.model, config.llm.base_url,
    )
    return True


def build_benchmark_config(milvus_uri: str = None) -> MemoryConfig:
    """Full benchmark config from legacy env names + MiniMax override."""
    config = MemoryConfig(
        llm=LLMConfig(
            _env_file=None,
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        ),
        embedding=EmbeddingConfig(
            _env_file=None, api_key=os.getenv("SILICONFLOW_API_KEY", "")
        ),
        store=StoreConfig(
            _env_file=None,
            uri=milvus_uri or os.getenv("MILVUS_URL", DEFAULT_MILVUS_URI),
        ),
    )
    apply_minimax_primary(config)
    return config
