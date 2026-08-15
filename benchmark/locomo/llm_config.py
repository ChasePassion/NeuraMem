"""Benchmark LLM provider configuration.

The benchmark runs entirely on a single OpenAI-compatible LLM provider,
aligned with the target architecture (architecture_target.md #8): no
fallback chain; the provider is selected via base_url + api_key + model.

When MINIMAX_API_KEY is set, every LLM call in the benchmark (memory
management, semantic consolidation, usage judging, answering, grading)
uses MiniMax-M3. Otherwise the .env primary (DEEPSEEK_*) is used as-is.
"""

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

MINIMAX_DEFAULT_BASE_URL = "https://api.minimaxi.com/v1"
MINIMAX_DEFAULT_MODEL = "MiniMax-M3"


def apply_minimax_primary(config: Any) -> bool:
    """Override the Memory primary LLM with MiniMax and drop the fallback.

    Returns True when MiniMax is configured and applied.
    """
    api_key = os.getenv("MINIMAX_API_KEY")
    if not api_key:
        return False
    config.llm_primary_api_key = api_key
    config.llm_primary_base_url = os.getenv("MINIMAX_BASE_URL", MINIMAX_DEFAULT_BASE_URL)
    config.llm_primary_model = os.getenv("MINIMAX_MODEL", MINIMAX_DEFAULT_MODEL)
    # Single-provider mode: no fallback chain (target architecture #8)
    config.llm_fallback_api_key = None
    config.llm_fallback_base_url = None
    config.llm_fallback_model = None
    logger.info(
        f"Benchmark LLM: MiniMax single provider "
        f"({config.llm_primary_model} @ {config.llm_primary_base_url})"
    )
    return True
