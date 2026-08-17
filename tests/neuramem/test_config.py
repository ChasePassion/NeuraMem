"""Unit tests for the layered pydantic-settings configuration (step 1).

Sub-configs read their own prefixed env vars; these tests clear the
relevant variables first so results do not depend on the developer's
environment (the repo .env only carries legacy-prefixed vars, which the
new schema ignores).
"""

import pytest
from pydantic import ValidationError

from neuramem.config import (
    EmbeddingConfig,
    LLMConfig,
    MemoryConfig,
    StoreConfig,
)

_ALL_NEW_ENV_VARS = [
    "LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL", "LLM_EXTRA_BODY",
    "EMBEDDING_API_KEY", "EMBEDDING_DIM", "STORE_URI", "RETRIEVAL_K_EPISODIC",
]


@pytest.fixture(autouse=True)
def _clean_new_env(monkeypatch):
    for var in _ALL_NEW_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


class TestMemoryConfig:
    def test_missing_required_sections_raises(self):
        """Misconfiguration fails at construction, not at first use (#8)."""
        with pytest.raises(ValidationError):
            MemoryConfig()

    def test_env_prefix_binding(self, monkeypatch):
        """LLM_BASE_URL etc. bind into the llm sub-config."""
        monkeypatch.setenv("LLM_BASE_URL", "https://api.deepseek.com")
        monkeypatch.setenv("LLM_API_KEY", "env-key")
        monkeypatch.setenv("LLM_MODEL", "deepseek-chat")
        monkeypatch.setenv("EMBEDDING_API_KEY", "embed-key")
        monkeypatch.setenv("STORE_URI", "http://milvus:19530")
        monkeypatch.setenv("RETRIEVAL_K_EPISODIC", "7")

        config = MemoryConfig()

        assert config.llm.base_url == "https://api.deepseek.com"
        assert config.llm.api_key.get_secret_value() == "env-key"
        assert config.llm.model == "deepseek-chat"
        assert config.embedding.api_key.get_secret_value() == "embed-key"
        assert config.store.uri == "http://milvus:19530"
        assert config.retrieval.k_episodic == 7

    def test_extra_body_env_parses_as_json_dict(self, monkeypatch):
        """LLM_EXTRA_BODY carries vendor params (W3 thinking-off setup)."""
        monkeypatch.setenv("LLM_BASE_URL", "https://x")
        monkeypatch.setenv("LLM_API_KEY", "k")
        monkeypatch.setenv("LLM_MODEL", "m")
        monkeypatch.setenv("EMBEDDING_API_KEY", "e")
        monkeypatch.setenv("STORE_URI", "u")
        monkeypatch.setenv(
            "LLM_EXTRA_BODY", '{"thinking": {"type": "disabled"}}'
        )

        config = MemoryConfig()

        assert config.llm.extra_body == {"thinking": {"type": "disabled"}}

    def test_legacy_env_vars_are_ignored(self, monkeypatch):
        """Legacy vars (DEEPSEEK_*, MILVUS_URL...) coexist with the new schema."""
        monkeypatch.setenv("DEEPSEEK_API_KEY", "legacy")
        monkeypatch.setenv("MILVUS_URL", "legacy")
        monkeypatch.setenv("LLM_BASE_URL", "https://x")
        monkeypatch.setenv("LLM_API_KEY", "k")
        monkeypatch.setenv("LLM_MODEL", "m")
        monkeypatch.setenv("EMBEDDING_API_KEY", "e")
        monkeypatch.setenv("STORE_URI", "u")

        config = MemoryConfig()

        assert config.llm.api_key.get_secret_value() == "k"

    def test_langfuse_disabled_by_default(self):
        """Null telemetry is the default (architecture #5)."""
        with pytest.raises(ValidationError):
            # llm section missing here on purpose; use explicit sections
            MemoryConfig()
        config = MemoryConfig(
            llm=LLMConfig(base_url="https://x", api_key="k", model="m"),
            embedding=EmbeddingConfig(api_key="e"),
            store=StoreConfig(uri="u"),
        )
        assert config.langfuse.enabled is False


class TestSubConfigValidation:
    def test_embedding_dim_must_be_positive(self):
        with pytest.raises(ValidationError):
            EmbeddingConfig(api_key="k", dim=0)

    def test_llm_config_requires_provider_fields(self):
        with pytest.raises(ValidationError):
            LLMConfig(base_url="https://x")  # api_key and model missing

    def test_sub_configs_constructible_directly_for_tests(self):
        # _env_file=None isolates from the developer's .env (which may carry
        # legacy LLM_EXTRA_BODY / LLM_MAX_RETRIES from the W3 setup)
        llm = LLMConfig(base_url="b", api_key="k", model="m", _env_file=None)
        assert llm.extra_body is None
        assert llm.max_retries == 3
