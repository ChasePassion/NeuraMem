"""Unit tests for the migrated benchmark pipeline (step 5, no LLM/Milvus)."""

import json

import pytest

from neuramem_benchmark.llm_config import apply_minimax_primary
from neuramem_benchmark.locomo import count_samples, load_locomo_qa_list, load_locomo_samples
from neuramem_benchmark.metrics import (
    evidence_recall,
    evidence_recall_detail,
    resolve_evidence_texts,
)

DATA = "data/locomo10.json"


class TestLoaders:
    def test_real_dataset_loads(self):
        samples = load_locomo_samples(DATA)
        assert len(samples) == 10
        assert count_samples(DATA) == 10
        assert samples[0]["user_id"] == "sample_0"

    def test_qa_list_excludes_category_5(self):
        qa_list = load_locomo_qa_list(DATA)
        assert qa_list
        assert all(q["category"] != "5" for q in qa_list)
        assert len({q["question_id"] for q in qa_list}) == len(qa_list)

    def test_sample_filter(self):
        qa_list = load_locomo_qa_list(DATA, sample_idx=3)
        assert qa_list and all(q["sample_index"] == 3 for q in qa_list)


class TestEvidenceMetrics:
    def _synthetic_sample(self):
        return {
            "conversation": {
                "session_1": [
                    {"speaker": "Alice", "text": "I adopted a golden retriever puppy last month."},
                    {"speaker": "Bob", "text": "That's wonderful news!"},
                ],
                "session_2": [
                    {"speaker": "Alice", "text": "short"},
                ],
            }
        }

    def test_pointer_resolution_openviking_rule(self):
        texts = resolve_evidence_texts(self._synthetic_sample(), ["D1:1"])
        # 1-based: D1:1 -> session_1[0], rendered speaker-first
        assert texts == ["Alice: I adopted a golden retriever puppy last month."]

    def test_out_of_range_and_short_skipped(self):
        sample = self._synthetic_sample()
        assert resolve_evidence_texts(sample, ["D1:99"]) == []
        assert resolve_evidence_texts(sample, ["D2:1"]) == []  # too short
        assert resolve_evidence_texts(sample, ["garbage"]) == []

    def test_recall_hit_miss_none(self):
        retrieved = ["Alice: I adopted a golden retriever puppy last month."]
        assert evidence_recall(retrieved, ["Alice: I adopted a golden retriever puppy last month."]) is True
        assert evidence_recall(["Bob: unrelated memory"], ["Alice: I adopted a golden retriever puppy."]) is False
        assert evidence_recall(retrieved, []) is None

    def test_detail_flags_per_evidence(self):
        retrieved = ["Alice: I adopted a golden retriever puppy."]
        evidence = ["Alice: I adopted a golden retriever puppy.", "Bob: unrelated fact."]
        assert evidence_recall_detail(retrieved, evidence) == [True, False]
        assert evidence_recall_detail([], evidence) == [False, False]
        assert evidence_recall_detail(retrieved, []) is None

    def test_recall_consistent_with_detail(self):
        retrieved = ["Bob: unrelated memory"]
        evidence = ["Alice: I adopted a golden retriever puppy."]
        detail = evidence_recall_detail(retrieved, evidence)
        assert evidence_recall(retrieved, evidence) is any(detail)

    def test_real_dataset_evidence_resolves(self):
        """Most real evidence pointers must resolve to utterance texts."""
        sample = load_locomo_samples(DATA, sample_idx=0)[0]
        qa = sample.get("qa", [])
        resolved = total = 0
        for item in qa:
            for _ in item.get("evidence", []) or []:
                total += 1
        texts = []
        for item in qa:
            texts.extend(resolve_evidence_texts(sample, item.get("evidence", [])))
        resolved = len(texts)
        if total:
            assert resolved / total > 0.5, (
                f"only {resolved}/{total} evidence pointers resolved — "
                "pointer rule may not match this dataset build"
            )


class TestLLMConfig:
    def test_minimax_applied_with_key(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "k")
        from neuramem.config import MemoryConfig, LLMConfig, EmbeddingConfig, StoreConfig

        config = MemoryConfig(
            llm=LLMConfig(_env_file=None, base_url="x", api_key="y", model="z"),
            embedding=EmbeddingConfig(_env_file=None, api_key="e"),
            store=StoreConfig(_env_file=None, uri="u"),
        )
        assert apply_minimax_primary(config) is True
        assert config.llm.model == "MiniMax-M3"
        assert config.llm.max_retries == 10  # W3 profile
        assert config.llm.extra_body == {"thinking": {"type": "disabled"}}

    def test_minimax_skipped_without_key(self, monkeypatch):
        monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
        from neuramem.config import MemoryConfig, LLMConfig, EmbeddingConfig, StoreConfig

        config = MemoryConfig(
            llm=LLMConfig(_env_file=None, base_url="x", api_key="y", model="z"),
            embedding=EmbeddingConfig(_env_file=None, api_key="e"),
            store=StoreConfig(_env_file=None, uri="u"),
        )
        assert apply_minimax_primary(config) is False
        assert config.llm.model == "z"


class TestManifestGate:
    def test_missing_manifest_blocks_eval(self, tmp_path):
        from neuramem_benchmark.runner import _check_manifests

        qa_list = [{"sample_index": 4}]
        with pytest.raises(SystemExit, match="manifest"):
            _check_manifests(qa_list, str(tmp_path))

    def test_present_manifest_passes(self, tmp_path):
        from neuramem_benchmark.runner import _check_manifests

        (tmp_path / "ingest_manifest_4.json").write_text("{}", encoding="utf-8")
        _check_manifests([{"sample_index": 4}], str(tmp_path))  # no raise


class TestTraceHelpers:
    def test_memory_dicts_shape_and_no_vector(self):
        from neuramem_benchmark.runner import _memory_dicts
        from neuramem.core.models import MemoryRecord

        record = MemoryRecord(
            id=7, user_id="u", memory_type="episodic", ts=1, chat_id="c",
            text="hello", group_id=3, vector=[0.1, 0.2],
        )
        assert _memory_dicts([record]) == [
            {
                "id": 7,
                "memory_type": "episodic",
                "text": "hello",
                "ts": 1,
                "chat_id": "c",
                "group_id": 3,
            }
        ]

    def test_parse_evidence_pointers(self):
        from neuramem_benchmark.runner import _parse_evidence_pointers

        assert _parse_evidence_pointers('["D1:9", "D1:11"]') == ["D1:9", "D1:11"]
        assert _parse_evidence_pointers("") == []
        assert _parse_evidence_pointers("not json") == []
        assert _parse_evidence_pointers('{"a": 1}') == []
