"""Unit tests for the canonical answer prompt and id-protocol robustness."""

import asyncio
import datetime

from neuramem.core.models import MemoryRecord
from neuramem.core.ports import LLMJsonResult
from neuramem.pipeline.usage_judge import UsageJudge
from neuramem.prompts import build_answer_prompt, extract_final_answer


def _records(n: int, start_id: int = 0) -> list:
    return [
        MemoryRecord(
            id=start_id + i, text=f"memory {start_id + i}", ts=1700000000 + i,
            chat_id="c", user_id="u", memory_type="episodic",
        )
        for i in range(n)
    ]


class TestBuildAnswerPrompt:
    def test_record_bullets_and_limit(self):
        prompt = build_answer_prompt("q?", _records(250), "2023")
        assert prompt.count("\n- memory ") == 200  # ANSWERER_MEMORY_LIMIT
        assert "- memory 0" in prompt and "- memory 199" in prompt
        assert "memory 200" not in prompt

    def test_empty_memories(self):
        prompt = build_answer_prompt("q?", [], "2023")
        assert "(No relevant memories found)" in prompt

    def test_string_and_dict_inputs(self):
        prompt = build_answer_prompt("q?", ["plain fact"], "2023")
        assert "- plain fact" in prompt
        prompt = build_answer_prompt(
            "q?", [{"text": "d", "ts": "2023-05-07T10:00:00"}], "2023"
        )
        expected_date = datetime.datetime(2023, 5, 7).strftime("%A, %B %d, %Y")
        assert f"({expected_date}) d" in prompt

    def test_temporal_anchor_2023_matches_legacy_template(self):
        prompt = build_answer_prompt("q?", ["m"], "2023")
        assert "around 2023. All events occurred in 2022-2024." in prompt
        assert "Never output 2025 or 2026." in prompt

    def test_temporal_anchor_shifts_with_reference_year(self):
        prompt = build_answer_prompt("q?", ["m"], "2026")
        assert "around 2026. All events occurred in 2025-2027." in prompt
        assert "Never output 2028 or 2029." in prompt

    def test_default_reference_is_current_year(self):
        prompt = build_answer_prompt("q?", ["m"])
        assert f"around {datetime.date.today().year}." in prompt


class TestExtractFinalAnswer:
    def test_answer_marker(self):
        assert extract_final_answer("step one\nANSWER: Paris") == "Paris"

    def test_think_block_then_marker(self):
        raw = "<think>reasoning</think>ANSWER: 42"
        assert extract_final_answer(raw) == "42"

    def test_plain_text_passthrough(self):
        assert extract_final_answer("  plain answer  ") == "plain answer"


class TestUsageJudgeIdProtocol:
    def test_malformed_and_hallucinated_ids_are_recorded(self):
        class FakeLLM:
            async def complete_json(self, system_prompt, user_message,
                                    default=None, *, call_label=None):
                # mixed output: valid (1), garbage ("x"), hallucinated (999),
                # duplicate valid (1)
                return LLMJsonResult(
                    parsed_data={"used_episodic_memory_ids": [1, "x", 999, 1]},
                    raw_response="{}", model="fake", success=True,
                )

        judge = UsageJudge(FakeLLM())
        judgment = asyncio.run(
            judge.judge_used_memories(_records(2, start_id=1), "q", "a")
        )
        assert judgment.used_ids == [1]  # deduped, garbage skipped
        assert judgment.dropped_ids == [999]  # hallucinated id recorded
        assert judgment.malformed_count == 1
        assert judgment.has_anomalies

    def test_parse_failure_returns_empty_judgment(self):
        class FakeLLM:
            async def complete_json(self, system_prompt, user_message,
                                    default=None, *, call_label=None):
                return LLMJsonResult(
                    parsed_data={}, raw_response="not json", model="fake",
                    success=False,
                )

        judge = UsageJudge(FakeLLM())
        judgment = asyncio.run(
            judge.judge_used_memories(_records(2), "q", "a")
        )
        assert judgment.used_ids == []
        assert not judgment.has_anomalies
