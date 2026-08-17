"""Usage judge pipeline: which episodic memories were actually used.

Migrated from processors/memory_usage_judge.py with the id-protocol
change (architecture_target.md #14 / ch. 11): candidates enter the prompt
with ids and the judge returns ``used_episodic_memory_ids``. This
replaces the legacy exact-text matching that silently dropped
assignments whenever the LLM paraphrased a memory text.

Conservative fallback preserved: any failure returns [] — the write-back
channel must never break the consumer's answer path.
"""

import json
import logging
from neuramem.core.models import MemoryRecord
from neuramem.core.ports import LLM
from neuramem.prompts import MEMORY_RELEVANCE_FILTER_PROMPT

logger = logging.getLogger(__name__)


class UsageJudge:
    """Judges which retrieved episodic memories an answer actually used."""

    def __init__(self, llm: LLM):
        self._llm = llm

    async def judge_used_memories(
        self,
        candidates: list[MemoryRecord],
        last_user: str,
        last_assistant: str,
    ) -> list[int]:
        if not candidates:
            return []
        try:
            input_data = {
                "episodic_memories": [
                    {"id": m.id, "text": m.text} for m in candidates
                ],
                "last_user": last_user,
                "last_assistant": last_assistant,
            }
            result = await self._llm.complete_json(
                system_prompt=MEMORY_RELEVANCE_FILTER_PROMPT,
                user_message=json.dumps(input_data, ensure_ascii=False),
                default={"used_episodic_memory_ids": []},
                call_label="usage_judge",
            )
            if not result.success:
                return []
            known_ids = {m.id for m in candidates}
            used = [
                int(i)
                for i in result.parsed_data.get("used_episodic_memory_ids", [])
                if int(i) in known_ids
            ]
            logger.info(
                "usage judgment: %d/%d episodic memories were used",
                len(used), len(candidates),
            )
            return used
        except Exception as e:  # noqa: BLE001 - never break the answer path
            logger.warning("Failed to judge memory usage: %s", e)
            return []
