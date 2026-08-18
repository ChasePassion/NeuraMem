"""Usage judge pipeline: which episodic memories were actually used.

Migrated from processors/memory_usage_judge.py with the id-protocol
change (architecture_target.md #14 / ch. 11): candidates enter the prompt
with ids and the judge returns ``used_episodic_memory_ids``. This
replaces the legacy exact-text matching that silently dropped
assignments whenever the LLM paraphrased a memory text.

Id-output anomalies are surfaced instead of silently dropped (#14
observability): hallucinated ids (not in the candidate set) and
malformed values (non-int entries) are counted and returned alongside
the usable ids — one bad entry no longer discards the whole judgment.

Conservative fallback preserved: any failure returns an empty judgment —
the write-back channel must never break the consumer's answer path.
"""

import json
import logging
from dataclasses import dataclass, field

from neuramem.core.models import MemoryRecord
from neuramem.core.ports import LLM
from neuramem.prompts import MEMORY_RELEVANCE_FILTER_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class UsageJudgment:
    """Judge output plus id-protocol anomaly record."""

    used_ids: list[int] = field(default_factory=list)
    dropped_ids: list[int] = field(default_factory=list)  # ints not in candidates
    malformed_count: int = 0  # entries that were not ints at all

    @property
    def has_anomalies(self) -> bool:
        return bool(self.dropped_ids) or self.malformed_count > 0


class UsageJudge:
    """Judges which retrieved episodic memories an answer actually used."""

    def __init__(self, llm: LLM):
        self._llm = llm

    async def judge_used_memories(
        self,
        candidates: list[MemoryRecord],
        last_user: str,
        last_assistant: str,
    ) -> UsageJudgment:
        if not candidates:
            return UsageJudgment()
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
                return UsageJudgment()
            known_ids = {m.id for m in candidates}
            raw_list = result.parsed_data.get("used_episodic_memory_ids", [])
            if not isinstance(raw_list, list):
                raw_list = []
            used: list[int] = []
            dropped: list[int] = []
            malformed = 0
            for item in raw_list:
                try:
                    used_id = int(item)
                except (TypeError, ValueError):
                    malformed += 1  # garbage entry: skip, keep the rest
                    continue
                if used_id in known_ids:
                    if used_id not in used:
                        used.append(used_id)
                else:
                    dropped.append(used_id)  # hallucinated id
            judgment = UsageJudgment(
                used_ids=used, dropped_ids=dropped, malformed_count=malformed
            )
            if judgment.has_anomalies:
                logger.warning(
                    "usage judge id anomalies: dropped=%s malformed=%d",
                    dropped, malformed,
                )
            logger.info(
                "usage judgment: %d/%d episodic memories were used",
                len(used), len(candidates),
            )
            return judgment
        except Exception as e:  # noqa: BLE001 - never break the answer path
            logger.warning("Failed to judge memory usage: %s", e)
            return UsageJudgment()
