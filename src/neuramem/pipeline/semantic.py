"""Semantic pipeline: pattern merging + conflict elimination (#20).

Migrated from processors/semantic_writer.py with one behavior change:
existing semantic memories enter the prompt WITH ids, and the writer may
return ``retired_semantic_ids`` — semantic memories contradicted by newer
evidence. The facade flips their ``retired`` flag (permanent retrieval
filter; physical deletion only via reset — the minimal conflict
elimination agreed in the plan).

Failures keep the legacy conservative fallback: no facts written, nothing
retired (#22 decision — semantic is not on the answer-critical path).
"""

import json
import logging
from dataclasses import dataclass, field

from neuramem.core.models import MemoryRecord
from neuramem.core.ports import LLM
from neuramem.prompts import SEMANTIC_MEMORY_WRITER_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class SemanticExtraction:
    """Result of one consolidation pass."""

    write_semantic: bool = False
    facts: list[str] = field(default_factory=list)
    retire_ids: list[int] = field(default_factory=list)


class SemanticWriter:
    """Extracts stable facts from episodic batches; flags conflicts."""

    def __init__(self, llm: LLM):
        self._llm = llm

    async def extract(
        self,
        episodic: list[MemoryRecord],
        existing_semantic: list[MemoryRecord],
    ) -> SemanticExtraction:
        consolidation_data = {
            "episodic_texts": [m.text for m in episodic],
            "existing_semantic": [
                {"id": m.id, "text": m.text} for m in existing_semantic
            ],
        }
        result = await self._llm.complete_json(
            system_prompt=SEMANTIC_MEMORY_WRITER_PROMPT,
            user_message=json.dumps(consolidation_data, ensure_ascii=False),
            default={
                "write_semantic": False,
                "facts": [],
                "retired_semantic_ids": [],
            },
            call_label="consolidate",
        )
        if not result.success:
            logger.warning(
                "semantic extraction parse failure; conservative no-op"
            )
            return SemanticExtraction()

        parsed = result.parsed_data
        known_ids = {m.id for m in existing_semantic}
        raw_retire = parsed.get("retired_semantic_ids", []) or []
        if not isinstance(raw_retire, list):
            raw_retire = []
        retire_ids = []
        for item in raw_retire:
            try:
                candidate = int(item)
            except (TypeError, ValueError):
                continue  # garbage from the model must not crash consolidation
            if candidate in known_ids:
                retire_ids.append(candidate)
        facts = [str(f) for f in parsed.get("facts", []) if f]
        write_semantic = bool(parsed.get("write_semantic", False))

        logger.info(
            "semantic extraction: episodic=%d existing=%d write=%s facts=%d retire=%d",
            len(episodic), len(existing_semantic), write_semantic,
            len(facts), len(retire_ids),
        )
        return SemanticExtraction(
            write_semantic=write_semantic, facts=facts, retire_ids=retire_ids
        )
