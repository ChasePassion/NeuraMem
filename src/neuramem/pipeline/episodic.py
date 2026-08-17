"""Episodic pipeline: LLM-driven CRUD decisions over episodic memories.

Migrated from processors/memory_manager.py. Behavior preserved:
- full candidate set (the whole user's episodic memories go into the
  decision context — candidate selection was dropped from the plan)
- the MiniMax-style bare-id delete compat ("delete": [1, 2])
- parse failure raises LLMParseError instead of a silent empty plan (#22)
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from neuramem.core.exceptions import LLMParseError
from neuramem.core.models import MemoryRecord
from neuramem.core.ports import LLM
from neuramem.prompts import EPISODIC_MEMORY_MANAGER

logger = logging.getLogger(__name__)


@dataclass
class MemoryOperation:
    """A single planned memory operation."""

    operation_type: str  # "add" | "update" | "delete"
    memory_id: Optional[int] = None
    text: Optional[str] = None
    old_text: Optional[str] = None


@dataclass
class EpisodicPlan:
    """Planned CRUD operations for one conversation turn."""

    operations: list[MemoryOperation] = field(default_factory=list)


class EpisodicManager:
    """Decides add/update/delete over the user's episodic memories."""

    def __init__(self, llm: LLM):
        self._llm = llm

    async def manage_memories(
        self,
        user_text: str,
        assistant_text: str,
        episodic_memories: list[MemoryRecord],
    ) -> EpisodicPlan:
        input_data = {
            "current_turn": {"user": user_text, "assistant": assistant_text},
            "episodic_memories": [
                {"id": m.id, "text": m.text} for m in episodic_memories
            ],
        }
        result = await self._llm.complete_json(
            system_prompt=EPISODIC_MEMORY_MANAGER,
            user_message=json.dumps(input_data, ensure_ascii=False),
            default={"add": [], "update": [], "delete": []},
            call_label="manage",
        )
        if not result.success:
            raise LLMParseError(model=result.model, raw_response=result.raw_response)

        response = result.parsed_data
        operations: list[MemoryOperation] = []
        for add_op in response.get("add", []):
            operations.append(MemoryOperation("add", text=add_op["text"]))
        for update_op in response.get("update", []):
            operations.append(
                MemoryOperation(
                    "update",
                    memory_id=update_op["id"],
                    old_text=update_op.get("old_text"),
                    text=update_op["new_text"],
                )
            )
        for delete_op in response.get("delete", []):
            # MiniMax-M3 may emit bare ids ("delete": [1, 2]) instead of
            # objects ({"id": 1}); accept both shapes
            memory_id = delete_op["id"] if isinstance(delete_op, dict) else delete_op
            operations.append(MemoryOperation("delete", memory_id=memory_id))

        logger.info(
            "episodic plan: adds=%d updates=%d deletes=%d",
            sum(1 for op in operations if op.operation_type == "add"),
            sum(1 for op in operations if op.operation_type == "update"),
            sum(1 for op in operations if op.operation_type == "delete"),
        )
        return EpisodicPlan(operations)
