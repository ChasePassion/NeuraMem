"""Episodic Memory Write Decider processor.

Determines whether conversation content should be stored as episodic memory.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any

from prompts import EPISODIC_MEMORY_WRITE_FILTER
from ..clients.llm import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class MemoryRecord:
    """A single episodic memory record to be written.
    
    In v2 schema, all information is stored in the text field.
    The text field should include: time, where (if applicable), who, thing, reason.
    """
    text: str


@dataclass
class WriteDecision:
    """Decision result from EpisodicWriteDecider.
    
    Attributes:
        write_episodic: Whether to write episodic memory
        records: List of memory records to write (empty if write_episodic is False)
    """
    write_episodic: bool
    records: List[MemoryRecord] = field(default_factory=list)


class EpisodicWriteDecider:
    """Episodic Memory Write Filter processor.
    
    Decides whether a conversation snippet should be written as episodic memory
    and produces structured content for those memories.
    
    Uses EPISODIC_MEMORY_WRITE_FILTER prompt to call LLM for decision.
    """
    
    def __init__(self, llm_client: LLMClient):
        """Initialize the write decider.
        
        Args:
            llm_client: LLM client for making decisions
        """
        self._llm = llm_client
        self._prompt = EPISODIC_MEMORY_WRITE_FILTER

    def decide(self, chat_id: str, turns: List[Dict[str, str]]) -> WriteDecision:
        """Decide whether conversation turns should be stored as episodic memory.
        
        Args:
            chat_id: Conversation/thread identifier
            turns: List of conversation turns, each with 'role' and 'content' keys
            
        Returns:
            WriteDecision with write_episodic flag and records to write
        """
        # Prepare input for LLM
        input_data = {
            "chat_id": chat_id,
            "turns": turns
        }
        user_message = json.dumps(input_data, ensure_ascii=False)
        
        # Default response for fallback
        default_response = {
            "write_episodic": False,
            "records": []
        }
        
        # Call LLM for decision
        response = self._llm.chat_json(
            system_prompt=self._prompt,
            user_message=user_message,
            default=default_response
        )
        
        # Parse response
        write_episodic = response.get("write_episodic", False)
        raw_records = response.get("records", [])
        
        # Convert raw records to MemoryRecord objects (v2 schema: text only)
        records = []
        for raw in raw_records:
            if isinstance(raw, dict):
                # In v2 schema, all information is in the text field
                text = raw.get("text", "")
                if text:
                    record = MemoryRecord(text=text)
                    records.append(record)
            elif isinstance(raw, str) and raw:
                # Support simple string records
                record = MemoryRecord(text=raw)
                records.append(record)
        
        logger.info(
            f"WriteDecider decision for chat_id={chat_id}: "
            f"write_episodic={write_episodic}, records_count={len(records)}"
        )
        
        return WriteDecision(
            write_episodic=write_episodic,
            records=records
        )
