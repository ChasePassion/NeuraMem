"""Episodic Memory Merger processor.

Merges two highly similar episodic memories into one consolidated record.
"""

import json
import logging
from typing import Dict, Any

from prompts import EPISODIC_MEMORY_MERGER_PROMPT
from ..clients.llm import LLMClient

logger = logging.getLogger(__name__)


class EpisodicMerger:
    """Episodic Memory Merger processor.
    
    Merges two highly similar episodic memory records into one consolidated
    record without losing any factual information.
    
    Uses EPISODIC_MEMORY_MERGER_PROMPT to call LLM for merging.
    """
    
    def __init__(self, llm_client: LLMClient):
        """Initialize the merger.
        
        Args:
            llm_client: LLM client for merging decisions
        """
        self._llm = llm_client
        self._prompt = EPISODIC_MEMORY_MERGER_PROMPT
    
    def merge(self, memory_a: Dict[str, Any], memory_b: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two episodic memories into one consolidated record.
        
        Args:
            memory_a: First episodic memory record
            memory_b: Second episodic memory record
            
        Returns:
            Merged memory record with source_chat_ids and merged_from_ids
        """
        # Prepare input for LLM
        input_data = {
            "A": memory_a,
            "B": memory_b
        }
        user_message = json.dumps(input_data, ensure_ascii=False)
        
        # Default response - use memory_a as base
        default_response = self._create_default_merge(memory_a, memory_b)
        
        # Call LLM for merging
        response = self._llm.chat_json(
            system_prompt=self._prompt,
            user_message=user_message,
            default=default_response
        )
        
        # Ensure required fields are present
        merged = self._ensure_merge_fields(response, memory_a, memory_b)
        
        logger.info(
            f"EpisodicMerger merged memories {memory_a.get('id')} and {memory_b.get('id')}"
        )
        
        return merged

    def _create_default_merge(
        self, 
        memory_a: Dict[str, Any], 
        memory_b: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a default merged record if LLM fails.
        
        In v2 schema, all information is stored in the text field.
        
        Args:
            memory_a: First memory record
            memory_b: Second memory record
            
        Returns:
            Default merged record (v2 schema)
        """
        # Combine text fields
        text_a = memory_a.get("text", "")
        text_b = memory_b.get("text", "")
        merged_text = f"{text_a} {text_b}".strip()
        
        return {
            "user_id": memory_a.get("user_id", ""),
            "memory_type": "episodic",
            "chat_id": memory_a.get("chat_id", ""),
            "text": merged_text,
        }
    
    def _ensure_merge_fields(
        self,
        merged: Dict[str, Any],
        memory_a: Dict[str, Any],
        memory_b: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Ensure merged record has all required fields.
        
        In v2 schema, only core fields are needed.
        
        Args:
            merged: Merged record from LLM
            memory_a: First original memory
            memory_b: Second original memory
            
        Returns:
            Merged record with all required fields (v2 schema)
        """
        # Ensure top-level fields (v2 schema)
        if "user_id" not in merged:
            merged["user_id"] = memory_a.get("user_id", "")
        if "memory_type" not in merged:
            merged["memory_type"] = "episodic"
        if "chat_id" not in merged:
            merged["chat_id"] = memory_a.get("chat_id", "")
        if "text" not in merged:
            # Combine text fields as fallback
            text_a = memory_a.get("text", "")
            text_b = memory_b.get("text", "")
            merged["text"] = f"{text_a} {text_b}".strip()
        
        return merged
