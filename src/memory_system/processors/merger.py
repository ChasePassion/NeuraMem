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
        
        Args:
            memory_a: First memory record
            memory_b: Second memory record
            
        Returns:
            Default merged record
        """
        # Use earliest time as canonical
        time_a = memory_a.get("metadata", {}).get("time", "")
        time_b = memory_b.get("metadata", {}).get("time", "")
        canonical_time = min(time_a, time_b) if time_a and time_b else (time_a or time_b)
        
        return {
            "user_id": memory_a.get("user_id", ""),
            "memory_type": "episodic",
            "chat_id": memory_a.get("chat_id", ""),
            "who": memory_a.get("who", "user"),
            "text": f"{memory_a.get('text', '')} {memory_b.get('text', '')}".strip(),
            "metadata": {
                "context": memory_a.get("metadata", {}).get("context", ""),
                "thing": f"{memory_a.get('metadata', {}).get('thing', '')} {memory_b.get('metadata', {}).get('thing', '')}".strip(),
                "time": canonical_time,
                "chatid": memory_a.get("chat_id", ""),
                "who": memory_a.get("who", "user"),
                "source_chat_ids": [
                    memory_a.get("chat_id", ""),
                    memory_b.get("chat_id", "")
                ],
                "merged_from_ids": [
                    memory_a.get("id"),
                    memory_b.get("id")
                ]
            }
        }
    
    def _ensure_merge_fields(
        self,
        merged: Dict[str, Any],
        memory_a: Dict[str, Any],
        memory_b: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Ensure merged record has all required fields.
        
        Args:
            merged: Merged record from LLM
            memory_a: First original memory
            memory_b: Second original memory
            
        Returns:
            Merged record with all required fields
        """
        # Ensure top-level fields
        if "user_id" not in merged:
            merged["user_id"] = memory_a.get("user_id", "")
        if "memory_type" not in merged:
            merged["memory_type"] = "episodic"
        if "chat_id" not in merged:
            merged["chat_id"] = memory_a.get("chat_id", "")
        if "who" not in merged:
            merged["who"] = memory_a.get("who", "user")
        if "text" not in merged:
            merged["text"] = ""
        
        # Ensure metadata exists
        if "metadata" not in merged:
            merged["metadata"] = {}
        
        metadata = merged["metadata"]
        
        # Ensure metadata fields
        if "context" not in metadata:
            metadata["context"] = ""
        if "thing" not in metadata:
            metadata["thing"] = ""
        if "chatid" not in metadata:
            metadata["chatid"] = merged.get("chat_id", "")
        if "who" not in metadata:
            metadata["who"] = merged.get("who", "user")
        
        # Use earliest time as canonical (Requirements 9.4)
        time_a = memory_a.get("metadata", {}).get("time", "")
        time_b = memory_b.get("metadata", {}).get("time", "")
        if "time" not in metadata or not metadata["time"]:
            metadata["time"] = min(time_a, time_b) if time_a and time_b else (time_a or time_b)
        else:
            # Ensure we use the earliest time even if LLM provided one
            if time_a and time_b:
                earliest = min(time_a, time_b)
                metadata["time"] = earliest
        
        # Ensure source tracking (Requirements 9.5)
        if "source_chat_ids" not in metadata:
            metadata["source_chat_ids"] = []
        
        chat_id_a = memory_a.get("chat_id", "")
        chat_id_b = memory_b.get("chat_id", "")
        
        if chat_id_a and chat_id_a not in metadata["source_chat_ids"]:
            metadata["source_chat_ids"].append(chat_id_a)
        if chat_id_b and chat_id_b not in metadata["source_chat_ids"]:
            metadata["source_chat_ids"].append(chat_id_b)
        
        # Ensure merged_from_ids
        if "merged_from_ids" not in metadata:
            metadata["merged_from_ids"] = []
        
        id_a = memory_a.get("id")
        id_b = memory_b.get("id")
        
        if id_a is not None and id_a not in metadata["merged_from_ids"]:
            metadata["merged_from_ids"].append(id_a)
        if id_b is not None and id_b not in metadata["merged_from_ids"]:
            metadata["merged_from_ids"].append(id_b)
        
        return merged
