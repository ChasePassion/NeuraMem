"""Episodic Memory Separator processor.

Rewrites two similar but distinct episodic memories to make their differences explicit.
"""

import json
import logging
from typing import Dict, Any, Tuple

from prompts import EPISODIC_MEMORY_SEPARATOR_PROMPT
from ..clients.llm import LLMClient

logger = logging.getLogger(__name__)


class EpisodicSeparator:
    """Episodic Memory Separator processor.
    
    Takes two episodic memory records that are semantically similar but describe
    different events, and rewrites them to make their differences explicit.
    
    Uses EPISODIC_MEMORY_SEPARATOR_PROMPT to call LLM for separation.
    """
    
    def __init__(self, llm_client: LLMClient):
        """Initialize the separator.
        
        Args:
            llm_client: LLM client for separation decisions
        """
        self._llm = llm_client
        self._prompt = EPISODIC_MEMORY_SEPARATOR_PROMPT
    
    def separate(
        self, 
        memory_a: Dict[str, Any], 
        memory_b: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Rewrite two similar memories to make their differences explicit.
        
        Args:
            memory_a: First episodic memory record
            memory_b: Second episodic memory record
            
        Returns:
            Tuple of (updated_memory_a, updated_memory_b) with rewritten fields
        """
        # Prepare input for LLM
        input_data = {
            "A": memory_a,
            "B": memory_b
        }
        user_message = json.dumps(input_data, ensure_ascii=False)
        
        # Default response - return originals unchanged
        default_response = {
            "memA": self._extract_updatable_fields(memory_a),
            "memB": self._extract_updatable_fields(memory_b)
        }
        
        # Call LLM for separation
        response = self._llm.chat_json(
            system_prompt=self._prompt,
            user_message=user_message,
            default=default_response
        )
        
        # Apply updates to original memories
        updated_a = self._apply_updates(memory_a, response.get("memA", {}))
        updated_b = self._apply_updates(memory_b, response.get("memB", {}))
        
        logger.info(
            f"EpisodicSeparator separated memories {memory_a.get('id')} and {memory_b.get('id')}"
        )
        
        return updated_a, updated_b
    
    def _extract_updatable_fields(self, memory: Dict[str, Any]) -> Dict[str, Any]:
        """Extract fields that can be updated during separation.
        
        Args:
            memory: Original memory record
            
        Returns:
            Dict with updatable fields
        """
        metadata = memory.get("metadata", {})
        return {
            "chat_id": memory.get("chat_id", ""),
            "who": memory.get("who", "user"),
            "text": memory.get("text", ""),
            "metadata": {
                "context": metadata.get("context", ""),
                "thing": metadata.get("thing", ""),
                "time": metadata.get("time", ""),
                "chatid": metadata.get("chatid", ""),
                "who": metadata.get("who", "user")
            }
        }
    
    def _apply_updates(
        self, 
        original: Dict[str, Any], 
        updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply LLM updates to original memory while preserving immutable fields.
        
        Args:
            original: Original memory record
            updates: Updates from LLM
            
        Returns:
            Updated memory record
        """
        # Start with a copy of original
        result = original.copy()
        
        # Update text if provided
        if "text" in updates and updates["text"]:
            result["text"] = updates["text"]
        
        # Update metadata fields that are allowed to change
        if "metadata" in updates:
            if "metadata" not in result:
                result["metadata"] = {}
            
            update_meta = updates["metadata"]
            
            # Only update context and thing (text-related fields)
            if "context" in update_meta and update_meta["context"]:
                result["metadata"]["context"] = update_meta["context"]
            if "thing" in update_meta and update_meta["thing"]:
                result["metadata"]["thing"] = update_meta["thing"]
            
            # Preserve immutable fields from original
            original_meta = original.get("metadata", {})
            result["metadata"]["time"] = original_meta.get("time", "")
            result["metadata"]["chatid"] = original_meta.get("chatid", "")
            result["metadata"]["who"] = original_meta.get("who", "user")
        
        return result
