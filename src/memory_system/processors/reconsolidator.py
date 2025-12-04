"""Episodic Memory Reconsolidator processor.

Updates episodic memories when they are retrieved and mentioned again in conversation.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List

from prompts import EPISODIC_MEMORY_RECONSOLIDATOR_PROMPTS
from ..clients.llm import LLMClient

logger = logging.getLogger(__name__)


class EpisodicReconsolidator:
    """Episodic Memory Reconsolidator processor.
    
    Updates a single episodic memory record when it is retrieved and mentioned
    again in the current conversation, enriching it with new information while
    preserving historical facts.
    
    Uses EPISODIC_MEMORY_RECONSOLIDATOR_PROMPTS to call LLM for reconsolidation.
    """
    
    def __init__(self, llm_client: LLMClient):
        """Initialize the reconsolidator.
        
        Args:
            llm_client: LLM client for reconsolidation decisions
        """
        self._llm = llm_client
        self._prompt = EPISODIC_MEMORY_RECONSOLIDATOR_PROMPTS
    
    def reconsolidate(
        self, 
        old_memory: Dict[str, Any], 
        current_context: str
    ) -> Dict[str, Any]:
        """Update an episodic memory with new context information.
        
        Args:
            old_memory: Existing episodic memory record
            current_context: Text snippet from current dialogue related to this memory
            
        Returns:
            Updated memory record with preserved immutable fields and appended updates
        """
        # Prepare input for LLM
        input_data = {
            "old_memory": old_memory,
            "current_context": current_context
        }
        user_message = json.dumps(input_data, ensure_ascii=False)
        
        # Default response - return original with update appended
        default_response = self._create_default_update(old_memory, current_context)
        
        # Call LLM for reconsolidation
        response = self._llm.chat_json(
            system_prompt=self._prompt,
            user_message=user_message,
            default=default_response
        )
        
        # Apply updates while preserving immutable fields
        updated = self._apply_reconsolidation(old_memory, response, current_context)
        
        logger.info(
            f"EpisodicReconsolidator updated memory {old_memory.get('id')}"
        )
        
        return updated

    def _create_default_update(
        self, 
        old_memory: Dict[str, Any], 
        current_context: str
    ) -> Dict[str, Any]:
        """Create a default update if LLM fails.
        
        In v2 schema, all information is stored in the text field.
        
        Args:
            old_memory: Original memory record
            current_context: Current context text
            
        Returns:
            Default updated record (v2 schema)
        """
        old_text = old_memory.get("text", "")
        # Append new context to existing text
        updated_text = f"{old_text} [更新: {current_context[:100]}]"
        
        return {
            "chat_id": old_memory.get("chat_id", ""),
            "text": updated_text,
        }
    
    def _apply_reconsolidation(
        self,
        old_memory: Dict[str, Any],
        response: Dict[str, Any],
        current_context: str
    ) -> Dict[str, Any]:
        """Apply reconsolidation updates while preserving immutable fields.
        
        In v2 schema, all information is stored in the text field.
        
        Args:
            old_memory: Original memory record
            response: LLM response with updates
            current_context: Current context for update tracking
            
        Returns:
            Updated memory record (v2 schema)
        """
        # Start with a copy of original
        result = old_memory.copy()
        
        # Preserve immutable top-level fields
        # (id, user_id, ts, memory_type are handled upstream)
        result["chat_id"] = old_memory.get("chat_id", "")
        
        # Update text if provided (v2 schema: all info in text)
        if "text" in response and response["text"]:
            result["text"] = response["text"]
        else:
            # Append context to existing text as fallback
            old_text = old_memory.get("text", "")
            result["text"] = f"{old_text} [更新: {current_context[:50]}...]"
        
        return result
