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
        
        Args:
            old_memory: Original memory record
            current_context: Current context text
            
        Returns:
            Default updated record
        """
        metadata = old_memory.get("metadata", {})
        existing_updates = metadata.get("updates", [])
        
        # Create new update entry
        new_update = {
            "time": datetime.now(timezone.utc).isoformat(),
            "desc": f"Updated with new context: {current_context[:100]}..."
        }
        
        return {
            "chat_id": old_memory.get("chat_id", ""),
            "who": old_memory.get("who", "user"),
            "text": old_memory.get("text", ""),
            "metadata": {
                "context": metadata.get("context", ""),
                "thing": metadata.get("thing", ""),
                "time": metadata.get("time", ""),
                "chatid": metadata.get("chatid", ""),
                "who": metadata.get("who", "user"),
                "updates": existing_updates + [new_update]
            }
        }
    
    def _apply_reconsolidation(
        self,
        old_memory: Dict[str, Any],
        response: Dict[str, Any],
        current_context: str
    ) -> Dict[str, Any]:
        """Apply reconsolidation updates while preserving immutable fields.
        
        Args:
            old_memory: Original memory record
            response: LLM response with updates
            current_context: Current context for update tracking
            
        Returns:
            Updated memory record
        """
        # Start with a copy of original
        result = old_memory.copy()
        
        # Preserve immutable top-level fields
        # (id, user_id, ts, memory_type are handled upstream)
        
        # Update text if provided
        if "text" in response and response["text"]:
            result["text"] = response["text"]
        
        # Handle metadata updates
        old_metadata = old_memory.get("metadata", {})
        response_metadata = response.get("metadata", {})
        
        if "metadata" not in result:
            result["metadata"] = {}
        
        # Update mutable metadata fields
        if "context" in response_metadata and response_metadata["context"]:
            result["metadata"]["context"] = response_metadata["context"]
        if "thing" in response_metadata and response_metadata["thing"]:
            result["metadata"]["thing"] = response_metadata["thing"]
        
        # Preserve immutable metadata fields (Requirements 4.2)
        result["metadata"]["time"] = old_metadata.get("time", "")
        result["metadata"]["chatid"] = old_metadata.get("chatid", "")
        result["metadata"]["who"] = old_metadata.get("who", "user")
        
        # Preserve chat_id and who at top level
        result["chat_id"] = old_memory.get("chat_id", "")
        result["who"] = old_memory.get("who", "user")
        
        # Handle updates array (Requirements 4.4)
        existing_updates = old_metadata.get("updates", [])
        response_updates = response_metadata.get("updates", [])
        
        # Ensure updates array grows
        if response_updates:
            # Use response updates if they include existing ones
            if len(response_updates) > len(existing_updates):
                result["metadata"]["updates"] = response_updates
            else:
                # Append new updates from response
                new_updates = [u for u in response_updates if u not in existing_updates]
                result["metadata"]["updates"] = existing_updates + new_updates
        else:
            # No updates in response, add a default one
            new_update = {
                "time": datetime.now(timezone.utc).isoformat(),
                "desc": f"Reconsolidated with context: {current_context[:50]}..."
            }
            result["metadata"]["updates"] = existing_updates + [new_update]
        
        # Ensure updates array has at least one more entry than before
        if len(result["metadata"]["updates"]) <= len(existing_updates):
            new_update = {
                "time": datetime.now(timezone.utc).isoformat(),
                "desc": f"Reconsolidated with context: {current_context[:50]}..."
            }
            result["metadata"]["updates"].append(new_update)
        
        return result
