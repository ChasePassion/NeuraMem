"""Memory Usage Judge - Determines which episodic memories were actually used in response generation.

This module implements the logic to judge which retrieved episodic memories
were actually used to generate the assistant's final reply, enabling precise
memory reconsolidation based on actual usage.
"""

import json
import logging
from typing import List, Dict, Any

from ..clients import LLMClient
from ..prompts import MEMORY_RELEVANCE_FILTER_PROMPT

logger = logging.getLogger(__name__)


class MemoryUsageJudge:
    """Judge which episodic memories were actually used in generating a response.
    
    This class analyzes the complete context (system prompt, memories, message history,
    and final reply) to determine which episodic memories were actually utilized
    in generating the assistant's response.
    """
    
    def __init__(self, llm_client: LLMClient):
        """Initialize the memory usage judge.
        
        Args:
            llm_client: LLM client for making judgment calls
        """
        self._llm_client = llm_client
    
    def judge_used_memories(
        self,
        episodic_memories: List[str],
        last_user: str,
        last_assistant: str
    ) -> List[str]:
        """Judge which episodic memories were actually used in the final reply.
        
        Only uses the most recent user message and assistant reply to determine
        which episodic memories were actually utilized in generating the response.
        
        Args:
            episodic_memories: List of episodic memory texts that were retrieved
            last_user: The most recent user message
            last_assistant: The assistant's complete reply to that message
            
        Returns:
            List of episodic memory texts that were actually used
        """
        if not episodic_memories:
            return []
        
        try:
            # Prepare input data with only the essential context
            input_data = {
                "episodic_memories": episodic_memories,
                "last_user": last_user,
                "last_assistant": last_assistant
            }
            
            # Use MEMORY_RELEVANCE_FILTER_PROMPT imported at module level
            
            # Call LLM to judge which memories were used
            response = self._llm_client.chat_json(
                system_prompt=MEMORY_RELEVANCE_FILTER_PROMPT,
                user_message=json.dumps(input_data, ensure_ascii=False),
                default={"used_episodic_memories": []}
            )
            
            # chat_json returns {"parsed_data": {...}, "raw_response": ..., ...}
            # Extract the actual parsed data
            parsed_data = response.get("parsed_data", {})
            used_memories = parsed_data.get("used_episodic_memories", [])
            
            logger.info(
                f"Memory usage judgment: {len(used_memories)}/{len(episodic_memories)} "
                f"episodic memories were actually used"
            )
            
            return used_memories
            
        except Exception as e:
            logger.warning(f"Failed to judge memory usage: {e}")
            # Conservative fallback: assume no memories were used
            return []
