"""Memory Usage Judge - Determines which episodic memories were actually used in response generation.

This module implements the logic to judge which retrieved episodic memories
were actually used to generate the assistant's final reply, enabling precise
memory reconsolidation based on actual usage.
"""

import json
import logging
from typing import List, Dict, Any

from ..clients import LLMClient

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
        system_prompt: str,
        episodic_memories: List[str],
        semantic_memories: List[str],
        message_history: List[Dict[str, str]],
        final_reply: str
    ) -> List[str]:
        """Judge which episodic memories were actually used in the final reply.
        
        Args:
            system_prompt: The system prompt sent to the assistant
            episodic_memories: List of episodic memory texts that were retrieved
            semantic_memories: List of semantic memory texts that were retrieved
            message_history: Full message history sent to the assistant
            final_reply: The assistant's final reply
            
        Returns:
            List of episodic memory texts that were actually used
        """
        if not episodic_memories:
            return []
        
        try:
            # Prepare input data according to MEMORY_RELEVANCE_FILTER_PROMPT requirements
            input_data = {
                "system_prompt": system_prompt,
                "episodic_memories": episodic_memories,
                "semantic_memories": semantic_memories,
                "message_history": message_history,
                "final_reply": final_reply
            }
            
            # Import the prompt
            from prompts import MEMORY_RELEVANCE_FILTER_PROMPT
            
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
