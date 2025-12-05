"""Semantic Memory Writer processor.

Extracts stable, long-term facts from episodic memories for semantic memory storage.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any

from prompts import SEMANTIC_MEMORY_WRITER_PROMPT
from ..clients.llm import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class SemanticExtraction:
    """Result from SemanticWriter extraction.
    
    Attributes:
        write_semantic: Whether to create semantic memories
        facts: List of extracted facts (strings) to store as semantic memories
    """
    write_semantic: bool
    facts: List[str] = field(default_factory=list)


class SemanticWriter:
    """Semantic Memory Writer processor.
    
    Performs pattern merging: analyzes multiple episodic memories together
    to extract stable, long-term facts that should be promoted to semantic memory.
    
    Uses batch processing to identify abstract patterns across concrete episodes.
    Uses SEMANTIC_MEMORY_WRITER_PROMPT to call LLM for extraction.
    """
    
    def __init__(self, llm_client: LLMClient):
        """Initialize the semantic writer.
        
        Args:
            llm_client: LLM client for fact extraction
        """
        self._llm = llm_client
        self._prompt = SEMANTIC_MEMORY_WRITER_PROMPT
    
    def extract(self, consolidation_data: Dict[str, List[str]]) -> SemanticExtraction:
        """Extract semantic facts from batch of episodic memories.
        
        New batch processing implementation: performs pattern merging across
        multiple episodic memories to extract stable, long-term semantic facts.
        
        Args:
            consolidation_data: Dictionary containing:
                - episodic_texts: List of text content from episodic memories
                - existing_semantic_texts: List of text content from existing semantic memories
            
        Returns:
            SemanticExtraction with write_semantic flag and extracted facts
        """
        # Prepare input for LLM (batch mode)
        user_message = json.dumps(consolidation_data, ensure_ascii=False)
        
        # Default response for fallback
        default_response = {
            "write_semantic": False,
            "facts": []
        }
        
        # Call LLM for batch extraction
        response = self._llm.chat_json(
            system_prompt=self._prompt,
            user_message=user_message,
            default=default_response
        )
        
        # Parse response
        write_semantic = response.get("write_semantic", False)
        raw_facts = response.get("facts", [])
        
        # Ensure facts are strings
        facts = [str(f) for f in raw_facts if f]
        
        logger.info(
            f"SemanticWriter batch extraction: "
            f"episodic_count={len(consolidation_data.get('episodic_texts', []))}, "
            f"write_semantic={write_semantic}, facts_count={len(facts)}"
        )
        
        return SemanticExtraction(
            write_semantic=write_semantic,
            facts=facts
        )
