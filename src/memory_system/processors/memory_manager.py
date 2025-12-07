"""Episodic Memory Manager for CRUD operations on episodic memories.

This module provides intelligent memory management capabilities including
adding, updating, and deleting episodic memories based on LLM decisions.
"""

import json
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ..clients.llm import LLMClient
from prompts import EPISODIC_MEMORY_MANAGER

from langfuse import observe, get_client


logger = logging.getLogger(__name__)


@dataclass
class MemoryOperation:
    """Represents a single memory operation."""
    operation_type: str  # "add", "update", "delete"
    memory_id: Optional[int] = None
    text: Optional[str] = None
    old_text: Optional[str] = None


@dataclass 
class MemoryManagementResult:
    """Result of memory management operations."""
    operations: List[MemoryOperation] = field(default_factory=list)
    added_ids: List[int] = field(default_factory=list)


class EpisodicMemoryManager:
    """Manages episodic memories with CRUD operations using LLM intelligence.
    
    This class replaces the old EpisodicWriteDecider and provides full
    CRUD capabilities for episodic memories.
    """
    
    def __init__(self, llm_client: LLMClient):
        """Initialize the memory manager.
        
        Args:
            llm_client: LLM client for making decisions
        """
        self._llm = llm_client
        self._prompt = EPISODIC_MEMORY_MANAGER
    
    @observe(as_type="chain")
    def manage_memories(
        self, 
        user_text: str, 
        assistant_text: str, 
        episodic_memories: List[Dict[str, Any]]
    ) -> MemoryManagementResult:
        """Manage memories based on current conversation and existing memories.
        
        Args:
            user_text: Current user input
            assistant_text: Current assistant response
            episodic_memories: List of existing episodic memories
            
        Returns:
            MemoryManagementResult with operations to perform
        """
        get_client().update_current_trace(
            tags=["memory_manager", "episodic_crud"],
            metadata={
                "operation": "manage_episodic_memories",
                "episodic_memories_count": len(episodic_memories),
                "user_text_length": len(user_text),
                "assistant_text_length": len(assistant_text)
            }
        )
        
        # 构造完整对话轮次
        current_turn = {
            "user": user_text,
            "assistant": assistant_text
        }
        
        # 调用LLM进行CRUD决策
        input_data = {
            "current_turn": current_turn,
            "episodic_memories": [{"id": mem["id"], "text": mem["text"]} 
                                for mem in episodic_memories]
        }
        
        response = self._llm.chat_json(
            system_prompt=self._prompt,
            user_message=json.dumps(input_data, ensure_ascii=False),
            default={"add": [], "update": [], "delete": []}
        )
        
        # 转换为操作列表
        operations = []
        
        # 处理添加操作
        for add_op in response.get("add", []):
            operations.append(MemoryOperation("add", text=add_op["text"]))
        
        # 处理更新操作
        for update_op in response.get("update", []):
            operations.append(MemoryOperation(
                "update", 
                memory_id=update_op["id"],
                old_text=update_op["old_text"], 
                text=update_op["new_text"]
            ))
        
        # 处理删除操作
        for delete_op in response.get("delete", []):
            operations.append(MemoryOperation("delete", memory_id=delete_op["id"]))
        
        result = MemoryManagementResult(operations)
        
        operation_counts = {
            "add": len([op for op in operations if op.operation_type == "add"]),
            "update": len([op for op in operations if op.operation_type == "update"]),
            "delete": len([op for op in operations if op.operation_type == "delete"])
        }
        
        get_client().update_current_trace(
            output={
                "operations_count": len(operations),
                "operation_counts": operation_counts,
                "operations": [
                    {
                        "type": op.operation_type,
                        "memory_id": op.memory_id,
                        "text_length": len(op.text) if op.text else 0
                    }
                    for op in operations
                ]
            },
            metadata={
                "success": True,
                "total_operations": len(operations),
                "llm_response_keys": list(response.keys())
            }
        )

        
        return result
