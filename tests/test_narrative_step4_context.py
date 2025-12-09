"""
测试步骤4：将记忆和context拼接

验证：_build_context_with_memories是否正确构建上下文
"""

import pytest
import time
from typing import List, Dict
from dataclasses import dataclass


@dataclass
class MockMemoryRecord:
    """模拟MemoryRecord"""
    id: int
    user_id: str
    memory_type: str
    ts: int
    chat_id: str
    text: str
    distance: float = 0.0


class TestStep4Context:
    """测试上下文构建步骤"""
    
    def test_build_context_includes_episodic_memories(self):
        """测试上下文是否包含情景记忆"""
        # 模拟记忆数据
        episodic_memories = [
            MockMemoryRecord(
                id=1,
                user_id="test_user",
                memory_type="episodic",
                ts=int(time.time()),
                chat_id="chat_1",
                text="用户正在学习Python编程"
            ),
            MockMemoryRecord(
                id=2,
                user_id="test_user",
                memory_type="episodic",
                ts=int(time.time()),
                chat_id="chat_1",
                text="用户喜欢喝咖啡"
            )
        ]
        
        semantic_memories = [
            MockMemoryRecord(
                id=3,
                user_id="test_user",
                memory_type="semantic",
                ts=int(time.time()),
                chat_id="chat_1",
                text="用户是一名程序员"
            )
        ]
        
        memories = {
            "episodic": episodic_memories,
            "semantic": semantic_memories
        }
        
        # 模拟历史消息
        history = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！有什么可以帮助你的？"}
        ]
        
        # 构建上下文（模拟demo/app.py中的逻辑）
        context = self._build_context_with_memories("我想学习编程", memories, history)
        
        print(f"构建的上下文:\n{context}")
        
        # 验证
        assert "用户正在学习Python编程" in context, "上下文应包含情景记忆1"
        assert "用户喜欢喝咖啡" in context, "上下文应包含情景记忆2"
        assert "用户是一名程序员" in context, "上下文应包含语义记忆"
        assert "我想学习编程" in context, "上下文应包含当前任务"
        
        print("✅ 步骤4测试通过：上下文正确包含所有记忆")
    
    def _build_context_with_memories(self, message: str, memories: Dict, history: List[Dict[str, str]]) -> str:
        """复制自demo/app.py的上下文构建逻辑"""
        context_parts = []
        
        # 1. 情景记忆部分
        context_parts.append("Here are the episodic memories:")
        episodic_memories = memories.get("episodic", [])
        if episodic_memories:
            for i, mem in enumerate(episodic_memories[:3], 1):
                context_parts.append(f"{i}. {mem.text}")
        else:
            context_parts.append("(No episodic memories)")
        context_parts.append("")
        
        # 2. 语义记忆部分
        context_parts.append("Here are the semantic memories:")
        semantic_memories = memories.get("semantic", [])
        if semantic_memories:
            for i, mem in enumerate(semantic_memories[:3], 1):
                context_parts.append(f"{i}. {mem.text}")
        else:
            context_parts.append("(No semantic memories)")
        context_parts.append("")
        
        # 3. 历史对话部分
        context_parts.append("Here are the history messages:")
        history_pairs = self._history_pairs(history)
        if history_pairs:
            for i, (user_msg, ai_msg) in enumerate(history_pairs[-3:], 1):
                context_parts.append(f"Turn {i}:")
                context_parts.append(f"  User: {user_msg}")
                context_parts.append(f"  Assistant: {ai_msg}")
        else:
            context_parts.append("(No history messages)")
        context_parts.append("")
        
        # 4. 当前任务
        context_parts.append("Here are the task:")
        context_parts.append(message)
        
        return "\n".join(context_parts)
    
    def _history_pairs(self, history: List[Dict[str, str]]) -> List[tuple]:
        """转换历史消息为对话对"""
        pairs = []
        last_user = None
        
        for msg in history:
            if msg.get("role") == "user":
                last_user = msg.get("content", "")
            elif msg.get("role") == "assistant" and last_user is not None:
                pairs.append((last_user, msg.get("content", "")))
                last_user = None
        
        return pairs


if __name__ == "__main__":
    test = TestStep4Context()
    test.test_build_context_includes_episodic_memories()
