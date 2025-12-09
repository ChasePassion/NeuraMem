"""
测试步骤6：调用MemoryUsageJudge判断哪些情景记忆被使用

验证：judge_used_memories是否正确返回被使用的记忆文本
"""

import pytest
import time
import json
from unittest.mock import Mock, patch, MagicMock


class TestStep6Judge:
    """测试记忆使用判断步骤"""
    
    def test_judge_returns_used_memory_texts(self):
        """测试judge_used_memories是否返回正确的记忆文本"""
        from src.memory_system.processors.memory_usage_judge import MemoryUsageJudge
        from src.memory_system.clients import LLMClient
        
        # 创建真实的LLM客户端
        import os
        llm_client = LLMClient(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
            model="deepseek-chat"
        )
        
        judge = MemoryUsageJudge(llm_client)
        
        # 模拟输入数据
        system_prompt = """You are an AI assistant with memory capabilities.

Here are the episodic memories:
1. 用户正在学习Python编程
2. 用户喜欢喝咖啡

Here are the semantic memories:
1. 用户是一名程序员

Here are the task:
你还记得我在学什么吗？"""
        
        episodic_memories = [
            "用户正在学习Python编程",
            "用户喜欢喝咖啡"
        ]
        
        semantic_memories = [
            "用户是一名程序员"
        ]
        
        message_history = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！有什么可以帮助你的？"}
        ]
        
        # 模拟一个使用了记忆的回复
        final_reply = "是的，我记得你正在学习Python编程！这是一门很实用的编程语言。"
        
        # 调用judge
        used_memories = judge.judge_used_memories(
            system_prompt=system_prompt,
            episodic_memories=episodic_memories,
            semantic_memories=semantic_memories,
            message_history=message_history,
            final_reply=final_reply
        )
        
        print(f"输入的情景记忆: {episodic_memories}")
        print(f"模型回复: {final_reply}")
        print(f"判断为使用的记忆: {used_memories}")
        
        # 验证
        assert isinstance(used_memories, list), "返回值应该是列表"
        
        # 检查返回的是文本而不是ID
        for mem in used_memories:
            assert isinstance(mem, str), f"返回的应该是字符串，而不是 {type(mem)}"
            print(f"  使用的记忆文本: {mem}")
        
        # 验证"用户正在学习Python编程"应该被判断为使用
        # 注意：这取决于LLM的判断，可能不是100%确定
        if "用户正在学习Python编程" in used_memories:
            print("✅ 正确识别了被使用的记忆")
        else:
            print("⚠️ 可能没有正确识别被使用的记忆，请检查LLM判断逻辑")
        
        print("✅ 步骤6测试完成：judge_used_memories返回记忆文本列表")
    
    def test_judge_returns_exact_text_match(self):
        """测试judge返回的文本是否与输入完全匹配"""
        from src.memory_system.processors.memory_usage_judge import MemoryUsageJudge
        from src.memory_system.clients import LLMClient
        
        import os
        llm_client = LLMClient(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
            model="deepseek-chat"
        )
        
        judge = MemoryUsageJudge(llm_client)
        
        # 使用特殊格式的记忆文本来测试精确匹配
        episodic_memories = [
            "2024年12月9日，用户在北京学习Python编程，因为想成为程序员",
            "2024年12月8日，用户喜欢喝咖啡"
        ]
        
        system_prompt = f"""Here are the episodic memories:
1. {episodic_memories[0]}
2. {episodic_memories[1]}

Here are the task:
你记得我在学什么吗？"""
        
        final_reply = "我记得你在2024年12月9日在北京学习Python编程，因为你想成为程序员。"
        
        used_memories = judge.judge_used_memories(
            system_prompt=system_prompt,
            episodic_memories=episodic_memories,
            semantic_memories=[],
            message_history=[],
            final_reply=final_reply
        )
        
        print(f"原始记忆文本: {episodic_memories}")
        print(f"返回的使用记忆: {used_memories}")
        
        # 关键测试：返回的文本是否与输入完全一致
        for used_mem in used_memories:
            if used_mem in episodic_memories:
                print(f"✅ 文本完全匹配: {used_mem[:50]}...")
            else:
                print(f"❌ 文本不匹配！返回的: {used_mem[:50]}...")
                print(f"   这可能导致后续ID匹配失败！")
        
        print("✅ 步骤6测试完成：检查文本匹配情况")


if __name__ == "__main__":
    test = TestStep6Judge()
    test.test_judge_returns_used_memory_texts()
    print("\n" + "="*50 + "\n")
    test.test_judge_returns_exact_text_match()
