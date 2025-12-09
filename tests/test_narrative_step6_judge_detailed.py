"""
详细测试步骤6：MemoryUsageJudge

深入分析为什么judge_used_memories返回空列表
"""

import pytest
import time
import json
import os


class TestStep6JudgeDetailed:
    """详细测试MemoryUsageJudge"""
    
    def test_judge_raw_llm_response(self):
        """测试LLM的原始响应"""
        from src.memory_system.clients import LLMClient
        from prompts import MEMORY_RELEVANCE_FILTER_PROMPT
        
        llm_client = LLMClient(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
            model="deepseek-chat"
        )
        
        # 构建输入数据
        episodic_memories = ["用户正在学习Python编程"]
        
        system_prompt = f"""Here are the episodic memories:
1. {episodic_memories[0]}

Here are the task:
你记得我在学什么吗？"""
        
        final_reply = "是的，我记得你正在学习Python编程！"
        
        input_data = {
            "system_prompt": system_prompt,
            "episodic_memories": episodic_memories,
            "semantic_memories": [],
            "message_history": [],
            "final_reply": final_reply
        }
        
        print("="*60)
        print("输入数据:")
        print("="*60)
        print(json.dumps(input_data, ensure_ascii=False, indent=2))
        
        print("\n" + "="*60)
        print("MEMORY_RELEVANCE_FILTER_PROMPT (前500字符):")
        print("="*60)
        print(MEMORY_RELEVANCE_FILTER_PROMPT[:500])
        
        print("\n" + "="*60)
        print("调用LLM...")
        print("="*60)
        
        # 直接调用chat_json
        response = llm_client.chat_json(
            system_prompt=MEMORY_RELEVANCE_FILTER_PROMPT,
            user_message=json.dumps(input_data, ensure_ascii=False),
            default={"used_episodic_memories": []}
        )
        
        print(f"LLM响应: {response}")
        print(f"响应类型: {type(response)}")
        
        used_memories = response.get("used_episodic_memories", [])
        print(f"used_episodic_memories: {used_memories}")
        
        if used_memories:
            print("✅ LLM正确返回了使用的记忆")
        else:
            print("❌ LLM返回空列表，需要检查prompt或输入格式")
    
    def test_judge_with_explicit_usage(self):
        """测试明确使用记忆的情况"""
        from src.memory_system.processors.memory_usage_judge import MemoryUsageJudge
        from src.memory_system.clients import LLMClient
        
        llm_client = LLMClient(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
            model="deepseek-chat"
        )
        
        judge = MemoryUsageJudge(llm_client)
        
        # 构建一个非常明确使用记忆的场景
        episodic_memories = [
            "2024年12月1日，用户在北京参加了Python编程培训班"
        ]
        
        system_prompt = f"""Here are the episodic memories:
1. {episodic_memories[0]}

Here are the task:
你记得我参加过什么培训吗？"""
        
        # 回复明确引用了记忆中的信息
        final_reply = "是的，我记得你在2024年12月1日在北京参加了Python编程培训班。"
        
        print("="*60)
        print("测试场景：明确使用记忆")
        print("="*60)
        print(f"情景记忆: {episodic_memories}")
        print(f"模型回复: {final_reply}")
        
        used_memories = judge.judge_used_memories(
            system_prompt=system_prompt,
            episodic_memories=episodic_memories,
            semantic_memories=[],
            message_history=[],
            final_reply=final_reply
        )
        
        print(f"judge返回: {used_memories}")
        
        if episodic_memories[0] in used_memories:
            print("✅ 正确识别了被使用的记忆")
        else:
            print("❌ 没有正确识别被使用的记忆")
            if used_memories:
                print(f"   返回的内容: {used_memories}")
                print(f"   期望的内容: {episodic_memories}")
    
    def test_judge_input_format(self):
        """测试输入格式是否正确"""
        from src.memory_system.clients import LLMClient
        from prompts import MEMORY_RELEVANCE_FILTER_PROMPT
        
        llm_client = LLMClient(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
            model="deepseek-chat"
        )
        
        # 简化的测试输入
        input_data = {
            "system_prompt": "Here are the episodic memories:\n1. 用户喜欢喝咖啡\n\nHere are the task:\n你记得我喜欢什么吗？",
            "episodic_memories": ["用户喜欢喝咖啡"],
            "semantic_memories": [],
            "message_history": [],
            "final_reply": "我记得你喜欢喝咖啡！"
        }
        
        print("="*60)
        print("测试简化输入")
        print("="*60)
        print(f"输入: {json.dumps(input_data, ensure_ascii=False)}")
        
        response = llm_client.chat_json(
            system_prompt=MEMORY_RELEVANCE_FILTER_PROMPT,
            user_message=json.dumps(input_data, ensure_ascii=False),
            default={"used_episodic_memories": []}
        )
        
        print(f"响应: {response}")
        
        used = response.get("used_episodic_memories", [])
        if "用户喜欢喝咖啡" in used:
            print("✅ 简化输入测试通过")
        else:
            print("❌ 简化输入测试失败")
    
    def test_llm_chat_json_method(self):
        """测试LLMClient的chat_json方法"""
        from src.memory_system.clients import LLMClient
        
        llm_client = LLMClient(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
            model="deepseek-chat"
        )
        
        # 简单的JSON测试
        system_prompt = "You are a helpful assistant. Always respond with valid JSON."
        user_message = 'Return a JSON object with key "test" and value "success"'
        
        print("="*60)
        print("测试chat_json方法")
        print("="*60)
        
        response = llm_client.chat_json(
            system_prompt=system_prompt,
            user_message=user_message,
            default={"test": "failed"}
        )
        
        print(f"响应: {response}")
        print(f"响应类型: {type(response)}")
        
        if response.get("test") == "success":
            print("✅ chat_json方法正常工作")
        else:
            print("⚠️ chat_json方法可能有问题")


if __name__ == "__main__":
    test = TestStep6JudgeDetailed()
    
    print("\n" + "#"*70)
    print("# 测试1: LLM原始响应")
    print("#"*70)
    test.test_judge_raw_llm_response()
    
    print("\n" + "#"*70)
    print("# 测试2: 明确使用记忆的场景")
    print("#"*70)
    test.test_judge_with_explicit_usage()
    
    print("\n" + "#"*70)
    print("# 测试3: 简化输入格式")
    print("#"*70)
    test.test_judge_input_format()
    
    print("\n" + "#"*70)
    print("# 测试4: chat_json方法")
    print("#"*70)
    test.test_llm_chat_json_method()
