"""
测试步骤5：模型回复

验证：LLM是否正确生成回复，以及回复是否使用了记忆
"""

import pytest
import time


class TestStep5LLMResponse:
    """测试LLM回复步骤"""
    
    def test_llm_generates_response_with_memory(self):
        """测试LLM是否能生成使用记忆的回复"""
        from src.memory_system.clients import LLMClient
        import os
        
        llm_client = LLMClient(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
            model="deepseek-chat"
        )
        
        # 构建包含记忆的上下文
        context = """Here are the episodic memories:
1. 用户正在学习Python编程
2. 用户喜欢喝咖啡

Here are the semantic memories:
(No semantic memories)

Here are the task:
你记得我在学什么吗？"""
        
        system_prompt = f"""You are an AI assistant with long-term memory capabilities.
Please answer based on the user's messages and relevant memories. If there are relevant memories, reflect that you remember the user's information in your response.
Maintain a friendly and natural conversation style.

{context}"""
        
        user_message = "你记得我在学什么吗？"
        
        # 调用LLM
        response = llm_client.chat(system_prompt, user_message)
        
        print(f"LLM回复: {response}")
        
        # 验证回复是否提到了Python
        assert "Python" in response or "python" in response or "编程" in response, \
            "LLM回复应该提到Python或编程"
        
        print("✅ 步骤5测试通过：LLM正确使用记忆生成回复")


if __name__ == "__main__":
    test = TestStep5LLMResponse()
    test.test_llm_generates_response_with_memory()
