#!/usr/bin/env python3
"""
测试流式输出功能的简单脚本
"""

import asyncio
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.memory_system.clients.llm import LLMClient
from src.memory_system.config import Config

async def test_streaming():
    """测试流式输出功能"""
    print("🧪 测试流式输出功能...")
    
    # 初始化 LLM 客户端
    config = Config()
    llm_client = LLMClient(
        api_key=config.deepseek_api_key,
        base_url=config.deepseek_base_url,
        model=config.deepseek_model
    )
    
    # 测试消息
    system_prompt = "你是一个友好的AI助手，请简单介绍一下自己。"
    user_message = "你好，请用流式方式回复我。"
    
    print(f"📝 用户消息: {user_message}")
    print("🤖 AI回复（流式）: ", end="", flush=True)
    
    try:
        # 测试流式输出
        response_text = ""
        for chunk in llm_client.chat_stream(system_prompt, user_message):
            print(chunk, end="", flush=True)
            response_text += chunk
        
        print("\n\n✅ 流式输出测试成功！")
        print(f"📊 完整回复长度: {len(response_text)} 字符")
        
        # 测试非流式输出对比
        print("\n🔄 测试非流式输出对比...")
        normal_response = llm_client.chat(system_prompt, user_message)
        print(f"📊 非流式回复长度: {len(normal_response)} 字符")
        
        if response_text.strip() == normal_response.strip():
            print("✅ 流式和非流式结果一致！")
        else:
            print("⚠️ 流式和非流式结果不一致，需要检查")
            
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = asyncio.run(test_streaming())
    if success:
        print("\n🎉 所有测试通过！")
    else:
        print("\n💥 测试失败！")
        sys.exit(1)
