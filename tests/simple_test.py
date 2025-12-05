#!/usr/bin/env python3
"""
简单的流式测试
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.memory_system.clients.llm import LLMClient

def test_stream():
    """测试流式输出"""
    print("🧪 测试流式输出...")
    
    # 直接使用环境变量
    import os
    api_key = os.environ.get('DEEPSEEK_API_KEY')
    if not api_key:
        print("❌ 未找到 DEEPSEEK_API_KEY 环境变量")
        return False
    
    try:
        client = LLMClient(api_key=api_key)
        
        print("📝 发送测试消息...")
        system_prompt = "你是一个友好的AI助手。"
        user_message = "请简单回复'测试成功'"
        
        print("🤖 AI回复: ", end="", flush=True)
        
        # 测试流式输出
        full_response = ""
        for chunk in client.chat_stream(system_prompt, user_message):
            print(chunk, end="", flush=True)
            full_response += chunk
        
        print(f"\n✅ 流式测试完成！")
        print(f"📊 回复内容: {full_response}")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_stream()
    if success:
        print("🎉 流式输出功能正常！")
    else:
        print("💥 流式输出测试失败！")
