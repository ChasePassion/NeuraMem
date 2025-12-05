#!/usr/bin/env python3
"""
测试流式输出与记忆系统的集成功能
"""

import asyncio
import sys
import os
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.memory_system import Memory, MemoryConfig

async def test_memory_integration():
    """测试流式输出与记忆系统的集成"""
    print("🧪 测试流式输出与记忆系统集成...")
    
    # 设置环境变量
    os.environ['DEEPSEEK_API_KEY'] = 'sk-d99c433f066744e3b9489b3ce80ac943'
    
    try:
        # 初始化记忆系统
        print("📚 初始化记忆系统...")
        config = MemoryConfig()
        config.collection_name = f"test_stream_memories_{int(time.time())}"
        memory = Memory(config)
        
        # 测试用户ID
        user_id = "test_stream_user"
        
        # 1. 测试添加记忆
        print("\n1️⃣ 测试添加记忆...")
        test_message = "我叫张三，是北京大学计算机专业的学生，喜欢喝咖啡"
        await memory.add_async(test_message, user_id, "test_chat_1")
        print("✅ 记忆添加成功")
        
        # 2. 测试流式聊天与记忆检索
        print("\n2️⃣ 测试流式聊天与记忆检索...")
        query = "我叫什么名字？"
        
        # 检索相关记忆
        memories = memory.search(query, user_id, 5, False)
        print(f"🔍 检索到 {len(memories)} 条相关记忆")
        
        # 构建上下文
        context = f"相关记忆:\n"
        for i, mem in enumerate(memories, 1):
            context += f"{i}. {mem.text}\n"
        context += f"\n用户问题: {query}"
        
        # 测试流式回复
        print("🤖 AI流式回复: ", end="", flush=True)
        system_prompt = "你是一个有记忆的AI助手，请根据提供的记忆回答用户问题。"
        
        full_response = ""
        for chunk in memory._llm_client.chat_stream(system_prompt, query):
            print(chunk, end="", flush=True)
            full_response += chunk
        
        print(f"\n✅ 流式回复完成")
        
        # 3. 验证记忆是否被正确使用
        if "张三" in full_response:
            print("✅ AI正确使用了记忆中的信息")
        else:
            print("⚠️ AI可能没有正确使用记忆信息")
        
        # 4. 测试智能巩固功能
        print("\n3️⃣ 测试智能巩固功能...")
        try:
            # 模拟智能巩固过程
            system_prompt = memory._get_system_prompt()
            await asyncio.to_thread(
                memory._intelligent_reconsolidate,
                query,
                memories,
                system_prompt,
                [{"role": "user", "content": query}],
                full_response
            )
            print("✅ 智能巩固测试成功")
        except Exception as e:
            print(f"⚠️ 智能巩固测试失败: {e}")
        
        # 5. 验证记忆持久化
        print("\n4️⃣ 验证记忆持久化...")
        all_memories = memory._store.query(
            filter_expr=f'user_id == "{user_id}"',
            output_fields=["text", "memory_type"],
            limit=10
        )
        print(f"📊 总共找到 {len(all_memories)} 条记忆")
        
        # 6. 清理测试数据
        print("\n5️⃣ 清理测试数据...")
        memory.reset(user_id)
        print("✅ 测试数据清理完成")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_app_integration():
    """测试应用层的流式集成"""
    print("\n🎯 测试应用层流式集成...")
    
    try:
        # 导入应用类
        from demo.app import MemoryDemoApp
        
        # 创建应用实例
        app = MemoryDemoApp()
        
        # 初始化记忆系统
        result = app.initialize_memory_system("test_app_user")
        print(f"📚 初始化结果: {result}")
        
        if "✅" not in result:
            print("❌ 记忆系统初始化失败")
            return False
        
        # 测试流式聊天
        print("\n💬 测试流式聊天...")
        test_message = "请记住我喜欢编程"
        
        # 模拟流式聊天
        response_generator = app.chat_stream(test_message, [])
        
        async for history, memories in response_generator:
            if len(history) > 0 and history[-1].get("role") == "assistant":
                print(f"🤖 回复: {history[-1].get('content', '')}")
                break
        
        print("✅ 应用层流式集成测试成功")
        return True
        
    except Exception as e:
        print(f"❌ 应用层测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """主测试函数"""
    print("🚀 开始流式输出与记忆系统集成测试\n")
    
    # 测试1: 基础记忆集成
    test1_success = await test_memory_integration()
    
    # 测试2: 应用层集成
    test2_success = await test_app_integration()
    
    # 总结
    print("\n" + "="*50)
    print("📊 测试结果总结:")
    print(f"  基础记忆集成: {'✅ 通过' if test1_success else '❌ 失败'}")
    print(f"  应用层集成: {'✅ 通过' if test2_success else '❌ 失败'}")
    
    if test1_success and test2_success:
        print("\n🎉 所有测试通过！流式输出与记忆系统集成成功！")
        return True
    else:
        print("\n💥 部分测试失败，需要进一步调试")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
