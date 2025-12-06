#!/usr/bin/env python3
"""
测试Langfuse监控功能的简单脚本

这个脚本验证：
1. Langfuse依赖是否正确安装
2. 配置是否正确加载
3. Observe装饰器是否正常工作
4. SessionId是否正确设置
"""

import os
import sys
import logging
import time
from typing import Dict, Any

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_langfuse_import():
    """测试Langfuse导入"""
    print("🔍 测试Langfuse导入...")
    try:
        from langfuse import observe, get_client
        print("✅ Langfuse导入成功")
        return True
    except ImportError as e:
        print(f"❌ Langfuse导入失败: {e}")
        return False

def test_config_loading():
    """测试配置加载"""
    print("\n🔍 测试配置加载...")
    try:
        from src.memory_system.config import MemoryConfig
        
        config = MemoryConfig()
        
        # 检查Langfuse相关配置
        print(f"  - Langfuse Secret Key: {'已设置' if config.langfuse_secret_key else '未设置'}")
        print(f"  - Langfuse Public Key: {'已设置' if config.langfuse_public_key else '未设置'}")
        print(f"  - Langfuse Base URL: {config.langfuse_base_url}")
        
        return True
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        return False

def test_memory_system_initialization():
    """测试记忆系统初始化"""
    print("\n🔍 测试记忆系统初始化...")
    try:
        from src.memory_system import Memory, MemoryConfig
        
        config = MemoryConfig()
        config.collection_name = "test_langfuse_monitoring"
        
        # 创建记忆系统实例
        memory = Memory(config)
        print("✅ 记忆系统初始化成功")
        
        # 测试基本功能
        test_user_id = "test_user"
        test_chat_id = "test_chat_001"
        test_text = "这是一个测试消息，用于验证Langfuse监控功能。"
        
        # 添加记忆
        print("  - 测试添加记忆...")
        memory_ids = memory.add(test_text, test_user_id, test_chat_id)
        print(f"    ✅ 添加记忆成功，ID: {memory_ids}")
        
        # 搜索记忆
        print("  - 测试搜索记忆...")
        search_results = memory.search("测试消息", test_user_id)
        print(f"    ✅ 搜索记忆成功，找到 {len(search_results)} 条结果")
        
        return True
    except Exception as e:
        print(f"❌ 记忆系统测试失败: {e}")
        return False

def test_observe_decorator():
    """测试Observe装饰器"""
    print("\n🔍 测试Observe装饰器...")
    try:
        from langfuse import observe, get_client
        
        # 更新当前trace
        get_client().update_current_trace(
            session_id=f"test_session_{int(time.time())}",
            tags=["test", "monitoring"],
            metadata={"test_function": "test_observe_decorator"}
        )
        
        print("✅ Observe装饰器测试成功")
        return True
    except Exception as e:
        print(f"❌ Observe装饰器测试失败: {e}")
        return False

def test_observe_decorator_with_wrapper():
    """测试Observe装饰器包装"""
    print("\n🔍 测试Observe装饰器包装...")
    try:
        from langfuse import observe, get_client
        
        # 使用装饰器包装一个简单函数
        @observe(as_type="test")
        def wrapped_test_function():
            get_client().update_current_trace(
                session_id=f"test_session_{int(time.time())}",
                tags=["test", "monitoring", "wrapper"],
                metadata={"test_function": "test_observe_decorator_with_wrapper"}
            )
            return True
        
        result = wrapped_test_function()
        
        if result:
            print("✅ Observe装饰器包装测试成功")
            return True
        else:
            print("❌ Observe装饰器包装测试失败")
            return False
    except Exception as e:
        print(f"❌ Observe装饰器包装测试失败: {e}")
        return False

def test_demo_app():
    """测试Demo应用"""
    print("\n🔍 测试Demo应用...")
    try:
        from demo.app import MemoryDemoApp
        
        app = MemoryDemoApp()
        
        # 初始化记忆系统
        result = app.initialize_memory_system("test_user")
        print(f"  - 初始化结果: {result}")
        
        return True
    except Exception as e:
        print(f"❌ Demo应用测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 开始Langfuse监控功能测试\n")
    
    tests = [
        ("Langfuse导入", test_langfuse_import),
        ("配置加载", test_config_loading),
        ("记忆系统初始化", test_memory_system_initialization),
        ("Observe装饰器", test_observe_decorator),
        ("Demo应用", test_demo_app),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name}测试异常: {e}")
            results.append((test_name, False))
    
    # 输出测试结果摘要
    print("\n" + "="*50)
    print("📊 测试结果摘要")
    print("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！Langfuse监控功能已成功集成。")
    else:
        print("⚠️ 部分测试失败，请检查配置和依赖。")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
