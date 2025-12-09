#!/usr/bin/env python3
"""
验证叙事记忆功能的简单脚本
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """测试导入"""
    try:
        from src.memory_system import Memory, MemoryConfig
        print("✅ 基本导入成功")
        return True
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        return False

def test_config():
    """测试配置"""
    try:
        from src.memory_system import MemoryConfig
        config = MemoryConfig()
        print(f"✅ 配置创建成功")
        print(f"   - 叙事相似度阈值: {config.narrative_similarity_threshold}")
        print(f"   - Collection名称: {config.collection_name}")
        return True
    except Exception as e:
        print(f"❌ 配置测试失败: {e}")
        return False

def test_memory_initialization():
    """测试Memory初始化"""
    try:
        from src.memory_system import Memory, MemoryConfig
        
        # 创建测试配置
        config = MemoryConfig()
        config.collection_name = "test_narrative_memories"
        
        # 初始化Memory
        memory = Memory(config)
        print("✅ Memory初始化成功")
        
        # 检查是否有narrative_manager
        if hasattr(memory, '_narrative_manager'):
            print("✅ NarrativeMemoryManager已集成")
        else:
            print("❌ NarrativeMemoryManager未找到")
            return False
            
        return True
    except Exception as e:
        print(f"❌ Memory初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_narrative_manager():
    """测试NarrativeMemoryManager"""
    try:
        from src.memory_system import Memory, MemoryConfig
        from src.memory_system.processors.narrative_memory_manager import NarrativeMemoryManager
        
        config = MemoryConfig()
        config.collection_name = "test_narrative_memories"
        
        memory = Memory(config)
        manager = memory._narrative_manager
        
        print("✅ NarrativeMemoryManager访问成功")
        
        # 测试groups collection创建
        groups_collection = manager._ensure_groups_collection("test_user")
        print(f"✅ Groups collection创建/访问成功: {groups_collection}")
        
        return True
    except Exception as e:
        print(f"❌ NarrativeMemoryManager测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🧪 开始验证叙事记忆功能...")
    print("=" * 50)
    
    tests = [
        ("导入测试", test_imports),
        ("配置测试", test_config),
        ("Memory初始化测试", test_memory_initialization),
        ("NarrativeMemoryManager测试", test_narrative_manager),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}:")
        if test_func():
            passed += 1
        else:
            print(f"   ⚠️  {test_name}失败")
    
    print("\n" + "=" * 50)
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！叙事记忆功能已成功实现")
        return True
    else:
        print("❌ 部分测试失败，需要检查实现")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
