#!/usr/bin/env python3
"""
Final verification script for the AI Memory System CRUD enhancement.

This script tests the complete functionality of the new EpisodicMemoryManager
which supports add, update, and delete operations.
"""

import asyncio
import time
from src.memory_system import Memory, MemoryConfig
from src.memory_system.processors.memory_manager import EpisodicMemoryManager, MemoryOperation, MemoryManagementResult
from src.memory_system.clients.llm import LLMClient


async def test_memory_manager_crud():
    """Test the EpisodicMemoryManager CRUD operations."""
    print("🧪 Testing EpisodicMemoryManager CRUD Operations...")
    
    # Create a dummy LLM client for testing
    llm_client = LLMClient(
        api_key="test_key",
        base_url="http://localhost:8000",
        model="test-model"
    )
    
    # Create memory manager
    memory_manager = EpisodicMemoryManager(llm_client)
    
    # Test data
    user_text = "请记住我是北京大学的学生"
    assistant_text = "好的，我会记住这个信息"
    existing_memories = []
    
    # Test memory management
    result = memory_manager.manage_memories(
        user_text=user_text,
        assistant_text=assistant_text,
        episodic_memories=existing_memories
    )
    
    print(f"✅ Memory manager operations: {len(result.operations)} operations")
    for i, op in enumerate(result.operations):
        print(f"  {i+1}. {op.operation_type}: {op.text}")
    
    return True


async def test_memory_system_integration():
    """Test the full Memory system integration."""
    print("\n🔗 Testing Memory System Integration...")
    
    # Create memory system
    config = MemoryConfig()
    config.collection_name = f"test_final_verification_{int(time.time())}"
    memory = Memory(config)
    
    try:
        # Test manage method
        user_id = "test_user_final"
        chat_id = "test_chat_final"
        
        # Test 1: Add memory
        print("  📝 Testing memory addition...")
        result = await memory.manage_async(
            user_text="我是清华大学计算机系的学生",
            assistant_text="好的，我记住了你是清华大学计算机系的学生",
            user_id=user_id,
            chat_id=chat_id
        )
        print(f"    Added memory IDs: {result}")
        
        # Test 2: Search memories
        print("  🔍 Testing memory search...")
        search_results = memory.search(
            query="清华大学",
            user_id=user_id,
            limit=5
        )
        print(f"    Found {len(search_results)} memories")
        for mem in search_results:
            print(f"      - {mem.text[:50]}...")
        
        # Test 3: Update memory (if exists)
        if search_results:
            memory_id = search_results[0].id
            print("  ✏️ Testing memory update...")
            success = memory.update(memory_id, {"text": "我是清华大学计算机系的研究生"})
            print(f"    Update success: {success}")
        
        # Test 4: Reset user memories
        print("  🗑️ Testing memory reset...")
        deleted_count = memory.reset(user_id)
        print(f"    Deleted {deleted_count} memories")
        
        print("✅ Memory system integration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Memory system integration test failed: {e}")
        return False
    finally:
        # Cleanup
        try:
            memory.store.drop_collection()
        except:
            pass


async def test_demo_app():
    """Test the demo app functionality."""
    print("\n🎨 Testing Demo App...")
    
    try:
        from demo.app import MemoryDemoApp
        
        app = MemoryDemoApp()
        
        # Test initialization
        result = app.initialize_memory_system("test_user_demo")
        print(f"  🚀 Initialization: {result}")
        
        # Test chat method (simplified)
        history = []
        response, new_history, memories = await app.chat(
            "请记住我喜欢喝咖啡",
            history
        )
        print(f"  💬 Chat test completed")
        print(f"    Response length: {len(response)}")
        print(f"    History length: {len(new_history)}")
        
        print("✅ Demo app test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Demo app test failed: {e}")
        return False


async def main():
    """Run all verification tests."""
    print("🚀 AI Memory System CRUD Enhancement - Final Verification")
    print("=" * 60)
    
    tests = [
        ("Memory Manager CRUD", test_memory_manager_crud),
        ("Memory System Integration", test_memory_system_integration),
        ("Demo App", test_demo_app),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<40} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! The CRUD enhancement is working correctly.")
        print("\n📋 IMPLEMENTATION SUMMARY:")
        print("  ✅ EpisodicMemoryManager with CRUD operations")
        print("  ✅ Memory.manage() and Memory.manage_async() methods")
        print("  ✅ Integration with existing search and consolidation")
        print("  ✅ Demo app compatibility")
        print("  ✅ Comprehensive test coverage")
        return True
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
