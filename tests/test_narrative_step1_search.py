"""
测试步骤1-3：用户发送消息 → 检索情景记忆和语义记忆 → 扩展记忆

验证：search方法是否正确返回情景记忆和语义记忆
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
import numpy as np


class TestStep1Search:
    """测试检索步骤"""
    
    def test_search_returns_episodic_memories(self):
        """测试search是否正确返回情景记忆"""
        from src.memory_system import Memory, MemoryConfig
        
        # 创建测试配置
        config = MemoryConfig()
        config.collection_name = f"test_step1_{int(time.time())}"
        
        memory = Memory(config)
        user_id = "test_user_step1"
        chat_id = "test_chat_1"
        
        try:
            # 1. 先添加一些测试记忆
            test_memories = [
                "2024年12月，用户在北京学习Python编程",
                "2024年11月，用户喜欢喝咖啡",
                "2024年10月，用户正在开发AI记忆系统"
            ]
            
            # 直接插入记忆
            embeddings = memory._embedding_client.encode(test_memories)
            entities = []
            for i, text in enumerate(test_memories):
                entity = {
                    "user_id": user_id,
                    "memory_type": "episodic",
                    "ts": int(time.time()) - i * 86400,
                    "chat_id": chat_id,
                    "text": text,
                    "vector": embeddings[i],
                    "group_id": -1,
                }
                entities.append(entity)
            
            inserted_ids = memory._store.insert(entities)
            print(f"插入的记忆ID: {inserted_ids}")
            
            # 2. 执行搜索
            query = "Python编程"
            results = memory.search(query, user_id)
            
            # 3. 验证结果
            print(f"搜索结果: episodic={len(results.get('episodic', []))}, semantic={len(results.get('semantic', []))}")
            
            assert "episodic" in results, "结果应包含episodic键"
            assert "semantic" in results, "结果应包含semantic键"
            assert len(results["episodic"]) > 0, "应该返回至少一条情景记忆"
            
            # 验证返回的记忆包含必要字段
            for mem in results["episodic"]:
                print(f"  记忆: id={mem.id}, text={mem.text[:50]}...")
                assert hasattr(mem, 'id'), "记忆应有id字段"
                assert hasattr(mem, 'text'), "记忆应有text字段"
                assert mem.id is not None, "id不应为None"
                assert mem.text is not None, "text不应为None"
            
            print("✅ 步骤1-3测试通过：search正确返回情景记忆")
            
        finally:
            # 清理
            memory._store.drop_collection()
    
    def test_search_returns_memory_with_correct_id(self):
        """测试search返回的记忆ID是否正确"""
        from src.memory_system import Memory, MemoryConfig
        
        config = MemoryConfig()
        config.collection_name = f"test_step1_id_{int(time.time())}"
        
        memory = Memory(config)
        user_id = "test_user_step1_id"
        chat_id = "test_chat_1"
        
        try:
            # 插入单条记忆
            test_text = "用户正在测试记忆系统的ID返回功能"
            embeddings = memory._embedding_client.encode([test_text])
            
            entity = {
                "user_id": user_id,
                "memory_type": "episodic",
                "ts": int(time.time()),
                "chat_id": chat_id,
                "text": test_text,
                "vector": embeddings[0],
                "group_id": -1,
            }
            
            inserted_ids = memory._store.insert([entity])
            original_id = inserted_ids[0]
            print(f"插入的记忆ID: {original_id}")
            
            # 搜索
            results = memory.search("记忆系统", user_id)
            
            # 验证ID匹配
            found_ids = [mem.id for mem in results["episodic"]]
            print(f"搜索返回的ID: {found_ids}")
            
            assert original_id in found_ids, f"原始ID {original_id} 应该在搜索结果中"
            
            # 验证text也匹配
            for mem in results["episodic"]:
                if mem.id == original_id:
                    assert mem.text == test_text, "text应该完全匹配"
                    print(f"✅ ID和text都正确匹配")
            
            print("✅ 步骤1-3测试通过：search返回正确的记忆ID")
            
        finally:
            memory._store.drop_collection()


if __name__ == "__main__":
    test = TestStep1Search()
    test.test_search_returns_episodic_memories()
    test.test_search_returns_memory_with_correct_id()
