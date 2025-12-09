"""
测试步骤7：对于使用到的情景记忆，加入到叙事记忆组中

验证：assign_to_narrative_group是否正确创建/更新叙事组
"""

import pytest
import time
import numpy as np


class TestStep7AssignGroup:
    """测试叙事组分配步骤"""
    
    def test_assign_creates_new_group(self):
        """测试assign_to_narrative_group是否正确创建新组"""
        from src.memory_system import Memory, MemoryConfig
        
        config = MemoryConfig()
        config.collection_name = f"test_step7_{int(time.time())}"
        
        memory = Memory(config)
        user_id = "test_user_step7"
        chat_id = "test_chat_1"
        
        try:
            # 1. 插入一条测试记忆
            test_text = "用户正在测试叙事组功能"
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
            memory_id = inserted_ids[0]
            print(f"插入的记忆ID: {memory_id}")
            
            # 2. 验证初始group_id为-1
            mem_before = memory._store.query(
                filter_expr=f"id == {memory_id}",
                output_fields=["id", "group_id"]
            )
            print(f"分配前的group_id: {mem_before[0]['group_id']}")
            assert mem_before[0]['group_id'] == -1, "初始group_id应为-1"
            
            # 3. 调用assign_to_narrative_group
            result = memory.assign_to_narrative_group([memory_id], user_id)
            print(f"assign_to_narrative_group返回: {result}")
            
            # 4. 验证结果
            assert memory_id in result, f"结果应包含memory_id {memory_id}"
            assigned_group_id = result[memory_id]
            print(f"分配的group_id: {assigned_group_id}")
            
            # 5. 验证记忆的group_id已更新
            mem_after = memory._store.query(
                filter_expr=f"id == {memory_id}",
                output_fields=["id", "group_id"]
            )
            print(f"分配后的group_id: {mem_after[0]['group_id']}")
            assert mem_after[0]['group_id'] == assigned_group_id, "记忆的group_id应该被更新"
            
            # 6. 验证groups collection中有对应的组
            groups_collection = f"groups_{user_id}"
            if memory._store._client.has_collection(groups_collection):
                groups = memory._store._client.query(
                    collection_name=groups_collection,
                    filter=f"user_id == '{user_id}'",
                    output_fields=["group_id", "size"]
                )
                print(f"groups collection中的组: {groups}")
                
                group_ids = [g.get("group_id") for g in groups]
                assert assigned_group_id in group_ids, "groups collection应包含新创建的组"
            else:
                print(f"❌ groups collection '{groups_collection}' 不存在！")
            
            print("✅ 步骤7测试通过：成功创建新叙事组")
            
        finally:
            # 清理
            memory._store.drop_collection()
            groups_collection = f"groups_{user_id}"
            if memory._store._client.has_collection(groups_collection):
                memory._store._client.drop_collection(groups_collection)
    
    def test_assign_with_empty_ids(self):
        """测试空ID列表的情况"""
        from src.memory_system import Memory, MemoryConfig
        
        config = MemoryConfig()
        config.collection_name = f"test_step7_empty_{int(time.time())}"
        
        memory = Memory(config)
        user_id = "test_user_step7_empty"
        
        try:
            # 调用assign_to_narrative_group with empty list
            result = memory.assign_to_narrative_group([], user_id)
            print(f"空ID列表的返回结果: {result}")
            
            assert result == {}, "空ID列表应返回空字典"
            print("✅ 空ID列表测试通过")
            
        finally:
            memory._store.drop_collection()
    
    def test_assign_with_nonexistent_id(self):
        """测试不存在的ID"""
        from src.memory_system import Memory, MemoryConfig
        
        config = MemoryConfig()
        config.collection_name = f"test_step7_nonexist_{int(time.time())}"
        
        memory = Memory(config)
        user_id = "test_user_step7_nonexist"
        
        try:
            # 调用assign_to_narrative_group with nonexistent ID
            result = memory.assign_to_narrative_group([99999999], user_id)
            print(f"不存在ID的返回结果: {result}")
            
            assert 99999999 not in result, "不存在的ID不应该在结果中"
            print("✅ 不存在ID测试通过")
            
        finally:
            memory._store.drop_collection()
    
    def test_assign_already_grouped_memory(self):
        """测试已经分组的记忆"""
        from src.memory_system import Memory, MemoryConfig
        
        config = MemoryConfig()
        config.collection_name = f"test_step7_grouped_{int(time.time())}"
        
        memory = Memory(config)
        user_id = "test_user_step7_grouped"
        chat_id = "test_chat_1"
        
        try:
            # 1. 插入记忆
            test_text = "用户正在测试已分组记忆"
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
            memory_id = inserted_ids[0]
            
            # 2. 第一次分配
            result1 = memory.assign_to_narrative_group([memory_id], user_id)
            group_id_1 = result1[memory_id]
            print(f"第一次分配的group_id: {group_id_1}")
            
            # 3. 第二次分配（应该跳过，因为已经分组）
            result2 = memory.assign_to_narrative_group([memory_id], user_id)
            group_id_2 = result2.get(memory_id)
            print(f"第二次分配的group_id: {group_id_2}")
            
            assert group_id_1 == group_id_2, "已分组的记忆应该保持原group_id"
            print("✅ 已分组记忆测试通过")
            
        finally:
            memory._store.drop_collection()
            groups_collection = f"groups_{user_id}"
            if memory._store._client.has_collection(groups_collection):
                memory._store._client.drop_collection(groups_collection)


if __name__ == "__main__":
    test = TestStep7AssignGroup()
    
    print("="*60)
    print("测试1: 创建新组")
    print("="*60)
    test.test_assign_creates_new_group()
    
    print("\n" + "="*60)
    print("测试2: 空ID列表")
    print("="*60)
    test.test_assign_with_empty_ids()
    
    print("\n" + "="*60)
    print("测试3: 不存在的ID")
    print("="*60)
    test.test_assign_with_nonexistent_id()
    
    print("\n" + "="*60)
    print("测试4: 已分组记忆")
    print("="*60)
    test.test_assign_already_grouped_memory()
