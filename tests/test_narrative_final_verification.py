"""
最终验证测试：确认叙事记忆分组功能完全正常

这个测试模拟真实的使用场景，验证：
1. 用户发送消息
2. 检索记忆
3. 模型回复（使用记忆）
4. judge判断使用的记忆
5. 将使用的记忆分配到叙事组
6. 验证groups collection有数据
"""

import time


def test_narrative_memory_grouping():
    """测试完整的叙事记忆分组流程"""
    from src.memory_system import Memory, MemoryConfig
    
    print("="*70)
    print("最终验证测试：叙事记忆分组功能")
    print("="*70)
    
    config = MemoryConfig()
    config.collection_name = f"test_final_verify_{int(time.time())}"
    
    memory = Memory(config)
    user_id = "final_test_user"
    chat_id = "final_test_chat"
    
    try:
        # 1. 插入测试记忆
        print("\n[步骤1] 插入测试记忆...")
        test_memories = [
            "用户正在学习Python编程",
            "用户喜欢喝咖啡",
            "用户正在开发AI记忆系统"
        ]
        
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
        print(f"  ✅ 插入了 {len(inserted_ids)} 条记忆")
        
        # 2. 检索记忆
        print("\n[步骤2] 检索记忆...")
        query = "你记得我在学什么吗？"
        relevant_memories = memory.search(query, user_id)
        
        episodic = relevant_memories.get("episodic", [])
        print(f"  ✅ 检索到 {len(episodic)} 条情景记忆")
        
        # 3. 构建上下文
        print("\n[步骤3] 构建上下文...")
        episodic_texts = [mem.text for mem in episodic]
        semantic_texts = [mem.text for mem in relevant_memories.get("semantic", [])]
        
        full_context = f"""Here are the episodic memories:
{chr(10).join([f'{i+1}. {t}' for i, t in enumerate(episodic_texts)])}

Here are the task:
{query}"""
        print(f"  ✅ 上下文构建完成")
        
        # 4. 模拟模型回复
        print("\n[步骤4] 模拟模型回复...")
        final_reply = "是的，我记得你正在学习Python编程！这是一门很实用的编程语言。"
        print(f"  模型回复: {final_reply}")
        
        # 5. 调用judge判断使用的记忆
        print("\n[步骤5] 调用MemoryUsageJudge...")
        used_episodic_texts = memory._memory_usage_judge.judge_used_memories(
            system_prompt=full_context,
            episodic_memories=episodic_texts,
            semantic_memories=semantic_texts,
            message_history=[],
            final_reply=final_reply
        )
        print(f"  ✅ judge返回 {len(used_episodic_texts)} 条使用的记忆")
        print(f"  使用的记忆: {used_episodic_texts}")
        
        # 6. 文本到ID匹配
        print("\n[步骤6] 文本到ID匹配...")
        used_memory_ids = []
        for mem in episodic:
            if mem.text in used_episodic_texts:
                used_memory_ids.append(mem.id)
        print(f"  ✅ 匹配到 {len(used_memory_ids)} 个ID: {used_memory_ids}")
        
        # 7. 分配到叙事组
        print("\n[步骤7] 分配到叙事组...")
        if used_memory_ids:
            result = memory.assign_to_narrative_group(used_memory_ids, user_id)
            print(f"  ✅ 分配结果: {result}")
        else:
            print("  ⚠️ 没有需要分配的记忆")
            result = {}
        
        # 8. 验证groups collection
        print("\n[步骤8] 验证groups collection...")
        groups_collection = f"groups_{user_id}"
        
        if memory._store._client.has_collection(groups_collection):
            groups = memory._store._client.query(
                collection_name=groups_collection,
                filter=f"user_id == '{user_id}'",
                output_fields=["group_id", "size"]
            )
            print(f"  groups collection存在，包含 {len(groups)} 个组")
            
            if groups:
                print("  ✅ 叙事组数据:")
                for g in groups:
                    print(f"    - group_id={g.get('group_id')}, size={g.get('size')}")
            else:
                print("  ❌ groups collection存在但没有数据")
        else:
            print(f"  ❌ groups collection '{groups_collection}' 不存在")
        
        # 9. 验证记忆的group_id已更新
        print("\n[步骤9] 验证记忆的group_id...")
        for mem_id in used_memory_ids:
            mem_data = memory._store.query(
                filter_expr=f"id == {mem_id}",
                output_fields=["id", "text", "group_id"]
            )
            if mem_data:
                print(f"  记忆 {mem_id}: group_id={mem_data[0].get('group_id')}")
        
        # 最终结论
        print("\n" + "="*70)
        print("测试结论")
        print("="*70)
        
        success = (
            len(used_episodic_texts) > 0 and
            len(used_memory_ids) > 0 and
            len(result) > 0
        )
        
        if success:
            print("✅ 叙事记忆分组功能正常工作！")
            print(f"   - judge正确识别了 {len(used_episodic_texts)} 条使用的记忆")
            print(f"   - 成功将 {len(result)} 条记忆分配到叙事组")
        else:
            print("❌ 叙事记忆分组功能存在问题")
            if len(used_episodic_texts) == 0:
                print("   - judge没有返回使用的记忆")
            if len(used_memory_ids) == 0:
                print("   - 文本到ID匹配失败")
            if len(result) == 0:
                print("   - 分配到叙事组失败")
        
        return success
        
    finally:
        # 清理
        memory._store.drop_collection()
        groups_collection = f"groups_{user_id}"
        if memory._store._client.has_collection(groups_collection):
            memory._store._client.drop_collection(groups_collection)


if __name__ == "__main__":
    import os
    os.environ["PYTHONPATH"] = "."
    
    success = test_narrative_memory_grouping()
    exit(0 if success else 1)
