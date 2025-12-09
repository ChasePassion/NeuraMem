"""
测试完整流程：模拟整个叙事记忆分组流程

这个测试模拟demo/app.py中_process_memory_async的完整逻辑，
用于定位问题出在哪个环节。
"""

import pytest
import time
import asyncio
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class MockMemoryRecord:
    """模拟MemoryRecord"""
    id: int
    user_id: str
    memory_type: str
    ts: int
    chat_id: str
    text: str
    distance: float = 0.0


class TestFullFlow:
    """测试完整流程"""
    
    def test_full_flow_with_real_components(self):
        """使用真实组件测试完整流程"""
        from src.memory_system import Memory, MemoryConfig
        
        config = MemoryConfig()
        config.collection_name = f"test_full_flow_{int(time.time())}"
        
        memory = Memory(config)
        user_id = "test_user_full_flow"
        chat_id = "test_chat_1"
        
        try:
            print("="*60)
            print("步骤1: 插入测试记忆")
            print("="*60)
            
            # 插入测试记忆
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
            print(f"插入的记忆ID: {inserted_ids}")
            print(f"插入的记忆文本: {test_memories}")
            
            print("\n" + "="*60)
            print("步骤2: 检索记忆")
            print("="*60)
            
            # 检索记忆
            query = "你记得我在学什么吗？"
            relevant_memories = memory.search(query, user_id)
            
            print(f"检索到的情景记忆数量: {len(relevant_memories.get('episodic', []))}")
            for mem in relevant_memories.get('episodic', []):
                print(f"  ID={mem.id}, text={mem.text}")
            
            print("\n" + "="*60)
            print("步骤3: 构建上下文")
            print("="*60)
            
            # 构建上下文
            episodic_texts = [mem.text for mem in relevant_memories.get("episodic", [])]
            semantic_texts = [mem.text for mem in relevant_memories.get("semantic", [])]
            
            full_context = f"""Here are the episodic memories:
{chr(10).join([f'{i+1}. {t}' for i, t in enumerate(episodic_texts)])}

Here are the semantic memories:
{chr(10).join([f'{i+1}. {t}' for i, t in enumerate(semantic_texts)]) if semantic_texts else '(No semantic memories)'}

Here are the task:
{query}"""
            
            print(f"构建的上下文:\n{full_context}")
            
            print("\n" + "="*60)
            print("步骤4: 模拟模型回复")
            print("="*60)
            
            # 模拟一个使用了记忆的回复
            final_reply = "是的，我记得你正在学习Python编程！这是一门很实用的编程语言。"
            print(f"模型回复: {final_reply}")
            
            print("\n" + "="*60)
            print("步骤5: 调用MemoryUsageJudge")
            print("="*60)
            
            # 调用judge
            used_episodic_texts = memory._memory_usage_judge.judge_used_memories(
                system_prompt=full_context,
                episodic_memories=episodic_texts,
                semantic_memories=semantic_texts,
                message_history=[],
                final_reply=final_reply
            )
            
            print(f"judge返回的使用记忆文本: {used_episodic_texts}")
            print(f"judge返回的文本数量: {len(used_episodic_texts)}")
            
            print("\n" + "="*60)
            print("步骤6: 文本到ID匹配 (关键步骤!)")
            print("="*60)
            
            # 文本到ID匹配（复制自demo/app.py）
            used_memory_ids = []
            for mem in relevant_memories.get("episodic", []):
                print(f"  检查记忆: ID={mem.id}, text='{mem.text}'")
                print(f"    是否在used_episodic_texts中: {mem.text in used_episodic_texts}")
                if mem.text in used_episodic_texts:
                    used_memory_ids.append(mem.id)
                    print(f"    ✅ 匹配成功，添加ID {mem.id}")
                else:
                    # 尝试更宽松的匹配
                    for used_text in used_episodic_texts:
                        if mem.text.strip() == used_text.strip():
                            print(f"    ⚠️ strip后匹配成功")
                        elif mem.text in used_text or used_text in mem.text:
                            print(f"    ⚠️ 部分匹配: '{used_text}'")
            
            print(f"\n匹配到的ID列表: {used_memory_ids}")
            
            if not used_memory_ids:
                print("❌ 没有匹配到任何ID！这是问题所在！")
                print("\n详细对比:")
                for i, used_text in enumerate(used_episodic_texts):
                    print(f"  judge返回[{i}]: '{used_text}' (repr: {repr(used_text)})")
                for mem in relevant_memories.get("episodic", []):
                    print(f"  原始记忆[{mem.id}]: '{mem.text}' (repr: {repr(mem.text)})")
            
            print("\n" + "="*60)
            print("步骤7: 调用assign_to_narrative_group")
            print("="*60)
            
            if used_memory_ids:
                result = memory.assign_to_narrative_group(used_memory_ids, user_id)
                print(f"assign_to_narrative_group返回: {result}")
                
                # 验证groups collection
                groups_collection = f"groups_{user_id}"
                if memory._store._client.has_collection(groups_collection):
                    groups = memory._store._client.query(
                        collection_name=groups_collection,
                        filter=f"user_id == '{user_id}'",
                        output_fields=["group_id", "size"]
                    )
                    print(f"groups collection中的组: {groups}")
                    
                    if groups:
                        print("✅ 叙事组创建成功！")
                    else:
                        print("❌ groups collection存在但没有组数据！")
                else:
                    print(f"❌ groups collection '{groups_collection}' 不存在！")
            else:
                print("⚠️ 由于没有匹配到ID，跳过assign_to_narrative_group")
            
            print("\n" + "="*60)
            print("总结")
            print("="*60)
            
            # 最终验证
            groups_collection = f"groups_{user_id}"
            has_groups = False
            if memory._store._client.has_collection(groups_collection):
                groups = memory._store._client.query(
                    collection_name=groups_collection,
                    filter=f"user_id == '{user_id}'",
                    output_fields=["group_id", "size"]
                )
                has_groups = len(groups) > 0
            
            if has_groups:
                print("✅ 完整流程测试通过：叙事组已创建")
            else:
                print("❌ 完整流程测试失败：叙事组未创建")
                print("\n可能的问题点:")
                print("  1. judge_used_memories返回空列表")
                print("  2. 文本匹配失败（文本格式不一致）")
                print("  3. assign_to_narrative_group执行失败")
            
        finally:
            # 清理
            memory._store.drop_collection()
            groups_collection = f"groups_{user_id}"
            if memory._store._client.has_collection(groups_collection):
                memory._store._client.drop_collection(groups_collection)


if __name__ == "__main__":
    test = TestFullFlow()
    test.test_full_flow_with_real_components()
