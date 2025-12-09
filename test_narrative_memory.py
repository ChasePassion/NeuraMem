#!/usr/bin/env python3
"""
测试叙事记忆功能的简单脚本
"""

import sys
import os
import time
import logging

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.memory_system import Memory, MemoryConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_narrative_memory():
    """测试叙事记忆功能"""
    print("🧠 开始测试叙事记忆功能...")
    
    try:
        # 1. 初始化记忆系统
        print("\n1. 初始化记忆系统...")
        config = MemoryConfig()
        config.collection_name = "test_narrative_memories"
        memory = Memory(config)
        print("✅ 记忆系统初始化成功")
        
        user_id = "test_user"
        
        # 2. 清空现有记忆
        print("\n2. 清空现有记忆...")
        memory.reset(user_id)
        print("✅ 记忆已清空")
        
        # 3. 添加一些情景记忆（模拟同一事件的不同方面）
        print("\n3. 添加情景记忆...")
        
        # 事件1：用户去咖啡店
        memories_1 = [
            "昨天下午3点，我在星巴克点了一杯拿铁咖啡",
            "星巴克的拿铁咖啡味道不错，价格是32元",
            "我在星巴克遇到了我的朋友小明，我们聊了工作"
        ]
        
        for i, text in enumerate(memories_1):
            embedding = memory._embedding_client.encode([text])[0]
            entity = {
                "user_id": user_id,
                "memory_type": "episodic",
                "ts": int(time.time()) + i,
                "chat_id": f"chat_1_{i}",
                "text": text,
                "vector": embedding,
                "group_id": -1  # 初始未分组
            }
            memory._store.insert([entity])
        
        print(f"✅ 添加了 {len(memories_1)} 条关于咖啡店的记忆")
        
        # 事件2：用户去图书馆
        memories_2 = [
            "今天上午9点，我去北京大学图书馆学习",
            "图书馆里很安静，适合学习编程",
            "我在图书馆借了一本关于人工智能的书"
        ]
        
        for i, text in enumerate(memories_2):
            embedding = memory._embedding_client.encode([text])[0]
            entity = {
                "user_id": user_id,
                "memory_type": "episodic",
                "ts": int(time.time()) + i + 100,
                "chat_id": f"chat_2_{i}",
                "text": text,
                "vector": embedding,
                "group_id": -1  # 初始未分组
            }
            memory._store.insert([entity])
        
        print(f"✅ 添加了 {len(memories_2)} 条关于图书馆的记忆")
        
        # 4. 测试搜索功能（应该返回种子记忆）
        print("\n4. 测试搜索功能...")
        
        # 搜索咖啡相关记忆
        results_coffee = memory.search("咖啡", user_id)
        print(f"搜索'咖啡'找到 {len(results_coffee['episodic'])} 条情景记忆:")
        for i, mem in enumerate(results_coffee['episodic']):
            print(f"  {i+1}. [ID:{mem.id}] {mem.text}")
        
        # 搜索图书馆相关记忆
        results_library = memory.search("图书馆", user_id)
        print(f"搜索'图书馆'找到 {len(results_library['episodic'])} 条情景记忆:")
        for i, mem in enumerate(results_library['episodic']):
            print(f"  {i+1}. [ID:{mem.id}] {mem.text}")
        
        # 5. 模拟记忆使用判断和叙事分组
        print("\n5. 模拟记忆使用判断和叙事分组...")
        
        # 假设前两条咖啡记忆被使用了
        used_memory_ids = [mem.id for mem in results_coffee['episodic'][:2]]
        print(f"模拟被使用的记忆ID: {used_memory_ids}")
        
        # 执行叙事分组
        group_assignments = memory.assign_to_narrative_group(used_memory_ids, user_id)
        print(f"叙事分组结果: {group_assignments}")
        
        # 6. 再次搜索，测试叙事组扩展
        print("\n6. 测试叙事组扩展...")
        results_coffee_expanded = memory.search("咖啡", user_id)
        print(f"搜索'咖啡'（扩展后）找到 {len(results_coffee_expanded['episodic'])} 条情景记忆:")
        for i, mem in enumerate(results_coffee_expanded['episodic']):
            print(f"  {i+1}. [ID:{mem.id}, Group:{getattr(mem, 'group_id', 'N/A')}] {mem.text}")
        
        # 7. 测试图书馆记忆的分组
        print("\n7. 为图书馆记忆创建叙事组...")
        used_library_ids = [mem.id for mem in results_library['episodic'][:2]]
        library_assignments = memory.assign_to_narrative_group(used_library_ids, user_id)
        print(f"图书馆叙事分组结果: {library_assignments}")
        
        # 8. 最终搜索测试
        print("\n8. 最终搜索测试...")
        final_results = memory.search("学习", user_id)
        print(f"搜索'学习'找到 {len(final_results['episodic'])} 条情景记忆:")
        for i, mem in enumerate(final_results['episodic']):
            print(f"  {i+1}. [ID:{mem.id}] {mem.text}")
        
        print("\n✅ 叙事记忆功能测试完成！")
        
        # 9. 显示统计信息
        print("\n9. 统计信息...")
        all_episodic = memory._store.query(
            filter_expr=f'user_id == "{user_id}" and memory_type == "episodic"',
            output_fields=["id", "group_id", "text"]
        )
        
        grouped_count = len([m for m in all_episodic if m.get("group_id", -1) != -1])
        ungrouped_count = len(all_episodic) - grouped_count
        
        print(f"总情景记忆数: {len(all_episodic)}")
        print(f"已分组记忆数: {grouped_count}")
        print(f"未分组记忆数: {ungrouped_count}")
        
        # 显示分组详情
        groups = {}
        for mem in all_episodic:
            group_id = mem.get("group_id", -1)
            if group_id != -1:
                if group_id not in groups:
                    groups[group_id] = []
                groups[group_id].append(mem)
        
        print(f"\n叙事组详情:")
        for group_id, members in groups.items():
            print(f"  组 {group_id}: {len(members)} 个成员")
            for mem in members:
                print(f"    - [ID:{mem['id']}] {mem['text']}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_narrative_memory()
    if success:
        print("\n🎉 所有测试通过！")
        sys.exit(0)
    else:
        print("\n💥 测试失败！")
        sys.exit(1)
