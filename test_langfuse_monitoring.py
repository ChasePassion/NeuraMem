#!/usr/bin/env python3
"""测试脚本：验证Langfuse监控manage方法模型原始输出的功能"""

import os
import sys
import json
from unittest.mock import Mock, patch

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from memory_system.clients.llm import LLMClient
from memory_system.processors.memory_manager import EpisodicMemoryManager
from memory_system.memory import Memory
from memory_system.config import MemoryConfig

def test_llm_client_chat_json():
    """测试LLMClient的chat_json方法是否返回结构化数据"""
    print("=== 测试LLMClient.chat_json方法 ===")
    
    # 创建模拟的LLM客户端
    with patch('memory_system.clients.llm.OpenAI') as mock_openai:
        # 模拟API响应
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"add": [{"text": "测试记忆"}], "update": [], "delete": []}'
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        # 创建LLMClient实例
        llm_client = LLMClient(api_key="test_key", base_url="http://test.com", model="test-model")
        
        # 调用chat_json方法
        result = llm_client.chat_json(
            system_prompt="测试系统提示",
            user_message="测试用户消息",
            default={"add": [], "update": [], "delete": []}
        )
        
        # 验证返回结构
        print(f"返回结果类型: {type(result)}")
        print(f"包含的键: {list(result.keys())}")
        
        expected_keys = ["parsed_data", "raw_response", "model", "success"]
        for key in expected_keys:
            if key in result:
                print(f"✓ 包含键: {key}")
            else:
                print(f"✗ 缺少键: {key}")
        
        print(f"解析后的数据: {result['parsed_data']}")
        print(f"原始响应: {result['raw_response']}")
        print(f"使用的模型: {result['model']}")
        print(f"是否成功: {result['success']}")
        
        return result

def test_memory_manager_with_langfuse():
    """测试EpisodicMemoryManager是否能正确记录LLM原始输出"""
    print("\n=== 测试EpisodicMemoryManager的Langfuse集成 ===")
    
    # 模拟Langfuse客户端
    with patch('memory_system.processors.memory_manager.get_client') as mock_get_client:
        mock_langfuse_client = Mock()
        mock_get_client.return_value = mock_langfuse_client
        
        # 创建模拟的LLM客户端
        with patch('memory_system.clients.llm.OpenAI') as mock_openai:
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = '{"add": [{"text": "测试记忆"}], "update": [], "delete": []}'
            
            mock_client = Mock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client
            
            # 创建EpisodicMemoryManager实例
            llm_client = LLMClient(api_key="test_key", base_url="http://test.com", model="test-model")
            memory_manager = EpisodicMemoryManager(llm_client)
            
            # 调用manage_memories方法
            result = memory_manager.manage_memories(
                user_text="用户输入",
                assistant_text="助手回复",
                episodic_memories=[]
            )
            
            # 验证Langfuse调用
            print(f"update_current_trace调用次数: {mock_langfuse_client.update_current_trace.call_count}")
            
            # 检查最后一次调用的参数
            if mock_langfuse_client.update_current_trace.call_count > 0:
                last_call = mock_langfuse_client.update_current_trace.call_args
                kwargs = last_call[1] if last_call else {}
                
                print("最后一次update_current_trace调用:")
                print(f"  - output键: {list(kwargs.get('output', {}).keys())}")
                
                output = kwargs.get('output', {})
                if 'llm_raw_output' in output:
                    print("  ✓ 包含llm_raw_output")
                    print(f"  - llm_raw_output长度: {len(output['llm_raw_output'])}")
                else:
                    print("  ✗ 缺少llm_raw_output")
                
                if 'llm_parsed_output' in output:
                    print("  ✓ 包含llm_parsed_output")
                else:
                    print("  ✗ 缺少llm_parsed_output")
                
                if 'llm_model' in output:
                    print("  ✓ 包含llm_model")
                else:
                    print("  ✗ 缺少llm_model")
                
                if 'llm_success' in output:
                    print("  ✓ 包含llm_success")
                else:
                    print("  ✗ 缺少llm_success")
            
            print(f"操作结果: {len(result.operations)} 个操作")
            return result

def test_memory_manage_integration():
    """测试Memory类的manage方法的完整集成"""
    print("\n=== 测试Memory.manage方法的完整集成 ===")
    
    # 模拟所有外部依赖
    with patch('memory_system.memory.get_client') as mock_get_client, \
         patch('memory_system.clients.llm.OpenAI') as mock_openai, \
         patch('memory_system.clients.embedding.OpenAI') as mock_embedding_openai, \
         patch('memory_system.clients.milvus_store.MilvusStore') as mock_milvus:
        
        # 模拟Langfuse客户端
        mock_langfuse_client = Mock()
        mock_get_client.return_value = mock_langfuse_client
        
        # 模拟LLM响应
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"add": [{"text": "测试记忆"}], "update": [], "delete": []}'
        
        mock_llm_client = Mock()
        mock_llm_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_llm_client
        
        # 模拟嵌入客户端
        mock_embedding_client = Mock()
        mock_embedding_client.embeddings.create.return_value = Mock(data=[[0.1, 0.2, 0.3]])
        mock_embedding_openai.return_value = mock_embedding_client
        
        # 模拟Milvus存储
        mock_store = Mock()
        mock_store.query.return_value = []  # 没有现有记忆
        mock_store.insert.return_value = [1]  # 插入一个记忆，返回ID 1
        mock_milvus.return_value = mock_store
        
        # 创建Memory实例
        config = MemoryConfig()
        config.deepseek_api_key = "test_key"
        config.siliconflow_api_key = "test_embedding_key"
        config.milvus_uri = "http://localhost:19530"
        
        memory = Memory(config)
        
        # 调用manage方法
        result = memory.manage(
            user_text="用户输入",
            assistant_text="助手回复",
            user_id="test_user",
            chat_id="test_chat"
        )
        
        # 验证结果
        print(f"manage方法返回的记忆ID: {result}")
        
        # 验证Langfuse调用
        total_calls = mock_langfuse_client.update_current_trace.call_count
        print(f"总update_current_trace调用次数: {total_calls}")
        
        # 检查最后一次调用（应该是Memory.manage的调用）
        if total_calls > 0:
            last_call = mock_langfuse_client.update_current_trace.call_args_list[-1]
            kwargs = last_call[1] if last_call else {}
            
            output = kwargs.get('output', {})
            print("最后一次update_current_trace调用:")
            print(f"  - output键: {list(output.keys())}")
            
            if 'operation_summary' in output:
                print("  ✓ 包含operation_summary")
                summary = output['operation_summary']
                print(f"  - decision_trace_available: {summary.get('decision_trace_available')}")
                print(f"  - added_count: {summary.get('added_count')}")
            else:
                print("  ✗ 缺少operation_summary")
        
        return result

def main():
    """主测试函数"""
    print("开始测试Langfuse监控manage方法模型原始输出的功能\n")
    
    try:
        # 测试1: LLMClient的chat_json方法
        test_llm_client_chat_json()
        
        # 测试2: EpisodicMemoryManager的Langfuse集成
        test_memory_manager_with_langfuse()
        
        # 测试3: Memory.manage方法的完整集成
        test_memory_manage_integration()
        
        print("\n=== 测试完成 ===")
        print("所有测试都已执行，请检查上述输出以验证功能是否正常工作。")
        
    except Exception as e:
        print(f"\n测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
