# NeuraMem 核心时序图

> **Goal**: 展示三条核心链路——(A) 记忆增强对话完整闭环；(B) search 混合检索+叙事扩展；(C) consolidate 语义提炼。
> **Type**: Dynamic / Runtime 时序图（现状）
> **Date**: 2025 年

## 图 A：/v1/chat 记忆增强对话闭环

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant API as FastAPI<br/>routers/chat.py
    participant M as Memory 门面<br/>memory.py
    participant Emb as EmbeddingClient
    participant Store as MilvusStore
    participant LLM as LLMClient
    participant EMM as EpisodicMemoryManager

    C->>API: POST /v1/chat<br/>{user_id, chat_id, message, history}
    activate API
    API->>M: search(query, user_id)<br/>(asyncio.to_thread)
    activate M
    M->>Emb: encode([query])
    Emb-->>M: query_vector (2560d)
    M->>Store: search 情景记忆 top-k<br/>filter: user_id & episodic
    Store-->>M: 种子 hits (含 group_id)
    M->>Store: query 叙事组扩展成员<br/>group_id in {...}
    Store-->>M: 组内全部成员
    M-->>API: {episodic: 种子+扩展, semantic: 全部}
    deactivate M

    API->>API: _build_context_with_memories<br/>拼记忆+历史+当前消息<br/>(截断前 5 条/类)
    API->>LLM: chat_stream_async<br/>(system=上下文, user=message)
    loop SSE 流式
        LLM-->>API: chunk
        API-->>C: data: {"type":"chunk"}
    end
    API-->>C: data: {"type":"done"}

    Note over API,EMM: 回答完成后 fire-and-forget（不阻塞响应）
    API--)M: manage_async(user_text, assistant_text)<br/>asyncio.create_task 后台
    M->>Store: query 用户全部情景记忆<br/>limit=10000（全量，成本隐患）
    Store-->>M: 全部情景记忆
    M->>EMM: manage_memories<br/>(asyncio.to_thread)
    EMM->>LLM: chat_json<br/>prompt=EPISODIC_MEMORY_MANAGER
    LLM-->>EMM: {"add":[...],"update":[...],"delete":[...]}
    EMM-->>M: MemoryOperation 列表
    M->>M: 执行 delete → update → add
    M->>Emb: encode(新增文本)
    M->>Store: insert(entities, group_id=-1)
    Store-->>M: added_ids
    deactivate API
```

## 图 B：search 内部机制（混合检索 + 叙事组扩展）

```mermaid
sequenceDiagram
    autonumber
    participant Caller as 调用方
    participant M as Memory.search
    participant Emb as EmbeddingClient
    participant Store as MilvusStore

    Caller->>M: search(query, user_id)
    M->>Emb: encode([query])
    Emb-->>M: query_vector
    M->>M: normalize(query_vector)

    par 并行检索
        M->>Store: search 语义记忆<br/>use_all_semantic=true → query 全部<br/>否则向量 top-k
        Store-->>M: semantic_memories
    and
        M->>Store: search 情景记忆种子 top-k<br/>filter: user_id & episodic
        Store-->>M: seeds (带 group_id)
    end

    Note over M: 收集种子中的 group_id<br/>(-1 未分组跳过)
    loop 每个有效 group_id
        M->>Store: query 组内全部成员
        Store-->>M: member_ids
    end

    M->>M: 种子 ∪ 成员 → 去重
    M->>Store: query 完整字段 (id in [...])
    Store-->>M: 完整记录
    M-->>Caller: {episodic: 种子优先排序, semantic}
```

## 图 C：consolidate 语义提炼（巩固流程）

```mermaid
sequenceDiagram
    autonumber
    participant API as FastAPI<br/>routers/memories.py
    participant M as Memory.consolidate
    participant Store as MilvusStore
    participant SW as SemanticWriter
    participant LLM as LLMClient
    participant Emb as EmbeddingClient

    API->>M: consolidate(user_id)<br/>(asyncio.to_thread)
    M->>Store: query 情景记忆 limit=1000
    Store-->>M: episodic_memories
    M->>Store: query 现有语义记忆 limit=1000
    Store-->>M: semantic_memories
    M->>SW: extract({episodic_texts, existing_semantic_texts})
    SW->>LLM: chat_json<br/>prompt=SEMANTIC_MEMORY_WRITER_PROMPT
    Note over LLM: 批量模式识别稳定模式<br/>与现有语义去重<br/>保守原则（证据不足不写）
    LLM-->>SW: {write_semantic, facts}
    SW-->>M: SemanticExtraction
    alt write_semantic and facts
        M->>Emb: encode(facts)
        M->>Store: insert 语义记忆 (group_id=-1)
    else 无新事实
        Note over M: 结束，semantic_created=0
    end
    M-->>API: ConsolidationStats
```

## 图例与阅读说明

- 实线 = 同步调用；虚线 = 异步/后台任务；`Note` = 关键机制说明
- 图 A 是产品闭环：检索 → 流式回答 → 自动写记忆；注意 manage 是全量拉取（limit=10000）
- 图 B 展示了叙事扩展的"种子 → 组 → 全成员"三步
- 图 C 展示了巩固阶段的批量模式提炼
