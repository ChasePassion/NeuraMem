# 叙事记忆处理实现详解

## 概述

本文档详细说明 NeuraMem 系统中叙事记忆（Narrative Memory）的完整实现，包括从上游 API 入口到底层存储的全链路代码流程。

## 核心概念

### 叙事记忆（Narrative Memory）

叙事记忆是将多个相关的情景记忆（Episodic Memory）组织在一起的能力。系统通过向量相似度将情景记忆分组，形成叙事组（Narrative Group），使得在检索时能够一次性返回整个相关事件序列。

### 记忆类型

- **Episodic Memory（情景记忆）**：具体的对话事件和上下文信息
- **Semantic Memory（语义记忆）**：从情景记忆中提取的长期事实和知识
- **Narrative Group（叙事组）**：一组相关的情景记忆集合

## 完整数据流

### 1. 上游入口 - 对话请求

**文件**: `src/api/routers/chat.py`

```python
@router.post("/v1/chat")
async def chat_stream(
    request: ChatRequest,
    memory: Memory = Depends(get_memory_system)
) -> StreamingResponse:
    """SSE streaming chat endpoint with memory-augmented responses.
    
    Flow:
    1. Search relevant memories for user
    2. Build context with memories and history
    3. Stream LLM response via SSE
    4. Async trigger memory management after completion
    """
    async def event_generator():
        accumulated_response = ""
        
        try:
            # 步骤 1: 搜索相关记忆
            relevant_memories = await asyncio.to_thread(
                memory.search,
                query=request.message,
                user_id=request.user_id
            )
            
            # 步骤 2: 构建上下文
            context = _build_context_with_memories(
                request.message,
                relevant_memories,
                request.history
            )
            
            # 步骤 3: 流式返回 LLM 响应
            async for chunk in memory._llm_client.chat_stream_async(
                system_prompt,
                request.message
            ):
                accumulated_response += chunk
                yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
            
            # 步骤 4: 发送完成事件
            yield f"data: {json.dumps({'type': 'done', 'full_content': accumulated_response})}\n\n"
            
            # 步骤 5: 异步触发记忆管理
            asyncio.create_task(
                _manage_memory_background(
                    memory=memory,
                    user_message=request.message,
                    assistant_message=accumulated_response,
                    user_id=request.user_id,
                    chat_id=request.chat_id
                )
            )
```

**关键点**:
- 使用 SSE (Server-Sent Events) 流式返回响应
- 记忆检索在主线程同步执行
- LLM 流式生成后，异步触发记忆管理

### 2. 记忆检索与叙事组扩展

**文件**: `src/memory_system/memory.py`

```python
def search(
    self,
    query: str,
    user_id: str
) -> Dict[str, List[MemoryRecord]]:
    """Search memories for a user with narrative group expansion."""
    
    # 生成查询向量
    query_vectors = self._embedding_client.encode([query])
    query_vector = query_vectors[0]
    q = normalize(np.array(query_vector))
    
    # 获取语义记忆
    if self._config.use_all_semantic:
        semantic_filter = f'user_id == "{user_id}" and memory_type == "semantic"'
        semantic_records = self._store.query(filter_expr=semantic_filter, limit=1000)
        semantic_memories = [self._hit_to_memory_record(hit) for hit in semantic_records]
    else:
        semantic_filter = f'user_id == "{user_id}" and memory_type == "semantic"'
        semantic_results = self._store.search(
            vectors=[query_vector],
            filter_expr=semantic_filter,
            limit=self._config.k_semantic
        )
        semantic_memories = [self._hit_to_memory_record(hit) 
                           for hit in semantic_results[0]]
    
    # 步骤 1: 向量检索情景记忆种子
    episodic_filter = f'user_id == "{user_id}" and memory_type == "episodic"'
    episodic_results = self._store.search(
        vectors=[q.tolist()],
        filter_expr=episodic_filter,
        limit=self._config.k_episodic,
        output_fields=["id", "group_id", "user_id", "memory_type", "ts", "chat_id", "text"],
    )
    
    seeds = episodic_results[0] if episodic_results and episodic_results[0] else []
    
    if not seeds:
        # 无种子，直接返回空情景记忆 + 语义记忆
        return {"episodic": [], "semantic": semantic_memories}
    
    # 步骤 2: 根据种子的 group_id 决定扩展哪些组
    expansion_group_ids = set()
    
    for hit in seeds:
        g_id = hit.get("group_id")
        # 只有 group_id >= 0 的才扩展，-1 表示未分组，不扩展
        if g_id == -1:
            continue
        expansion_group_ids.add(g_id)
    
    # 步骤 3: 拉出这些扩展组的所有成员
    expanded_member_ids = set()
    
    for g_id in expansion_group_ids:
        members_res = self._store.query(
            filter_expr=f"group_id == {g_id} and user_id == '{user_id}'",
            output_fields=["id"],
        )
        member_ids = [row["id"] for row in members_res]
        # 不限制每组的记忆数
        expanded_member_ids.update(member_ids)
    
    # 步骤 4: 合并种子 + 扩展成员 → 去重 → 拉完整内容
    seed_ids = {hit["id"] for hit in seeds}
    all_ids = seed_ids | expanded_member_ids
    
    if not all_ids:
        final_memories = []
    else:
        id_list = list(all_ids)
        
        mem_res = self._store.query(
            filter_expr=f"id in {id_list} and user_id == '{user_id}'",
            output_fields=["id", "user_id", "memory_type", "ts", "chat_id", "text", "group_id"],
        )
        
        id2row = {row["id"]: row for row in mem_res}
        
        final_memories = []
        
        # 先放种子，保证它们在 prompt 里靠前（按相似度排序）
        for hit in seeds:
            row = id2row.get(hit["id"])
            if row:
                final_memories.append(self._hit_to_memory_record(row))
        
        # 再放扩展成员（去掉已经是种子的）
        for mid in expanded_member_ids:
            if mid in seed_ids:
                continue
            row = id2row.get(mid)
            if row:
                final_memories.append(self._hit_to_memory_record(row))
    
    return {
        "episodic": final_memories,  # 种子 + 叙事组扩展
        "semantic": semantic_memories
    }
```

**关键点**:
1. **向量检索种子**: 先检索 top-k 条最相关的情景记忆作为种子
2. **组扩展**: 根据种子的 `group_id` 找出所有相关组
3. **成员拉取**: 获取这些组的所有成员记忆
4. **合并去重**: 合并种子和扩展成员，去除重复，保留完整内容
5. **排序**: 种子按相似度排序在前，扩展成员在后

### 3. 记忆管理（CRUD）

**文件**: `src/memory_system/memory.py`

```python
async def manage_async(
    self,
    user_text: str,
    assistant_text: str,
    user_id: str,
    chat_id: str,
    metadata: Optional[Dict[str, Any]] = None
) -> List[int]:
    """Manage memories with CRUD operations based on conversation."""
    
    # 查询用户所有情景记忆
    episodic_filter = f'user_id == "{user_id}" and memory_type == "episodic"'
    episodic_memories = await asyncio.to_thread(
        self._store.query, filter_expr=episodic_filter, limit=10000
    )
    
    # 调用记忆管理器进行 LLM 决策
    result = await asyncio.to_thread(
        self._memory_manager.manage_memories,
        user_text=user_text,
        assistant_text=assistant_text,
        episodic_memories=episodic_memories
    )
    
    # 执行 CRUD 操作
    added_ids = []
    
    # 处理删除操作
    for op in result.operations:
        if op.operation_type == "delete":
            await asyncio.to_thread(self.delete, op.memory_id, user_id)
    
    # 处理更新操作
    for op in result.operations:
        if op.operation_type == "update":
            await asyncio.to_thread(self.update, op.memory_id, {"text": op.text}, user_id)
    
    # 处理添加操作
    add_operations = [op for op in result.operations if op.operation_type == "add"]
    if add_operations:
        add_texts = [op.text for op in add_operations]
        
        # Embedding + Milvus insert 在线程池中执行
        embeddings = await asyncio.to_thread(self._embedding_client.encode, add_texts)
        
        current_ts = int(time.time())
        entities = []
        
        for i, text in enumerate(add_texts):
            entity = {
                "user_id": user_id,
                "memory_type": "episodic",
                "ts": current_ts,
                "chat_id": chat_id,
                "text": text,
                "vector": embeddings[i],
                "group_id": -1,  # 新记忆默认未分组
            }
            entities.append(entity)
        
        added_ids = await asyncio.to_thread(self._store.insert, entities)
    
    return added_ids
```

**文件**: `src/memory_system/processors/memory_manager.py`

```python
class EpisodicMemoryManager:
    """Manages episodic memories with CRUD operations using LLM intelligence."""
    
    def manage_memories(
        self, 
        user_text: str, 
        assistant_text: str, 
        episodic_memories: List[Dict[str, Any]]
    ) -> MemoryManagementResult:
        """Manage memories based on current conversation and existing memories."""
        
        # 构建完整对话轮次
        current_turn = {
            "user": user_text,
            "assistant": assistant_text
        }
        
        # 调用 LLM 进行 CRUD 决策
        input_data = {
            "current_turn": current_turn,
            "episodic_memories": [{"id": mem["id"], "text": mem["text"]} 
                                for mem in episodic_memories]
        }
        
        llm_response = self._llm.chat_json(
            system_prompt=self._prompt,
            user_message=json.dumps(input_data, ensure_ascii=False),
            default={"add": [], "update": [], "delete": []}
        )
        
        # 提取解析后的数据
        response = llm_response["parsed_data"]
        
        # 转换为操作列表
        operations = []
        
        # 处理添加操作
        for add_op in response.get("add", []):
            operations.append(MemoryOperation("add", text=add_op["text"]))
        
        # 处理更新操作
        for update_op in response.get("update", []):
            operations.append(MemoryOperation(
                "update", 
                memory_id=update_op["id"],
                old_text=update_op["old_text"], 
                text=update_op["new_text"]
            ))
        
        # 处理删除操作
        for delete_op in response.get("delete", []):
            operations.append(MemoryOperation("delete", memory_id=delete_op["id"]))
        
        return MemoryManagementResult(operations)
```

**关键点**:
- LLM 决策需要添加、更新或删除哪些情景记忆
- 新增记忆的 `group_id` 默认为 -1（未分组）
- 记忆管理在后台异步执行，不影响主响应流

### 4. 记忆使用判断

**文件**: `src/memory_system/processors/memory_usage_judge.py`

```python
class MemoryUsageJudge:
    """Judge which episodic memories were actually used in generating a response."""
    
    def judge_used_memories(
        self,
        episodic_memories: List[str],
        last_user: str,
        last_assistant: str
    ) -> List[str]:
        """Judge which episodic memories were actually used in the final reply.
        
        Args:
            episodic_memories: List of episodic memory texts that were retrieved
            last_user: The most recent user message
            last_assistant: The assistant's complete reply to that message
            
        Returns:
            List of episodic memory texts that were actually used
        """
        if not episodic_memories:
            return []
        
        try:
            input_data = {
                "episodic_memories": episodic_memories,
                "last_user": last_user,
                "last_assistant": last_assistant
            }
            
            # 调用 LLM 判断哪些记忆被实际使用
            response = self._llm_client.chat_json(
                system_prompt=MEMORY_RELEVANCE_FILTER_PROMPT,
                user_message=json.dumps(input_data, ensure_ascii=False),
                default={"used_episodic_memories": []}
            )
            
            parsed_data = response.get("parsed_data", {})
            used_memories = parsed_data.get("used_episodic_memories", [])
            
            return used_memories
            
        except Exception as e:
            logger.warning(f"Failed to judge memory usage: {e}")
            # 保守回退：假设没有记忆被使用
            return []
```

**关键点**:
- 只分析最近一轮对话（用户问题 + 助手回答）
- LLM 判断哪些记忆实际上被用于生成回答
- 只有实际使用的记忆才会被分配到叙事组

### 5. 叙事组分配

**文件**: `src/memory_system/processors/narrative_memory_manager.py`

```python
class NarrativeMemoryManager:
    """Manager for narrative memory grouping operations."""
    
    def assign_to_narrative_group(self, memory_ids: List[int], user_id: str) -> Dict[int, int]:
        """将被使用的情景记忆分配到叙事组。
        
        Args:
            memory_ids: 被MemoryUsageJudge判断为实际使用的情景记忆ID列表
            user_id: 用户标识
            
        Returns:
            Dict[int, int] - memory_id到group_id的映射
        """
        if not memory_ids:
            return {}
        
        results = {}
        created_groups = 0
        reused_groups = 0
        
        # 确保组集合存在
        self._store.create_groups_collection(user_id, dim=self._config.embedding_dim)
        
        for memory_id in memory_ids:
            try:
                # 步骤 1: 检查是否已分组
                mem_res = self._store.query(
                    filter_expr=f"id == {memory_id} and user_id == '{user_id}'",
                    output_fields=["id", "group_id", "vector"],
                )
                
                if not mem_res:
                    logger.warning(f"Memory {memory_id} not found, skipping")
                    continue
                
                current_group_id = mem_res[0]["group_id"]
                v_mem = np.array(mem_res[0]["vector"])
                v_mem = normalize(v_mem)
                
                # 如果已经分组，跳过
                if current_group_id != -1:
                    logger.debug(f"Memory {memory_id} already in group {current_group_id}")
                    results[memory_id] = current_group_id
                    continue
                
                # 步骤 2: 在 groups 上做 ANN 搜索，找最相似组
                group_hits = self._store.search_groups(
                    user_id=user_id,
                    vector=v_mem.tolist(),
                    limit=1
                )
                
                best_group = group_hits[0] if group_hits else None
                
                # 步骤 3: 阈值判断：新建组 or 加入已有组
                threshold = self._config.narrative_similarity_threshold
                
                if best_group is None or best_group["sim"] < threshold:
                    # 步骤 3.1: 新建组
                    group_id = self._store.insert_group(
                        user_id=user_id,
                        centroid_vector=v_mem.tolist(),
                        size=1
                    )
                    
                    if group_id is None:
                        logger.error(f"Failed to create group for memory {memory_id}")
                        continue
                    
                    # 更新 memory 的 group_id
                    self._store.update_memory_group_id(memory_id, group_id, user_id)
                    
                    logger.info(f"Created new group {group_id} for memory {memory_id}")
                    results[memory_id] = group_id
                    created_groups += 1
                    
                else:
                    # 步骤 3.2: 加入已有组（重算中心）
                    group_id = best_group["group_id"]
                    
                    if group_id is None:
                        logger.error(f"Invalid group_id (None) from search_groups for memory {memory_id}")
                        continue
                    
                    # 1) 更新 memories.group_id
                    self._store.update_memory_group_id(memory_id, group_id, user_id)
                    
                    # 2) 重算这个组的 centroid_vector & size（精确版）
                    members_res = self._store.query(
                        filter_expr=f"group_id == {group_id} and user_id == '{user_id}'",
                        output_fields=["id", "vector"],
                    )
                    vectors = [row["vector"] for row in members_res]
                    size = len(vectors)
                    
                    if vectors:
                        new_centroid = normalize(np.mean(np.array(vectors), axis=0))
                        self._store.update_group(
                            user_id=user_id,
                            group_id=group_id,
                            centroid_vector=new_centroid.tolist(),
                            size=size
                        )
                    
                    logger.info(f"Added memory {memory_id} to existing group {group_id} (size: {size})")
                    results[memory_id] = group_id
                    reused_groups += 1
                    
            except Exception as e:
                logger.error(f"Failed to assign memory {memory_id} to narrative group: {e}")
                continue
        
        return results
```

**关键点**:
1. **已分组检查**: 如果记忆已有 `group_id`，跳过
2. **相似组搜索**: 在叙事组集合中搜索最相似的组
3. **阈值判断**:
   - 相似度低于阈值 → 新建组
   - 相似度高于阈值 → 加入已有组
4. **中心更新**: 加入已有组后，重新计算组的中心向量和大小

### 6. 记忆删除时的叙事组清理

**文件**: `src/memory_system/processors/narrative_memory_manager.py`

```python
def delete_memory_from_group(self, memory_id: int, user_id: str) -> None:
    """删除记忆时同步更新叙事组。"""
    try:
        # 步骤 1: 查出 group_id
        res = self._store.query(
            filter_expr=f"id == {memory_id} and user_id == '{user_id}'",
            output_fields=["group_id"],
        )
        
        if not res:
            logger.warning(f"Memory {memory_id} not found for group cleanup")
            return
        
        group_id = res[0]["group_id"]
        
        # 步骤 2: 如有必要，更新或删除组
        if group_id != -1:
            members_res = self._store.query(
                filter_expr=f"group_id == {group_id} and user_id == '{user_id}'",
                output_fields=["id", "vector"],
            )
            n = len(members_res)
            
            if n == 0:
                # 该组已经空了，删除组
                self._store.delete_group(user_id, group_id)
                logger.info(f"Deleted empty group {group_id}")
            else:
                vectors = [row["vector"] for row in members_res]
                if vectors:
                    new_centroid = normalize(np.mean(np.array(vectors), axis=0))
                    self._store.update_group(
                        user_id=user_id,
                        group_id=group_id,
                        centroid_vector=new_centroid.tolist(),
                        size=n
                    )
                    logger.info(f"Updated group {group_id} centroid (size: {n})")
                    
    except Exception as e:
        logger.error(f"Failed to cleanup group for memory {memory_id}: {e}")
```

**关键点**:
- 删除记忆前先检查其所属组
- 如果组为空（无成员），删除该组
- 如果组非空，重新计算中心向量

### 7. 底层存储层 - Milvus

**文件**: `src/memory_system/clients/milvus_store.py`

#### 主记忆集合 Schema

```python
SCHEMA_FIELDS = [
    ("id", DataType.INT64, {"is_primary": True, "auto_id": True}),
    ("user_id", DataType.VARCHAR, {"max_length": 128}),
    ("memory_type", DataType.VARCHAR, {"max_length": 32}),  # "episodic" or "semantic"
    ("ts", DataType.INT64, {}),  # Unix timestamp
    ("chat_id", DataType.VARCHAR, {"max_length": 128}),
    ("text", DataType.VARCHAR, {"max_length": 65535}),
    ("vector", DataType.FLOAT_VECTOR, {"dim": 2560}),
    ("group_id", DataType.INT64, {"default_value": -1}),  # 叙事组 ID
]
```

#### 叙事组集合 Schema

```python
GROUP_SCHEMA_FIELDS = [
    ("group_id", DataType.INT64, {"is_primary": True, "auto_id": True}),
    ("user_id", DataType.VARCHAR, {"max_length": 128}),
    ("centroid_vector", DataType.FLOAT_VECTOR, {"dim": 2560}),  # 组中心向量
    ("size", DataType.INT64, {}),  # 当前组内成员数量
]
```

#### 关键操作

**插入记忆**:
```python
def insert(self, entities: List[Dict[str, Any]]) -> List[int]:
    """Insert memory records."""
    # 确保 group_id 总是存在
    for ent in entities:
        ent.setdefault("group_id", -1)
    
    result = self._client.insert(
        collection_name=self._collection_name,
        data=entities
    )
    
    ids = result.get("ids", [])
    return list(ids) if ids else []
```

**搜索叙事组**:
```python
def search_groups(
    self,
    user_id: str,
    vector: List[float],
    limit: int = 1
) -> List[Dict[str, Any]]:
    """Search for similar groups by centroid vector."""
    groups_collection = self._get_groups_collection_name(user_id)
    
    if not self._client.has_collection(groups_collection):
        return []
    
    results = self._client.search(
        collection_name=groups_collection,
        data=[vector],
        anns_field="centroid_vector",
        limit=limit,
        search_params={"metric_type": "IP", "params": {"nprobe": 10}},
        filter=f"user_id == '{user_id}'",
        output_fields=["group_id", "size"],
    )
    
    hits = results[0] if results else []
    groups = []
    for hit in hits:
        entity = hit.get("entity", {})
        group_id = entity.get("group_id") or hit.get("id")
        groups.append({
            "group_id": group_id,
            "sim": hit.get("distance", 0),  # 对于 normalized vectors，这是余弦相似度
            "size": entity.get("size", 0),
        })
    
    return groups
```

**插入新组**:
```python
def insert_group(
    self,
    user_id: str,
    centroid_vector: List[float],
    size: int = 1
) -> Optional[int]:
    """Insert a new group."""
    groups_collection = self._get_groups_collection_name(user_id)
    
    # 确保集合存在
    self.create_groups_collection(user_id, dim=len(centroid_vector))
    
    result = self._client.insert(
        collection_name=groups_collection,
        data=[{
            "user_id": user_id,
            "centroid_vector": centroid_vector,
            "size": size,
        }]
    )
    
    primary_keys = result.get("ids", []) or result.get("primary_keys", [])
    group_id = primary_keys[0] if primary_keys else None
    
    return group_id
```

**更新组**:
```python
def update_group(
    self,
    user_id: str,
    group_id: int,
    centroid_vector: Optional[List[float]] = None,
    size: Optional[int] = None
) -> bool:
    """Update a group's centroid and/or size."""
    groups_collection = self._get_groups_collection_name(user_id)
    
    if not self._client.has_collection(groups_collection):
        return False
    
    try:
        # 获取现有记录
        existing = self._client.query(
            collection_name=groups_collection,
            filter=f"group_id == {group_id}",
            output_fields=["*"]
        )
        
        if not existing:
            return False
        
        record = existing[0].copy()
        
        if centroid_vector is not None:
            record["centroid_vector"] = centroid_vector
        if size is not None:
            record["size"] = size
        
        self._client.upsert(
            collection_name=groups_collection,
            data=[record]
        )
        
        return True
        
    except Exception as e:
        logger.warning(f"Failed to update group {group_id}: {e}")
        return False
```

**更新记忆的 group_id**:
```python
def update_memory_group_id(self, memory_id: int, group_id: int, user_id: str) -> bool:
    """Update a memory's group_id field."""
    try:
        # 获取现有记录
        existing = self._client.query(
            collection_name=self._collection_name,
            filter=f"id == {memory_id} and user_id == '{user_id}'",
            output_fields=["*"]
        )
        
        if not existing:
            return False
        
        record = existing[0].copy()
        record["group_id"] = group_id
        
        self._client.upsert(
            collection_name=self._collection_name,
            data=[record]
        )
        
        return True
        
    except Exception as e:
        logger.warning(f"Failed to update memory {memory_id} group_id: {e}")
        return False
```

**删除组**:
```python
def delete_group(self, user_id: str, group_id: int) -> bool:
    """Delete a group."""
    groups_collection = self._get_groups_collection_name(user_id)
    
    if not self._client.has_collection(groups_collection):
        return False
    
    try:
        self._client.delete(
            collection_name=groups_collection,
            filter=f"group_id == {group_id} and user_id == '{user_id}'"
        )
        return True
    except Exception as e:
        logger.warning(f"Failed to delete group {group_id}: {e}")
        return False
```

### 8. 语义记忆提取（可选）

**文件**: `src/memory_system/memory.py`

```python
def consolidate(self, user_id: Optional[str] = None) -> ConsolidationStats:
    """Run consolidation process for memories.
    
    Performs batch pattern merging: analyzes multiple episodic memories together
    to extract stable, long-term semantic facts.
    """
    stats = ConsolidationStats()
    
    # 查询情景记忆
    if user_id:
        episodic_filter = f'user_id == "{user_id}" and memory_type == "episodic"'
        semantic_filter = f'user_id == "{user_id}" and memory_type == "semantic"'
    else:
        episodic_filter = 'memory_type == "episodic"'
        semantic_filter = 'memory_type == "semantic"'
    
    episodic_memories = self._store.query(filter_expr=episodic_filter, limit=1000)
    semantic_memories = self._store.query(filter_expr=semantic_filter, limit=1000)
    
    stats.memories_processed = len(episodic_memories)
    
    # 批量处理数据
    episodic_texts = [mem.get("text", "") for mem in episodic_memories]
    existing_semantic_texts = [mem.get("text", "") for mem in semantic_memories]
    
    consolidation_data = {
        "episodic_texts": episodic_texts,
        "existing_semantic_texts": existing_semantic_texts
    }
    
    # 调用批量模式合并
    extraction = self._semantic_writer.extract(consolidation_data)
    
    # 创建新的语义记忆
    if extraction.write_semantic and extraction.facts:
        source_memory = episodic_memories[0] if episodic_memories else {}
        self._create_semantic_memories(source_memory, extraction.facts)
        stats.semantic_created += len(extraction.facts)
    
    return stats
```

**文件**: `src/memory_system/processors/semantic_writer.py`

```python
class SemanticWriter:
    """Semantic Memory Writer processor.
    
    Performs pattern merging: analyzes multiple episodic memories together
    to extract stable, long-term facts that should be promoted to semantic memory.
    """
    
    def extract(self, consolidation_data: Dict[str, List[str]]) -> SemanticExtraction:
        """Extract semantic facts from batch of episodic memories.
        
        Args:
            consolidation_data: Dictionary containing:
                - episodic_texts: List of text content from episodic memories
                - existing_semantic_texts: List of text content from existing semantic memories
            
        Returns:
            SemanticExtraction with write_semantic flag and extracted facts
        """
        user_message = json.dumps(consolidation_data, ensure_ascii=False)
        
        default_response = {
            "write_semantic": False,
            "facts": []
        }
        
        # 调用 LLM 进行批量提取
        result = self._llm.chat_json(
            system_prompt=self._prompt,
            user_message=user_message,
            default=default_response
        )
        
        parsed = result.get("parsed_data", {})
        write_semantic = parsed.get("write_semantic", False)
        raw_facts = parsed.get("facts", [])
        
        # 确保 facts 是字符串列表
        facts = [str(f) for f in raw_facts if f]
        
        return SemanticExtraction(
            write_semantic=write_semantic,
            facts=facts
        )
```

## 完整流程图

```
用户请求 (POST /v1/chat)
    │
    ├─→ 检索记忆 (Memory.search)
    │       ├─→ 生成查询向量
    │       ├─→ 向量检索情景记忆种子 (k_episodic)
    │       ├─→ 获取语义记忆 (k_semantic 或全部)
    │       ├─→ 叙事组扩展
    │       │   ├─→ 提取种子的 group_id (排除 -1)
    │       │   ├─→ 查询这些组的所有成员
    │       │   └─→ 合并种子 + 扩展成员 → 去重 → 拉完整内容
    │       └─→ 返回 {"episodic": [...], "semantic": [...]}
    │
    ├─→ 构建上下文（记忆 + 历史）
    │
    ├─→ 流式 LLM 响应 (SSE)
    │
    └─→ 异步记忆管理 (Memory.manage_async)
            ├─→ 查询所有情景记忆
            ├─→ LLM CRUD 决策 (EpisodicMemoryManager)
            │       ├─→ 决定添加哪些新记忆
            │       ├─→ 决定更新哪些记忆
            │       └─→ 决定删除哪些记忆
            │
            ├─→ 执行 CRUD 操作
            │       ├─→ 删除记忆 (触发叙事组清理)
            │       │       └─→ 检查组状态
            │       │           ├─→ 空组 → 删除组
            │       │           └─→ 非空 → 重算中心
            │       │
            │       ├─→ 更新记忆
            │       │       └─→ 删除旧记忆 + 插入新记忆 (group_id = -1)
            │       │
            │       └─→ 添加记忆
            │               └─→ 生成 embedding + 插入 (group_id = -1)
            │
            └─→ [可选] 记忆使用判断 + 叙事组分配
                    ├─→ MemoryUsageJudge.judge_used_memories
                    │       └─→ 判断哪些记忆被实际使用
                    │
                    └─→ NarrativeMemoryManager.assign_to_narrative_group
                            ├─→ 遍历被使用的记忆
                            │       ├─→ 检查是否已分组
                            │       ├─→ 搜索相似叙事组
                            │       ├─→ 阈值判断
                            │       │   ├─→ 低于阈值 → 新建组
                            │       │   │       ├─→ insert_group
                            │       │   │       └─→ update_memory_group_id
                            │       │   │
                            │       │   └─→ 高于阈值 → 加入已有组
                            │       │               ├─→ update_memory_group_id
                            │       │               ├─→ 查询组成员
                            │       │               ├─→ 重算中心向量
                            │       │               └─→ update_group
                            │       │
                            │       └─→ 返回 memory_id → group_id 映射
                            │
                            └─→ [可选] 语义记忆提取
                                    └─→ SemanticWriter.extract
                                            └─→ 批量模式合并，提取长期事实
```

## 关键设计决策

### 1. 为什么需要叙事记忆？

- **上下文连贯性**: 单个情景记忆可能只包含部分信息，通过叙事组可以返回完整的事件序列
- **避免信息碎片化**: 相关的记忆被组织在一起，检索时能获得更全面的上下文
- **模仿人类记忆**: 人类记忆是以事件序列而非孤立事实存储的

### 2. 为什么只对"实际使用"的记忆分组？

- **减少噪声**: 只有被 LLM 真正使用的记忆才具有实际价值
- **提高相关性**: 避免将无关的记忆误分组
- **资源效率**: 减少不必要的分组计算

### 3. 为什么使用阈值判断新建/加入？

- **灵活性**: 允许新主题的创建（新建组）
- **连续性**: 允许相似主题的延续（加入已有组）
- **可调性**: 通过阈值控制组的粒度

### 4. 为什么删除记忆时要处理叙事组？

- **数据一致性**: 删除成员后组状态可能需要更新
- **资源清理**: 空组应该被删除以节省存储
- **准确性**: 组中心向量需要重新计算以反映当前成员

### 5. 为什么叙事组集合按用户隔离？

- **隐私隔离**: 不同用户的记忆不应该相互影响
- **性能优化**: 搜索时只需在一个用户的组集合中查找
- **独立性**: 每个用户的叙事结构独立演化

## 配置参数

**文件**: `src/memory_system/config.py`

```python
@dataclass
class MemoryConfig:
    # 叙事记忆相关
    narrative_similarity_threshold: float = 0.7  # 叙事组相似度阈值
    
    # 检索相关
    k_episodic: int = 5        # 检索的情景记忆种子数量
    k_semantic: int = 3        # 检索的语义记忆数量
    use_all_semantic: bool = False  # 是否使用所有语义记忆
    
    # 其他配置
    embedding_dim: int = 2560  # 向量维度
    # ... 其他配置
```

## 性能考虑

1. **向量检索**: 使用 Milvus 的 ANN (Approximate Nearest Neighbor) 索引，性能优秀
2. **组扩展**: 通过 `group_id` 过滤直接查询，不需要额外的向量计算
3. **批量操作**: 插入、更新、删除都支持批量操作
4. **异步处理**: 记忆管理在后台异步执行，不阻塞主响应
5. **索引优化**: 
   - 主记忆集合对 `vector` 建立了 COSINE 索引
   - 叙事组集合对 `centroid_vector` 建立了 IP (Inner Product) 索引

## 扩展性

1. **多组支持**: 系统可以轻松扩展到支持更复杂的记忆分组策略
2. **动态阈值**: 可以根据记忆使用情况动态调整相似度阈值
3. **组层次结构**: 未来可以支持组的嵌套或层次结构
4. **其他记忆类型**: 框架支持添加新的记忆类型（如程序记忆、情绪记忆等）

## 测试建议

1. **单元测试**:
   - `NarrativeMemoryManager.assign_to_narrative_group` 的各种情况
   - `NarrativeMemoryManager.delete_memory_from_group` 的清理逻辑
   - `Memory.search` 的叙事组扩展逻辑

2. **集成测试**:
   - 完整的对话流程：请求 → 检索 → 响应 → 记忆管理 → 分组
   - 删除记忆后的叙事组状态
   - 语义记忆提取流程

3. **性能测试**:
   - 大量记忆下的检索性能
   - 大量叙事组下的搜索性能
   - 并发情况下的数据一致性

## 总结

NeuraMem 的叙事记忆系统通过以下机制实现：

1. **检索时扩展**: 在检索时根据种子记忆的 `group_id` 扩展到整个组
2. **使用时分组**: 只有被实际使用的记忆才会被分配到叙事组
3. **相似度驱动**: 使用向量相似度和阈值判断新建/加入已有组
4. **自动维护**: 删除记忆时自动处理叙事组的清理和更新
5. **双层存储**: 主记忆集合存储所有记忆，叙事组集合存储组的元数据

这个设计既保证了记忆检索的全面性（通过组扩展），又避免了无关记忆的干扰（通过使用判断），是一个平衡了召回率和精度的实用方案。
