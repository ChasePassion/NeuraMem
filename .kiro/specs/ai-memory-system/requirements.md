# Requirements Document

## Introduction

本系统是一个基于认知心理学和记忆模型设计的AI长期记忆系统。系统模拟人类大脑的海马体功能，实现情景记忆（episodic）和语义记忆（semantic）的编码、巩固、检索和再巩固。系统使用Milvus向量数据库存储记忆，通过OpenRouter调用大语言模型进行记忆处理，遵循"善于记忆、善于遗忘"的原则，倾向于增加记忆信息而非减少。

## Glossary

- **Memory System**: AI记忆系统，负责管理情景记忆和语义记忆的完整生命周期
- **Episodic Memory**: 情景记忆，记录具体事件+时间+地点+自我经历的记忆类型
- **Semantic Memory**: 语义记忆，存储稳定的、可复用的长期事实（身份、习惯、偏好等）
- **Milvus**: 向量数据库，用于存储和检索记忆的embedding向量及元数据
- **Embedding**: 文本的向量表示，用于语义相似度计算
- **Consolidation**: 巩固阶段，离线处理记忆的合并、分离、语义提取和遗忘
- **Reconsolidation**: 再巩固，检索记忆后根据当前上下文更新记忆内容
- **Pattern Separation**: 图案分离，将相似但不同的记忆在向量空间中拉开距离
- **hit_count**: 记忆被检索命中的次数，用于衡量记忆重要性
- **EpisodicWriteDecider**: 情景记忆写入决策器，判断对话是否需要写入情景记忆
- **SemanticWriter**: 语义记忆写入器，从情景记忆中提取长期事实
- **EpisodicMerger**: 情景记忆合并器，合并描述同一事件的多条记忆
- **EpisodicSeparator**: 情景记忆分离器，区分相似但不同的记忆
- **EpisodicReconsolidator**: 情景记忆再巩固器，根据新上下文更新已有记忆

- **OpenRouter**: 模型调用平台，提供embedding和LLM服务
- **T_merge_high**: 合并阈值上限（0.85），高于此值考虑合并
- **T_amb_low**: 模糊阈值下限（0.65），低于此值视为不同事件

## Requirements

### Requirement 1

**User Story:** As a developer, I want to initialize and configure the memory system, so that I can connect to Milvus and OpenRouter services for memory operations.

#### Acceptance Criteria

1. WHEN the Memory class is instantiated THEN the system SHALL connect to Milvus using the provided URI and create the memories collection if it does not exist
2. WHEN the memories collection is created THEN the system SHALL define the schema with fields: id (INT64 auto_id), user_id (VARCHAR), memory_type (VARCHAR), ts (INT64), chat_id (VARCHAR), who (VARCHAR), text (VARCHAR), vector (FLOAT_VECTOR dim=2560), hit_count (INT64), metadata (JSON)
3. WHEN the system initializes THEN the system SHALL configure OpenRouter client with the provided API key and base URL for embedding model (qwen/qwen3-embedding-4b) and LLM model (x-ai/grok-4.1-fast:free)
4. WHEN enable_dynamic_field is set to True THEN the system SHALL allow additional fields in metadata without schema changes

### Requirement 2

**User Story:** As a user, I want to add memories from conversations, so that the system can remember important information about me.

#### Acceptance Criteria

1. WHEN a user sends a message THEN the system SHALL call EpisodicWriteDecider to determine if the content should be stored as episodic memory
2. WHEN EpisodicWriteDecider returns write_episodic=true THEN the system SHALL generate embedding for each record's text field using the embedding model
3. WHEN storing an episodic memory THEN the system SHALL populate user_id, ts (current timestamp), chat_id, memory_type="episodic", hit_count=0, and metadata fields (context, thing, time, chatid, who)
4. WHEN the content is pure chitchat, objective knowledge questions, or meaningless fragments THEN the system SHALL NOT store it as episodic memory
5. WHEN the user explicitly requests "remember this" THEN the system SHALL store the information as episodic memory

### Requirement 3

**User Story:** As a user, I want to search my memories during conversations, so that the AI can provide contextually relevant responses.

#### Acceptance Criteria

1. WHEN a search query is received THEN the system SHALL generate an embedding vector for the query text
2. WHEN searching memories THEN the system SHALL retrieve both episodic and semantic memories filtered by user_id
3. WHEN retrieving memories THEN the system SHALL return up to k_semantic (default 5) semantic memories and k_episodic (default 5) episodic memories
4. WHEN ranking search results THEN the system SHALL consider similarity score, memory_type (semantic weighted higher), time decay (recent episodic more important), and hit_count
5. WHEN a memory is retrieved and used THEN the system SHALL increment its hit_count by 1

### Requirement 4

**User Story:** As a user, I want my episodic memories to be updated when new related information emerges, so that my memories stay current and complete.

#### Acceptance Criteria

1. WHEN an episodic memory is retrieved and used in conversation THEN the system SHALL call EpisodicReconsolidator with the old memory and current context
2. WHEN reconsolidating a memory THEN the system SHALL preserve metadata.time (original event start time), chat_id, who, and metadata.who unchanged
3. WHEN reconsolidating a memory THEN the system SHALL update metadata.context, metadata.thing, and text to reflect the chronological evolution of the event
4. WHEN reconsolidating a memory THEN the system SHALL append to metadata.updates array with time and description of the update
5. WHEN reconsolidation produces new text THEN the system SHALL regenerate the embedding vector and update the record in Milvus

### Requirement 5

**User Story:** As a system administrator, I want to run consolidation processes periodically, so that memories are organized, merged, and cleaned up.

#### Acceptance Criteria

1. WHEN consolidation is triggered THEN the system SHALL iterate through all episodic memories and find similar candidates using vector similarity search
2. WHEN two episodic memories have cosine similarity >= T_merge_high (0.85) AND same who AND pass time/chat_id constraints THEN the system SHALL call EpisodicMerger to merge them
3. WHEN two episodic memories have T_amb_low (0.65) <= similarity < T_merge_high (0.85) THEN the system SHALL call EpisodicSeparator to rewrite them for better distinction
4. WHEN merging memories THEN the system SHALL delete the original two records and insert one merged record with source_chat_ids and merged_from_ids in metadata
5. WHEN separating memories THEN the system SHALL update both records' text, metadata.context, and metadata.thing, then regenerate embeddings

### Requirement 6

**User Story:** As a user, I want stable facts about me to be extracted into semantic memory, so that my long-term attributes are easily accessible.

#### Acceptance Criteria

1. WHEN consolidation runs THEN the system SHALL call SemanticWriter for each episodic memory to check for extractable facts
2. WHEN an episodic memory contains stable identity information (major, job, location, research direction) THEN the system SHALL extract it as a semantic fact
3. WHEN an episodic memory contains long-term interests or habits (repeated preferences, ongoing hobbies) THEN the system SHALL extract it as a semantic fact
4. WHEN an episodic memory has hit_count > 10 and describes stable user attributes THEN the system SHALL extract it as a semantic fact
5. WHEN the content is temporary states, short-term plans, conversation style preferences, or uncertain speculations THEN the system SHALL NOT extract it as semantic memory
6. WHEN creating a semantic memory THEN the system SHALL set memory_type="semantic" and populate metadata with fact, source_chatid, and first_seen fields

### Requirement 7

**User Story:** As a developer, I want a clean API interface similar to mem0, so that I can easily integrate the memory system into applications.

#### Acceptance Criteria

1. WHEN calling Memory.add(text, user_id, chat_id) THEN the system SHALL process the text through EpisodicWriteDecider and store qualifying memories
2. WHEN calling Memory.search(query, user_id, limit) THEN the system SHALL return ranked memories matching the query
3. WHEN calling Memory.update(memory_id, data) THEN the system SHALL update the specified memory record
4. WHEN calling Memory.delete(memory_id) THEN the system SHALL remove the specified memory record
5. WHEN calling Memory.reset(user_id) THEN the system SHALL delete all memories for the specified user
6. WHEN implementing the Memory class THEN the system SHALL use factory pattern for Embedding model, Vector store (Milvus), and LLM instantiation
7. WHEN implementing the Memory class THEN the system SHALL support synchronous calls and leave extension points for future asynchronous implementation

### Requirement 9

**User Story:** As a developer, I want the system to handle time and chat_id constraints properly during merge decisions, so that only truly related memories are combined.

#### Acceptance Criteria

1. WHEN two memories have the same chat_id THEN the system SHALL allow merge if time difference is within 30 minutes
2. WHEN two memories have different chat_ids THEN the system SHALL allow merge only if time difference is within 7 days AND semantic analysis confirms same event
3. WHEN the who field differs between two memories THEN the system SHALL NOT merge them
4. WHEN merging memories THEN the system SHALL use the earliest metadata.time as the canonical event start time
5. WHEN merging memories THEN the system SHALL record both original chat_ids in metadata.source_chat_ids

### Requirement 10

**User Story:** As a developer, I want proper error handling and logging, so that I can debug and monitor the memory system effectively.

#### Acceptance Criteria

1. WHEN Milvus connection fails THEN the system SHALL raise a descriptive exception with connection details
2. WHEN OpenRouter API call fails THEN the system SHALL retry up to 3 times with exponential backoff before raising an exception
3. WHEN LLM returns invalid JSON THEN the system SHALL log the error and return a safe default response
4. WHEN a memory operation completes THEN the system SHALL log the operation type, user_id, and affected memory count
5. WHEN consolidation runs THEN the system SHALL log statistics including memories processed, merged, separated, promoted to semantic, and deleted
