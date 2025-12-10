# NeuraMem API Documentation

NeuraMem 记忆系统 REST API 接口文档。

## 快速开始

```bash
# 启动服务
cd e:\code\NeuraMem
uvicorn src.api.main:app --reload --port 8000

# 访问文档
# Swagger UI: http://localhost:8000/docs
# ReDoc: http://localhost:8000/redoc
```

---

## API 端点

### 健康检查

#### `GET /v1/health`

检查服务状态。

**Response** `200 OK`
```json
{
  "status": "ok",
  "version": "1.0.0"
}
```

---

### 对话 (SSE 流式)

#### `POST /v1/chat`

基于记忆增强的流式对话接口，返回 Server-Sent Events。

**Request Body**
```json
{
  "user_id": "string",      // 必填：用户标识
  "chat_id": "string",      // 必填：对话标识
  "message": "string",      // 必填：用户消息
  "history": [              // 可选：历史消息
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

**Response** `200 OK` (text/event-stream)

```
data: {"type": "chunk", "content": "你"}

data: {"type": "chunk", "content": "好"}

data: {"type": "done", "full_content": "你好，有什么可以帮您的吗？"}

```

**SSE 事件类型**
| type | 描述 |
|------|------|
| `chunk` | 流式 token |
| `done` | 完成，包含完整回复 |
| `error` | 错误信息 |

**前端示例 (JavaScript)**
```javascript
const response = await fetch('/v1/chat', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    user_id: 'user123',
    chat_id: 'chat456',
    message: '你好'
  })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  
  const lines = decoder.decode(value).split('\n');
  for (const line of lines) {
    if (line.startsWith('data: ')) {
      const data = JSON.parse(line.slice(6));
      if (data.type === 'chunk') {
        console.log(data.content);
      }
    }
  }
}
```

---

### 记忆管理

#### `POST /v1/memories/manage`

根据对话内容智能管理记忆（添加/更新/删除）。

**Request Body**
```json
{
  "user_id": "string",        // 必填：用户标识
  "chat_id": "string",        // 必填：对话标识
  "user_text": "string",      // 必填：用户输入
  "assistant_text": "string"  // 必填：助手回复
}
```

**Response** `200 OK`
```json
{
  "added_ids": [123, 124],
  "success": true
}
```

---

#### `POST /v1/memories/search`

搜索用户相关记忆，支持叙事组扩展。

**Request Body**
```json
{
  "user_id": "string",  // 必填：用户标识
  "query": "string"     // 必填：搜索内容
}
```

**Response** `200 OK`
```json
{
  "episodic": [
    {
      "id": 123,
      "user_id": "user123",
      "memory_type": "episodic",
      "ts": 1702233600,
      "chat_id": "chat456",
      "text": "用户说他是北京大学的学生",
      "group_id": 5
    }
  ],
  "semantic": [
    {
      "id": 456,
      "user_id": "user123",
      "memory_type": "semantic",
      "ts": 1702233700,
      "chat_id": "chat456",
      "text": "用户是北京大学计算机系的学生",
      "group_id": -1
    }
  ]
}
```

---

#### `DELETE /v1/memories/{memory_id}`

删除单条记忆。

**Path Parameters**
- `memory_id` (int): 记忆 ID

**Query Parameters**
- `user_id` (string, **必填**): 用户标识（用于所有权验证）

**Example**
```bash
DELETE /v1/memories/123?user_id=user123
```

**Response** `200 OK`
```json
{
  "success": true,
  "deleted_count": 1
}
```

**Error Response** `404 Not Found`
```json
{
  "error_code": "MEMORY_NOT_FOUND",
  "detail": "Memory with id 123 not found"
}
```

---

#### `DELETE /v1/memories/reset`

删除用户的所有记忆。

**Request Body**
```json
{
  "user_id": "string"  // 必填：用户标识
}
```

**Response** `200 OK`
```json
{
  "success": true,
  "deleted_count": 42
}
```

---

#### `POST /v1/memories/consolidate`

触发记忆整合，从情景记忆中提取语义事实。

**Request Body**
```json
{
  "user_id": "string"  // 可选：指定用户，空则整合所有
}
```

**Response** `200 OK`
```json
{
  "memories_processed": 15,
  "semantic_created": 3
}
```

---

## 错误码

| HTTP Status | Error Code | 描述 |
|-------------|------------|------|
| 404 | `MEMORY_NOT_FOUND` | 记忆不存在或无权访问 |
| 502 | `LLM_SERVICE_ERROR` | LLM 服务不可用 |
| 503 | `DB_CONNECTION_ERROR` | 数据库连接失败 |
| 500 | `INTERNAL_ERROR` | 内部错误 |

---

## 数据模型

### MemoryResponse

```typescript
interface MemoryResponse {
  id: number;           // 记忆 ID
  user_id: string;      // 用户标识
  memory_type: string;  // "episodic" | "semantic"
  ts: number;           // Unix 时间戳
  chat_id: string;      // 对话标识
  text: string;         // 记忆内容
  group_id: number;     // 叙事组 ID，-1 表示未分组
}
```

### ChatMessage

```typescript
interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}
```

---

## 配置环境变量

```bash
# API 配置
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=http://localhost:3000,http://localhost:8080

# 数据库
MILVUS_URL=http://localhost:19530

# LLM
DEEPSEEK_API_KEY=your_api_key
DEEPSEEK_BASE_URL=https://api.deepseek.com

# Embedding
SILICONFLOW_API_KEY=your_api_key
```
