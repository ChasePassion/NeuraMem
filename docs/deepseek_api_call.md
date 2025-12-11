简单讲，这个页面就是 DeepSeek 版的「/chat/completions」接口说明，基本等价于 OpenAI 的 Chat Completions，只是多了一些 DeepSeek 自己的东西（思考模式、前缀续写、硬盘缓存统计等）。我分块给你拆一下。

---

## 1. 这个接口干嘛用？

* URL：`POST /chat/completions`
* 作用：根据你提供的多轮对话 `messages`，生成后续的 assistant 回复。([api-docs.deepseek.com][1])
* 典型用法：聊天机器人、Agent、多轮问答、工具调用（function calling）等。

你可以基本把它当作「OpenAI /v1/chat/completions 的 DeepSeek 版」。

---

## 2. Request：核心字段说明

### 2.1 messages：对话消息列表（必填）

`messages: [...]` 是一个数组，每一条都是一条消息，有 4 种角色：

* `system`：系统提示词，设定模型整体行为和风格。
* `user`：用户说的话。
* `assistant`：模型之前的回复（做多轮对话时要带上）。
* `tool`：工具（function）执行完之后，返回给模型看的结果。([api-docs.deepseek.com][1])

每种消息的结构：

**① system**

```jsonc
{
  "role": "system",
  "content": "你是一个写代码的助手",
  "name": "setup-1"    // 可选，用来区分多个 system
}
```

* `name` 可选，用来标记是谁发的这条消息（在同一角色下区分不同来源）。([api-docs.deepseek.com][1])

**② user**

```jsonc
{
  "role": "user",
  "content": "帮我写一个快速排序",
  "name": "user-1"   // 可选
}
```

**③ assistant**

```jsonc
{
  "role": "assistant",
  "content": "这是我上一次的回答……",
  "name": "bot-1",        // 可选
  "prefix": false,        // Beta
  "reasoning_content": null  // 只给 deepseek-reasoner 用
}
```

这里有两个 DeepSeek 特有的字段：([api-docs.deepseek.com][1])

* `prefix` (Beta)

  * 如果设为 `true`，表示**强制模型在新回答里，以这条 assistant 消息的 content 作为前缀继续写**。
  * 使用这个功能时，需要把 base_url 改成 `https://api.deepseek.com/beta`。 ([api-docs.deepseek.com][1])

* `reasoning_content` (Beta)

  * 用于 `deepseek-reasoner` 模型，在「对话前缀续写」场景下，作为上一轮的「思维链内容」输入。
  * 使用时要求 `prefix=true`。([api-docs.deepseek.com][1])

**④ tool**

```jsonc
{
  "role": "tool",
  "tool_call_id": "call_xxx",
  "content": "这是函数实际执行后的结果"
}
```

* `tool_call_id`：这条 tool 消息是对哪一次函数调用（tool call）的响应，ID 对应 response 里的 `tool_calls[*].id`。([api-docs.deepseek.com][1])

---

### 2.2 model：用哪个模型（必填）

```jsonc
"model": "deepseek-chat"
```

可选值目前文档里写的是：([api-docs.deepseek.com][1])

* `deepseek-chat`：通用聊天模型。
* `deepseek-reasoner`：带显式「思考模式」的模型（会输出 reasoning_content）。

---

### 2.3 thinking：控制“思考模式”

```jsonc
"thinking": {
  "type": "enabled"  // or "disabled"
}
```

* `type: "enabled"`：开启思考模式（主要给 `deepseek-reasoner` 用），模型会在 response 里额外给出一段 `reasoning_content`，表示中间推理过程。
* `type: "disabled"`：关闭思考模式，只要最终答案。([api-docs.deepseek.com][1])

如果你只是想便宜好用、不看推理过程，通常用 `deepseek-chat` + 不设置 `thinking` 就行。

---

### 2.4 采样与长度控制

这些参数和 OpenAI 几乎一样：

1. **max_tokens**

```jsonc
"max_tokens": 512
```

* 限制一次生成的最大 token 数（不含输入）。超了就会 `finish_reason = "length"`。([api-docs.deepseek.com][1])

2. **temperature**（采样温度）

```jsonc
"temperature": 0.7  // 0~2，默认 1
```

* 越高越随机，越低越保守。官方建议：调 `temperature` 或 `top_p` 二选一，不要一起乱调。([api-docs.deepseek.com][1])

3. **top_p**

```jsonc
"top_p": 0.9  // 0~1，默认 1
```

* nucleus sampling。只从概率前 top_p 的 token 内采样。([api-docs.deepseek.com][1])

4. **frequency_penalty / presence_penalty**

```jsonc
"frequency_penalty": 0,
"presence_penalty": 0
```

* 范围 -2~2。正值会惩罚重复 token，使模型少重复（frequency）或更愿意引入新话题（presence）。([api-docs.deepseek.com][1])

5. **stop**

```jsonc
"stop": ["</END>", "用户："]
// 或 "stop": "某个字符串"
```

* 字符串或字符串数组，最多 16 个。生成遇到这些就停止。([api-docs.deepseek.com][1])

---

### 2.5 response_format：强制输出 JSON

```jsonc
"response_format": {
  "type": "json_object" // or "text"
}
```

* `type = "json_object"` 时，模型会被强制生成**合法 JSON**。([api-docs.deepseek.com][1])
* 官方提醒两个坑：([api-docs.deepseek.com][1])

  1. 你还需要在 system / user 里明确说“请只输出 JSON”，不然模型可能疯狂打空白直到 token 上限，看起来像卡死。
  2. 如果 `finish_reason = "length"`，说明被截断，JSON 可能是不完整的，需要你自己检查。

---

### 2.6 stream & stream_options：流式输出

```jsonc
"stream": true,
"stream_options": {
  "include_usage": true
}
```

* `stream = true`：返回 SSE 流，每一行 `data: {json...}`，最后一行 `data: [DONE]`。([api-docs.deepseek.com][1])
* `stream_options.include_usage = true`：在最后一个 `[DONE]` 之前，多发一块只含 `usage` 的统计数据（`choices` 为空）。其他块的 `usage` 都是 `null`。([api-docs.deepseek.com][1])

这点对你排查 TTFT 和流式耗时挺有用（可以区分“每块输出很慢”还是“token 很多导致总时间长”）。

---

### 2.7 tools & tool_choice：函数调用（Tool Calls）

#### 2.7.1 tools：声明函数

```jsonc
"tools": [
  {
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "获取某地天气",
      "parameters": {
        "type": "object",
        "properties": {
          "city": { "type": "string" }
        },
        "required": ["city"]
      },
      "strict": false  // 可选，默认 false
    }
  }
]
```

* 目前只支持 `type = "function"`。([api-docs.deepseek.com][1])
* `parameters` 用 JSON Schema 描述输入参数。
* `strict = true` 时启用“严格模式”，API 会强制输出严格符合 JSON Schema（这是 Beta 特性）。([api-docs.deepseek.com][1])

#### 2.7.2 tool_choice：控制要不要调用函数

```jsonc
"tool_choice": "auto"
// 或
"tool_choice": "none"
// 或
"tool_choice": "required"
// 或指定某个函数：
"tool_choice": {
  "type": "function",
  "function": { "name": "get_weather" }
}
```

* `none`：不会调用任何 tool，只正常回答。
* `auto`：模型可以自己决定要不要调用 tool（默认值，有 tool 才会默认 `auto`）。
* `required`：必须调用至少一个 tool。
* 指定函数：强制调用某一个 tool。([api-docs.deepseek.com][1])

---

### 2.8 logprobs / top_logprobs：返回 token 级概率

```jsonc
"logprobs": true,
"top_logprobs": 5    // 0~20
```

* `logprobs = true` 时，响应里 `message.logprobs` 会包含每个输出 token 的 log prob 列表。([api-docs.deepseek.com][1])
* `top_logprobs = N`：每个位置返回 top N 的候选 token 及其 log prob，用于做更细粒度的分析或采样可视化。

---

## 3. Response：返回结构拆解

非流式（`stream=false`）时，服务器返回一个完整的 `chat completion` 对象：([api-docs.deepseek.com][1])

### 3.1 顶层字段

* `id`：本次对话补全的唯一 ID。
* `object`: 固定是 `"chat.completion"`。
* `created`: Unix 时间戳（秒）。
* `model`: 实际使用的模型名。
* `system_fingerprint`: 标识后端配置（方便排查问题）。([api-docs.deepseek.com][1])
* `usage`: token 用量统计：([api-docs.deepseek.com][1])

  * `prompt_tokens`: 输入 prompt 的总 token 数。
  * `completion_tokens`: 输出的 token 数。
  * `prompt_cache_hit_tokens`: 命中上下文缓存的输入 token。
  * `prompt_cache_miss_tokens`: 没命中的输入 token。
  * `total_tokens`: 总和。

这个 `prompt_cache_hit_tokens / miss_tokens` 就是它「上下文硬盘缓存」功能的配套统计，你能看到自己的请求有多少 token 是直接复用缓存的。([api-docs.deepseek.com][1])

---

### 3.2 choices：模型的一个或多个候选答案

`choices` 是一个数组，一般你只要第 0 个：([api-docs.deepseek.com][1])

* `index`：第几个答案。

* `finish_reason`：为什么停下：([api-docs.deepseek.com][1])

  * `stop`：正常结束（或者遇到 stop 字符串）。
  * `length`：到达 `max_tokens` 或上下文长度上限，被截断。
  * `content_filter`：被内容安全过滤。
  * `tool_calls`：因为要调用工具而停。
  * `insufficient_system_resource`：系统算力不足中断。

* `message`：真正重要的东西在这里：([api-docs.deepseek.com][1])

  * `role`: 永远是 `"assistant"`。
  * `content`: 模型给你的文字回复。
  * `reasoning_content`: 只在 `deepseek-reasoner` 且开启思考模式时存在，是最终答案前的推理文本。
  * `tool_calls`: 如果模型决定调用工具，会在这里给出调用信息：

```jsonc
"tool_calls": [
  {
    "id": "call_xxx",
    "type": "function",
    "function": {
      "name": "get_weather",
      "arguments": "{\"city\": \"Guangzhou\"}"
    }
  }
]
```

* `arguments` 注意是 **JSON 字符串**，你得在自己代码里再 `json.loads` 一下，并做好参数校验，因为模型可能会乱造字段。([api-docs.deepseek.com][1])

如果请求了 `logprobs`，`message` 下还会有：

* `logprobs.content`: 每个 token 的概率信息。
* 如果是 reasoner，`reasoning_content` 也有自己的 `logprobs` 结构。([api-docs.deepseek.com][1])

---

## 4. 流式响应大致长什么样（SSE）

文档只写了：

* `stream = true` 时，以 SSE 格式逐块发送；
* 每块是 `data: {json...}`；
* 最后一块是 `data: [DONE]`；
* 如果 `stream_options.include_usage = true`，倒数第二块会是一条 usage 统计，`choices` 为空数组。([api-docs.deepseek.com][1])

你这边用 Python / Node 处理 SSE，逻辑跟 OpenAI 兼容。

---

## 5. 最小调用示例（Python 风格）

给你一个简单示例（伪代码，用 requests 写）：

```python
import requests
import json

url = "https://api.deepseek.com/chat/completions"
headers = {
    "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
    "Content-Type": "application/json",
}

data = {
    "model": "deepseek-chat",
    "messages": [
        {"role": "system", "content": "你是一个严谨的编程助手"},
        {"role": "user", "content": "帮我用 Python 写一个快速排序，并简单解释思路"},
    ],
    "max_tokens": 512,
    "temperature": 0.3,
}

resp = requests.post(url, headers=headers, data=json.dumps(data))
resp.raise_for_status()
result = resp.json()
print(result["choices"][0]["message"]["content"])
```

如果你要用 `deepseek-reasoner` 并看推理过程：

```python
data = {
    "model": "deepseek-reasoner",
    "thinking": {"type": "enabled"},
    "messages": [
        {"role": "system", "content": "你是一个会详细展示推理过程的数学助手"},
        {"role": "user", "content": "证明一下根号2是无理数"},
    ],
}
```

然后从 `choices[0].message.reasoning_content` 拿推理过程，从 `content` 拿最终答案。([api-docs.deepseek.com][1])

---

如果你接下来准备在你自己的项目里接 DeepSeek，我可以按你现有的 FastAPI / Next.js / Node 客户端，帮你写一版适配代码，并顺便把思考模式、tool calls、JSON 模式这些都配置好。

[1]: https://api-docs.deepseek.com/zh-cn/api/create-chat-completion "对话补全 | DeepSeek API Docs"
