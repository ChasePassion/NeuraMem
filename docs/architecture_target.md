# NeuraMem 目标架构（优化后）

> **Goal**: 展示 NeuraMem 重构后的目标架构，并逐项说明相对现状的调整及其与三大目标（架构干净 / 生产接口与包 / 评测 pipeline）的对应关系。
> **定位**: 公开通用的 AI 记忆库（与 mem0 / Zep / MemGPT 同类定位）——独立发行、通用 API（manage/search/consolidate）、公开评测（LoCoMo），不绑定任何业务项目；parlasoul-backend 是首个真实消费者，其业务改造只经通用机制（metadata / filter / 端口注入）接入，不进核心模型。
> **Type**: Container/Component 级架构图（目标态）
> **Date**: 2026-08-17（rev2：合入评审修订——LLMConfig 逃生舱、流式 usage 前提、重试预算收敛、labeled usage 聚合、两段式闭环接口、benchmark 资产迁移；详见调整表 #21/#22 与第十一章）

## 〇、原架构图（对照参考）

> 重构前的现状架构，与下方的目标架构对比阅读。

```mermaid
flowchart TB
    subgraph ACTORS["调用方"]
        App["上层应用 / Agent<br/>REST / SSE"]
        Demo["Gradio Demo<br/>demo/app.py"]
    end

    subgraph API["API 层 src/api/ · FastAPI"]
        direction TB
        Main["main.py<br/>app 装配 · CORS · lifespan"]
        ChatR["routers/chat.py<br/>POST /v1/chat · SSE"]
        MemR["routers/memories.py<br/>manage/search/consolidate<br/>delete/reset"]
        Deps["deps.py<br/>get_memory_system()<br/>lru_cache 单例"]
        Schemas["schemas.py<br/>Pydantic 模型"]
    end

    subgraph CORE["核心库 src/memory_system/"]
        Facade["Memory 门面<br/>memory.py<br/>manage/search/consolidate<br/>CRUD 执行 · 日志 · 遥测"]
        Prompts["prompts.py<br/>4 个 LLM Prompt（核心资产）"]
        subgraph PROC["Processors 业务处理层"]
            EMM["EpisodicMemoryManager<br/>LLM 决策 CRUD"]
            SW["SemanticWriter<br/>情景 → 语义提炼"]
            MUJ["MemoryUsageJudge<br/>判断记忆是否被使用"]
            NMM["NarrativeMemoryManager<br/>叙事聚类 · 质心维护"]
        end
        subgraph CLI["Clients 基础设施层"]
            LLMC["LLMClient<br/>主备 fallback · 重试<br/>chat_json 解析"]
            EMB["EmbeddingClient<br/>Qwen3-Embedding-4B<br/>dim=2560 硬编码"]
            MS["MilvusStore<br/>memories 集合<br/>groups_{user_id} 集合"]
        end
        Retry["utils/retry.py<br/>RetryExecutor"]
    end

    subgraph EXT["外部依赖"]
        LLMAPI["DeepSeek API<br/>OpenRouter API"]
        EMBAPI["SiliconFlow API<br/>Embedding"]
        MILVUS[("Milvus 向量库<br/>memories / groups_{user_id}")]
        LF["Langfuse<br/>可观测性"]
    end

    App -->|"REST / SSE"| ChatR
    App -->|"REST"| MemR
    Demo -->|"直接调用（绕过 API）"| Facade

    ChatR --> Deps
    MemR --> Deps
    Deps --> Facade
    Schemas -.-> ChatR
    Schemas -.-> MemR

    Facade --> EMM
    Facade --> SW
    Facade --> MUJ
    Facade --> NMM
    Facade --> LLMC
    Facade --> EMB
    Facade --> MS
    Facade --> Prompts
    EMM -->|"chat_json"| LLMC
    SW -->|"chat_json"| LLMC
    MUJ -->|"chat_json"| LLMC
    NMM --> MS
    LLMC --> Retry
    EMB --> Retry

    LLMC -->|"HTTPS"| LLMAPI
    EMB -->|"HTTPS"| EMBAPI
    MS -->|"gRPC/HTTP"| MILVUS
    Facade -.->|"observe 遥测侵入"| LF
    EMM -.->|"observe"| LF
    SW -.->|"observe"| LF
    NMM -.->|"observe"| LF

    classDef problem stroke:#d64541,stroke-width:2px,stroke-dasharray:5 3
    classDef asset fill:#e8f0fe,stroke:#4285f4
    classDef ext fill:#f5f5f5,stroke:#9e9e9e
    class Facade,LF problem
    class Prompts,Retry asset
    class LLMAPI,EMBAPI,MILVUS ext
```

## 一、目标架构图

```mermaid
flowchart TB
    subgraph ACTORS["调用方"]
        App["Agent 应用 / 第三方<br/>REST / SSE"]
        Demo["Gradio Demo<br/>demo/app.py（瘦身）"]
    end

    subgraph SRV["服务层 neuramem_server（可独立部署）"]
        direction TB
        Routers["routers/<br/>chat · memories"]
        Schemas["schemas.py<br/>Pydantic 契约"]
        DI["deps.py<br/>组件装配 · 依赖注入"]
        AppF["app.py<br/>lifespan · CORS · 异常"]
        Routers --> DI
        Schemas -.-> Routers
    end

    subgraph CORE["核心库 neuramem（可 pip 安装）"]
        direction TB
        Facade["memory.py<br/>薄编排门面（公共 API 入口）"]
        subgraph APP["Application 编排层"]
            Retr["pipeline/retrieval.py<br/>Retriever 检索+叙事扩展"]
            Ep["pipeline/episodic.py<br/>LLM 决策 CRUD+候选选择"]
            Sem["pipeline/semantic.py<br/>语义提炼（增量）"]
            Nar["pipeline/narrative.py<br/>叙事聚类·质心"]
            Jud["pipeline/usage_judge.py<br/>判断被用记忆（再巩固）"]
        end
        subgraph DOM["Domain 领域层（零 IO 依赖）"]
            Models["core/models.py<br/>Pydantic 领域模型"]
            Ports["core/ports.py<br/>VectorStore · LLM<br/>Embedder · Telemetry"]
            Ex["core/exceptions.py<br/>core/retry.py"]
        end
        Prompts["prompts.py<br/>4 个 LLM Prompt（资产保留）"]
        Cfg["config.py<br/>pydantic-settings 分层校验"]
    end

    subgraph ADAPT["适配器层（实现 Ports，可替换）"]
        direction TB
        MS["store/milvus.py<br/>MilvusStore（schema 兼容）"]
        IM["store/inmemory.py<br/>InMemoryStore（测试用）"]
        LLMA["llm/openai_adapter.py<br/>OpenAI-compatible<br/>单 provider · 全 async"]
        EMBA["embed/openai_adapter.py<br/>原生 async"]
        TL["telemetry/langfuse.py<br/>LangfuseTelemetry"]
        TLN["telemetry/null.py + memory.py<br/>NullTelemetry（默认）<br/>InMemoryTelemetry（测试/评测）"]
    end

    subgraph EXT["外部依赖"]
        MILVUS[("Milvus<br/>memories + groups<br/>（均单集合）")]
        LLMAPI["OpenAI-compatible LLM API<br/>任意服务商（base_url 配置）"]
        EMBAPI["Embedding API<br/>SiliconFlow"]
        LF["Langfuse 可观测性"]
    end

    subgraph BM["评测 pipeline neuramem_benchmark/（LoCoMo 专用）"]
        direction TB
        AD["locomo.py<br/>数据集加载与解析"]
        RUN["runner.py<br/>对话重放 → manage/search"]
        MET["metrics.py + judge.py<br/>recall@k · LLM-as-judge"]
        RPT["report.py<br/>JSON + Markdown 报告"]
    end

    App -->|"HTTPS REST/SSE"| Routers
    Demo -->|"直接调用（可选）"| Facade

    DI -->|"组件装配"| Facade
    Facade --> Retr
    Facade --> Ep
    Facade --> Sem
    Facade --> Nar
    Facade --> Jud
    Jud -->|"被用记忆"| Nar
    Facade --> Prompts
    Facade --> Cfg
    Retr --> Ports
    Ep --> Ports
    Sem --> Ports
    Nar --> Ports
    Jud --> Ports
    Models -.-> Facade
    Models -.-> APP
    Ex -.-> APP
    Ports --> MS
    Ports --> IM
    Ports --> LLMA
    Ports --> EMBA
    Ports --> TL
    Ports --> TLN
    MS -->|"gRPC/HTTP"| MILVUS
    LLMA -->|"HTTPS"| LLMAPI
    EMBA -->|"HTTPS"| EMBAPI
    TL -->|"HTTPS"| LF
    Facade -.->|"可选遥测（经 Ports）"| TL
    RUN -.->|"评测：消费公共 API"| Facade
    AD --> RUN
    RUN --> MET
    MET --> RPT

    classDef dom fill:#e8f0fe,stroke:#4285f4,stroke-width:2px
    classDef app fill:#e6f4ea,stroke:#34a853
    classDef port fill:#fef7e0,stroke:#f9ab00,stroke-dasharray:4 3
    classDef adap fill:#f1f3f4,stroke:#9e9e9e
    classDef ext fill:#eceff1,stroke:#607d8b,stroke-dasharray:2 2
    classDef asset fill:#fce8e6,stroke:#d93025,stroke-width:2px
    class Models,Ports,Ex dom
    class Facade,Retr,Ep,Sem,Nar,Jud app
    class MS,IM,LLMA,EMBA,TL,TLN adap
    class MILVUS,LLMAPI,EMBAPI,LF ext
    class Prompts asset
```

## 二、核心调整（现状 → 目标）

| # | 现状 | 目标 | 对应要求 |
|---|---|---|---|
| 1 | 不可安装：无 pyproject.toml，源码直跑 uvicorn | **单发行物 pyproject.toml**：neuramem 核心库 + neuramem_server 服务层，extras 区分 [server] [benchmark] [test] | 生产接口/包 |
| 2 | memory_system 内部无边界，Facade 734 行 | **领域层（core）与应用层（pipeline）分离**：core 零 IO 依赖，pipeline 只依赖 core 的 Ports | 架构干净 |
| 3 | 无抽象接口，Facade 直接 new 具体 client | **端口-适配器（Hexagonal）**：core/ports.py 定义 VectorStore / LLM / Embedder / Telemetry 四个 Protocol，Milvus/OpenAI/Langfuse 均为可替换适配器 | 架构干净 + 二次开发 |
| 4 | search 叙事扩展逻辑（约 120 行）内嵌 Facade | **独立 Retriever**（pipeline/retrieval.py），Facade 只做编排 | 架构干净 |
| 5 | @observe 遥测侵入全部核心代码，与 Langfuse 强耦合 | **Telemetry 端口 + Null/Langfuse 双适配器**，默认 Null，核心代码零遥测依赖 | 架构干净 + 二次开发 |
| 6 | store 返回裸 dict，API 层手工转换 | **Pydantic 模型贯穿三层**（core/models.py），删除手工字段拷贝 | 架构干净 |
| 7 | 同步/异步混用，asyncio.to_thread 多层嵌套 | **async-first**：store/embed 原生 async，同步方法降级为薄 wrapper。注意 pymilvus 无原生 async——store 适配器内部以 to_thread 包装，线程池大小显式进 StoreConfig（隐藏配置点显式化） | 架构干净 |
| 8 | 配置为 dataclass + os.getenv，无校验；双通道 fallback（主 DeepSeek + 备 OpenRouter/GLM） | **pydantic-settings 分层配置**（LLM/Embed/Store/Retrieval 子模型）+ 校验；**完全删除 fallback**，LLM 收敛为单一 OpenAI-compatible provider（base_url + api_key + model + extra_body 逃生舱，见 6.1——vendor 专属参数如 thinking 开关是用户策略，不属于 compat 检测，必须可配置） | 生产接口/包 |
| 9 | 2560 维三处硬编码，换模型必须同步改 | **维度单一来源**：config 显式值 + 启动探测校验（首条 embed 比对维度，不一致即 fail-fast），schema 构建统一引用该值；不做运行时推导 | 架构干净 |
| 10 | 测试依赖真 Milvus | **InMemoryStore 适配器**：实现同一 VectorStore 端口，单元/属性测试零外部依赖 | 生产接口/包 |
| 11 | 无评测体系 | **neuramem_benchmark/ 独立评测包**（顶层名避开 `benchmark` 通用名，防安装后命名空间撞名）：LoCoMo 专用（数据加载 + 对话重放 runner + recall@k 规则指标 + LLM-as-judge + 报告），不做多数据集抽象 | 评测 pipeline |
| 12 | demo 与 routers 重复实现上下文构建 | demo 复用核心库公共 API，删除重复逻辑 | 架构干净 |
| 13 | API 层与核心库同包、不可单独部署 | **neuramem_server 独立包**：服务层与库分离，库可被任意应用 import | 生产接口/包 |
| 14 | 再巩固闭环缺失：usage_judge → 叙事分组的调用只存在于 demo 路径（且靠私有属性伸手 + 文本精确匹配映射 id），服务层触发 manage 后记忆永远停在 group_id=-1，检索的叙事组扩展形同虚设 | **完整闭环：两段式公共 API**——`search_async` 返回 `SearchResult`（两段间的关联令牌）→ 消费者用自己的 LLM 生成回答 → `report_usage_async(result, answer)` 触发 usage_judge→assign；回答编排归消费者，server/demo/benchmark 共用库接口（设计见第十一章） | 架构干净 + 业务正确性 |
| 15 | 每用户一个 groups_{user_id} 集合（Milvus 集合数上限千级，多租户规模不可行） | **单一 groups 集合 + user_id 字段过滤**（与主集合同构），集合数恒定；存量 groups_{user_id} 集合提供一次性迁移命令，benchmark 环境可直接 reset 重导 | 架构干净 + 生产接口 |
| 16 | user_id 直接字符串拼接进 Milvus 过滤表达式（含引号即破坏表达式/注入类风险） | API 层校验 user_id 格式（如 ^[A-Za-z0-9_-]{1,64}$）+ store 层提供结构化过滤器 | 生产接口/包 |
| 17 | update 用 delete+add 实现（记忆 id 变化、两步之间崩溃丢数据） | **Milvus upsert 原地更新**（id 稳定、单步原子；narrative 已用 upsert 改 group_id，可行性已验证） | 生产接口/包 |
| 18 | 流式回答不解析 usage，token 成本不可见 | **LLM 适配器聚合 usage**（prompt/completion/cache/reasoning tokens + 按定价表算 cost），供监控与评测成本统计（参考 pi-mono parseChunkUsage，见 6.5）。两个落地前提：流式必须显式请求 usage（`stream_options.include_usage`，compat 门控——OpenAI 官方不给、MiniMax/DeepSeek 恰好给，不能假设最后 chunk 自带）；usage 聚合须支持 call_label 分桶 + contextvars 归属（见 6.5，W3 分组件 cache hit rate 报告依赖） | 评测 pipeline + 生产 |
| 19 | search 对每个叙事组各发一次 query（N+1） | 合并为单次 group_id in [...] 查询 | 架构干净 |
| 20 | consolidate 只增不减：旧语义记忆被新事实否定时仍保留 | 增量合并 + **冲突淘汰（最简版）**：新事实与旧语义记忆矛盾时给旧记录打 retired 标记（动态字段 + upsert 原地写入），检索层**永久过滤** retired 记录；不设宽限期、确认机制与自动扫尾，物理删除只由 reset 手动触发 | 业务正确性 |
| 21 | 叙事组清理时机错误：`Memory.delete` 先清理后删记录，`delete_memory_from_group` 的成员查询在记忆仍在库时执行——质心把垂死记忆的向量算了进去、size 多计 1，且 `n==0` 删空组分支永远不可达（孤儿组永久累积） | 成员查询排除被删记忆：`group_id == G and user_id == U and id != {memory_id}`；空组删除分支随之复活。补单测：删除组内最后一名成员 → 组被清除（2026-08 评审中发现，已先于重构在现状代码修复，重构继承） | 业务正确性 |
| 22 | chat_json 解析失败静默 no-op：解析失败回退 default 但 `success` 恒为 True，调用方无法区分"LLM 决定什么都不做"与"输出是垃圾"——该轮记忆变化无声丢失，只留一条日志 | `success` 如实反映解析结果；解析失败带纠正提示重试一次（多数格式失败可修复）；manage 路径解析失败显式抛错（server/demo 边界捕获记日志，benchmark ingest 每轮隔离异常并计数，不中断样本重放）。usage_judge/semantic 维持各自既有的保守降级语义（2026-08 已修复） | 业务正确性 + 可观测 |

## 三、三大要求的落实

### 1. 架构干净

- 依赖单向且向内：server → 编排 → domain ← 适配器；domain 不 import 任何 IO 依赖，可在无外部服务下完整测试
- 每个类职责单一：Facade 只剩编排（约为原 1/3 体量），检索/编码/巩固/聚类/判断各有归属
- 遥测、存储、LLM、Embedding 全部接口化，新增实现（如 Chroma、本地模型）只加适配器，不动业务代码

### 2. 生产接口/包

- pip install neuramem 即得核心库：from neuramem import Memory, MemoryConfig, MemoryRecord；顶层 __init__.py 只导出公共 API，附带 py.typed
- 存储格式兼容：Milvus collection schema 不变，老数据直接可用；/v1/* REST 契约不变
- 服务层可独立部署（neuramem_server），也可被现有 FastAPI 应用挂载（app.include_router）
- 配置即代码：pydantic-settings 校验 + .env 覆盖，错误配置启动即失败而非运行期爆炸

### 3. 评测 pipeline

- **评测数据集仅 LoCoMo**（Long-Term Conversation Memory，对话记忆评测集）：locomo.py 单文件负责数据加载与解析（多 session 对话轮次 + 标注 QA + evidence），**不做多数据集 adapter 抽象**——当前只有单一评测目标，通用框架属于过度设计；将来确有第二个数据集需求时再按需抽象（YAGNI）
- 两级指标分离：**recall@k**（不调 LLM，验证检索出的记忆是否包含 ground-truth evidence）+ **端到端 accuracy**（LLM-as-judge 按 LoCoMo 官方 rubric 0/4 打分）
- **无记忆 baseline 对照组**：同一批 QA 跑"纯 LLM 回答（不注入记忆）"作为对照，报告两组准确率差（memory uplift）——量化记忆系统本身带来的增益，是评测报告最有说服力的输出
- 对话重放 runner：按时间序把对话喂给 manage_async 模拟真实使用，再对 QA 执行 search → 回答 → 打分，支持抽样与固定 seed 复现、--max-queries 成本控制
- **存量资产迁移（重组而非重写）**：现状 `benchmark/locomo/` 的 9 个文件（run_eval / run_benchmark / judge / rejudge / stat_results / llm_config / locomo_prompts / import_to_neuramem / RUN_RECORD）按新结构归位；RUN_RECORD 的 W 编号连续性与其中 6.1 对比规则（同工作流才可比）原样延续，作为评测史的权威记录
- **必须移植的硬功能**（W1–W3 实战验证过的，缺一不可）：多进程并行与样本选择参数、CSV 按 sample_index+question_id 去重合并、rejudge 重判、KV cache 分组件报告（依赖 6.5 的 labeled 聚合，口径见 RUN_RECORD 6.2）
- **ingest 完整性校验内建**：重放完成自动核对 Milvus 记忆数与 "Import complete" 日志行，未达标直接 fail 而非继续 eval——s8 残缺 ingest 事故（32.7% 假分数）的教训从人肉 checklist 变成 runner 的内建断言（详见 RUN_RECORD 8.1）
- 评测对象用公共 API（Memory 接口）抽象，未来可与 mem0 等对照系统横向对比

## 四、目标目录结构

```
NeuraMem/
├── pyproject.toml              # 单发行物；extras: [server] [benchmark] [test]
├── src/
│   ├── neuramem/               # 核心库（可安装、py.typed）
│   │   ├── memory.py           # 薄编排门面（公共 API）
│   │   ├── config.py           # pydantic-settings 分层配置
│   │   ├── prompts.py          # 保留原样（核心资产）
│   │   ├── core/               # 领域层：models / ports / exceptions / retry
│   │   ├── pipeline/           # retrieval / episodic / semantic(+淘汰) / narrative / usage_judge 闭环
│   │   ├── llm/  embed/  store/  telemetry/   # 适配器（各含具体实现）
│   │   └── py.typed
│   └── neuramem_server/        # FastAPI 服务层（可独立部署）
│       ├── app.py  routers/  schemas.py  deps.py  exceptions.py
├── neuramem_benchmark/         # 评测 pipeline（LoCoMo 专用；顶层名避开通用词 benchmark，防安装后命名空间撞名）
│   ├── locomo.py  runner.py  metrics.py  judge.py  report.py
├── demo/                       # 瘦身：复用核心库公共 API（不进 wheel，extras 只管依赖）
└── tests/                      # 单元/属性测试（InMemoryStore 免外部依赖）+ telemetry conformance + 集成
```

## 五、与原架构的对应关系（迁移锚点）

| 原路径 | 新路径 | 说明 |
|---|---|---|
| src/memory_system/memory.py | src/neuramem/memory.py | 瘦身为编排 |
| src/memory_system/processors/memory_manager.py | src/neuramem/pipeline/episodic.py | +候选选择 |
| src/memory_system/processors/semantic_writer.py | src/neuramem/pipeline/semantic.py | +增量合并 |
| src/memory_system/processors/narrative_memory_manager.py | src/neuramem/pipeline/narrative.py | 逻辑不变 |
| src/memory_system/processors/memory_usage_judge.py | src/neuramem/pipeline/usage_judge.py | 协议 text→id：候选记忆带 id 进 prompt，返回 used ids（消除 demo 的精确文本匹配；行为变更，W4 验证） |
| （search 叙事扩展内嵌段） | src/neuramem/pipeline/retrieval.py | 新抽出 |
| src/memory_system/clients/llm.py | src/neuramem/llm/openai_adapter.py | 删 fallback · 单 provider · 全 async |
| src/memory_system/clients/embedding.py | src/neuramem/embed/openai_adapter.py | 原生 async |
| src/memory_system/clients/milvus_store.py | src/neuramem/store/milvus.py | schema 兼容 + groups 单集合改造（#15） |
| — | src/neuramem/store/inmemory.py | 新增 |
| src/memory_system/config.py | src/neuramem/config.py | pydantic-settings |
| src/memory_system/exceptions.py | src/neuramem/core/exceptions.py | 不变 |
| src/memory_system/utils/retry.py | src/neuramem/core/retry.py | 不变 |
| src/memory_system/prompts.py | src/neuramem/prompts.py | 原样保留 |
| src/api/* | src/neuramem_server/* | 契约不变；chat 路由不再伸手 `memory._llm_client`——server 的 deps 装配自己的 LLM 适配器实例回答，Memory 只走两段式接口 |
| （chat.py / demo 的回答编排段） | 消费者侧胶水 | search_async → 消费者回答 → report_usage_async；记忆块格式化收进 SearchResult.render（见第十一章） |
| benchmark/locomo/*（9 文件） | neuramem_benchmark/* | 重组而非重写；硬功能移植清单见三.3 |

---

## 六、LLM 客户端设计（OpenAI-compatible · 参考 pi-mono）

> pi-mono（Vercel AI SDK monorepo）参考实现：
> - packages/ai/src/api/openai-completions.ts —— createClient（SDK + baseURL 注入）、detectCompat（服务商差异检测）、buildParams
> - packages/ai/src/utils/provider-retry.ts —— 单 provider 重试策略

### 6.1 单 provider，无 fallback

- 现状的 primary + fallback 双通道（DeepSeek 主 + OpenRouter 备；且 fallback_model 配置缺失，备用通道实际从未生效）**整体删除**
- 目标：一个 LLMClient 只对接一个 OpenAI-compatible 服务商，连接参数完全由配置决定：

```python
class LLMConfig(BaseModel):
    base_url: str  # any OpenAI-compatible endpoint (api.deepseek.com, api.minimaxi.com, self-hosted vLLM, ...)
    api_key: str
    model: str
    max_retries: int = 3
    extra_body: Optional[dict] = None  # vendor-specific params, e.g. {"thinking": {"type": "disabled"}}
```

- **extra_body 是逃生舱，不是可选项**：W3 全程依赖 `LLM_EXTRA_BODY={"thinking":{"type":"disabled"}}` 关 MiniMax-M3 思考。thinking 开/关是**用户策略**，不属于 detectCompat 的协议差异范畴——没有这个通道，换推理行为就得改代码，违反 8.3 配置化原则。原样透传（OpenAI SDK extra_body 逃生舱），库不解析内容

- 对应 pi-mono createClient 模式：官方 OpenAI SDK + baseURL 注入，即 `openai.OpenAI(api_key=..., base_url=...)`，不手写 HTTP 协议
- 切换服务商 = 改配置，不涉及代码；多服务商并存 = 多个 LLMClient 实例（由使用方显式创建），系统自身不做自动切换
- Embedding 同理：EmbeddingClient 已是 OpenAI-compatible（base_url + api_key + model），维持单 provider 模式

### 6.2 兼容性检测（pi-mono detectCompat 思想）

- 按 base_url / provider 指纹自动适配服务商协议差异，例如：
  - deepseek.com → 请求参数用 max_tokens（而非 max_completion_tokens）、thinking 内容走 deepseek 专属格式
  - 拒绝 `stream_options` 参数的服务商 → 流式不带 `stream_options.include_usage`（见 6.5，对应 pi-mono 的 supportsUsageInStreaming 门控）
  - 其余 → 默认 OpenAI 行为
- 初期只需 DeepSeek 指纹 + 默认分支两条路径，随实际接入的服务商增长再扩展
- 自动检测结果可被显式配置覆盖（对应 pi-mono 的 model.compat 覆盖机制）

### 6.3 重试策略（pi-mono provider-retry 模式）

- **只重试可恢复错误**：408 / 409 / 429 / 5xx，或服务端 x-should-retry 头显式指示；其余 4xx 一律不重试，直接抛错
- **重试预算单点拥有（关掉 SDK 内置重试）**：OpenAI SDK client 以 `max_retries=0` 构造，重试全部由 RetryExecutor 负责——否则 SDK 默认 2 次 × executor N 次相乘（现状最坏 30 次请求），预算既不可预期也不可观测。pi-mono 同样以 SDK maxRetries:0 + 自管重试包装实现（理由：SDK 内置重试的 sleep 不响应 abort）
- **退避策略**：优先尊重服务端 retry-after / retry-after-ms 头；否则指数退避（0.5 * 2^n，上限 8s）+ 25% 随机抖动，避免重试风暴
- **retry-after 上限护栏**：服务端要求的等待超过上限（默认 60s，可配置，0 关闭）直接抛结构化错误，而不是无限顺从傻等（对应 pi-mono validateServerRetryDelayMs）
- **可中断**：重试等待可被取消（asyncio task cancellation），避免请求取消后仍空等
- **场景档位**：交互场景默认 3 次 × 8s cap 足够；benchmark 重放场景 429 风暴下预算应显著更大（W3 实测稳定档位：进程内 10 次 × 30s cap），由 runner 预设，不改库默认
- 全部重试用尽后抛出结构化错误（含 HTTP status / headers / body），由上层决定处理方式

### 6.4 对现有代码的影响

| 项 | 动作 |
|---|---|
| RetryExecutor（utils/retry.py） | 保留，但重试范围收敛为 408/409/429/5xx（现状是 retryable_exceptions 默认全 Exception，过宽）；SDK 内置重试关闭（见 6.3） |
| LLMClient 双通道逻辑 | 删除 fallback client / fallback_model / 主备切换分支 |
| config 的 llm_fallback_* / glm_* 字段 | 删除，收敛为单一 LLMConfig（含 extra_body，见 6.1） |
| chat / chat_json / chat_stream_async 签名 | 不变，processors 与 prompts 零改动；chat_json 的 success 语义修正与解析修复重试见调整表 #22 |
| UsageStats（现状 llm.py 内） | 保留并升级为 usage 聚合的一等组件：call_label 分桶 + 归属存储从 threading.local 换 contextvars（见 6.5） |

### 6.5 Token 用量计量（参考 pi-mono parseChunkUsage）

pi-mono 参考：packages/ai/src/api/openai-completions.ts 中 stream 内对 chunk.usage 的处理（约 449-461 行）与 parseChunkUsage（1374-1412 行）。

- **流式场景**：OpenAI-compatible 协议在流的最后一个 chunk 携带完整 usage——遍历 chunk 时发现 `chunk.usage` 即解析并聚合（一次会话只出现一次）；**兼容分支**：部分服务商（如 Moonshot）把 usage 放在 `choice.usage` 而非 `chunk.usage`，两个位置都检查。**前提：usage 必须显式请求**——请求参数带 `stream_options: {"include_usage": true}`（compat 门控，拒绝该参数的服务商降级不发；对应 pi-mono buildParams 的 supportsUsageInStreaming）。不能假设"最后 chunk 自带 usage"：OpenAI 官方流式默认不给（必须请求），MiniMax/DeepSeek 恰好无条件给——W3 能采到 KV cache 数据是 provider 运气，换服务商就会静默变空
- **字段映射**（parseChunkUsage 语义）：
  - input = prompt_tokens - cache_read - cache_write；output = completion_tokens
  - cache_read = prompt_tokens_details.cached_tokens（或 prompt_cache_hit_tokens）；cache_write = prompt_tokens_details.cache_write_tokens
  - reasoning = completion_tokens_details.reasoning_tokens（OpenAI 已包含在 completion_tokens 内，不重复相加）
  - total = input + output + cache_read + cache_write
  - **缓存语义**：cached_tokens 是缓存命中（读），不从中扣除 cache_write，否则合规服务商被低估（pi-mono 注释引用的 OpenRouter/DS4 契约）
- **成本计算**：usage 各分量 × 模型定价表（区分 input / output / cache_read / cache_write 单价），得到单次调用 cost
- **用途**：① 评测报告输出每样本 token 与成本（成本控制）；② 生产侧监控告警；③ chat_json / chat 响应同步带回 usage（非流式接口）
- **落地形态**：LLM 适配器返回 `{ content / stream chunks, usage: { input, output, cacheRead, cacheWrite, reasoning, total, cost } }`，由 telemetry 记录，不侵入 processors 业务逻辑

#### 6.5.1 与 Langfuse 的分工（不冲突，边界明确）

两者看似重叠（都在收集 token 用量），实际是"**数据生产 vs 数据消费**"的关系，按以下规则划清边界：

1. **LLM 适配器是 usage 的唯一解析点**：无论是否启用 Langfuse，适配器都解析 usage（pi-mono 模式）。原因：评测/CI/本地默认跑 Null telemetry，没有 Langfuse 实例——若用量只依赖 Langfuse 自动 instrumentation，评测就拿不到 token 与成本数据。评测成本统计是 6.5 的第一用途，必须脱离 Langfuse 成立。
2. **Langfuse 只是消费者之一**：LangfuseTelemetry 适配器把适配器解析好的结构化 usage 写入 Langfuse generation（traces 层面仍记录完整链路）；NullTelemetry 则写入本地结构化日志。两者同一数据源，只是出口不同。
3. **禁止双份解析**：不启用 langfuse 的 OpenAI SDK 自动 instrumentation（langfuse 包装的 OpenAI 客户端 / 自动钩子），否则同一请求的 usage 会被解析两次、上报两次，Langfuse 后台统计翻倍。约定：LLM 调用一律走原生 OpenAI SDK（6.1 的 baseURL 注入），usage 由适配器统一解析后经 Telemetry 端口上报。
4. **数据流**：LLM API 响应 → LLM 适配器解析 usage（唯一解析点）→ Telemetry 端口 → NullTelemetry（本地日志，评测直接消费）/ LangfuseTelemetry（写入 Langfuse）。
5. **评测独立于遥测**：benchmark 的 runner 直接读 LLM 适配器返回的 usage 生成成本报告，不经过 Langfuse；Langfuse 只服务于生产可观测性。

#### 6.5.2 labeled 聚合与并发归属（评审补充，W3 验证过的能力不得丢失）

- **per-call-type label 是 usage 设计的一等公民**：每次调用可携带 `call_label`（`answer` / `judge` / `usage_judge` / `manage` / `consolidate`），聚合器按 label 分桶——W3 报告的"分组件 cache hit rate"（answer 42.45 / usage_judge 39.34 / manage 9.17 / consolidate 1.57）依赖此能力，口径定义见 RUN_RECORD 6.2。目标设计的"适配器返回 usage、telemetry 记录"如果不保留 labeled 聚合，benchmark 的分项报告直接消失
- **并发归属用 contextvars，不用 threading.local**：现状 `UsageStats.thread_snapshot()` 以 threading.local 实现按线程归属（benchmark 多线程 worker 各取各的增量）。async-first 化后 threading.local 对 async task **静默失效**——不报错、只把用量归错桶。contextvars 对 thread 与 task 都生效，benchmark 从多线程迁到 asyncio 也平滑
- **定价表维护位置**：仓库内 JSON（覆盖常用模型：DeepSeek / MiniMax / GLM 等）+ 配置覆盖；字段名留 tiered rates（按累计用量分档）与 cache_write 长时写加价（Anthropic 1h 2x）的扩展位——首版平面价格即可，但字段别写死

#### 6.5.3 错误规范化与调试钩子（pi-mono 可选借鉴）

- **结构化错误形状**对齐 pi-mono `NormalizedProviderError`：`{status, body, message}`，body 截断（4000 字符）防日志爆炸——6.3 说"抛结构化错误"但没定形状，直接对齐这个
- **on_request / on_response 钩子**（与 telemetry 分离的请求/响应观察点）：调试 provider quirk 时极有用（例如确认 extra_body 实际发出了什么、服务端到底回了什么头）。优先级低，实现期按需

---

## 七、Telemetry 设计（参考 pi-mono）

> pi-mono 参考：packages/telemetry（@earendil-works/pi-telemetry）——厂商中立的遥测契约包：TelemetryContext / TelemetrySpan 回调契约、NOOP 默认实现、InMemory 参考实现、adapter conformance 测试、声明式 typed schema。

### 7.1 设计原则（从 pi-mono 提炼）

1. **厂商中立契约**：核心只定义 span 生命周期（startSpan / addEvent / setAttributes / setStatus），不依赖任何具体后端——无 exporter、无全局 current-span 状态、无后端 SDK 依赖。OpenTelemetry / Sentry / Langfuse / 日志都是适配器。
2. **显式上下文，非隐式全局**：telemetry context 作为参数显式传递（默认 NOOP），不依赖全局单例。对比现状的 get_client() 隐式全局——不可测试、无法多实例隔离。
3. **遥测不影响业务**：span 是诊断数据不是业务状态；NOOP 实现零副作用；span 回调异常不得改变业务结果；业务异常自动转为 span error status。
4. **Schema 与实现分离**：各包定义自己的 domain schema（span 名、start/end 属性、必填标记、事件、父级约束），类型系统保证埋点与 schema 一致。
5. **span 生命周期由 startSpan 拥有（无公开 end()）**：pi-mono 的 startSpan 以回调结算——回调返回即 span 结算、异常自动置 error status。Python 侧等价实现为上下文管理器语义（`async with telemetry.span(...) as span:`），杜绝"start 了忘 end"的泄漏路径；父 span 结算后子 span 自动转投 NOOP

### 7.2 NeuraMem 落地映射

| pi-mono | neuramem |
|---|---|
| TelemetryContext / TelemetrySpan 契约 | core/ports.py：Telemetry 协议（start_span / add_event / set_attributes / set_status） |
| NOOP_TELEMETRY_CONTEXT | telemetry/null.py：默认零开销 |
| InMemoryTelemetryContext | telemetry/memory.py：内存参考实现（spans/events/status 记录，异常自动记 error status） |
| adapter conformance 测试 | tests/telemetry/conformance.py：对任意适配器跑统一契约测试（第三方实现适配器时复用） |
| typed schema（TS 类型推断） | 首版用 Pydantic 定义 span 契约（span 名 / 属性集 / 必填标记），测试约束一致性 |
| 显式 context 参数传递 | 依赖注入：Memory 构造时注入 Telemetry（等价，避免隐式全局） |

### 7.3 注入与使用

```python
# 默认：无遥测（零开销，评测/测试/本地）
memory = Memory(config, telemetry=NullTelemetry())

# 生产：Langfuse 适配器（server 层装配）
memory = Memory(config, telemetry=LangfuseTelemetry(secret_key=..., public_key=..., host=...))

# 测试/评测：内存实现（可断言 span 树、生成报告）
memory = Memory(config, telemetry=InMemoryTelemetry())
```

- pipeline 组件通过构造注入拿到 Telemetry（或由 Facade 统一传入），核心代码零遥测依赖
- 遥测点只发生在端口边界：每次 LLM 调用、检索、manage / consolidate 各是一个 span
- usage（6.5）解析后作为 span 属性经同一端口上报（与 Langfuse 的分工见 6.5.1）

### 7.4 评测与遥测的关系

- benchmark runner 使用 InMemoryTelemetry 收集 span 树（检索命中、LLM 调用次数、usage、耗时），评测报告不依赖任何外部遥测后端
- Langfuse 只服务生产可观测性；评测独立于遥测（6.5.1 第 5 条）

---

## 八、通用性设计原则（第三方不侵入的六项机制）

> 定位（见文档头）：公开通用 AI 记忆库。本节定义"通用"在工程上的落地：六项机制保证第三方消费者只写胶水、不写侵入。
> 统一判据：换一个消费者还需要吗？——需要 → 进核心（机制化）；不需要 → 走通用机制；只属于某个消费者 → 留在消费者侧。

### 8.1 端口抽象（可替换性）——缺了它，换组件 = 改库

（对应调整表 #3；Telemetry 端口详见第七章）

- **改什么**：core/ports.py 定义四个 Protocol——VectorStore / LLM / Embedder / Telemetry；库内所有业务代码只依赖端口
- **为什么缺了它必然侵入**：端口是"可替换性"的唯一载体。parlasoul 想用自己的多模型 LLM 网关，现状只能侵入 llm.py（改出 _ModelRoute / LLMTextResult，13KB→29KB）；想换 PGVector/Chroma、换 embedding 服务商、换自研遥测（parlasoul 有 33KB observability.py），同样只能改库
- **有了端口后**：换组件 = 写一个实现端口的适配器（纯胶水）注入，库零改动

### 8.2 metadata / filter 透传（数据扩展）——缺了它，业务字段只能硬写进库

（对应调整表 #16）

- **改什么**：manage_async(..., metadata: dict) 写入通道 + search(..., filter: dict) 过滤通道；库只搬运、不解析字段含义
- **为什么缺了它必然侵入**：第三方需要带自己的持久化字段（parlasoul 的 character_id 在 memory.py 出现 83 处、污染每个方法签名）——没有透传通道，业务字段只能硬编码进库的实体构造和 filter 构建
- **落地**：metadata 展开写入 Milvus 动态字段（enable_dynamic_field=True 已具备，零迁移）；filter 为结构化 dict，由 store 安全编译成表达式（顺带消除字符串拼接注入风险）
- **边界**：metadata 字段只做过滤圈定，不参与向量相似度；字段语义的解释权归调用方
- **评测 provenance**：LoCoMo ingest 通过同一 metadata 通道写入扁平 `provenance_*` 字段（sample/session/turn/date/pointer）；benchmark 根据这些字段计算来源指针 recall，旧数据没有 provenance 时结果保持 unknown，不伪造 miss

### 8.3 配置化（行为参数化）——缺了它，调行为 = 改代码

（对应调整表 #8）

- **改什么**：pydantic-settings 分层配置（LLMConfig / EmbeddingConfig / StoreConfig / RetrievalConfig），校验 + .env 覆盖；行为参数（k 值、阈值、开关、top-k、时间窗口）全部配置化
- **关键：prompt 可替换**——第三方想改记忆决策规则（"只记用户说的公司名"），通过配置注入自定义 prompt，而不是改 prompts.py（现状 self._prompt 硬编码在库内）
- **为什么缺了它必然侵入**：行为差异是最普遍的定制需求，它不应该是代码改动；prompt 硬编码 = 决策规则定制 = 必须改库

### 8.4 async-first（消费模型契约）——缺了它，第三方被迫写并发补丁

（对应调整表 #7）

- **改什么**：公共 API 原生 async（manage_async / search_async / consolidate_async），同步方法降级为薄 wrapper；store / embed 底层原生 async（async 自底向上传染，否则每层都要 to_thread）
- **为什么缺了它必然侵入**：库内部同步阻塞从外面修不了——parlasoul 被迫写 concurrency.py（run_blocking + anyio 线程池限流 4 workers），且只能改库内部
- **边界**：主流消费模型（FastAPI 等 async 应用）零成本；同步消费者（脚本/CLI）付一次薄包装
- **同步 wrapper 实现约束**：`asyncio.run` 在已有事件循环的上下文会直接抛 `RuntimeError`——同步 wrapper 的实现策略与限制必须写进 API 文档（独立线程跑 loop，或明确声明不得在 async 上下文调用）
- **命名惯例**：保持 `_async` 后缀（与现状代码及 benchmark 脚本一致）。备选方案"async 方法无后缀 + `_sync` 后缀薄包装"（更贴 Python 社区惯例）已评估：收益是口味层面的，不抵迁移噪音与文档/脚本联动改动，放弃

### 8.5 单 provider 无 fallback（策略与机制分离）

（对应调整表 #8）

- **改什么**：LLM 收敛为单一 OpenAI-compatible provider（base_url + api_key + model 配置）
- **为什么这是通用化的一部分**：fallback / 多模型路由是策略，不是机制——"DeepSeek 挂了切 GLM"是每个消费者自己的业务决策（且各不相同）。库内置 fallback = 替所有消费者做同一个做死的决定
- **落地**：第三方用自己的多模型路由实现 Ports.LLM 注入；库只提供机制（重试针对可恢复错误、结构化错误），不提供策略

### 8.6 评测（能力验证）——缺了它，通用库的"可信"无从谈起

（对应调整表 #11；baseline 见第三章）

- **改什么**：neuramem_benchmark/（LoCoMo 专用）：对话重放 → manage/search → recall@k（规则）+ 端到端 accuracy（LLM-as-judge）+ 无记忆 baseline（memory uplift）
- **为什么这是通用化的一部分**：通用库对第三方是黑盒承诺（"记忆能力是对的"）；评测把承诺变成可验证数字：每次发版跑 LoCoMo 作为回归基线，第三方升级前可自行跑
- **与 10.1 的关系**：semver 管签名契约，LoCoMo 管行为契约——两者配套才是完整承诺

### 8.7 第三方集成检查清单（决策树）

| 第三方需求 | 对应机制 | 落点（胶水） |
|---|---|---|
| 换 LLM / 存储 / embedding / 遥测 | 端口（8.1） | 写适配器注入 |
| 带自己的业务字段 | metadata / filter（8.2） | 传参，库零改动 |
| 调行为参数 / 换决策规则 | 配置化 + prompt 替换（8.3） | 改配置 |
| 并发环境集成 | async-first（8.4） | 直接 await，无补丁 |
| 自己的多模型路由策略 | 单 provider + 端口（8.5） | 实现 Ports.LLM |
| 验证能力再决定用不用 | 评测（8.6） | 跑 benchmark |

判据：需求在接口两侧 → 胶水；需求穿透接口触及库的 schema / 流程 / 组件 / 自发行为 → 说明库缺机制，此时"侵入"不是第三方的错，而是**向上游提交需求的最强信号**（自下而上的通用化来源）。

---

## 九、分层职责说明

> 每层一句话定位 + 核心职责 + 依赖规则。对应"一、目标架构图"的四个子图 + 两个外部层。

| 层 | 定位 | 核心职责 | 依赖规则 |
|---|---|---|---|
| **core/（领域层）** | 纯逻辑，零 IO | models（Pydantic 领域模型，贯穿三层）、ports（VectorStore / LLM / Embedder / Telemetry 四个协议）、retry / exceptions 原语 | **不 import 任何 IO 库**（openai / pymilvus / fastapi / langfuse 一律不出现） |
| **pipeline/ + memory.py（编排层）** | 业务编排 | 五个组件：retrieval（检索+叙事扩展）、episodic（LLM 决策 CRUD+候选选择）、semantic（提炼+淘汰）、narrative（聚类）、usage_judge（再巩固）；memory.py 薄 Facade 只做组装与编排 | **只依赖 core/ports**，不知道任何具体适配器 |
| **llm/ embed/ store/ telemetry/（适配器层）** | 外部世界的实现 | 每个端口至少一个实现：OpenAI-compatible LLM/Embedding、Milvus + InMemory store、Null/InMemory/Langfuse telemetry | **实现 ports**，由装配层（deps）注入，不反向依赖编排层 |
| **neuramem_server/（服务层）** | HTTP 适配壳 | routers（REST/SSE 契约）、schemas（请求响应模型）、deps（组件装配/依赖注入）、exceptions（领域异常 → HTTP 状态码） | 依赖核心库公共 API；**不含业务逻辑** |
| **neuramem_benchmark/（评测层）** | 能力验证 | LoCoMo 数据加载 → 对话重放 → recall@k + LLM-judge + 无记忆 baseline → 报告 | 消费公共 API（Memory 门面），黑盒驱动 |
| **消费者层（外部，如 parlasoul）** | 胶水 | 写适配器 / 配置 / 包装（MemoryRepository、Telemetry 实现、filter 传参） | 只写胶水不写侵入（第八章六项机制保证） |

**依赖方向（总纲）**：消费者 → server → 编排 → 领域 ← 适配器。箭头永远向内，任何一层不反向依赖外层。

---

## 十、定位与消费者策略

> 定位见文档头：公开通用 AI 记忆库。本节定义对消费者的承诺与切换路径。

### 10.1 公共 API 承诺（semver 契约 + 行为回归）

- **签名契约（semver）**：Memory 门面（manage / search / consolidate / delete / reset / assign_to_narrative_group）与数据类型（MemoryRecord / ConsolidationStats）是版本化契约：
  - PATCH：只修 bug，不改变行为；MINOR：向后兼容新增；MAJOR：允许破坏性变更
  - 演进规则：**加方法不动签名**；已有方法只加带默认值的关键字参数；破坏性变更须 major 版本 + 迁移指南
- **行为契约（LoCoMo 评测）**：签名不变不代表行为不变（prompt/检索算法/阈值改动都不改签名）——每次发版跑 benchmark 作为回归基线，准确率/成本趋势可见
- 两者配套：**semver 管"能不能调用"，评测管"调用了对不对"**，缺一不可

### 10.2 parlasoul 切换路径

1. 删除本地 src/memory_system/ 副本
2. pip install neuramem（或 -e 开发模式）
3. import 变更：from src.memory_system.* → from neuramem.*
4. 三层适配：
   - MemoryRepository 保留（改用 neuramem 导入；错误映射 / 超时 / 注入模式不变）
   - LLM 走端口注入（LLMGatewayService 实现 Ports.LLM，或复用 neuramem OpenAI-compatible 适配器）
   - Telemetry 注入（observability 实现 Telemetry 端口，或使用 Null/InMemory）
5. 验收：parlasoul tests + evaluation/ 全绿 = 切换成功；功能对照逐项核对（吸收清单见 docs/parlasoul_backport.md）

---

## 十一、两段式闭环接口设计（回答编排落点）

> 对应调整表 #14/#12 的接口设计。核心决策：**回答生成归消费者，库提供一对公共调用完成再巩固闭环**，`SearchResult` 作为两段之间的关联令牌（correlation token）。
>
> 现状证据（设计出发点）：server `/v1/chat` 完全没有跑闭环（只有 search → 回答 → 后台 manage，记忆永远 group_id=-1）；demo 的闭环靠私有属性伸手（`memory._memory_usage_judge`）+ 精确文本匹配把 judge 返回的 text 映射回 id（LLM 轻微改写即静默丢 assign）；回答用的 LLM 也是掏 `memory._llm_client` 私有属性。

### 11.1 接口形态

```python
# core/models.py
class RetrievalTraceHit(BaseModel):
    memory_id: int
    memory_type: str
    group_id: int
    distance: float | None
    score: float | None
    is_seed: bool
    source: str


class RetrievalTrace(BaseModel):
    status: str
    elapsed_ms: int | None
    seed_ids: list[int]
    expanded_ids: list[int]
    semantic_ids: list[int]
    expanded_group_ids: list[int]
    hits: list[RetrievalTraceHit]


class SearchResult(BaseModel):
    query: str
    user_id: str
    episodic: list[MemoryRecord]
    semantic: list[MemoryRecord]
    retrieval_trace: RetrievalTrace

    def render(self, max_episodic: int | None = None, max_semantic: int | None = None) -> str:
        """Format the memory block for an answer prompt (pure, no IO).

        Defaults render everything search returned (retrieval config owns
        quantity); consumers may cap explicitly (e.g. server keeps [:5]).
        """


class UsageReport(BaseModel):
    judged_candidates: int
    used_memory_ids: list[int]
    assignments: dict[int, int]  # memory_id -> group_id


# memory.py (facade) — the two-phase pair
result = await memory.search_async(query, user_id)
answer = my_own_llm(result.render(), my_history, query)   # consumer's model/prompt/streaming
report = await memory.report_usage_async(result, answer)  # judge -> assign, closes the loop
```

### 11.2 设计要点

- **render() 只格式化记忆块**（编号 episodic/semantic 列表，收编现状 chat.py 与 demo 的重复段）；历史消息与用户消息**不进 render**——历史是消费者会话状态（服务端请求体 / demo 内存 / benchmark 数据集）。#12 去重边界的精确定义：**库消除"记忆相关"的重复，不消除"调 LLM"的重复**
- **retrieval_trace 是 transient correlation data**：`SearchResult` 额外携带 seed/expanded/semantic ids、group 扩展集合、每条命中的 distance/score/source 和检索耗时；它服务评测与诊断，不进入 `MemoryRecord` 持久化，也不改变 REST response schema
- **render() 默认不截断**：W3 口径下 semantic 是全量返回（USE_ALL_SEMANTIC=true），现状 server chat.py 的 `[:5]` 截断与检索配置互相打架（检索侧刻意全量、拼 prompt 侧丢弃）。目标接口单一职责——**数量由检索配置决定，render 只负责格式化**，默认渲染 search 返回的全部；server 维持现状需显式传参（`render(max_episodic=5, max_semantic=5)`）。这同时保证 benchmark 接入新 API 后不改变 W3 的注入行为（semantic 全量进 prompt），W4 vs W3 对比口径不被污染
- **report_usage_async 是薄编排**（Facade 只做编排）：空候选直接返回空 report 不发 LLM；judge 异常吞掉返回空 report（对齐现状保守降级——回写通道绝不能打断消费者的回答路径）；可安全重复调用，不承诺去重
- **judge 协议 text→id**：候选记忆带 id 进 prompt，返回 `used_episodic_memory_ids: [1, 3, 7]`（id 数组形态的稳定性已被 W3 验证——memory_manager 为 MiniMax-M3 的 bare-id delete 做的兼容是同一教训）。行为变更，W4 验证
- **usage 标签保留**：report_usage 内的 judge 调用带 `call_label="usage_judge"`（6.5.2）
- **server 连带修正**：chat 路由不再伸手 `memory._llm_client`——deps.py 从同一份配置装配自己那个 LLM 适配器实例用于回答；Memory 只走两段式接口。server 就是一个普通消费者（恰好同仓分发、共享配置）
- **闭环是"增强回路"不是"正确性回路"**：消费者不调 report_usage，系统退化为开环（search 只有种子无组扩展、manage/consolidate 照常），不出错、只是检索质量停在纯向量 top-k 水位——这是两段式（而非库内自己闭环）没有正确性代价的根基

### 11.3 消费者接入

| 消费者 | Phase 1 | 回答 | Phase 2 |
|---|---|---|---|
| server /v1/chat | `search_async` | 自己的 LLM 适配器（SSE 流式） | `report_usage_async`（done 后后台 task，现状缺失的闭环补上）+ `manage_async` |
| demo | `search_async` | 自己的 LLM | 同上 |
| benchmark runner | `search_async` | runner 的 answer 调用 | `report_usage_async`（W3 流程的等价复现） |
| parlasoul | `search_async` | 它自己的模型网关 | `report_usage_async` |

- benchmark 是这套 API 的第一个全量消费者——W4 一跑，接口不完备立刻暴露（内建 dogfooding 校验）
- 可选 REST 加法：`POST /v1/usage_report`（body 为 SearchResult + answer_text），服务"用 /v1/search 拿记忆、自己生成回答"的外部客户端；向后兼容新增，符合 10.1 MINOR 规则；第一版可不加（主要消费者都是进程内）

### 11.4 明确不做的（边界论证）

- **不建 pipeline/answer.py、不提供 Memory.answer()**：回答模型 / prompt / 流式形态是消费者核心差异（parlasoul 有自己的网关），库接管反而逼人绕开
- **SearchResult 不持有 Memory 引用**（不做 `result.report_usage()` 方法糖）：保持纯数据可序列化——REST 透传、benchmark 落盘、跨进程都成立；方法糖省一行代码的代价是模型层反向依赖编排层，违反第九章依赖规则
