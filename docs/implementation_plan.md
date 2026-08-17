# NeuraMem 重构实施计划（implementation plan）

> **目标**: 按 docs/architecture_target.md（rev2）完成从 src/memory_system 到 neuramem 包化的重构，最终以 W4 跑分验收。
> **分步依据**: 第九章依赖方向——消费者 → server → 编排 → 领域 ← 适配器。步骤沿依赖方向**由内向外**推进：先建零依赖的领域层，再实现适配器，然后编排层，最后切换消费者（server/demo）并由评测验收。每一步只动一层，且结束时 main 分支完整可运行。
> **迁移模式**: strangler——Step 1–3 期间 src/neuramem（新）与 src/memory_system（旧）并存，旧路径保持 W3 可复现；Step 4 切换消费者并删除旧包；Step 5 评测迁移 + W4。
> **Date**: 2026-08-17

## 〇、开始前的一次性动作

1. **打 tag 固化 W3 基线**：`git tag benchmark-2026-08-17-69.16-pre-refactor`——旧包删除前它是 W3 的唯一复现入口，删除后是代码考古入口。
2. 确认服务器 Milvus 中 W3 数据（10 样本 + groups_{user_id} 集合）无需保留：W4 反正重新 ingest，旧 groups 集合按 #15 的结论直接弃用（benchmark 环境 reset 重导，不写迁移）。

## 一、Step 1 —— 包骨架 + 领域层（core/）+ 配置

**架构依据**: core 是依赖箭头的最内层（零 IO 依赖），一切后续步骤都依赖它；它本身不依赖任何其他步骤的产物。

**范围**:

| 动作 | 文件 |
|---|---|
| 新建单发行物 | `pyproject.toml`（neuramem 库 + extras `[server] [benchmark] [test]`，demo/benchmark 不进 wheel） |
| 包骨架 | `src/neuramem/__init__.py`（只导出公共 API + py.typed） |
| 领域模型 | `core/models.py`：MemoryRecord / SearchResult（含 render，默认不截断）/ UsageReport / ConsolidationStats，全部 Pydantic |
| 端口定义 | `core/ports.py`：VectorStore / LLM / Embedder / Telemetry 四个 Protocol。**端口设计时预留 demo 需要的组查询公开接口**（现状 demo 伸手 `_store._client` 直查 groups 集合，端口缺这个能力 Step 4 就得返工） |
| 原语迁移 | `core/exceptions.py`（含 LLMParseError）、`core/retry.py`（RetryExecutor 迁移 + 重试范围收敛 408/409/429/5xx + retry-after 上限护栏，见 6.3） |
| 配置 | `config.py`：pydantic-settings 分层（LLMConfig 含 extra_body / EmbeddingConfig / StoreConfig / RetrievalConfig），embedding 维度显式值单一来源 |

**验证**: core 纯逻辑单测（render 格式化、config 校验 fail-fast、retry 分类）；`pip install -e .` 冒烟 import。
**完成判据**: 新包可安装可 import；旧路径零改动（全测试现状不变）。

## 二、Step 2 —— 适配器层（llm / embed / store / telemetry）

**架构依据**: 适配器实现端口、只依赖 core；不碰业务逻辑，与 Step 3 无耦合。

**范围**:

| 适配器 | 从哪来 | 关键动作 |
|---|---|---|
| `llm/openai_adapter.py` | `clients/llm.py` | 删 fallback 双通道；SDK 以 `max_retries=0` 构造（重试预算单点归 RetryExecutor）；流式带 `stream_options.include_usage`（compat 门控）；detectCompat 首版（deepseek 指纹 + 默认分支）；UsageStats 归属从 threading.local 换 **contextvars**（6.5.2，call_label 分桶保留）；错误规范化 `{status, body, message}` + body 截断（6.5.3） |
| `embed/openai_adapter.py` | `clients/embedding.py` | 原生 async；dim 来自 config（消灭 `_dim = 2560` 硬编码） |
| `store/milvus.py` | `clients/milvus_store.py` | schema 兼容（memories 集合字段不变）；**结构化 filter dict 编译**（#16，顺带统一现状单双引号混用）；**groups 单集合化**（#15，废除 groups_{user_id}）；**upsert 原地更新替代 delete+add**（#17）；dim 参数化（消灭 SCHEMA_FIELDS / create_collection / create_groups_collection 三处 2560） |
| `store/inmemory.py` | 新增 | VectorStore 内存实现：暴力余弦 top-k + filter 求值 + groups 语义 + 动态字段 |
| `telemetry/` | 现状 @observe 全部拆除 | null.py（默认零开销）/ memory.py（InMemory，span 上下文管理器语义）/ langfuse.py；conformance 测试随附 |

**验证**: InMemoryStore 全量单测；Milvus 集成冒烟（用服务器实例，memories 集合新建临时名验证后删除）；telemetry conformance；**新旧 store 行为等价对照**（同一组数据/查询，两边结果一致）。
**完成判据**: 四端口各有实现；InMemoryStore 让单测零外部依赖成立。

## 三、Step 3 —— pipeline 编排层 + 薄 Facade（行为变更集中地）

**架构依据**: 编排层只依赖 core/ports（第九章）；所有行为契约变更（judge id 协议、冲突淘汰、闭环）在此层与基础设施解耦地落地。**这是 W4 分数风险最集中的一步。**

**范围**:

| 组件 | 从哪来 | 关键动作 |
|---|---|---|
| `pipeline/retrieval.py` | memory.py search 内嵌段（~120 行） | 抽出检索 + 叙事扩展；N+1 组查询合并为单次 `group_id in [...]`（#19）；返回 SearchResult |
| `pipeline/episodic.py` | processors/memory_manager.py | 迁移；维持全量候选（候选选择已从方案删除，不改） |
| `pipeline/semantic.py` | processors/semantic_writer.py | 迁移 + **冲突淘汰最简版**（#20：矛盾旧语义记忆打 retired 标记（动态字段 + upsert），检索永久过滤；物理删除仅手动 reset） |
| `pipeline/narrative.py` | processors/narrative_memory_manager.py | 迁移（含 #21 修复与新 single-groups 存储）；组操作走新 store 端口 |
| `pipeline/usage_judge.py` | processors/memory_usage_judge.py | **协议 text→id**：MEMORY_RELEVANCE_FILTER_PROMPT 改造（候选带 id，返回 used ids）——prompt 资产变更记录在案 |
| `memory.py`（Facade） | memory.py 734 行瘦身 | 两段式闭环：`search_async` → SearchResult、`report_usage_async(result, answer)` → judge→assign（空候选不发 LLM、judge 异常吞掉、call_label 保留）；manage/consolidate/delete/reset 编排；@observe 全部替换为 Telemetry 端口注入 |

**验证**: InMemoryStore + DummyLLM（tests/properties/dummy_llm.py 可复用）全流程单测：manage → search → report_usage 断言 assign 被调、冲突淘汰打标、retired 过滤；冒烟脚本对比新旧库在同一样本上的 search 输出形态。
**完成判据**: 新库独立可用，公共 API（manage/search/consolidate/report_usage/delete/reset）全部就位；旧路径仍未动。

## 四、Step 4 —— 服务层（neuramem_server）+ demo 瘦身 + 删旧包

**架构依据**: server 是核心库的第一个消费者（依赖向内：server → 编排 → 领域）；切换全部消费者与删除旧包必须同一步完成，避免长期双轨。

**范围**:

| 动作 | 细节 |
|---|---|
| `src/api/*` → `src/neuramem_server/*` | REST/SSE 契约不变（/v1/* 请求响应 schema 逐一对照）；routers/schemas/deps/exceptions 结构按第四章目录 |
| chat 路由两段式改造 | search_async → 自己的 LLM 适配器流式回答（deps 从同一份配置装配独立实例，**消灭 `memory._llm_client` 伸手**）→ done 后 report_usage_async + manage_async 后台任务（闭环补上，#14） |
| memories 路由 | ownership 预检查不再用 `memory.store` 私有（Facade 加公开方法或走端口）；user_id 格式校验（`^[A-Za-z0-9_-]{1,64}$`，#16） |
| demo 瘦身 | 复用 SearchResult.render / report_usage_async；删除 `_llm_client` / `_memory_usage_judge` / `_store` / `_store._client` 全部伸手；叙事组面板改走端口公开组查询（Step 1 预留） |
| 删除旧包 | `git rm -r src/memory_system`；全仓 import 清理 |

**验证**: REST 契约测试（用旧 schemas.py 的请求/响应模型做快照对照）；SSE 手测一轮对话（含 group_id 从 -1 变有效的闭环验证）；demo 手测。
**完成判据**: 旧包不存在，全仓无 `src.memory_system` import，server/demo 全走新库。

## 五、Step 5 —— 评测迁移（neuramem_benchmark）+ W4 验收

**架构依据**: 评测是黑盒消费者 + 行为契约的验收方（8.6 / 10.1）；放最后，因为它是重构的最终裁判，且 W4 口径依赖前四步的稳定产物。

**范围**:

| 动作 | 细节 |
|---|---|
| `benchmark/locomo/` → `neuramem_benchmark/` | 9 文件重组为 locomo.py / runner.py / metrics.py / judge.py / report.py（重组非重写） |
| 硬功能移植（缺一不可） | 多进程并行与样本选择、CSV 按 sample_index+question_id 去重合并、rejudge、KV cache 分组件报告（contextvars 归属重写 thread_snapshot 逻辑，**口径对齐 RUN_RECORD 6.2**） |
| runner 改走公共 API | `memory._memory_usage_judge` 伸手 + text→id 手工映射（run_eval.py L164-178）替换为 `report_usage_async`；答案 LLM 不再复用 `memory._llm_client`，runner 自建（同配置） |
| ingest 完整性校验内建 | 重放完成自动核对 Milvus 记忆数 + 完成日志行，未达标 fail（s8 教训，RUN_RECORD 8.1） |
| 杂项 | run_benchmark.py 硬编码 `PYTHON_EXEC` 改可配；locomo_prompts 的 200 条注入上限原样保留（W3 口径） |
| **跑 W4** | 模型/prompt/口径锁死同 W3（MiniMax-M3 thinking off + 逐字一致的答题/判分模板）；同机同参数；补跑 no-memory baseline（memory uplift）与 recall@k；结果落 RUN_RECORD（6.1 的 W4 行）与 README |

**验证**: W4 全量（10 样本 ingest + eval + 合并 + 报告）。
**完成判据**: W4 数字落档，W4 vs W3 对比（RUN_RECORD 6.1 已声明可比）成立。

## 六、步骤依赖与并行性

```text
Step 1 (core)  →  Step 2 (adapters)  →  Step 3 (pipeline)  →  Step 4 (server+switch)  →  Step 5 (benchmark+W4)
```

严格串行，无并行捷径——每步的输入是上一步的产物（依赖方向决定）。唯一可交错的部分：Step 2 的四个适配器彼此独立可并行；Step 5 的评测重组可在 Step 4 进行期间起步（它消费的公共 API 在 Step 3 末已冻结），但 W4 必须等 Step 4 完成。

## 七、风险与注意事项

1. **Step 3 是分数风险集中地**：judge id 协议、冲突淘汰、闭环接入全部是行为变更，方向对不对只有 W4 能裁决。缓解：Step 3 完成后先用 **1–2 个样本小规模试跑**（新库 ingest+eval），对比 W3 同样本分数提前发现回归，不要等 Step 5 才第一次见分数。
2. **KV cache 统计的归属迁移是隐形坑**：run_eval 现在靠 `threading.local` 每线程快照归桶；Step 2 换 contextvars、Step 5 runner 若改 asyncio 并发，口径必须与 RUN_RECORD 6.2 逐项对齐，否则 W4 的 cache 数字与 W3 不可比。
3. **#22 修复重试会波及判分成本**：judge.py 走 chat_json，判分解析失败现在会多打一次修复调用（成本 ×2 on garbage）；判分失败仍保守记 WRONG，与 W3 口径一致，但 rejudge/judge 的预算评估要把重试算进去。
4. **demo 的私有伸手比架构图显示的深**：`_store._client` 直查 groups 集合、每用户独立 memories 集合（demo_memories_{user_id}）。Step 1 端口设计必须预留组查询公开接口，否则 Step 4 返工。
5. **存量测试的处置**：13 处失败是存量（properties 部分依赖真 Milvus / v1 字段残留）。原则：依赖 InMemoryStore 能复活的按新端口重写（不硬移植），test_stream.py 直接删。测试债在本重构内清偿，不带入新包。
6. **W3 复现窗口在 Step 4 关闭**：删旧包前旧路径随时可复现 W3（安全网）；删除后只能靠 Step 〇 的 tag。若 W4 出现大滑坡需要 A/B 定位，用 tag checkout 旧代码跑对照。
7. **服务器与本地同步**：沿用 GitHub 部署惯例（服务器 git pull），每步合并后同步；Milvus 服务器上 W4 前建议清掉 W3 的 10 个样本数据与 groups 集合（reset 重导），避免新旧 schema 数据混存。
