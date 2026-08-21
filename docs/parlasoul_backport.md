# parlasoul-backend 改造吸收清单（backport）

> **目的**：重写 neuramem 时逐项兑现 parlasoul-backend 对 memory_system 的改造，确保切换后零能力损失。
> **来源**：E:\code\parlasoul-backend\src\memory_system\ 与 NeuraMem 现状对比（2025-08 分析）。
> **配套**：架构规格见 docs/architecture_target.md（六章 LLM / 七章 Telemetry / 八章通用性 / 十章 10.2 切换路径）。
> **验收**：parlasoul tests + evaluation/ 全绿 = 切换成功；下表每项在 neuramem 有落点。

## 一、回流核心（通用能力 → neuramem 标准功能）

| # | 来源（parlasoul） | 能力 | neuramem 落点 | 状态 |
|---|---|---|---|---|
| 1 | clients/llm.py: LLMTextResult | 结构化 LLM 返回（text / model / usage_details / cost_details，frozen dataclass） | 6.5 返回形态正式定义 | 待办 |
| 2 | clients/llm.py: chat_with_usage / pop_last_stream_usage_details / pop_last_stream_cost_details | 流式 usage/cost 的捕获与取出 | 6.5 usage 聚合（适配器返回 { content, usage }，不再用 pop 模式） | 待办 |
| 3 | clients/llm.py: _extract_usage_details | 多字段名兼容解析：input（prompt_tokens/input_tokens/request_tokens/input）、output、cached（cached_tokens/cache_read_*）、cache_write（cache_creation_input_tokens）、total 修正（cached 未计入时补齐） | 6.5 字段映射（parseChunkUsage Python 版） | 待办 |
| 4 | clients/llm.py: _extract_cost_details | cost 提取（cost / total_cost / cost_usd） | 6.5 成本计算 | 待办 |
| 5 | clients/llm.py: _provider_name_from_base_url / _is_openrouter_base_url / _is_dashscope_base_url | base_url 指纹识别服务商 | 6.2 compat 检测首版 | 待办 |
| 6 | clients/llm.py: _normalize_model_name | deepseek/ 前缀剥离等模型名归一化 | 6.2 compat 的一部分 | 待办 |
| 7 | clients/llm.py: _provider_extra_body | 服务商专属请求体扩展（OpenRouter 等） | 6.2 compat 的一部分（extra_body 配置） | 待办 |
| 8 | clients/llm.py: _normalize_terminal_error / _extract_auth_error_detail | 错误归一化 + 认证错误识别 | 6.3 结构化错误（status/headers/body + AuthenticationError 子类） | 待办 |
| 9 | exceptions.py: LLMAuthenticationError | 认证失败专属异常（model/provider/original_error） | 6.3 结构化异常子类 | 待办 |
| 10 | memory.py: judge_and_assign_narrative_group_async | 再巩固闭环：judge → assign，fire-and-forget 安全（never raises、best-effort 返回 used_ids） | pipeline 标准编排（调整表 #14）；run_blocking 换成原生 async | 待办 |
| 11 | memory.py: MemoryRetrievalTrace | 检索 provenance：query_fingerprint / seed_ids / expanded_ids / semantic_ids / distances / is_seed / 耗时 / trace_status（不推断、不填零） | `core/models.py: RetrievalTrace` + `RetrievalTraceHit`；`SearchResult` 始终携带 transient trace | 已完成（字段名按当前 API 收敛） |
| 12 | memory.py: search_with_trace | 带 trace 的检索入口（生产可选，评测必需） | `search_async` 默认生成 `SearchResult.retrieval_trace`，无需第二个入口 | 已完成（不另设入口） |
| 13 | memory.py: MemoryRecord 扩展字段 distance / is_seed / source_turn_id | 检索命中距离、种子标记、来源 turn（评测 provenance） | score/distance/is_seed/source 进入 transient `RetrievalTraceHit`；来源 turn 走 `metadata.provenance_*`，不污染通用 `MemoryRecord` | 已完成（采用通用机制） |
| 14 | memory.py: ConsolidationStats 扩展（physical_before/after、logical_before/after、action_counts、duration_ms、batch_fingerprint） | 增量合并 + 冲突淘汰的统计基线（before/after 对比） | 调整表 #20 的统计形态 | 待办 |
| 15 | processors/memory_usage_judge.py: MemoryUsageTrace + judge_used_memory_ids（返回 ids 而非 texts）+ trace_sink 回调 | judge 判定溯源（retrieved → used 映射、judge_status，评测钩子） | pipeline/usage_judge.py（#14）+ 7.4 评测 | 待办 |
| 16 | processors/narrative_memory_manager.py: cleanup_group | 组清理独立方法（供删除/更新时复用） | pipeline/narrative.py | 待办 |
| 17 | prompts.py: CHAT_TITLE_PROMPT | 会话标题生成 prompt | prompts.py 资产 | 待办 |
| 18 | utils/__init__.py: escape_milvus_string | 过滤表达式字符串转义 | 调整表 #16 结构化过滤器的过渡实现（首版可先用转义） | 待办 |
| 19 | memory.py: _run_coroutine_sync | 同步包装的安全化：检测运行中事件循环（比裸 asyncio.run 安全） | 同步 wrapper 的实现规范（8.4） | 待办 |
| 20 | config.py: llm_timeout_seconds / llm_max_retries / llm_retry_base_delay_seconds / milvus_token / group_collection_name | 配置项（超时、重试参数、Milvus token、组集合名） | 8.3 配置化（LLMConfig / StoreConfig） | 待办 |

## 二、走通用机制（不进核心模型）

| 业务需求 | parlasoul 现状 | neuramem 机制（8.2） |
|---|---|---|
| character_id 维度（每角色独立记忆） | 硬编码进 MemoryRecord 必填字段 + 全部方法签名 + build_filter（83 处） | metadata 透传（写入）+ filter 透传（查询）；MemoryRecord 不引入必填业务字段 |

## 三、留在消费者侧（parlasoul 保留，不回流）

| 资产 | 处置 |
|---|---|
| repositories/memory_repository.py | 保留（改用 neuramem import；错误映射 / 超时 / memory_factory 注入模式不变） |
| services/llm_gateway_service.py + core/llm_registry.py | 保留；实现 Ports.LLM 注入 neuramem（其多模型路由、认证、成本策略是消费者策略，见 8.5） |
| observability.py | 保留；实现 Telemetry 端口注入（第七章） |
| evaluation/ | 保留；业务场景评测，与 neuramem benchmark/（LoCoMo）并存 |
| core/config.py（pydantic settings） | 保留；neuramem 配置通过其读取并注入（8.3） |

## 四、重写时删除（async-first 后不需要）

| 来源 | 内容 | 原因 |
|---|---|---|
| concurrency.py（run_blocking + anyio CapacityLimiter） | 同步阻塞调用的线程池限流补丁 | 8.4：store/embed 原生 async 后阻塞消失，补丁整个删除 |
| memory.py 内所有 run_blocking(...) 包装 | 同上 | 同上 |

## 五、验收标准

1. parlasoul pytest 全绿（含 hypothesis 属性测试）
2. parlasoul evaluation/ 业务评测基线不倒退
3. 功能对照：上表 #1–#20 每项在 neuramem 有落点（状态从"待办"改为"已吸收"）
4. character_id 功能经 metadata/filter 通道等价工作（§二 对照）
5. 删除项确认无残留引用（§四）

## 六、吸收顺序建议（按重写阶段）

| 阶段 | 吸收项 |
|---|---|
| 领域层（core） | #9 异常子类、#13 MemoryRecord 扩展字段、#14 stats 形态 |
| 适配器层（llm） | #1–#8 全部（LLMTextResult / usage / cost / compat / 错误） |
| 编排层（pipeline） | #10 再巩固闭环、#11–#12 检索 trace、#15 judge trace、#16 cleanup_group |
| 资产与配置 | #17 prompts、#18 转义过渡、#19 同步 wrapper 规范、#20 配置项 |
| 删除 | §四 concurrency 相关 |
