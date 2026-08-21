# NeuraMem LoCoMo Benchmark — Historical Run Record

> 本文件记录 2026-08-14/15 首次完整跑分（--all-samples）的环境与配置，用于系统升级后复现同口径跑分并做前后对比。
> W3（完整闭环 + MiniMax-M3）跑分记录见第 8 节（2026-08-17，69.16%）。
> 当前稳定 ID 重跑结果见第 10 节；日常运行入口和目录结构见 docs/benchmark.md。
> API keys 一律不写入本文件，均从 .env 读取；下文 `<from .env>` 表示该变量的值只存在于 .env。

## 1. 本次跑分概要

| 项 | 值 |
|---|---|
| 跑分时间 | 2026-08-14 22:13 起，完整跑分结束于次日 00:3x |
| 数据 | data/locomo10.json（10 样本，1986 QA，排除 cat-5 后 1540 题） |
| 命令 | 见第 5 节 |
| **记忆工作流** | **W1 episodic-only 基线**（定义见 6.1） |
| 总体准确率 | **54.48%**（839/1540 CORRECT，W1 工作流） |
| 分项 | 1-multi-hop 56.74% / 2-temporal 47.35% / 3-open-domain 48.96% / 4-single-hop 57.07% |
| 平均单题时延 | 2.05s（search + answer + judge） |
| 结果文件 | result/locomo_neuramem_all_results.csv、result/summary.txt |

## 2. 代码状态

```text
git commit : 449cc14 (2025-12-12) — baseline HEAD
workspace : 存在未提交改动，跑分实际使用的工作区内容（非 HEAD 内容）
```

**跑分前必须确认的未提交/未跟踪内容**（升级或重跑前建议先 git add + git commit 保存）：

- benchmark/ 整个目录当前是 untracked（含本文件、run_benchmark.py、run_eval.py、import_to_neuramem.py、locomo_prompts.py、judge.py、rejudge.py、stat_results.py）
- 跑分相关的已修改文件：
  - src/memory_system/clients/milvus_store.py（连接重试 + 超时：connect_timeout=30、connect_retries=5、retry_backoff=3）
  - src/memory_system/clients/llm.py / embedding.py（重试默认 LLM_MAX_RETRIES=10、LLM_BASE_DELAY=1.0，可用环境变量覆盖）
  - src/memory_system/utils/retry.py（退避上限 max_delay=30s + ±20% 抖动）
  - src/memory_system/config.py、src/memory_system/memory.py（历史改动）
  - tests/test_clients.py（fallback 测试显式 max_retries=3）

## 3. 运行环境

| 项 | 值 |
|---|---|
| 操作系统 | Windows（PowerShell） |
| Python 解释器 | E:\Anaconda\envs\Langchain_learn\python.exe（conda env Langchain_learn） |
| Milvus | 远程 standalone **v2.5.9** @ 117.72.161.187:19530（docker，etcd/minio healthy） |
| 集合 | memories（另有 memories_dev、groups_dev 历史集合） |

关键依赖版本（pip list 摘录）：

```text
pymilvus      2.6.4
grpcio        1.76.0
openai        2.29.0
langfuse      3.10.5
python-dotenv 1.1.1
numpy         2.3.1
pydantic      2.12.5
protobuf      6.33.1
tqdm          4.67.1
```

## 4. 配置变量（.env，值从 .env 读取）

| 变量 | 本次值（非敏感部分） | 用途 |
|---|---|---|
| MILVUS_URL | http://117.72.161.187:19530 | 向量库地址 |
| SILICONFLOW_API_KEY | <from .env> | Embedding API key |
| SILICONFLOW_BASE_URL | https://api.siliconflow.cn/v1 | Embedding endpoint |
| SILICONFLOW_EMBEDDING_MODEL | Qwen/Qwen3-Embedding-4B（dim 2560） | 检索向量模型 |
| DEEPSEEK_API_KEY | <from .env> | 主 LLM API key（W1 基线使用；W3 起未用） |
| DEEPSEEK_BASE_URL | https://api.deepseek.com | 主 LLM endpoint（W1 基线） |
| DEEPSEEK_MODEL | deepseek-chat | 主 LLM（W1 基线：回答 + 判分共用） |
| MINIMAX_API_KEY | <from .env> | **MiniMax key（W3 起全链路唯一 LLM）** |
| MINIMAX_BASE_URL | https://api.minimaxi.com/v1 | MiniMax OpenAI 兼容 endpoint（官方文档） |
| MINIMAX_MODEL | MiniMax-M3 | 评测全链路模型（manage/consolidate/usage judge/answer/judge） |
| LLM_EXTRA_BODY | {"thinking":{"type":"disabled"}} | 关闭 M3 thinking（与 OpenViking 官方口径对齐；未设置则 M3 默认开启） |
| OPENROUTER_API_KEY | <from .env> | 备用 LLM key（W3 起评测禁用 fallback，单 provider 模式） |
| OPENROUTER_BASE_URL | https://openrouter.ai/api/v1 | 备用 LLM endpoint |
| GLM_API_KEY | <from .env> | GLM 备用 key |
| GLM_BASE_URL | https://open.bigmodel.cn/api/coding/paas/v4 | GLM endpoint |
| GLM_MODEL | glm-4.6v | GLM 模型 |
| LANGFUSE_TRACING_ENABLED | false | 本次跑分关闭 tracing |
| LANGFUSE_SECRET_KEY / PUBLIC_KEY / BASE_URL | <from .env> | 仅 tracing 开启时使用 |
| K_SEMANTIC | 5 | 语义检索 top-k（W1 基线 ingest 不产生 semantic，实际为 0；W2 完整工作流 consolidate 后全量注入） |
| K_EPISODIC | 5 | 情景检索 top-k（W1 实际上下文 = episodic top-5；W2 还会被叙事组扩展） |
| USE_ALL_SEMANTIC | true | 若存在 semantic 则全量注入 prompt |
| NARRATIVE_SIMILARITY_THRESHOLD | 0.8（默认） | 叙事分组相似度阈值（W1 未触发分组，扩展=0；W2 由 usage judge 驱动） |

**记忆工作流（Memory Workflow）—— 跑分核心变量之一，定义与分数映射见 6.1。**

## 5. 跑分命令（复现用）

```powershell
cd E:\code\NeuraMem
& "E:\Anaconda\envs\Langchain_learn\python.exe" benchmark/locomo/run_benchmark.py --all-samples --threads 10 --ingest-parallel 4 --output result/locomo_neuramem_all_results.csv
```

流程：STEP1 ingest（10 样本并行 4 路，每样本独立子进程）→ STEP2 eval（1540 题，10 线程，answer+judge 内联）→ STEP3 stat（stat_results.py，排除 cat-5）。

### 5.1 命令与工作流对应关系

- W1（episodic-only 基线）：`run_benchmark.py` 命令 + 未接 consolidate/usage judge 的旧版评测脚本（commit 1cd4092 之前的 benchmark 代码）；
- W2（完整系统闭环）：`run_benchmark.py` 命令 + commit e754dad 及之后的 benchmark 代码；
- 升级后重跑请明确你要对比的是哪个工作流，输出文件不要相互覆盖。

## 6. 评测口径（与 OpenViking 的差异，升级对比时保持一致）

| 项 | 本次 NeuraMem 设置 | OpenViking 设置 |
|---|---|---|
| 数据 | data/locomo10.json（与官方 locomo10.json 内容一致） | 同一份数据 |
| 答题 prompt | 与 OpenViking locomo_prompts.py 逐字一致（Step1-7） | 相同 |
| 检索 | episodic 向量 top-5（k_episodic=5），无 rerank，无时间排序 | 50 召回 + rerank top10 + 30k 字符预算，按 created_at 时间升序 |
| reference_date | 硬编码 2023 | 每个样本最后一个 session 的实际日期 |
| 判分 prompt | OpenViking lenient 模板（partial credit/paraphrase/日期±14天），无 evidence | 同一 lenient 模板，默认带 evidence 原文，另有 --strict-prompt |
| 判分模型 | W1/W2: deepseek-chat；W3: MiniMax-M3（见 6.1） | doubao-seed-2-0-pro（火山方舟） |
| cat-5 adversarial | 排除 | 排除 |
| preprocess | cat-3 取分号前第一项 | 相同 |

> 结论：54.48% 是 NeuraMem 记忆系统本体（自身检索 + deepseek lenient judge）的成绩；**不可直接与 OpenViking 官方 README 的 80–83%（Agent+OpenViking 记忆，doubao judge）横向比较**。

### 6.2 KV Cache（前缀缓存）命中率统计

自 2026-08-16 起，benchmark 输出 KV cache（前缀缓存）命中率统计，实现依据 docs/architecture_target.md #18 / 6.5（pi-mono parseChunkUsage 语义）：

- **数据来源**：MiniMax OpenAI 兼容接口对 ≥512 token 的输入自动做前缀缓存（无需请求参数），并在 `usage.prompt_tokens_details.cached_tokens` 返回命中 token 数（实测 MiniMax-M3 每次调用均返回该字段，命中时延迟明显下降）。
- **采集点**：LLMClient 内部统一解析每次成功调用的 usage（唯一解析点，不依赖 Langfuse），覆盖 answer / judge / usage judge 及叙事分组等全部 LLM 调用；`chat_json` 返回 dict 额外携带 `usage` 字段，流式路径解析最后一个 chunk 的 usage。
- **口径**：命中率 = Σ cache_read_tokens / Σ prompt_tokens（token 加权，prompt_tokens = input + cache_read + cache_write）。每次调用同时兼容 DeepSeek 风格顶层字段（`prompt_cache_hit_tokens`，SDK extra="allow" 保留）。
- **按调用类型区分（整个记忆系统口径）**：LLM 调用带 `call_label`，覆盖记忆系统自身的全部 LLM 环节：
  - **ingest 阶段**：`manage`（EpisodicMemoryManager CRUD 决策）、`consolidate`（SemanticWriter 语义提炼）——由 import_to_neuramem.py 在 ingest 结束时写入 `result/ingest_usage_stats*.json`；
  - **eval 阶段**：`usage_judge`（MemoryUsageJudge）、`answer`（记忆增强回答生成，prompt 内嵌检索记忆）——随 QA CSV 落盘；
  - **judge 判分不属于记忆系统**（评测工具），单独报告，不计入记忆系统命中率。
- **主指标**：**Memory System Hit Rate = ingest（manage+consolidate）+ eval（usage_judge+answer）合并的 token 加权命中率**；报告同时拆分 ingest / eval 两段、judge（排除项）与 overall（参考）。
- **落盘**：`run_eval.py` 输出 CSV 每行新增 `cache_hit_tokens` / `cache_prompt_tokens`（该题全部调用增量）、`answer_cache_*`（answer 增量）、`memory_cache_*`（usage_judge+answer 增量；按线程归属，多线程并发下互不串扰）；`stat_results.py` 汇总 CSV 与 `--ingest-usage-dir`（默认 result/）下的 ingest_usage_stats*.json，输出上述口径，同时写入 summary.txt。
- **说明**：首次调用也可能出现 cached_tokens（平台侧全局缓存）；benchmark 场景中 answer 调用命中主要来自固定 system prompt 与多题间重复检索的记忆内容，该指标可反映检索输出的重复度与缓存收益。

### 6.1 记忆工作流变量（Memory Workflow as a Benchmark Variable）

记忆系统在评测中的工作形态是**核心变量之一**：评测脚本决定系统哪些能力被激活，因此不同工作流跑出的分数**不可直接对比**。**LLM 模型也是变量**（不同模型的推理能力直接影响分数）。目前定义四种工作流：

| 工作流 | 摄取阶段（Phase 1） | 评测阶段（Phase 2，每题） | 状态 | 分数 |
|---|---|---|---|---|
| **W1 episodic-only 基线** | manage（episodic CRUD），无 consolidate | search（episodic top-5 向量）→ LLM 回答 → LLM 判分 | 已跑（2026-08-14） | **54.48%** |
| **W2 完整系统闭环** | manage（episodic CRUD）+ **每 7 个 session consolidate 一次**（模拟周期巩固；末尾不补——无后续消费方） | search（episodic top-5 + 叙事组扩展 + semantic 全量）→ LLM 回答 → usage judge（判断哪些检索记忆被用到）→ assign_to_narrative_group（用到的记忆聚成叙事组）→ LLM 判分 | 代码已就绪，**分数待重跑** | 待定 |
| **W3 完整闭环 + MiniMax-M3** | 同 W2 | 同 W2；**全链路 LLM = MiniMax-M3**（api.minimaxi.com/v1，OpenAI 兼容；`LLM_EXTRA_BODY={"thinking":{"type":"disabled"}}` 关闭 thinking，与 OpenViking 官方口径对齐；manage/consolidate/usage judge/answer/judge 全部走 M3，评测禁用 fallback 单 provider 模式） | 已跑（2026-08-16/17，服务器分批，见第 8 节） | **69.16%** |
| **W4 重构后系统** | 同 W3 + **冲突淘汰**（consolidate 增量合并，矛盾旧语义记忆打 retired 标记并永久过滤，物理删除仅手动 reset） | 同 W3：**模型、答题/判分 prompt、评测口径全部不变**；差异仅在实现——闭环经两段式公共 API（search_async + report_usage_async）接入、usage judge 协议 text→id、闭环保真修复（见 docs/architecture_target.md #14/#20/#21/#22 与第十一章） | 已跑（2026-08-18/19，本地分批；`--no-memory` baseline 组待跑） | **65.97%** |

```text
W1 流程: manage -> search(top-5) -> answer -> judge                                    (deepseek-chat)
W2 流程: manage -> consolidate -> search(top-5 + 组扩展 + semantic) -> answer -> usage judge -> 叙事分组 -> judge   (deepseek-chat)
W3 流程: 同 W2，全链路 LLM 换成 MiniMax-M3（thinking 关闭：LLM_EXTRA_BODY={"thinking":{"type":"disabled"}}）
W4 流程: 同 W3 + consolidate 冲突淘汰（retired 标记 + 永久过滤）；实现为重构后代码（两段式闭环、judge id 协议、闭环保真修复）
```

与 demo/app.py 的对应关系：consolidate = run_consolidation 按钮；usage judge + 叙事分组 = _process_memory 的 reconsolidation 闭环；search 叙事组扩展 = 混合检索的叙事链扩展。

**对比规则**：
- 系统代码升级对比（如升级前 vs 升级后）：必须使用**同一工作流**，否则分数差异无法归因；
- **W4 vs W3 可直接对比**：模型（MiniMax-M3 thinking off）、答题/判分 prompt、评测口径全部不变，差值归因于重构（闭环保真修复、judge id 协议）与新增冲突淘汰机制——这是重构效果的主对比组；
- W2 vs W1 的分数差距 = 完整系统相对子集的增益，可作为单独结论报告，但不得与系统升级混为一谈；
- 报告任何分数时都必须声明工作流（如 54.48%@W1 / 69.16%@W3）。

验证记录（冒烟，2026-08-15）：
- 摄取：sample_1 摄取 8 sessions，第 7 个 session 后 consolidate 一次（48 episodic → +4 semantic），末尾无 consolidate；
- 评测：5 题评测中 expanded=1→2→4（叙事组扩展随分组增长而生效），每题目时约 7s（含一次 usage judge LLM 调用）。

注意：W2 的分数**尚未跑出**（54.48% 是 W1 基线）；重跑 W2 后应在本节补充新分数，输出文件建议使用独立文件名（如 locomo_neuramem_full_results.csv），保留 W1 基线 CSV 以便逐题对比。

## 7. 升级后重跑注意事项

1. 升级前先 git add -A && git commit（尤其 benchmark/ 目前 untracked），必要时打 tag 记录本次跑分代码基线；
2. 用同一份 .env（或同值新 env），确认 LANGFUSE_TRACING_ENABLED=false、K_EPISODIC=5、USE_ALL_SEMANTIC=true；
3. 确认 Milvus 117.72.161.187:19530 可用（v2.5.9），集合名 memories；
4. 重跑命令与第 5 节一致；ingest 会自动 reset 各样本（sample_0~sample_9）后重写，不产生跨次污染；
5. 对比口径：新旧两次跑分之间只允许系统代码差异，评测脚本 / .env / 数据 / 服务器不应变化；
6. **声明工作流**：每次跑分必须记录使用的工作流（W1 / W2），同一工作流下才能做系统升级对比（见 6.1）；
7. 若升级包含检索策略变化（如 k、rerank、时间排序），在第 6 节表格中追加记录，避免对比时混淆。

## 8. W3 完整跑分记录（2026-08-16/17，服务器）

| 项 | 值 |
|---|---|
| 跑分时间 | 2026-08-16 20:10 ~ 2026-08-17 09:08（ingest 与 eval 分批跨限额窗口执行） |
| 运行位置 | 服务器 `/root/neuramem`（GitHub 部署，HEAD=1ce529f，.venv；Milvus 同机） |
| 数据 | data/locomo10.json（同 W1；实际评出 1537/1540——s2/s4/s9 各 1 题偶发丢失，影响 0.2%） |
| 工作流 | **W3 完整闭环 + MiniMax-M3**（thinking 关闭） |
| **总体准确率** | **69.16%**（1063/1537） |
| 分项 | 1-multi-hop 71.79% / 2-temporal 61.68% / 3-open-domain 60.42% / 4-single-hop 72.14% |
| 平均单题时延 | 19.11s（含 usage judge / 叙事分组 / judge 判分，及 429 限流重试） |
| 结果文件 | result/locomo_neuramem_w3_results.csv、result/summary_w3.txt（已从服务器同步回本地） |

W1 → W3 对比：multi-hop +15.05 / temporal +14.33 / open-domain +11.46 / single-hop +15.07 / 总体 **+14.68pp**。注意 W1→W3 同时改变了**工作流**（episodic-only → 完整闭环）与**模型**（deepseek-chat → MiniMax-M3）两个变量，增益不可单独归因（见 6.1 对比规则；W2 未单独跑分）。

各样本准确率：s0 74.3 / s1 75.3 / s2 78.8 / s3 63.3 / s4 75.7 / s5 61.0 / s6 80.7 / s7 58.1 / s8 64.7 / s9 65.0。

### 8.1 执行方式（受 MiniMax 5h 限额约束，分批跑）

- **ingest**：cron 串行全量任务中断后，改为按样本 nohup 单独执行；每样本完成即落盘 `result/ingest_usage_stats_N.json`。
- **eval**：每样本独立进程 `run_eval.py --sample N --threads 2~4 --output result/w3_eval_sN.csv`；最后按 (sample_index, question_id) 去重合并为 `locomo_neuramem_w3_results.csv`。
- **事故记录**：2026-08-16 下午手动重跑 s6–s9 ingest 时 s8 被中断（残留 201 条残缺记忆），导致 s8 首轮 eval 仅 32.7%（平均检索命中 5.4 条/题）；8-17 上午重跑 s8 的 ingest+eval 后恢复 64.7%。**教训：样本 ingest 完成后必须核对 Milvus 记忆数与日志「Import complete」行，再进入 eval 阶段。**
- **429 限流**：MiniMax 偶发 429 Too Many Requests，SDK 重试退避可全部消化；5 进程 × 2 线程（总并发 10）时依然稳定。

### 8.2 KV cache 命中率（W3 首次输出，口径见 6.2）

- **Memory System Hit Rate = 19.06%**（token 加权）：answer 42.45% / usage_judge 39.34% / manage 9.17% / consolidate 1.57%；judge 48.38% 按口径排除。
- 覆盖范围：eval 阶段覆盖全部 10 样本；ingest 阶段 usage JSON 仅保留 s6/s7/s8/s9 四份（s0–s5 的统计因 8-16 14:00 串行任务中断且当时为进程末尾统一落盘而丢失）。
- 观察：eval 阶段命中率（~40%+）显著高于 ingest 阶段 manage（~9%）——answer 的 system prompt 与跨题重复检索的记忆内容形成稳定前缀，而 manage 每次上下文为「新对话 + 检索记忆」，前缀重复度低。

## 9. 工具链迁移记录（2026-08-18，重构 Step 5）

`benchmark/locomo/` 迁移至 `neuramem_benchmark/`（本目录），W 编号与对比规则原样延续。入口变化：

| 旧入口 | 新入口 |
|---|---|
| `python benchmark/locomo/import_to_neuramem.py` | `python -m neuramem_benchmark.ingest` |
| `python benchmark/locomo/run_eval.py` | `python -m neuramem_benchmark.runner` |
| `python benchmark/locomo/stat_results.py` | `python -m neuramem_benchmark.report` |
| `python benchmark/locomo/run_benchmark.py` | `python -m neuramem_benchmark.run_benchmark` |
| `python benchmark/locomo/judge.py` / `rejudge.py` | `python -m neuramem_benchmark.judge` / `python -m neuramem_benchmark.judge --include-wrong` |

行为要点：
- **ingest 完整性 manifest**（8.1 教训内建化）：每个样本 ingest 结束写 `result/ingest_manifest_{idx}.json`（含最终 Milvus 计数）；runner 默认拒绝无 manifest 的样本（`--no-manifest-check` 可越过）。
- **eval 闭环走公共 API**：`search_async` → runner 自答 → `report_usage_async`（judge id 协议在库内），旧 `_memory_usage_judge` 伸手与 text 匹配删除。
- **并发模型**：线程 → asyncio semaphore；usage 归属 contextvars scope（每题一个 scope，快照即增量），CSV 列名与 W3 完全一致。
- **新增 `evidence_recall` 列**：OpenViking 指针解析规则（D{session}:{1-based 位置} → "speaker: text"）+ 子串命中，W4 校准。
- **per-question trace JSONL**：runner 默认随 CSV 写 `<output 前缀>.trace.jsonl`，每题一行——完整 answer prompt（system+user）、检索记忆全文（episodic/semantic 分列）、evidence 逐指针命中、分段耗时（retrieval/answer/usage_judge/judge）、按 label 的 usage 明细、判分 raw 输出。错题归因与性能分析数据集（`--no-trace` 关闭）。
- **答题 prompt 单一源**：canonical `build_answer_prompt` 上移至 `src/neuramem/prompts.py`，benchmark（ref="2023"，快照验证字节不变）/ server `/v1/chat` / demo（ref=当前年，不截断、回写喂 `extract_final_answer` 后的最终答案）三消费者同构——评测分数与产品行为一致。模板两句 LoCoMo 时间硬编码参数化为年份窗口（ref=2023 时还原原句）。
- **id 协议异常留痕（#14 可观测）**：usage judge / semantic writer 返回的幻觉 id（不在候选集）与坏值（非整数）不再静默丢弃——逐项容错（一个坏值不再丢整题判定），记入 `UsageReport.dropped_ids/malformed_count`（runner trace 的 `usage_report` 区块）、WARNING 日志；consolidate 淘汰时记录被淘汰 id 与原文。
- **llm_config**：`MINIMAX_API_KEY` 存在时套用 W3 档位（api.minimaxi.com / MiniMax-M3 / thinking off / 10 次 × base 1s × cap 30s）；`MINIMAX_BASE_URL`/`MINIMAX_MODEL` 可覆盖（本地可用国内端点 api.minimax.chat）。
- 环境变量兼容 legacy 名（DEEPSEEK_* / SILICONFLOW_* / MILVUS_URL）。

## 10. Stable-ID rerun (2026-08-21/22)

After the Milvus primary-key migration (`auto_id=False` plus application-side IDs), samples 0-9 were rerun with the current full closed-loop benchmark and MiniMax-M3. Category 5 adversarial questions remain excluded.

| Metric | Value |
|---|---:|
| Samples completed | 10 |
| Graded questions | 1540 |
| Correct | 1029 |
| Wrong | 511 |
| Overall accuracy | **66.82%** |
| Weighted average latency | 11.84s |
| Failed ingest turns | 1 (s8) |
| Trace ghost IDs | 0 for s0-s9 |

Per-sample results and the reproducible scorecard command are maintained in [docs/benchmark.md](../docs/benchmark.md). The generated files under `result/locomo_full_rerun/` are intentionally ignored by Git.
