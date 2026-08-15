# NeuraMem LoCoMo Benchmark — Run Record

> 本文件记录 2026-08-14/15 首次完整跑分（--all-samples）的环境与配置，用于系统升级后复现同口径跑分并做前后对比。
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

### 6.1 记忆工作流变量（Memory Workflow as a Benchmark Variable）

记忆系统在评测中的工作形态是**核心变量之一**：评测脚本决定系统哪些能力被激活，因此不同工作流跑出的分数**不可直接对比**。**LLM 模型也是变量**（不同模型的推理能力直接影响分数）。目前定义三种工作流：

| 工作流 | 摄取阶段（Phase 1） | 评测阶段（Phase 2，每题） | 状态 | 分数 |
|---|---|---|---|---|
| **W1 episodic-only 基线** | manage（episodic CRUD），无 consolidate | search（episodic top-5 向量）→ LLM 回答 → LLM 判分 | 已跑（2026-08-14） | **54.48%** |
| **W2 完整系统闭环** | manage（episodic CRUD）+ **每 7 个 session consolidate 一次**（模拟周期巩固；末尾不补——无后续消费方） | search（episodic top-5 + 叙事组扩展 + semantic 全量）→ LLM 回答 → usage judge（判断哪些检索记忆被用到）→ assign_to_narrative_group（用到的记忆聚成叙事组）→ LLM 判分 | 代码已就绪，**分数待重跑** | 待定 |
| **W3 完整闭环 + MiniMax-M3** | 同 W2 | 同 W2；**全链路 LLM = MiniMax-M3**（api.minimaxi.com/v1，OpenAI 兼容；`LLM_EXTRA_BODY={"thinking":{"type":"disabled"}}` 关闭 thinking，与 OpenViking 官方口径对齐；manage/consolidate/usage judge/answer/judge 全部走 M3，评测禁用 fallback 单 provider 模式） | 代码已就绪（llm_config.py apply_minimax_primary），**分数待重跑** | 待定 |

```text
W1 流程: manage -> search(top-5) -> answer -> judge                                    (deepseek-chat)
W2 流程: manage -> consolidate -> search(top-5 + 组扩展 + semantic) -> answer -> usage judge -> 叙事分组 -> judge   (deepseek-chat)
W3 流程: 同 W2，全链路 LLM 换成 MiniMax-M3（thinking 关闭：LLM_EXTRA_BODY={"thinking":{"type":"disabled"}}）
```

与 demo/app.py 的对应关系：consolidate = run_consolidation 按钮；usage judge + 叙事分组 = _process_memory 的 reconsolidation 闭环；search 叙事组扩展 = 混合检索的叙事链扩展。

**对比规则**：
- 系统代码升级对比（如升级前 vs 升级后）：必须使用**同一工作流**，否则分数差异无法归因；
- W2 vs W1 的分数差距 = 完整系统相对子集的增益，可作为单独结论报告，但不得与系统升级混为一谈；
- 报告任何分数时都必须声明工作流（如 54.48%@W1）。

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
