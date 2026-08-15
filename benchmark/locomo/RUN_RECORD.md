# NeuraMem LoCoMo Benchmark — Run Record

> 本文件记录 2026-08-14/15 首次完整跑分（--all-samples）的环境与配置，用于系统升级后复现同口径跑分并做前后对比。
> API keys 一律不写入本文件，均从 .env 读取；下文 `<from .env>` 表示该变量的值只存在于 .env。

## 1. 本次跑分概要

| 项 | 值 |
|---|---|
| 跑分时间 | 2026-08-14 22:13 起，完整跑分结束于次日 00:3x |
| 数据 | data/locomo10.json（10 样本，1986 QA，排除 cat-5 后 1540 题） |
| 命令 | 见第 5 节 |
| 总体准确率 | **54.48%**（839/1540 CORRECT） |
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
| DEEPSEEK_API_KEY | <from .env> | 主 LLM API key |
| DEEPSEEK_BASE_URL | https://api.deepseek.com | 主 LLM endpoint |
| DEEPSEEK_MODEL | deepseek-chat | 主 LLM（回答 + 判分共用） |
| OPENROUTER_API_KEY | <from .env> | 备用 LLM key |
| OPENROUTER_BASE_URL | https://openrouter.ai/api/v1 | 备用 LLM endpoint |
| GLM_API_KEY | <from .env> | GLM 备用 key |
| GLM_BASE_URL | https://open.bigmodel.cn/api/coding/paas/v4 | GLM endpoint |
| GLM_MODEL | glm-4.6v | GLM 模型 |
| LANGFUSE_TRACING_ENABLED | false | 本次跑分关闭 tracing |
| LANGFUSE_SECRET_KEY / PUBLIC_KEY / BASE_URL | <from .env> | 仅 tracing 开启时使用 |
| K_SEMANTIC | 5 | 语义检索 top-k（本次 ingest 不产生 semantic，实际为 0） |
| K_EPISODIC | 5 | 情景检索 top-k（本次实际检索上下文 = episodic top-5） |
| USE_ALL_SEMANTIC | true | 若存在 semantic 则全量注入 prompt |
| NARRATIVE_SIMILARITY_THRESHOLD | 0.8（默认） | 叙事分组（本次 ingest 无分组，扩展=0） |

## 5. 跑分命令（复现用）

```powershell
cd E:\code\NeuraMem
& "E:\Anaconda\envs\Langchain_learn\python.exe" benchmark/locomo/run_benchmark.py --all-samples --threads 10 --ingest-parallel 4 --output result/locomo_neuramem_all_results.csv
```

流程：STEP1 ingest（10 样本并行 4 路，每样本独立子进程）→ STEP2 eval（1540 题，10 线程，answer+judge 内联）→ STEP3 stat（stat_results.py，排除 cat-5）。

## 6. 评测口径（与 OpenViking 的差异，升级对比时保持一致）

| 项 | 本次 NeuraMem 设置 | OpenViking 设置 |
|---|---|---|
| 数据 | data/locomo10.json（与官方 locomo10.json 内容一致） | 同一份数据 |
| 答题 prompt | 与 OpenViking locomo_prompts.py 逐字一致（Step1-7） | 相同 |
| 检索 | episodic 向量 top-5（k_episodic=5），无 rerank，无时间排序 | 50 召回 + rerank top10 + 30k 字符预算，按 created_at 时间升序 |
| reference_date | 硬编码 2023 | 每个样本最后一个 session 的实际日期 |
| 判分 prompt | OpenViking lenient 模板（partial credit/paraphrase/日期±14天），无 evidence | 同一 lenient 模板，默认带 evidence 原文，另有 --strict-prompt |
| 判分模型 | deepseek-chat（与回答同一 client） | doubao-seed-2-0-pro（火山方舟） |
| cat-5 adversarial | 排除 | 排除 |
| preprocess | cat-3 取分号前第一项 | 相同 |

> 结论：54.48% 是 NeuraMem 记忆系统本体（自身检索 + deepseek lenient judge）的成绩；**不可直接与 OpenViking 官方 README 的 80–83%（Agent+OpenViking 记忆，doubao judge）横向比较**。

## 7. 升级后重跑注意事项

1. 升级前先 git add -A && git commit（尤其 benchmark/ 目前 untracked），必要时打 tag 记录本次跑分代码基线；
2. 用同一份 .env（或同值新 env），确认 LANGFUSE_TRACING_ENABLED=false、K_EPISODIC=5、USE_ALL_SEMANTIC=true；
3. 确认 Milvus 117.72.161.187:19530 可用（v2.5.9），集合名 memories；
4. 重跑命令与第 5 节一致；ingest 会自动 reset 各样本（sample_0~sample_9）后重写，不产生跨次污染；
5. 对比口径：新旧两次跑分之间只允许系统代码差异，评测脚本 / .env / 数据 / 服务器不应变化；
6. 若升级包含检索策略变化（如 k、rerank、时间排序），在第 6 节表格中追加记录，避免对比时混淆。
