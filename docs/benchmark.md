# Benchmark Guide

NeuraMem 的 LoCoMo benchmark 分成三个阶段：

1. `ingest` 回放对话并写入情景记忆，同时生成 ingest manifest；
2. `runner` 执行检索、回答、usage judge、叙事分组和最终判分，并写出 CSV 与 trace JSONL；
3. `report` 汇总单个 CSV，`scorecard` 汇总多个 `sample_NN` 目录。

`result/` 是运行产物目录，已被 `.gitignore` 忽略。历史运行记录保存在 [`neuramem_benchmark/RUN_RECORD.md`](../neuramem_benchmark/RUN_RECORD.md)；本文件只描述当前代码和推荐操作。

## Module Layout

| Module | Responsibility |
| --- | --- |
| `neuramem_benchmark/locomo.py` | Dataset loading and QA normalization |
| `neuramem_benchmark/llm_config.py` | Benchmark-specific provider and retry profile |
| `neuramem_benchmark/ingest.py` | Phase 1 ingest and completeness manifest |
| `neuramem_benchmark/runner.py` | Phase 2 evaluation and per-question trace |
| `neuramem_benchmark/grading.py` | Shared judge prompt, LLM call, and response parsing |
| `neuramem_benchmark/report.py` | Report for one evaluation CSV |
| `neuramem_benchmark/scorecard.py` | Cross-sample Markdown/JSON scorecard and optional ghost-ID check |
| `neuramem_benchmark/run_benchmark.py` | Cross-platform one-shot pipeline |
| `scripts/locomo_batch.ps1` | Windows resumable per-sample orchestration |

The PowerShell runner is intentionally separate from `run_benchmark.py`: it owns Windows process supervision, per-sample logs, resumable state, and worker status files. The Python entrypoint remains the portable single-command pipeline.

## Current Results

This is the stable-ID rerun completed on 2026-08-21/22. It uses the full closed loop with MiniMax-M3, excludes category 5 adversarial questions, and uses `auto_id=False` Milvus collections with application-generated IDs.

Overall: **66.82%** (`1029/1540`), weighted average latency `11.84s` per graded question.

| Category | Correct | Wrong | Total | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| 1-multi-hop | 198 | 84 | 282 | 70.21% |
| 2-temporal | 185 | 136 | 321 | 57.63% |
| 3-open-domain | 53 | 43 | 96 | 55.21% |
| 4-single-hop | 593 | 248 | 841 | 70.51% |

| Sample | Sessions | Memories added | Graded | Correct | Accuracy | Failed turns |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| s0 | 19 | 163 | 152 | 99 | 65.13% | 0 |
| s1 | 19 | 136 | 81 | 52 | 64.20% | 0 |
| s2 | 32 | 281 | 152 | 120 | 78.95% | 0 |
| s3 | 29 | 250 | 199 | 124 | 62.31% | 0 |
| s4 | 29 | 278 | 178 | 135 | 75.84% | 0 |
| s5 | 28 | 250 | 123 | 91 | 73.98% | 0 |
| s6 | 31 | 333 | 150 | 113 | 75.33% | 0 |
| s7 | 30 | 252 | 191 | 114 | 59.69% | 0 |
| s8 | 25 | 166 | 156 | 86 | 55.13% | 1 |
| s9 | 30 | 202 | 158 | 95 | 60.13% | 0 |

All ten workflows completed and the final trace-to-store verification reported `ghosts=0` for every sample. s8 completed with one skipped ingest turn and should be treated as the only sample with a non-zero ingest completeness warning.

## Recommended Commands

Run from the repository root after configuring `.env` and connecting to Milvus.

### Full portable run

```powershell
python -m neuramem_benchmark.run_benchmark `
  --all-samples `
  --threads 10 `
  --ingest-parallel 4 `
  --output result/locomo_neuramem_all_results.csv
```

This writes one combined CSV and is suitable for a clean, one-shot run.

### Resumable Windows run

Use one sample or a small batch when the LLM quota or network is the limiting factor:

```powershell
powershell -File scripts/locomo_batch.ps1 `
  -SampleIndices 5,6 `
  -BatchSize 2 `
  -DataPath data/locomo10.json `
  -OutputRoot result/locomo_full_rerun `
  -PythonPath E:/Anaconda/python.exe `
  -Threads 2
```

Each sample directory contains:

```text
sample_NN/
  ingest.stderr.log
  ingest_manifest_N.json
  ingest_usage_stats_N.json
  eval.csv
  eval.trace.jsonl
  summary.txt
  worker_status.json
```

The manifest is the ingest gate. `runner` refuses to evaluate a sample when the manifest is missing. `scorecard` can infer completion from the manifest, eval CSV, and summary when a manually launched worker did not write `worker_status.json`.

### Build a scorecard

```powershell
python -m neuramem_benchmark.scorecard `
  --root result/locomo_full_rerun `
  --output result/locomo_full_rerun/scorecard.md `
  --json-output result/locomo_full_rerun/scorecard.json
```

To verify that every episodic ID written into a trace still exists in Milvus:

```powershell
python -m neuramem_benchmark.scorecard `
  --root result/locomo_full_rerun `
  --verify-ghosts `
  --milvus-uri http://117.72.161.187:19530
```

### Run one phase manually

```powershell
python -m neuramem_benchmark.ingest `
  --input data/locomo10.json `
  --sample 5 `
  --usage-output-dir result/locomo_full_rerun/sample_05

python -m neuramem_benchmark.runner `
  --input data/locomo10.json `
  --sample 5 `
  --output result/locomo_full_rerun/sample_05/eval.csv `
  --manifest-dir result/locomo_full_rerun/sample_05 `
  --threads 2

python -m neuramem_benchmark.report `
  --input result/locomo_full_rerun/sample_05/eval.csv `
  --ingest-usage-dir result/locomo_full_rerun/sample_05
```

Do not evaluate a partially ingested sample. Confirm the manifest, `Import complete`, and `failed_turns` before starting `runner`.
