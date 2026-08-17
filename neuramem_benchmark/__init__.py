"""neuramem_benchmark — LoCoMo evaluation pipeline (repo-local, not packaged).

Restructured from benchmark/locomo per implementation plan step 5
("重组而非重写"): locomo data loading, ingest replay with completeness
manifests, two-phase eval runner, recall@k metrics, judge/rejudge tools,
report generation, one-click orchestration. RUN_RECORD.md moves with it
(W numbering continuity preserved).
"""
