"""LoCoMo benchmark pipeline.

Public command modules:

* ``ingest``: replay conversation data and write a completeness manifest.
* ``runner``: evaluate QA rows and write CSV plus trace JSONL.
* ``report``: summarize one evaluation CSV.
* ``scorecard``: aggregate per-sample run directories and verify trace ids.
* ``run_benchmark``: cross-platform one-shot orchestration.

The Windows resumable workflow lives in ``scripts/locomo_batch.ps1``.
"""
