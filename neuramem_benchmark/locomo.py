"""LoCoMo dataset loading and parsing for ingest and evaluation."""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_INPUT = "data/locomo10.json"
_FALLBACK_PATH = Path("E:/code/locomo/data/locomo10.json")


def _resolve_path(path: str) -> str:
    if os.path.exists(path):
        return path
    if _FALLBACK_PATH.exists():
        return str(_FALLBACK_PATH)
    raise FileNotFoundError(f"LoCoMo dataset file not found: {path}")


def _load_data(path: str) -> list[dict[str, Any]]:
    with open(_resolve_path(path), "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"LoCoMo dataset must be a JSON list: {path}")
    return data


def load_locomo_samples(
    path: str, sample_idx: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Load raw conversation samples (ingest side)."""
    samples = []
    for idx, item in enumerate(_load_data(path)):
        normalized = dict(item)
        normalized["sample_index"] = idx
        normalized["user_id"] = f"sample_{idx}"
        samples.append(normalized)
    if sample_idx is not None:
        return [s for s in samples if s["sample_index"] == sample_idx]
    return samples


def load_locomo_qa_list(
    path: str, sample_idx: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Load QA pairs (eval side); adversarial category 5 excluded (W1-W3 口径)."""
    qa_list = []
    for s_idx, sample in enumerate(_load_data(path)):
        if sample_idx is not None and s_idx != sample_idx:
            continue
        sample_id = sample.get("sample_id", f"sample_{s_idx}")
        user_id = f"sample_{s_idx}"
        for q_idx, qa in enumerate(sample.get("qa", [])):
            if str(qa.get("category", "")) == "5":
                continue
            qa_list.append({
                "sample_index": s_idx,
                "sample_id": sample_id,
                "user_id": user_id,
                "question_id": f"{sample_id}_q{q_idx}",
                "category": str(qa.get("category", "")),
                "question": qa.get("question", ""),
                "answer": str(qa.get("answer", "")),
                "evidence": json.dumps(qa.get("evidence", []), ensure_ascii=False),
            })
    return qa_list


def count_samples(data_path: str) -> int:
    """Sample count in the dataset file (0 if unreadable)."""
    try:
        return len(_load_data(data_path))
    except Exception:  # noqa: BLE001 - caller falls back to serial ingest
        return 0
