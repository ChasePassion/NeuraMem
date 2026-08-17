"""Retrieval-level metrics: recall@k over LoCoMo evidence pointers.

LoCoMo evidence items are dialogue pointers ("D1:3" = dialogue 1,
utterance 3). Resolution follows the OpenViking reference exactly
(benchmark/locomo/openviking/run_eval.py get_evidence_text): 1-based
utterance index, rendered as "speaker: text" — the same shape ingest
stores, so plain substring containment decides the hit. Unresolvable
pointers are skipped rather than guessed.
"""

import re
from typing import Any, Dict, List, Optional

_EVIDENCE_POINTER = re.compile(r"^D(\d+):(\d+)$")

# evidence utterances shorter than this are too generic to substring-match
_MIN_EVIDENCE_LEN = 8


def resolve_evidence_texts(sample: Dict[str, Any], evidence: List[Any]) -> List[str]:
    """Resolve 'D{n}:{k}' pointers to 'speaker: text' utterances (1-based)."""
    conv = sample.get("conversation", {})
    texts: List[str] = []
    for item in evidence or []:
        match = _EVIDENCE_POINTER.match(str(item).strip())
        if not match:
            continue
        session_num, turn_num = int(match.group(1)), int(match.group(2))
        session = conv.get(f"session_{session_num}", [])
        if not isinstance(session, list):
            continue
        msg_index = turn_num - 1  # 1-based, OpenViking rule
        if not 0 <= msg_index < len(session):
            continue
        utterance = session[msg_index]
        if isinstance(utterance, dict):
            text = str(utterance.get("text", "")).strip()
            if len(text) < _MIN_EVIDENCE_LEN:
                continue
            speaker = str(utterance.get("speaker", ""))
            texts.append(f"{speaker}: {text}" if speaker else text)
    return texts


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def evidence_recall(
    retrieved_texts: List[str], evidence_texts: List[str]
) -> Optional[bool]:
    """True when at least one evidence utterance appears in the retrieved set.

    Returns None when no evidence text could be resolved (question skipped
    in the aggregate rate rather than counted as a miss).
    """
    if not evidence_texts:
        return None
    normalized_retrieved = [_normalize(t) for t in retrieved_texts if t]
    for evidence_text in evidence_texts:
        needle = _normalize(evidence_text)
        if any(needle in retrieved for retrieved in normalized_retrieved):
            return True
    return False
