"""Retrieval-level metrics: recall@k over LoCoMo evidence pointers.

LoCoMo evidence items are dialogue pointers ("D1:3" = dialogue 1,
utterance 3). Resolution follows the OpenViking reference exactly
(benchmark/locomo/openviking/run_eval.py get_evidence_text): 1-based
utterance index, rendered as "speaker: text" — the same shape ingest
stores, so plain substring containment decides the hit. Unresolvable
pointers are skipped rather than guessed.
"""

import re
from typing import Any, Dict, List, Optional, Sequence

_EVIDENCE_POINTER = re.compile(r"^D(\d+):(\d+)$")
_PROVENANCE_POINTER = re.compile(
    r"^D(?P<start_session>\d+):(?P<start_turn>\d+)"
    r"(?:-D(?P<end_session>\d+):(?P<end_turn>\d+))?$"
)

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


def evidence_recall_detail(
    retrieved_texts: List[str], evidence_texts: List[str]
) -> Optional[List[bool]]:
    """Per-evidence hit flags (error attribution: WHICH evidence missed).

    Returns None when no evidence text could be resolved (question skipped
    in the aggregate rate rather than counted as a miss).
    """
    if not evidence_texts:
        return None
    normalized_retrieved = [_normalize(t) for t in retrieved_texts if t]
    return [
        any(_normalize(evidence_text) in retrieved
            for retrieved in normalized_retrieved)
        for evidence_text in evidence_texts
    ]


def evidence_recall(
    retrieved_texts: List[str], evidence_texts: List[str]
) -> Optional[bool]:
    """True when at least one evidence utterance appears in the retrieved set."""
    detail = evidence_recall_detail(retrieved_texts, evidence_texts)
    if detail is None:
        return None
    return any(detail)


def _expand_provenance_pointer(value: Any) -> set[str]:
    match = _PROVENANCE_POINTER.match(str(value).strip())
    if not match:
        return set()
    start_session = int(match.group("start_session"))
    start_turn = int(match.group("start_turn"))
    end_session = int(match.group("end_session") or start_session)
    end_turn = int(match.group("end_turn") or start_turn)
    if start_session != end_session:
        return {
            f"D{start_session}:{start_turn}",
            f"D{end_session}:{end_turn}",
        }
    return {
        f"D{start_session}:{turn}"
        for turn in range(start_turn, end_turn + 1)
    }


def _record_provenance_pointers(record: Any) -> set[str]:
    metadata = getattr(record, "metadata", None)
    if not isinstance(metadata, dict):
        return set()
    pointers = _expand_provenance_pointer(metadata.get("provenance_pointer"))
    if pointers:
        return pointers

    session = metadata.get("provenance_session")
    start = metadata.get("provenance_turn_start")
    end = metadata.get("provenance_turn_end", start)
    try:
        start_pointer = f"D{int(session)}:{int(start)}"
        end_pointer = f"D{int(session)}:{int(end)}"
    except (TypeError, ValueError):
        return set()
    return _expand_provenance_pointer(
        start_pointer if start_pointer == end_pointer
        else f"{start_pointer}-{end_pointer}"
    )


def provenance_recall_detail(
    retrieved_records: Sequence[Any], evidence_pointers: Sequence[Any]
) -> Optional[List[bool]]:
    """Match evidence pointers against persisted record provenance.

    Returns None when the evaluated records carry no provenance metadata, so
    old stores are not incorrectly reported as provenance misses.
    """
    normalized_evidence = [
        str(pointer).strip()
        for pointer in evidence_pointers or []
        if _EVIDENCE_POINTER.match(str(pointer).strip())
    ]
    if not normalized_evidence:
        return None
    retrieved_pointers = set()
    for record in retrieved_records:
        retrieved_pointers.update(_record_provenance_pointers(record))
    if not retrieved_pointers:
        return None
    return [pointer in retrieved_pointers for pointer in normalized_evidence]


def provenance_recall(
    retrieved_records: Sequence[Any], evidence_pointers: Sequence[Any]
) -> Optional[bool]:
    """True when at least one evidence pointer was retrieved by source."""
    detail = provenance_recall_detail(retrieved_records, evidence_pointers)
    if detail is None:
        return None
    return any(detail)
