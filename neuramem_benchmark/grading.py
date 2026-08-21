"""Shared LoCoMo judge prompt, response parsing, and LLM call helpers."""

from __future__ import annotations

import json
from typing import Any

from neuramem.llm.openai_adapter import OpenAILLM
from neuramem_benchmark.locomo_prompts import (
    JUDGE_SYSTEM_PROMPT,
    get_judge_prompt,
    preprocess_answer,
)


def category_number(value: Any) -> int:
    """Return a valid numeric category for a CSV or QA row."""

    return int(value) if str(value).isdigit() else 1


def parse_judge_response(raw: str) -> tuple[str, str]:
    """Parse a judge response, tolerating fences, thinking tags, and plain text."""

    clean = (raw or "").strip()
    if clean.startswith("```"):
        clean = "\n".join(clean.split("\n")[1:])
        clean = clean.split("```", 1)[0].strip()

    think_open = "<think>"
    think_close = "</think>"
    if think_open in clean and think_close in clean:
        clean = clean.split(think_close, 1)[-1].strip()

    start, end = clean.find("{"), clean.rfind("}")
    if start != -1 and end > start:
        clean = clean[start : end + 1]
    try:
        payload = json.loads(clean)
    except json.JSONDecodeError:
        upper = raw.upper()
        label = "CORRECT" if (
            '"CORRECT"' in upper or ("CORRECT" in upper and "WRONG" not in upper)
        ) else "WRONG"
        return label, raw[:300]

    label = str(payload.get("label", "")).strip().upper()
    reasoning = str(payload.get("reasoning", "")).strip()
    return ("CORRECT" if "CORRECT" in label else "WRONG"), reasoning


async def judge_row(
    llm: OpenAILLM,
    row: dict[str, Any],
    *,
    call_label: str = "judge",
) -> tuple[str, str, str]:
    """Judge one row and return ``(label, reasoning, raw_response)``."""

    category = category_number(row.get("category", "1"))
    prompt = get_judge_prompt(
        category=category,
        question=row.get("question", ""),
        answer=preprocess_answer(category, row.get("answer", "")),
        response=row.get("response", ""),
    )
    try:
        response = await llm.complete(
            JUDGE_SYSTEM_PROMPT,
            prompt,
            call_label=call_label,
        )
    except Exception as exc:  # noqa: BLE001 - one failed row must remain auditable
        return "WRONG", f"Judge call failed: {exc}", ""

    raw = response.content or ""
    label, reasoning = parse_judge_response(raw)
    return label, reasoning, raw
