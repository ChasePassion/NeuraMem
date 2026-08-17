"""Structured filter compilation for the Milvus adapter (#16).

The library never builds Milvus filter expressions by string
concatenation of caller input: MemoryFilter is compiled here with literal
escaping and identifier validation. The InMemory adapter evaluates the
same filter in Python (see inmemory.filter_matches).
"""

import re
from typing import Optional

from neuramem.core.models import MemoryFilter

# dynamic-field names must be plain identifiers — anything else would let
# caller-controlled metadata inject expression syntax
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def escape_string(value: str) -> str:
    """Escape a string literal for a Milvus filter expression."""
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def format_literal(value) -> str:
    """Format a Python value as a Milvus literal."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    return escape_string(str(value))


def compile_filter(flt: Optional[MemoryFilter]) -> str:
    """Compile a MemoryFilter into a Milvus boolean expression.

    Returns an empty string for None/empty filters (Milvus treats "" as
    no filter).
    """
    if flt is None:
        return ""
    clauses = []
    if flt.user_id is not None:
        clauses.append(f"user_id == {escape_string(flt.user_id)}")
    if flt.memory_type is not None:
        clauses.append(f"memory_type == {escape_string(flt.memory_type)}")
    if flt.group_id is not None:
        clauses.append(f"group_id == {flt.group_id}")
    if flt.group_id_in is not None:
        if flt.group_id_in:
            ids = ", ".join(str(g) for g in flt.group_id_in)
            clauses.append(f"group_id in [{ids}]")
        else:
            clauses.append("group_id == -2")  # empty set matches nothing
    if flt.id_in is not None:
        if flt.id_in:
            ids = ", ".join(str(i) for i in flt.id_in)
            clauses.append(f"id in [{ids}]")
        else:
            clauses.append("id == -2")  # empty set matches nothing
    if flt.id_not is not None:
        clauses.append(f"id != {flt.id_not}")
    if flt.retired is not None:
        clauses.append(f"retired == {format_literal(flt.retired)}")
    if flt.metadata is not None:
        for key, value in flt.metadata.items():
            if not _IDENTIFIER.match(key):
                raise ValueError(
                    f"metadata filter key {key!r} is not a valid field name"
                )
            clauses.append(f"{key} == {format_literal(value)}")
    return " and ".join(clauses)
