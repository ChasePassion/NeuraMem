"""In-memory telemetry — reference implementation for tests and benchmark.

Contract (pi-mono packages/telemetry, mapped to Python async context
managers):
- the span context manager owns settlement: clean exit ends as "ok",
  exception ends as "error" unless an explicit status was set
- last explicit set_status wins over the automatic error status
- post-settlement calls on a span handle are inert
- recorded data is diagnostic only: recording failures never propagate
"""

import copy
import logging
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Optional

from neuramem.core.models import SpanStatus

logger = logging.getLogger(__name__)


@dataclass
class RecordedSpan:
    """Detached snapshot of one span (safe to hand out)."""

    id: int
    name: str
    parent_id: Optional[int]
    attributes: dict = field(default_factory=dict)
    events: list = field(default_factory=list)
    status: Optional[SpanStatus] = None  # explicit status if any
    final_status: SpanStatus = field(default_factory=lambda: SpanStatus())


@dataclass
class _LiveSpan:
    """Mutable record while the span is in flight."""

    id: int
    name: str
    parent_id: Optional[int]
    attributes: dict = field(default_factory=dict)
    events: list = field(default_factory=list)
    status: Optional[SpanStatus] = None
    settled: bool = False


class _SpanHandle:
    """The TelemetrySpan handed to the span body."""

    def __init__(self, record: _LiveSpan):
        self._record = record

    def add_event(self, name: str, attributes: Optional[dict] = None) -> None:
        if self._record.settled:
            return
        self._record.events.append({"name": name, "attributes": attributes})

    def set_attributes(self, attributes: dict) -> None:
        if self._record.settled:
            return
        self._record.attributes.update(attributes)

    def set_status(self, status: SpanStatus) -> None:
        if self._record.settled:
            return
        self._record.status = status


class InMemoryTelemetry:
    """Telemetry port implementation recording spans in process memory."""

    def __init__(self) -> None:
        self._spans: dict[int, _LiveSpan] = {}
        self._sequence = 0
        self._current: ContextVar[Optional[int]] = ContextVar(
            "neuramem_current_span", default=None
        )

    @asynccontextmanager
    async def start_span(self, name: str, attributes: Optional[dict] = None):
        self._sequence += 1
        parent_id = self._current.get()
        record = _LiveSpan(
            id=self._sequence,
            name=name,
            parent_id=parent_id,
            attributes=dict(attributes or {}),
        )
        self._spans[record.id] = record
        token = self._current.set(record.id)
        try:
            yield _SpanHandle(record)
            final = record.status or SpanStatus(status="ok")
        except BaseException as exc:
            # business exception becomes span error status unless an
            # explicit status was set; the exception itself propagates
            final = record.status or SpanStatus(
                status="error", error_message=f"{type(exc).__name__}: {exc}"
            )
            raise
        finally:
            record.settled = True
            record.final_status = final  # type: ignore[assignment]
            self._current.reset(token)

    def get_spans(self) -> list[RecordedSpan]:
        """Detached snapshots in start order."""
        snapshots = []
        for record in self._spans.values():
            snapshots.append(
                RecordedSpan(
                    id=record.id,
                    name=record.name,
                    parent_id=record.parent_id,
                    attributes=copy.deepcopy(record.attributes),
                    events=copy.deepcopy(record.events),
                    status=copy.copy(record.status),
                    final_status=copy.copy(record.final_status),
                )
            )
        return snapshots
