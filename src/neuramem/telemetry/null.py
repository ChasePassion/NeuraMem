"""Null telemetry — the default, zero cost (architecture_target.md ch. 7)."""

from contextlib import asynccontextmanager
from typing import Optional

from neuramem.core.models import SpanStatus


class NullSpan:
    """Inert span: every call is a no-op."""

    def add_event(self, name: str, attributes: Optional[dict] = None) -> None:
        pass

    def set_attributes(self, attributes: dict) -> None:
        pass

    def set_status(self, status: SpanStatus) -> None:
        pass


_NULL_SPAN = NullSpan()


class NullTelemetry:
    """Telemetry port implementation that records nothing.

    The span context manager still runs the business body exactly once —
    telemetry never changes behavior (7.1.3).
    """

    @asynccontextmanager
    async def start_span(self, name: str, attributes: Optional[dict] = None):
        yield _NULL_SPAN
