"""Langfuse telemetry adapter — production observability consumer.

Mapping onto the langfuse v3 SDK (Langfuse.start_span / LangfuseSpan
update / create_event / end):
- start_span -> LangfuseSpan.create_span (root or child of the current
  span via langfuse's own context)
- add_event -> span.create_event(name=...)
- set_attributes -> span.update(metadata=merged)
- set_status -> span.update(level/status_message)
- settlement -> span.end()

Every langfuse call is wrapped defensively (7.1.3: telemetry must never
change business outcomes) — SDK API drift degrades to logging, not to
exceptions in the request path. Per 6.5.1 the LLM adapter remains the
single usage parsing point; this adapter only forwards what it is given.
"""

import logging
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Optional

from neuramem.core.models import SpanStatus

logger = logging.getLogger(__name__)


class _LangfuseSpanHandle:
    def __init__(self, lf_span):
        self._lf_span = lf_span

    def add_event(self, name: str, attributes: Optional[dict] = None) -> None:
        try:
            self._lf_span.create_event(name=name)
        except Exception as e:  # noqa: BLE001 - telemetry stays passive
            logger.debug("langfuse create_event failed: %s", e)

    def set_attributes(self, attributes: dict) -> None:
        try:
            current = getattr(self._lf_span, "metadata", None)
            merged = dict(current or {})
            merged.update(attributes)
            self._lf_span.update(metadata=merged)
        except Exception as e:  # noqa: BLE001
            logger.debug("langfuse update failed: %s", e)

    def set_status(self, status: SpanStatus) -> None:
        try:
            if status.status == "error":
                self._lf_span.update(level="ERROR", status_message=status.error_message)
            else:
                self._lf_span.update(level="DEFAULT")
        except Exception as e:  # noqa: BLE001
            logger.debug("langfuse set_status failed: %s", e)


class LangfuseTelemetry:
    """Telemetry port implementation writing to Langfuse."""

    def __init__(self, secret_key: str, public_key: str, host: Optional[str] = None):
        from langfuse import Langfuse  # imported lazily: core stays SDK-free

        kwargs = {"secret_key": secret_key, "public_key": public_key}
        if host:
            kwargs["host"] = host
        self._client = Langfuse(**kwargs)
        self._current: ContextVar[Optional[object]] = ContextVar(
            "neuramem_langfuse_span", default=None
        )

    @asynccontextmanager
    async def start_span(self, name: str, attributes: Optional[dict] = None):
        parent = self._current.get()
        lf_span = None
        try:
            if parent is not None:
                lf_span = parent.start_span(name=name, metadata=attributes)
            else:
                lf_span = self._client.start_span(name=name, metadata=attributes)
        except Exception as e:  # noqa: BLE001 - telemetry stays passive
            logger.debug("langfuse start_span failed: %s", e)
            from neuramem.telemetry.null import NullSpan

            yield NullSpan()
            return
        token = self._current.set(lf_span)
        try:
            yield _LangfuseSpanHandle(lf_span)
        finally:
            self._current.reset(token)
            try:
                lf_span.end()
            except Exception as e:  # noqa: BLE001
                logger.debug("langfuse end failed: %s", e)
