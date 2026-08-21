"""neuramem - general-purpose AI memory library.

Public API surface (versioned contract, see docs/architecture_target.md 10.1):
- Memory facade (manage / search / report_usage / consolidate / delete / reset)
- MemoryConfig / MemoryRecord / SearchResult / UsageReport / ConsolidationStats
"""

from neuramem.config import MemoryConfig
from neuramem.core.models import (
    ConsolidationStats,
    MemoryRecord,
    RetrievalTrace,
    RetrievalTraceHit,
    SearchResult,
    UsageReport,
)
from neuramem.memory import Memory

__version__ = "1.0.0a1"

__all__ = [
    "ConsolidationStats",
    "Memory",
    "MemoryConfig",
    "MemoryRecord",
    "RetrievalTrace",
    "RetrievalTraceHit",
    "SearchResult",
    "UsageReport",
    "__version__",
]
