"""neuramem - general-purpose AI memory library.

Public API surface (versioned contract, see docs/architecture_target.md 10.1):
- MemoryConfig / MemoryRecord / SearchResult / UsageReport / ConsolidationStats

The Memory facade lands with the pipeline layer (implementation plan step 3).
"""

from neuramem.config import MemoryConfig
from neuramem.core.models import (
    ConsolidationStats,
    MemoryRecord,
    SearchResult,
    UsageReport,
)

__version__ = "1.0.0a1"

__all__ = [
    "ConsolidationStats",
    "MemoryConfig",
    "MemoryRecord",
    "SearchResult",
    "UsageReport",
    "__version__",
]
