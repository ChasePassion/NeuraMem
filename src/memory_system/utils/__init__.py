"""Utility modules for AI Memory System."""

import numpy as np

from .retry import RetryExecutor


def normalize(vec) -> np.ndarray:
    """Normalize a vector to unit length."""
    vec = np.asarray(vec)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


__all__ = ["RetryExecutor", "normalize"]

