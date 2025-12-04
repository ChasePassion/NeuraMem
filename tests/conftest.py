"""Pytest configuration and shared fixtures for AI Memory System tests."""

import pytest
from hypothesis import settings, Phase

# Configure hypothesis for CI - run at least 100 examples per property test
settings.register_profile("ci", max_examples=100, deadline=None)
settings.register_profile("dev", max_examples=5, deadline=None)  # Faster for development
settings.load_profile("dev")
