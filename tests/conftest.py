"""Pytest configuration and shared fixtures for router_headroom tests.

This module provides centralized fixtures and markers for testing the headroom-ai
compression implementation with support for base, code, and relevance compression modes.
"""

import os
from types import SimpleNamespace
from unittest import mock

import pytest

import router_headroom as headroom


# =============================================================================
# Shared Constants
# =============================================================================

MODEL = "test-model"
BASE_POLICY = {
    "options": {"num_ctx": 1000},
    "reserved_output_tokens": 100,
    "safety_headroom_tokens": 100,
    "recovery_strategy": "",
}
CODE_POLICY = {
    "options": {
        "num_ctx": 1000,
        "compress_user_messages": True,
        "target_ratio": 0.7,
    },
    "reserved_output_tokens": 100,
    "safety_headroom_tokens": 100,
    "recovery_strategy": "",
}

MESSAGES = [{"role": "user", "content": "hello"}]

# Policy with drop_oldest strategy for recovery tests
DROP_OLDEST_POLICY = {
    "options": {"num_ctx": 1000},
    "reserved_output_tokens": 100,
    "safety_headroom_tokens": 100,
    "recovery_strategy": "drop_oldest",
}

# Policy with summarize_history strategy for recovery tests
SUMMARIZE_POLICY = {
    "options": {"num_ctx": 1000},
    "reserved_output_tokens": 100,
    "safety_headroom_tokens": 100,
    "recovery_strategy": "summarize_history",
}

# Policy with high budget for tests that expect no rejection
HIGH_BUDGET_POLICY = {
    "options": {"num_ctx": 10000},
    "reserved_output_tokens": 100,
    "safety_headroom_tokens": 100,
    "recovery_strategy": "",
}


# =============================================================================
# Shared Fixtures
# =============================================================================

@pytest.fixture
def mock_compress_result():
    """Factory fixture to create a CompressResult-like SimpleNamespace."""
    def _create(messages=None, tokens_before=10, tokens_after=5, transforms=None):
        return SimpleNamespace(
            messages=messages or MESSAGES,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            transforms_applied=transforms or ["content_router"],
        )
    return _create


@pytest.fixture
def mock_compress_result_dict():
    """Factory fixture to create a dict-based CompressResult."""
    def _create(messages=None, tokens_before=10, tokens_after=5, transforms=None):
        return {
            "messages": messages or MESSAGES,
            "tokens_before": tokens_before,
            "tokens_after": tokens_after,
            "transforms_applied": transforms or ["content_router"],
        }
    return _create


@pytest.fixture
def mock_compress_list():
    """Factory fixture to create a plain list result."""
    def _create(messages=None):
        return messages or MESSAGES.copy()
    return _create


@pytest.fixture
def patch_headroom_compress(mock_compress_result):
    """Patch headroom.compress with a CompressResult return value."""
    def _patch(result=None):
        if result is None:
            result = mock_compress_result()
        return mock.patch.object(headroom, "_headroom_compress", return_value=result)
    return _patch


@pytest.fixture
def patch_headroom_compress_list(mock_compress_list):
    """Patch headroom.compress with a plain list return value."""
    def _patch(messages=None):
        return mock.patch.object(headroom, "_headroom_compress", return_value=mock_compress_list(messages))
    return _patch


@pytest.fixture
def patch_headroom_compress_dict(mock_compress_result_dict):
    """Patch headroom.compress with a dict return value."""
    def _patch(result=None):
        if result is None:
            result = mock_compress_result_dict()
        return mock.patch.object(headroom, "_headroom_compress", return_value=result)
    return _patch


@pytest.fixture
def patch_token_count():
    """Patch tokenizer.count_prompt_tokens with configurable side effects."""
    def _patch(side_effect=None, return_value=None):
        if side_effect is not None:
            return mock.patch.object(headroom, "_token_count", side_effect=side_effect)
        elif return_value is not None:
            return mock.patch.object(headroom, "_token_count", return_value=return_value)
        else:
            return mock.patch.object(headroom, "_token_count", return_value=10)
    return _patch


@pytest.fixture
def patch_headroom_disabled():
    """Patch HEADROOM_ENABLED to disable compression."""
    return mock.patch.dict(os.environ, {"HEADROOM_ENABLED": "0"}, clear=True)


@pytest.fixture
def patch_headroom_enabled():
    """Patch HEADROOM_ENABLED to enable compression."""
    return mock.patch.dict(os.environ, {"HEADROOM_ENABLED": "1"}, clear=True)


# =============================================================================
# Pytest Markers Configuration
# =============================================================================

# These markers are defined in pyproject.toml or pytest.ini
# Usage: @pytest.mark.base, @pytest.mark.code, @pytest.mark.relevance
#        @pytest.mark.compression, @pytest.mark.error_handling, @pytest.mark.integration


def pytest_configure(config):
    """Register custom markers for pytest."""
    config.addinivalue_line(
        "markers", "base: Tests for base compression (headroom-ai without extras)"
    )
    config.addinivalue_line(
        "markers", "code: Tests for code compression ([code] extra with tree-sitter)"
    )
    config.addinivalue_line(
        "markers", "compression: Tests for compression pipeline functionality"
    )
    config.addinivalue_line(
        "markers", "error_handling: Tests for error handling scenarios"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests for full pipelines"
    )
