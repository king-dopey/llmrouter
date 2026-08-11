"""Integration tests for headroom-ai compression with real library.

These tests use the actual headroom library (not mocked) to verify the API contract.
These tests require headroom-ai to be installed (as specified in requirements.txt).
"""

import os
from unittest import mock

import pytest

import router_headroom as headroom

from conftest import BASE_POLICY, MODEL


# Fail if headroom is not installed - these tests require the real library
if headroom._headroom_compress is None:
    raise RuntimeError(
        "headroom library is not installed. "
        "Integration tests require headroom-ai to be installed (see requirements.txt)."
    )


@pytest.mark.integration
class TestHeadroomIntegration:
    """Integration tests with real headroom library."""

    def test_basic_compression_with_messages_list(self):
        """Test basic compression with a simple messages list."""
        messages = [
            {"role": "user", "content": "What is AI?"},
        ]
        
        with mock.patch.dict(os.environ, {"HEADROOM_ENABLED": "1", "HEADROOM_RELEVANCE_ENABLED": "0"}):
            result = headroom.check_and_trim(messages, MODEL, BASE_POLICY)
        
        assert result.rejected is False
        assert result.messages is not None
        assert len(result.messages) > 0

    def test_compression_with_model_parameter(self):
        """Test compression with model parameter."""
        messages = [
            {"role": "user", "content": "Explain machine learning."},
        ]
        
        with mock.patch.dict(os.environ, {"HEADROOM_ENABLED": "1", "HEADROOM_RELEVANCE_ENABLED": "0"}):
            result = headroom.check_and_trim(messages, MODEL, BASE_POLICY)
        
        assert result.rejected is False
        assert result.messages is not None

    def test_compression_with_headroom_query_for_relevance(self):
        """Test compression with headroom_query for relevance scoring."""
        messages = [
            {"role": "user", "content": "What is deep learning?"},
        ]
        
        with mock.patch.dict(os.environ, {"HEADROOM_ENABLED": "1", "HEADROOM_RELEVANCE_ENABLED": "1"}):
            result = headroom.check_and_trim(messages, MODEL, BASE_POLICY)
        
        assert result.rejected is False
        assert result.messages is not None

    def test_compression_with_multimodal_content(self):
        """Test compression with multimodal content (list of dicts)."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}},
                ],
            }
        ]
        
        with mock.patch.dict(os.environ, {"HEADROOM_ENABLED": "1", "HEADROOM_RELEVANCE_ENABLED": "1"}):
            result = headroom.check_and_trim(messages, MODEL, BASE_POLICY)
        
        assert result.rejected is False
        assert result.messages is not None

    def test_compression_with_tool_call_messages(self):
        """Test compression with tool call messages."""
        messages = [
            {"role": "user", "content": "Search for Python tutorials."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "search", "arguments": '{"q": "python"}'},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_123",
                "content": '{"results": [{"title": "Python Tutorial", "score": 0.95}]}',
            },
        ]
        
        with mock.patch.dict(os.environ, {"HEADROOM_ENABLED": "1", "HEADROOM_RELEVANCE_ENABLED": "0"}):
            result = headroom.check_and_trim(messages, MODEL, BASE_POLICY)
        
        assert result.rejected is False
        assert result.messages is not None

    def test_compression_with_empty_messages(self):
        """Test compression with empty messages list."""
        messages = []
        
        with mock.patch.dict(os.environ, {"HEADROOM_ENABLED": "1"}):
            result = headroom.check_and_trim(messages, MODEL, BASE_POLICY)
        
        assert result.rejected is False
        assert result.messages == []
        assert result.prompt_tokens == 0

    def test_compression_with_no_user_messages(self):
        """Test compression with no user messages (only system/assistant)."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "assistant", "content": "How can I help you?"},
        ]
        
        with mock.patch.dict(os.environ, {"HEADROOM_ENABLED": "1", "HEADROOM_RELEVANCE_ENABLED": "1"}):
            result = headroom.check_and_trim(messages, MODEL, BASE_POLICY)
        
        assert result.rejected is False
        assert result.messages is not None

    def test_compression_with_dict_content(self):
        """Test compression when a message contains dict content."""
        messages = [
            {"role": "user", "content": "Check this file"},
            {"role": "assistant", "content": {"path": "file.py", "mode": "read"}},
        ]
        
        with mock.patch.dict(os.environ, {"HEADROOM_ENABLED": "1", "HEADROOM_RELEVANCE_ENABLED": "1"}):
            result = headroom.check_and_trim(messages, MODEL, BASE_POLICY)
        
        assert result.rejected is False
        assert result.messages is not None

    def test_compression_with_list_content(self):
        """Test compression when a message contains list content."""
        messages = [
            {"role": "user", "content": "List content"},
            {"role": "assistant", "content": ["item1", "item2", "item3"]},
        ]
        
        with mock.patch.dict(os.environ, {"HEADROOM_ENABLED": "1", "HEADROOM_RELEVANCE_ENABLED": "1"}):
            result = headroom.check_and_trim(messages, MODEL, BASE_POLICY)
        
        assert result.rejected is False
        assert result.messages is not None

    def test_compression_disabled(self):
        """Test compression when HEADROOM_ENABLED=0."""
        messages = [
            {"role": "user", "content": "What is AI?"},
        ]
        
        with mock.patch.dict(os.environ, {"HEADROOM_ENABLED": "0"}):
            result = headroom.check_and_trim(messages, MODEL, BASE_POLICY)
        
        assert result.rejected is False
        assert result.messages == messages  # Original messages returned
        assert result.trimmed is False

    def test_compression_with_large_context(self):
        """Test compression with large context that may exceed budget."""
        # Create a large message that might exceed the budget
        large_content = "x" * 10000
        messages = [
            {"role": "user", "content": large_content},
        ]
        
        with mock.patch.dict(os.environ, {"HEADROOM_ENABLED": "1", "HEADROOM_RELEVANCE_ENABLED": "0"}):
            result = headroom.check_and_trim(messages, MODEL, BASE_POLICY)
        
        # Should either compress successfully or reject if still too large
        assert result.messages is not None
