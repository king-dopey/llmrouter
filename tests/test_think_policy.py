"""Tests for think policy module with pytest markers."""

import os

import pytest

from policy import (
    ThinkPolicyConfig,
    _combined_message_chars,
    load_think_policy_config,
    parse_think_override,
    should_enable_think,
)


@pytest.mark.policy
class TestThinkPolicyCharThreshold:
    """Tests for character threshold logic."""

    def test_below_threshold_keeps_think_enabled(self):
        """Test that content below threshold keeps think enabled."""
        body = {
            "messages": [{"role": "user", "content": "short"}],
            "tools": [],
        }
        config = load_think_policy_config()
        assert should_enable_think(body, override=None, config=config) is True

    def test_above_threshold_disables_think(self):
        """Test that content above threshold disables think."""
        long_text = "A" * 12000  # Above default threshold of 12000
        body = {
            "messages": [{"role": "user", "content": long_text}],
            "tools": [],
        }
        config = load_think_policy_config()
        assert should_enable_think(body, override=None, config=config) is False

    def test_custom_char_threshold(self):
        """Test custom character threshold via environment variable."""
        os.environ["DISABLE_THINK_CHAR_THRESHOLD"] = "50"
        try:
            config = load_think_policy_config()
            assert config.char_threshold == 50
        finally:
            del os.environ["DISABLE_THINK_CHAR_THRESHOLD"]


@pytest.mark.policy
class TestThinkPolicyOverride:
    """Tests for X-Ollama-Think header override parsing."""

    def test_parse_override_true(self):
        """Test parsing 'true' override."""
        assert parse_think_override("true") == "true"

    def test_parse_override_false(self):
        """Test parsing 'false' override."""
        assert parse_think_override("FALSE") == "false"

    def test_parse_override_tool_call_true(self):
        """Test parsing 'tool_call_true' override."""
        assert parse_think_override("tool_call_true") == "tool_call_true"

    def test_parse_override_none(self):
        """Test parsing None override."""
        assert parse_think_override(None) is None

    def test_parse_override_invalid_raises_error(self):
        """Test that invalid override raises ValueError."""
        with pytest.raises(ValueError, match="X-Ollama-Think must be"):
            parse_think_override("maybe")

    def test_override_forces_value_no_tools(self):
        """Test that override forces think value when no tools present."""
        body_short = {
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [],
        }
        config = load_think_policy_config()
        # Override="true" forces think enabled when no tools
        assert should_enable_think(body_short, override="true", config=config) is True
        # Override="false" forces think disabled when no tools
        assert should_enable_think(body_short, override="false", config=config) is False


@pytest.mark.policy
class TestThinkPolicyConfigLoading:
    """Tests for environment variable configuration."""

    def test_load_think_policy_config_default(self):
        """Test loading config with default values."""
        config = load_think_policy_config()
        assert config.char_threshold == 12000


@pytest.mark.policy
class TestThinkPolicyEdgeCases:
    """Tests for empty messages, missing tools, None values."""

    def test_empty_messages_list(self):
        """Test with empty messages list."""
        body = {"messages": [], "tools": []}
        config = load_think_policy_config()
        assert should_enable_think(body, override=None, config=config) is True

    def test_missing_messages_key(self):
        """Test with missing messages key."""
        body = {"tools": []}
        config = load_think_policy_config()
        assert should_enable_think(body, override=None, config=config) is True

    def test_missing_tools_key(self):
        """Test with missing tools key."""
        body = {"messages": [{"role": "user", "content": "hi"}]}
        config = load_think_policy_config()
        assert should_enable_think(body, override=None, config=config) is True

    def test_none_messages(self):
        """Test with None messages."""
        body = {"messages": None, "tools": []}
        config = load_think_policy_config()
        assert should_enable_think(body, override=None, config=config) is True

    def test_none_tools(self):
        """Test with None tools."""
        body = {"messages": [{"role": "user", "content": "hi"}], "tools": None}
        config = load_think_policy_config()
        assert should_enable_think(body, override=None, config=config) is True


@pytest.mark.policy
class TestThinkPolicyHelperFunctions:
    """Tests for helper functions."""

    def test_combined_message_chars(self):
        """Test combining message character count."""
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]
        chars = _combined_message_chars(messages)
        assert chars == 10


@pytest.mark.policy
class TestToolCallThinkSafetyInvariant:
    """Tests for configuration-driven tool_call_think safety invariant."""

    def test_model_with_tool_call_think_false_and_tools_returns_false(self):
        """Test that model with tool_call_think=False and tools returns False."""
        body = {
            "messages": [{"role": "user", "content": "Use a tool."}],
            "tools": [{"type": "function", "function": {"name": "read_file"}}],
        }
        config = load_think_policy_config()
        result = should_enable_think(
            body,
            override=None,
            config=config,
            model_name="any-model",
            tool_call_think=False,
        )
        assert result is False

    def test_model_with_tool_call_think_false_without_tools_returns_default(self):
        """Test that model with tool_call_think=False without tools returns default."""
        body = {
            "messages": [{"role": "user", "content": "Just chat."}],
            "tools": [],
        }
        config = load_think_policy_config()
        result = should_enable_think(
            body,
            override=None,
            config=config,
            default_think=True,
            model_name="any-model",
            tool_call_think=False,
        )
        assert result is True

    def test_model_with_tool_call_think_false_cannot_override_with_tool_call_true(self):
        """Test that tool_call_think=False is absolute safety — no override allowed."""
        body = {
            "messages": [{"role": "user", "content": "Use a tool."}],
            "tools": [{"type": "function", "function": {"name": "read_file"}}],
        }
        config = load_think_policy_config()
        result = should_enable_think(
            body,
            override="tool_call_true",
            config=config,
            model_name="any-model",
            tool_call_think=False,
        )
        assert result is False  # Safety invariant cannot be bypassed

    def test_model_with_tools_no_policy_returns_false_failsafe(self):
        """Test that model with tools and no explicit policy → fail-safe to false."""
        body = {
            "messages": [{"role": "user", "content": "Use a tool."}],
            "tools": [{"type": "function", "function": {"name": "read_file"}}],
        }
        config = load_think_policy_config()
        result = should_enable_think(
            body,
            override=None,
            config=config,
            model_name="any-model",
            tool_call_think=None,  # No explicit policy
        )
        assert result is False  # Fail-safe

    def test_model_with_tools_and_tool_call_true_override(self):
        """Test that model with tools and valid override → returns true."""
        body = {
            "messages": [{"role": "user", "content": "Use a tool."}],
            "tools": [{"type": "function", "function": {"name": "read_file"}}],
        }
        config = load_think_policy_config()
        result = should_enable_think(
            body,
            override="tool_call_true",
            config=config,
            model_name="any-model",
            tool_call_think=None,
        )
        assert result is True  # Valid override

    def test_model_with_tools_and_invalid_override_denied(self):
        """Test that model with tools and invalid override → denied."""
        body = {
            "messages": [{"role": "user", "content": "Use a tool."}],
            "tools": [{"type": "function", "function": {"name": "read_file"}}],
        }
        config = load_think_policy_config()
        result = should_enable_think(
            body,
            override="true",  # Invalid for tools
            config=config,
            model_name="any-model",
            tool_call_think=None,
        )
        assert result is False  # Denied by default

    def test_model_with_tool_call_think_false_and_safe_tool_still_returns_false(self):
        """Test that model with tool_call_think=False and safe tool still returns False."""
        body = {
            "messages": [{"role": "user", "content": "Search files."}],
            "tools": [{"type": "function", "function": {"name": "file_search"}}],
        }
        config = load_think_policy_config()
        result = should_enable_think(
            body,
            override=None,
            config=config,
            model_name="any-model",
            tool_call_think=False,
        )
        # Safety invariant triggers before name-pattern check
        assert result is False


@pytest.mark.policy
class TestReactAlignedPolicy:
    """Tests for ReAct-aligned policy: each turn evaluated on its own merits."""

    def test_conversation_history_with_tool_refs_does_not_disable_thinking(self):
        """Test that tool references in conversation history do NOT disable thinking.

        Per ReAct paradigm, reasoning should be interleaved with action.
        A summary/analysis turn after tool use should have thinking enabled.
        """
        body = {
            "messages": [
                {"role": "user", "content": "Search for news about AI"},
                {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "web_search"}}]},
                {"role": "tool", "name": "web_search", "content": "AI news results..."},
                {"role": "user", "content": "Summarize the key findings"},
            ],
            "tools": [],
        }
        config = load_think_policy_config()
        # No tools in current request, content below threshold → default_think (True)
        assert should_enable_think(body, override=None, config=config) is True

    def test_summary_request_does_not_disable_thinking(self):
        """Test that summary keywords in user message do NOT disable thinking.

        Per ReAct paradigm, summary/analysis tasks benefit from reasoning.
        """
        body = {
            "messages": [{"role": "user", "content": "Please summarize this."}],
            "tools": [],
        }
        config = load_think_policy_config()
        assert should_enable_think(body, override=None, config=config) is True

    def test_tl_dr_request_does_not_disable_thinking(self):
        """Test that 'tl;dr' in user message does NOT disable thinking."""
        body = {
            "messages": [{"role": "user", "content": "tl;dr this text."}],
            "tools": [],
        }
        config = load_think_policy_config()
        assert should_enable_think(body, override=None, config=config) is True

    def test_long_content_with_summary_request_uses_char_threshold(self):
        """Test that long content + summary request disables think ONLY via char threshold.

        The summary keywords themselves do not disable thinking.
        Only the char_threshold heuristic applies.
        """
        long_text = "A" * 12000  # Above default threshold
        body = {
            "messages": [
                {"role": "tool", "name": "web_fetch", "content": long_text},
                {"role": "user", "content": "Please summarize this with bullet summary."},
            ]
        }
        config = load_think_policy_config()
        # Disabled because combined chars exceed threshold, NOT because of summary patterns
        assert should_enable_think(body, override=None, config=config) is False

    def test_tool_activity_does_not_affect_threshold_logic(self):
        """Test that tool activity in history does NOT affect threshold logic.

        The old logic used tool activity to lower the effective threshold.
        The new logic only uses the absolute char_threshold.
        """
        long_text = "A" * 60  # Half of default threshold
        body = {
            "messages": [
                {"role": "tool", "name": "web_fetch", "content": long_text},
                {"role": "user", "content": "Please summarize this."},
            ]
        }
        config = load_think_policy_config()
        # Combined chars (60 + ~22) well below 12000 threshold → think enabled
        assert should_enable_think(body, override=None, config=config) is True

    def test_no_tools_no_override_below_threshold_returns_default(self):
        """Test that request with no tools, no override, below threshold returns default."""
        body = {
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [],
        }
        config = load_think_policy_config()
        assert should_enable_think(body, override=None, config=config, default_think=True) is True
        assert should_enable_think(body, override=None, config=config, default_think=False) is False
