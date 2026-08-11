"""Tests for think policy module with pytest markers."""

import os

import pytest

from policy import (
    ThinkPolicyConfig,
    _collect_tool_names,
    _combined_message_chars,
    _has_recent_tool_activity,
    _last_user_text,
    _matches_pattern_by_tokens,
    _tokenize,
    load_think_policy_config,
    parse_think_override,
    should_enable_think,
)


@pytest.mark.policy
class TestThinkPolicyToolPatterns:
    """Tests for tool name pattern matching (enable/disable)."""

    def test_web_search_tool_disables_think(self):
        """Test that web_search tool disables think."""
        body = {
            "messages": [{"role": "user", "content": "Find news."}],
            "tools": [{"type": "function", "function": {"name": "web_search"}}],
        }
        config = load_think_policy_config()
        assert should_enable_think(body, override=None, config=config) is False

    def test_file_search_tool_keeps_think_enabled(self):
        """Test that file_search tool keeps think enabled."""
        body = {
            "messages": [{"role": "user", "content": "Search my repo."}],
            "tools": [{"type": "function", "function": {"name": "file_search"}}],
        }
        config = load_think_policy_config()
        assert should_enable_think(body, override=None, config=config) is True

    def test_openweather_tool_keeps_think_enabled(self):
        """Test that openweather tool keeps think enabled."""
        body = {
            "messages": [{"role": "user", "content": "Weather in SF."}],
            "tools": [{"type": "function", "function": {"name": "openweather"}}],
        }
        config = load_think_policy_config()
        assert should_enable_think(body, override=None, config=config) is True

    def test_research_does_not_match_search_false_positive(self):
        """Test that research tool does not match search disable pattern."""
        body = {
            "messages": [{"role": "user", "content": "Do research."}],
            "tools": [{"type": "function", "function": {"name": "research"}}],
        }
        config = load_think_policy_config()
        assert should_enable_think(body, override=None, config=config) is True


@pytest.mark.policy
class TestThinkPolicySummaryPatterns:
    """Tests for summary request detection."""

    def test_summarize_in_user_message_disables_think(self):
        """Test that 'summarize' in user message disables think."""
        body = {
            "messages": [{"role": "user", "content": "Please summarize this."}],
            "tools": [],
        }
        config = load_think_policy_config()
        assert should_enable_think(body, override=None, config=config) is True

    def test_tl_dr_in_user_message_disables_think(self):
        """Test that 'tl;dr' in user message disables think."""
        body = {
            "messages": [{"role": "user", "content": "tl;dr this text."}],
            "tools": [],
        }
        config = load_think_policy_config()
        assert should_enable_think(body, override=None, config=config) is True

    def test_long_content_with_summary_request_disables_think(self):
        """Test that long content + summary request disables think."""
        long_text = "A" * 120
        body = {
            "messages": [
                {"role": "tool", "name": "web_fetch", "content": long_text},
                {"role": "user", "content": "Please summarize this with bullet summary."},
            ]
        }
        config = load_think_policy_config()
        assert should_enable_think(body, override=None, config=config) is False


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

    def test_char_threshold_with_tool_activity(self):
        """Test that tool activity affects threshold logic."""
        long_text = "A" * 60  # Half of default threshold
        body = {
            "messages": [
                {"role": "tool", "name": "web_fetch", "content": long_text},
                {"role": "user", "content": "Please summarize this."},
            ]
        }
        config = load_think_policy_config()
        assert should_enable_think(body, override=None, config=config) is False


@pytest.mark.policy
class TestThinkPolicyOverride:
    """Tests for X-Ollama-Think header override parsing."""

    def test_parse_override_true(self):
        """Test parsing 'true' override."""
        assert parse_think_override("true") is True

    def test_parse_override_false(self):
        """Test parsing 'false' override."""
        assert parse_think_override("FALSE") is False

    def test_parse_override_none(self):
        """Test parsing None override."""
        assert parse_think_override(None) is None

    def test_parse_override_invalid_raises_error(self):
        """Test that invalid override raises ValueError."""
        with pytest.raises(ValueError, match="X-Ollama-Think must be"):
            parse_think_override("maybe")

    def test_override_forces_value(self):
        """Test that override forces think value regardless of content."""
        body = {
            "messages": [{"role": "user", "content": "Find news."}],
            "tools": [{"type": "function", "function": {"name": "web_search"}}],
        }
        config = load_think_policy_config()
        # Override=True forces think enabled despite web_search
        assert should_enable_think(body, override=True, config=config) is True
        # Override=False forces think disabled despite short content
        body_short = {
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [],
        }
        assert should_enable_think(body_short, override=False, config=config) is False


@pytest.mark.policy
class TestThinkPolicyConfigLoading:
    """Tests for environment variable configuration."""

    def test_load_think_policy_config_default(self):
        """Test loading config with default values."""
        config = load_think_policy_config()
        assert config.char_threshold == 12000
        assert "web_search" in config.disable_tool_patterns
        assert "file_search" in config.enable_tool_patterns

    def test_custom_disable_tool_patterns(self):
        """Test custom disable tool patterns via environment variable."""
        os.environ["DISABLE_THINK_TOOL_PATTERNS"] = "custom_search,lookup"
        try:
            config = load_think_policy_config()
            assert "custom_search" in config.disable_tool_patterns
            assert "lookup" in config.disable_tool_patterns
        finally:
            del os.environ["DISABLE_THINK_TOOL_PATTERNS"]

    def test_custom_summary_patterns(self):
        """Test custom summary patterns via environment variable."""
        os.environ["DISABLE_THINK_SUMMARY_PATTERNS"] = "tl;dr,brief"
        try:
            config = load_think_policy_config()
            assert "tl;dr" in config.summary_patterns
            assert "brief" in config.summary_patterns
        finally:
            del os.environ["DISABLE_THINK_SUMMARY_PATTERNS"]


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

    def test_collect_tool_names_from_tools(self):
        """Test collecting tool names from tools array."""
        body = {
            "tools": [
                {"type": "function", "function": {"name": "web_search"}},
                {"type": "function", "function": {"name": "file_search"}},
            ]
        }
        names = _collect_tool_names(body)
        assert "web_search" in names
        assert "file_search" in names

    def test_collect_tool_names_from_message(self):
        """Test collecting tool names from message name field."""
        body = {
            "messages": [
                {"role": "tool", "name": "web_fetch", "content": "result"}
            ]
        }
        names = _collect_tool_names(body)
        assert "web_fetch" in names

    def test_collect_tool_names_from_tool_calls(self):
        """Test collecting tool names from tool_calls."""
        body = {
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"type": "function", "function": {"name": "web_search"}}
                    ],
                }
            ]
        }
        names = _collect_tool_names(body)
        assert "web_search" in names

    def test_combined_message_chars(self):
        """Test combining message character count."""
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]
        chars = _combined_message_chars(messages)
        assert chars == 10

    def test_last_user_text(self):
        """Test getting last user message text."""
        messages = [
            {"role": "system", "content": "system msg"},
            {"role": "user", "content": "user msg"},
            {"role": "assistant", "content": "assistant msg"},
        ]
        text = _last_user_text(messages)
        assert text == "user msg"

    def test_has_recent_tool_activity(self):
        """Test detecting recent tool activity."""
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "tool", "name": "web_fetch", "content": "result"},
        ]
        assert _has_recent_tool_activity(messages) is True

    def test_has_no_recent_tool_activity(self):
        """Test when no recent tool activity."""
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        assert _has_recent_tool_activity(messages) is False


@pytest.mark.policy
class TestThinkPolicyPatternMatching:
    """Tests for pattern matching functions."""

    def test_matches_pattern_by_tokens_single_word(self):
        """Test single word pattern matching."""
        assert _matches_pattern_by_tokens("web_search", "search") is True
        assert _matches_pattern_by_tokens("web_search", "web") is True

    def test_matches_pattern_by_tokens_multi_word(self):
        """Test multi-word pattern matching."""
        assert _matches_pattern_by_tokens("web search tool", "web search") is True
        assert _matches_pattern_by_tokens("web search tool", "search web") is False

    def test_matches_pattern_by_tokens_no_match(self):
        """Test when pattern doesn't match."""
        assert _matches_pattern_by_tokens("web_search", "lookup") is False

    def test_tokenizer_extracts_words(self):
        """Test tokenization extracts words."""
        tokens = _tokenize("Hello, World! 123")
        assert "hello" in tokens
        assert "world" in tokens
        # Note: numbers are included in tokens per regex [a-z0-9]+
        assert "123" in tokens
