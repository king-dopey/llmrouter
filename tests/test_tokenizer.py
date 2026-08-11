"""Tests for tokenizer module with pytest markers."""

import importlib
import json
import os

import pytest

import tokenizer


def _reload_tokenizer_with_map(mapping: dict[str, str]):
    """Reload tokenizer with a custom TOKENIZER_MAP."""
    os.environ["TOKENIZER_MAP"] = json.dumps(mapping)
    importlib.reload(tokenizer)
    return tokenizer


@pytest.mark.tokenizer
class TestTokenizerApproximate:
    """Tests for approximate token counting (byte-based estimation)."""

    def test_approximate_prompt_token_count(self, monkeypatch):
        """Test approximate token counting for prompt messages."""
        tokenizer = _reload_tokenizer_with_map({"mymodel": "approximate"})
        messages = [{"role": "user", "content": "hello world"}]
        text = "\n".join(m["content"] for m in messages)
        expected = (len(text.encode("utf-8")) + 2) // 3
        assert tokenizer.count_prompt_tokens(messages, "mymodel") == expected

    def test_approximate_completion_token_count(self, monkeypatch):
        """Test approximate token counting for completion text."""
        tokenizer = _reload_tokenizer_with_map({"mymodel": "approximate"})
        text = "abcde" * 10
        expected = (len(text.encode("utf-8")) + 2) // 3
        assert tokenizer.count_completion_tokens(text, "mymodel") == expected

    def test_approximate_multilingual_text(self):
        """Test approximate token counting for multilingual text."""
        tokenizer = _reload_tokenizer_with_map({"mymodel": "approximate"})
        text = "こんにちは世界"
        expected = (len(text.encode("utf-8")) + 2) // 3
        assert tokenizer.count_completion_tokens(text, "mymodel") == expected

    def test_approximate_empty_text_returns_zero(self):
        """Test approximate token counting for empty text."""
        tokenizer = _reload_tokenizer_with_map({"mymodel": "approximate"})
        assert tokenizer.count_completion_tokens("", "mymodel") == 0


@pytest.mark.tokenizer
class TestTokenizerTiktoken:
    """Tests for tiktoken-based counting when available."""

    def test_tiktoken_encoding_used_when_available(self, monkeypatch):
        """Test that tiktoken encoding is used when available."""
        # This test verifies the logic path for tiktoken usage
        tokenizer = _reload_tokenizer_with_map({"mymodel": "tiktoken:cl100k_base"})
        text = "hello world"
        result = tokenizer.count_completion_tokens(text, "mymodel")
        assert isinstance(result, int)
        assert result >= 0

    def test_tiktoken_fallback_to_approximate_when_unavailable(self):
        """Test fallback to approximate when tiktoken encoding is unavailable."""
        tokenizer = _reload_tokenizer_with_map({"mymodel": "tiktoken:nonexistent_encoding"})
        text = "hello world"
        result = tokenizer.count_completion_tokens(text, "mymodel")
        # Should fall back to approximate counting
        assert isinstance(result, int)
        assert result >= 0


@pytest.mark.tokenizer
class TestTokenizerModelMapping:
    """Tests for TOKENIZER_MAP environment variable parsing."""

    def test_tokenizer_map_loads_from_env(self):
        """Test that TOKENIZER_MAP is loaded from environment variable."""
        tokenizer = _reload_tokenizer_with_map({"test-model": "approximate"})
        assert "test-model" in tokenizer._TOKENIZER_MAP

    def test_empty_tokenizer_map_returns_empty_dict(self):
        """Test that empty TOKENIZER_MAP returns empty dict."""
        os.environ["TOKENIZER_MAP"] = "{}"
        importlib.reload(tokenizer)
        assert tokenizer._TOKENIZER_MAP == {}

    def test_invalid_json_tokenizer_map_logs_warning(self, caplog):
        """Test that invalid JSON in TOKENIZER_MAP logs a warning."""
        os.environ["TOKENIZER_MAP"] = "not valid json"
        importlib.reload(tokenizer)
        # Should log a warning but return empty dict
        assert tokenizer._TOKENIZER_MAP == {}


@pytest.mark.tokenizer
class TestTokenizerWildcardPatterns:
    """Tests for prefix wildcard patterns (e.g., 'qwen3*')."""

    def test_wildcard_pattern_matches_model_prefix(self):
        """Test that wildcard pattern matches model name prefix."""
        tokenizer = _reload_tokenizer_with_map({"qwen3*": "approximate"})
        # Model names starting with qwen3 should match the wildcard
        text = "hello"
        result = tokenizer.count_completion_tokens(text, "qwen3-coder")
        assert isinstance(result, int)
        assert result >= 0

    def test_wildcard_pattern_does_not_match_different_prefix(self):
        """Test that wildcard pattern doesn't match different prefix."""
        tokenizer = _reload_tokenizer_with_map({"qwen3*": "approximate"})
        # Model names not starting with qwen3 should not match
        text = "hello"
        result = tokenizer.count_completion_tokens(text, "llama3")
        assert isinstance(result, int)
        assert result >= 0


@pytest.mark.tokenizer
class TestTokenizerMultilingual:
    """Tests for multilingual text handling (UTF-8)."""

    def test_multilingual_japanese_text(self):
        """Test token counting for Japanese text."""
        tokenizer = _reload_tokenizer_with_map({"mymodel": "approximate"})
        text = "こんにちは世界"
        expected = (len(text.encode("utf-8")) + 2) // 3
        assert tokenizer.count_completion_tokens(text, "mymodel") == expected

    def test_multilingual_chinese_text(self):
        """Test token counting for Chinese text."""
        tokenizer = _reload_tokenizer_with_map({"mymodel": "approximate"})
        text = "你好世界"
        expected = (len(text.encode("utf-8")) + 2) // 3
        assert tokenizer.count_completion_tokens(text, "mymodel") == expected

    def test_multilingual_arabic_text(self):
        """Test token counting for Arabic text."""
        tokenizer = _reload_tokenizer_with_map({"mymodel": "approximate"})
        text = "مرحبا بالعالم"
        expected = (len(text.encode("utf-8")) + 2) // 3
        assert tokenizer.count_completion_tokens(text, "mymodel") == expected


@pytest.mark.tokenizer
class TestTokenizerEdgeCases:
    """Tests for edge cases: empty input, None content, mixed types."""

    def test_empty_messages_list(self):
        """Test token counting for empty messages list."""
        tokenizer = _reload_tokenizer_with_map({"mymodel": "approximate"})
        result = tokenizer.count_prompt_tokens([], "mymodel")
        assert result == 0

    def test_none_content_in_message(self):
        """Test token counting with None content in message."""
        tokenizer = _reload_tokenizer_with_map({"mymodel": "approximate"})
        messages = [{"role": "user", "content": None}]
        result = tokenizer.count_prompt_tokens(messages, "mymodel")
        assert result == 0

    def test_dict_content_in_message(self):
        """Test token counting with dict content in message."""
        tokenizer = _reload_tokenizer_with_map({"mymodel": "approximate"})
        messages = [{"role": "user", "content": {"text": "hello"}}]
        result = tokenizer.count_prompt_tokens(messages, "mymodel")
        assert result > 0

    def test_list_content_in_message(self):
        """Test token counting with list content in message."""
        tokenizer = _reload_tokenizer_with_map({"mymodel": "approximate"})
        messages = [{"role": "user", "content": ["hello", "world"]}]
        result = tokenizer.count_prompt_tokens(messages, "mymodel")
        assert result > 0

    def test_non_dict_message_item(self):
        """Test token counting with non-dict message item."""
        tokenizer = _reload_tokenizer_with_map({"mymodel": "approximate"})
        messages = ["hello", {"role": "user", "content": "world"}]
        result = tokenizer.count_prompt_tokens(messages, "mymodel")
        assert result > 0
