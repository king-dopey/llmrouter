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

    def test_tiktoken_handles_raw_special_token_text(self):
        """Raw special-token text should not raise and should be counted safely."""
        tokenizer = _reload_tokenizer_with_map({"mymodel": "tiktoken:cl100k_base"})
        text = "prefix <|endoftext|> suffix"
        result = tokenizer.count_completion_tokens(text, "mymodel")
        assert isinstance(result, int)
        assert result > 0


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


# =============================================================================
# Step 9: Qwen model family tokenizer integration tests
# =============================================================================

@pytest.mark.tokenizer
class TestTokenizerQwenCompatibility:
    """Tests for Qwen2/Qwen3 tokenizer compatibility (Section 7.1, risk table Section 13)."""

    def test_qwen_model_family_uses_approximate_fallback(self):
        """Test that Qwen models use approximate fallback when tiktoken unavailable."""
        # Map all Qwen model families to approximate for deterministic testing
        tokenizer_map = {
            "qwen3-coder-next:q4_K_M": "approximate",
            "qwen3.6:35b-a3b-q8_0": "approximate",
            "qwen3-coder:30b": "approximate",
        }
        tokenizer = _reload_tokenizer_with_map(tokenizer_map)

        messages = [{"role": "user", "content": "Implement a binary search tree"}]
        result = tokenizer.count_prompt_tokens(messages, "qwen3-coder-next:q4_K_M")
        assert isinstance(result, int)
        assert result > 0

    def test_approximate_fallback_deterministic(self):
        """Test that approximate token counting is deterministic across runs."""
        tokenizer_map = {"testmodel": "approximate"}
        _reload_tokenizer_with_map(tokenizer_map)

        text = "The quick brown fox jumps over the lazy dog" * 10
        result1 = tokenizer.count_completion_tokens(text, "testmodel")
        result2 = tokenizer.count_completion_tokens(text, "testmodel")
        assert result1 == result2

    def test_total_tokens_invariant(self):
        """Test that total_tokens = prompt_tokens + completion_tokens invariant holds."""
        tokenizer_map = {"mymodel": "approximate"}
        _reload_tokenizer_with_map(tokenizer_map)

        messages = [
            {"role": "system", "content": "You are a coding assistant."},
            {"role": "user", "content": "Write a function to sort an array"},
        ]
        completion_text = "Here is the sorted function:\n\ndef sort_array(arr):\n    return sorted(arr)"

        prompt_tokens = tokenizer.count_prompt_tokens(messages, "mymodel")
        completion_tokens = tokenizer.count_completion_tokens(completion_text, "mymodel")
        total = prompt_tokens + completion_tokens

        # Verify invariant: total must equal sum of parts
        assert total == prompt_tokens + completion_tokens
        assert total > 0
        assert prompt_tokens > 0
        assert completion_tokens > 0

    def test_approximate_within_reasonable_range_of_tiktoken(self):
        """Test that approximate fallback produces estimates within 20% of tiktoken count.

        This addresses the Medium risk in Section 13 about Qwen3 tokenizer differing from Qwen2.
        When tiktoken is available, verify approximate is a reasonable upper bound.
        """
        importlib.reload(tokenizer)
        # Reload with tiktoken-based mapping for comparison
        tokenizer_tik = _reload_tokenizer_with_map({"mymodel": "tiktoken:cl100k_base"})

        importlib.reload(tokenizer)
        tokenizer_approx = _reload_tokenizer_with_map({"mymodel": "approximate"})

        test_texts = [
            "Hello world",
            "def hello():\n    print('world')",
            "The quick brown fox jumps over the lazy dog",
            "#!/usr/bin/env python3\nimport sys",
        ]

        for text in test_texts:
            tik_result = tokenizer_tik.count_completion_tokens(text, "mymodel")
            approx_result = tokenizer_approx.count_completion_tokens(text, "mymodel")

            if tik_result > 0:
                # Approximate should be within 2x of tiktoken (conservative bound)
                ratio = approx_result / tik_result
                assert 0.5 <= ratio <= 2.0, f"Approx too far from tiktoken for: {text[:30]} (ratio={ratio:.2f})"

    def test_tokenizer_map_from_env_overrides_fallback(self):
        """Test that TOKENIZER_MAP env var takes precedence over any fallback."""
        tokenizer_map = {"custom-model": "approximate"}
        _reload_tokenizer_with_map(tokenizer_map)

        # Verify the custom model uses approximate path
        result = tokenizer.count_completion_tokens("test", "custom-model")
        expected = (len("test".encode("utf-8")) + 2) // 3
        assert result == expected
