import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from policy import ThinkPolicyConfig, parse_think_override, should_enable_think


def _config(threshold: int = 200) -> ThinkPolicyConfig:
    return ThinkPolicyConfig(
        disable_tool_patterns=["web_search", "search", "browser", "browse", "serp", "http_get", "http", "fetch", "scrape"],
        enable_tool_patterns=["file_search", "retrieval", "rag", "openweather", "weather"],
        summary_patterns=["summarize", "tl;dr", "tldr", "bullet summary", "key takeaways"],
        char_threshold=threshold,
    )


class ThinkPolicyTests(unittest.TestCase):
    def test_web_search_tool_disables_think(self):
        body = {
            "messages": [{"role": "user", "content": "Find news."}],
            "tools": [{"type": "function", "function": {"name": "web_search"}}],
        }
        self.assertFalse(should_enable_think(body, override=None, config=_config()))

    def test_file_search_tool_keeps_think_enabled(self):
        body = {
            "messages": [{"role": "user", "content": "Search my repo."}],
            "tools": [{"type": "function", "function": {"name": "file_search"}}],
        }
        self.assertTrue(should_enable_think(body, override=None, config=_config()))

    def test_openweather_tool_keeps_think_enabled(self):
        body = {
            "messages": [{"role": "user", "content": "Weather in SF."}],
            "tools": [{"type": "function", "function": {"name": "openweather"}}],
        }
        self.assertTrue(should_enable_think(body, override=None, config=_config()))

    def test_summarize_and_long_content_disables_think(self):
        long_text = "A" * 120
        body = {
            "messages": [
                {"role": "tool", "name": "web_fetch", "content": long_text},
                {"role": "user", "content": "Please summarize this with bullet summary."},
            ]
        }
        self.assertFalse(should_enable_think(body, override=None, config=_config(threshold=200)))

    def test_override_forces_value(self):
        body = {
            "messages": [{"role": "user", "content": "Normal prompt."}],
            "tools": [{"type": "function", "function": {"name": "web_search"}}],
        }
        self.assertTrue(should_enable_think(body, override=True, config=_config()))
        self.assertFalse(should_enable_think(body, override=False, config=_config()))

    def test_research_does_not_match_search_false_positive(self):
        body = {
            "messages": [{"role": "user", "content": "Do research."}],
            "tools": [{"type": "function", "function": {"name": "research"}}],
        }
        self.assertTrue(should_enable_think(body, override=None, config=_config()))

    def test_parse_override(self):
        self.assertTrue(parse_think_override("true"))
        self.assertFalse(parse_think_override("FALSE"))
        self.assertIsNone(parse_think_override(None))
        with self.assertRaises(ValueError):
            parse_think_override("maybe")


if __name__ == "__main__":
    unittest.main()