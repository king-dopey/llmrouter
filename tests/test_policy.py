import unittest
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from policy import ThinkPolicyConfig, parse_think_override, should_enable_think


def _config(threshold: int = 200) -> ThinkPolicyConfig:
    return ThinkPolicyConfig(char_threshold=threshold)


class ThinkPolicyTests(unittest.TestCase):
    def test_web_search_tool_disables_think(self):
        body = {
            "messages": [{"role": "user", "content": "Find news."}],
            "tools": [{"type": "function", "function": {"name": "web_search"}}],
        }
        self.assertFalse(should_enable_think(body, override=None, config=_config()))

    def test_file_search_tool_keeps_think_enabled(self):
        # With tools present and no explicit policy, fail-safe to False
        body = {
            "messages": [{"role": "user", "content": "Search my repo."}],
            "tools": [{"type": "function", "function": {"name": "file_search"}}],
        }
        self.assertFalse(should_enable_think(body, override=None, config=_config()))
        
        # With explicit tool_call_think=True, should return True
        self.assertTrue(should_enable_think(body, override=None, config=_config(), tool_call_think=True))

    def test_openweather_tool_keeps_think_enabled(self):
        # With tools present and no explicit policy, fail-safe to False
        body = {
            "messages": [{"role": "user", "content": "Weather in SF."}],
            "tools": [{"type": "function", "function": {"name": "openweather"}}],
        }
        self.assertFalse(should_enable_think(body, override=None, config=_config()))
        
        # With explicit tool_call_think=True, should return True
        self.assertTrue(should_enable_think(body, override=None, config=_config(), tool_call_think=True))

    def test_long_content_disables_think(self):
        # Test that long content disables thinking via char_threshold
        # Summary keywords no longer affect thinking (ReAct-aligned policy)
        long_text = "A" * 250  # Exceeds threshold of 200
        body = {
            "messages": [
                {"role": "tool", "name": "web_fetch", "content": long_text},
                {"role": "user", "content": "Please summarize this with bullet summary."},
            ]
        }
        # Combined chars > 200 threshold → thinking disabled
        self.assertFalse(should_enable_think(body, override=None, config=_config(threshold=200)))
        
        # Test that summary request with content below threshold enables thinking
        short_text = "A" * 50  # Below threshold
        body_short = {
            "messages": [
                {"role": "tool", "name": "web_fetch", "content": short_text},
                {"role": "user", "content": "Please summarize this."},
            ]
        }
        # Combined chars < 200 threshold → thinking enabled (ReAct-aligned)
        self.assertTrue(should_enable_think(body_short, override=None, config=_config(threshold=200)))

    def test_override_forces_value(self):
        # With tools present, override="true" is invalid and denied (fail-safe to False)
        body = {
            "messages": [{"role": "user", "content": "Normal prompt."}],
            "tools": [{"type": "function", "function": {"name": "web_search"}}],
        }
        self.assertFalse(should_enable_think(body, override="true", config=_config()))
        
        # With tools present, override="tool_call_true" is valid and returns True
        self.assertTrue(should_enable_think(body, override="tool_call_true", config=_config()))
        
        # With tools present, override="false" is invalid and denied (fail-safe to False)
        self.assertFalse(should_enable_think(body, override="false", config=_config()))
        
        # Without tools, override works normally
        body_no_tools = {
            "messages": [{"role": "user", "content": "Normal prompt."}],
            "tools": [],
        }
        self.assertTrue(should_enable_think(body_no_tools, override="true", config=_config()))
        self.assertFalse(should_enable_think(body_no_tools, override="false", config=_config()))

    def test_research_does_not_match_search_false_positive(self):
        # With tools present and no explicit policy, fail-safe to False
        body = {
            "messages": [{"role": "user", "content": "Do research."}],
            "tools": [{"type": "function", "function": {"name": "research"}}],
        }
        self.assertFalse(should_enable_think(body, override=None, config=_config()))
        
        # With explicit tool_call_think=True, should return True
        self.assertTrue(should_enable_think(body, override=None, config=_config(), tool_call_think=True))

    def test_parse_override(self):
        self.assertEqual(parse_think_override("true"), "true")
        self.assertEqual(parse_think_override("FALSE"), "false")
        self.assertEqual(parse_think_override("tool_call_true"), "tool_call_true")
        self.assertIsNone(parse_think_override(None))
        with self.assertRaises(ValueError):
            parse_think_override("maybe")

    def test_safety_invariant_cannot_be_overridden(self):
        """Rule 1: tool_call_think=False with tools present is absolute safety invariant."""
        body = {
            "messages": [{"role": "user", "content": "Use a tool."}],
            "tools": [{"type": "function", "function": {"name": "web_search"}}],
        }
        # tool_call_think=False with tools present → always False
        self.assertFalse(should_enable_think(body, override=None, config=_config(), tool_call_think=False))
        # Even with override attempt, safety invariant cannot be bypassed
        self.assertFalse(should_enable_think(body, override="tool_call_true", config=_config(), tool_call_think=False))
        self.assertFalse(should_enable_think(body, override="true", config=_config(), tool_call_think=False))
        self.assertFalse(should_enable_think(body, override="false", config=_config(), tool_call_think=False))

    def test_invalid_override_with_tools_denied(self):
        """Rule 2: Invalid override values with tools present are denied (fail-safe to False)."""
        body = {
            "messages": [{"role": "user", "content": "Use a tool."}],
            "tools": [{"type": "function", "function": {"name": "web_search"}}],
        }
        # Invalid override values with tools → denied (fail-safe)
        self.assertFalse(should_enable_think(body, override="true", config=_config()))
        self.assertFalse(should_enable_think(body, override="false", config=_config()))
        # Only valid override is "tool_call_true"
        self.assertTrue(should_enable_think(body, override="tool_call_true", config=_config()))

    def test_threshold_boundary(self):
        """Rule 3: char_threshold boundary conditions for non-tool requests."""
        # Exactly at threshold → disabled
        body = {"messages": [{"role": "user", "content": "A" * 200}], "tools": []}
        self.assertFalse(should_enable_think(body, override=None, config=_config(threshold=200)))
        
        # Just below threshold → enabled
        body = {"messages": [{"role": "user", "content": "A" * 199}], "tools": []}
        self.assertTrue(should_enable_think(body, override=None, config=_config(threshold=200)))
        
        # Just above threshold → disabled
        body = {"messages": [{"role": "user", "content": "A" * 201}], "tools": []}
        self.assertFalse(should_enable_think(body, override=None, config=_config(threshold=200)))

    def test_empty_vs_missing_tools(self):
        """Verify behavior with empty tools array vs. missing tools key."""
        # Empty tools array → treated as no tools (Rule 3 applies)
        body_empty = {
            "messages": [{"role": "user", "content": "Short message"}],
            "tools": [],
        }
        self.assertTrue(should_enable_think(body_empty, override=None, config=_config()))
        
        # Missing tools key → treated as no tools (Rule 3 applies)
        body_missing = {
            "messages": [{"role": "user", "content": "Short message"}],
        }
        self.assertTrue(should_enable_think(body_missing, override=None, config=_config()))
        
        # Both should behave identically
        self.assertEqual(
            should_enable_think(body_empty, override=None, config=_config()),
            should_enable_think(body_missing, override=None, config=_config())
        )


class TestAuditLogging(unittest.TestCase):
    """Tests for security audit logging."""

    def test_safety_invariant_logs_warning(self):
        """Verify safety invariant enforcement is logged."""
        body = {
            "messages": [{"role": "user", "content": "Use a tool."}],
            "tools": [{"type": "function", "function": {"name": "web_search"}}],
        }

        with patch('policy.logger') as mock_logger:
            result = should_enable_think(
                body,
                override=None,
                config=_config(),
                model_name="test-model",
                tool_call_think=False
            )

            self.assertFalse(result)
            mock_logger.warning.assert_called()
            call_args = mock_logger.warning.call_args
            self.assertIn("Safety invariant enforced", call_args[0][0])
            self.assertEqual(call_args[1]['extra']['event_type'], "safety_invariant")
            self.assertEqual(call_args[1]['extra']['model'], "test-model")
            self.assertEqual(call_args[1]['extra']['decision'], False)
            self.assertEqual(call_args[1]['extra']['reason'], "safety_invariant")

    def test_invalid_override_logs_warning(self):
        """Verify invalid override attempts are logged."""
        body = {
            "messages": [{"role": "user", "content": "Use a tool."}],
            "tools": [{"type": "function", "function": {"name": "web_search"}}],
        }

        with patch('policy.logger') as mock_logger:
            result = should_enable_think(
                body,
                override="true",  # Invalid for tools
                config=_config(),
                model_name="test-model"
            )

            self.assertFalse(result)
            mock_logger.warning.assert_called()
            call_args = mock_logger.warning.call_args
            self.assertIn("Invalid override denied", call_args[0][0])
            self.assertEqual(call_args[1]['extra']['event_type'], "override_denied")
            self.assertEqual(call_args[1]['extra']['model'], "test-model")
            self.assertEqual(call_args[1]['extra']['decision'], False)
            self.assertEqual(call_args[1]['extra']['reason'], "invalid_override")

    def test_fail_safe_logs_warning(self):
        """Verify fail-safe activation is logged."""
        body = {
            "messages": [{"role": "user", "content": "Use a tool."}],
            "tools": [{"type": "function", "function": {"name": "web_search"}}],
        }

        with patch('policy.logger') as mock_logger:
            result = should_enable_think(
                body,
                override=None,
                config=_config(),
                model_name="test-model",
                tool_call_think=None  # No explicit policy
            )

            self.assertFalse(result)
            mock_logger.warning.assert_called()
            call_args = mock_logger.warning.call_args
            self.assertIn("Fail-safe activated", call_args[0][0])
            self.assertEqual(call_args[1]['extra']['event_type'], "fail_safe")
            self.assertEqual(call_args[1]['extra']['model'], "test-model")
            self.assertEqual(call_args[1]['extra']['decision'], False)
            self.assertEqual(call_args[1]['extra']['reason'], "no_explicit_policy")


if __name__ == "__main__":
    unittest.main()