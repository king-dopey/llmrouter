"""Tests for tool argument serialization/deserialization edge cases.

This module tests the _serialize_tool_arguments_for_headroom() and
_deserialize_tool_arguments() functions for robustness against malformed input,
structural anomalies, and resource exhaustion attacks.
"""

import json
import sys
import uuid
from typing import Any
from unittest import mock

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(uuid.uuid4()))

from app import _serialize_tool_arguments_for_headroom
from router_headroom import _deserialize_tool_arguments


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def valid_tool_call_dict():
    """Standard tool call with dict arguments."""
    return {
        "id": "call_123",
        "type": "function",
        "function": {
            "name": "test_function",
            "arguments": {"key": "value", "nested": {"inner": "data"}}
        }
    }


@pytest.fixture
def valid_tool_call_list():
    """Tool call with list arguments."""
    return {
        "id": "call_456",
        "type": "function",
        "function": {
            "name": "list_function",
            "arguments": ["item1", "item2", {"nested": "list_item"}]
        }
    }


@pytest.fixture
def valid_tool_call_string():
    """Tool call with string arguments (already serialized)."""
    return {
        "id": "call_789",
        "type": "function",
        "function": {
            "name": "string_function",
            "arguments": '{"key": "value"}'
        }
    }


@pytest.fixture
def sample_messages_with_tool_calls():
    """Sample messages with tool calls for round-trip testing."""
    return [
        {
            "role": "assistant",
            "content": "Let me check that for you.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "arguments": {"query": "test query", "num_results": 5}
                    }
                }
            ]
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": '{"results": ["result1", "result2"]}'
        }
    ]


@pytest.fixture
def large_payload():
    """Generate large argument strings for resource limit tests."""
    def _create(size_kb=1024):  # Default 1MB
        large_string = "x" * (size_kb * 1024)
        return {
            "id": "call_large",
            "type": "function",
            "function": {
                "name": "large_function",
                "arguments": {"data": large_string}
            }
        }
    return _create


# =============================================================================
# Circular References & Self-Referential Structures Tests
# =============================================================================

@pytest.mark.serialization
class TestCircularReferences:
    """Tests for circular reference handling in serialization."""
    
    def test_circular_dict_reference_returns_empty_dict(self):
        """Test that circular dict references fall back to {} without crashing."""
        # Create a circular reference (intentional for testing)
        circular_dict: dict[str, Any] = {"key": "value"}  # type: ignore[assignment]
        circular_dict["self"] = circular_dict  # type: ignore[assignment]
        
        tool_call = {
            "id": "call_circular",
            "type": "function",
            "function": {
                "name": "circular_func",
                "arguments": circular_dict
            }
        }
        
        messages = [{"role": "assistant", "tool_calls": [tool_call]}]
        
        with mock.patch("app.logger") as mock_logger:
            result = _serialize_tool_arguments_for_headroom(messages)
            
            # Should not crash and should log a warning
            mock_logger.warning.assert_called()
            
            # Verify the function returned a result
            assert isinstance(result, list)
            assert len(result) == 1
            
            # The arguments should be serialized to "{}" due to circular reference
            tool_calls = result[0].get("tool_calls", [])
            assert len(tool_calls) == 1
            args = tool_calls[0]["function"]["arguments"]
            assert args == "{}"

    def test_circular_list_reference_returns_empty_dict(self):
        """Test that circular list references fall back to {} without crashing."""
        # Create a circular list reference (intentional for testing)
        circular_list: list[Any] = [1, 2, 3]
        circular_list.append(circular_list)  # type: ignore[arg-type]
        
        tool_call = {
            "id": "call_circular_list",
            "type": "function",
            "function": {
                "name": "circular_list_func",
                "arguments": circular_list
            }
        }
        
        messages = [{"role": "assistant", "tool_calls": [tool_call]}]
        
        with mock.patch("app.logger") as mock_logger:
            result = _serialize_tool_arguments_for_headroom(messages)
            
            # Should not crash and should log a warning
            mock_logger.warning.assert_called()
            
            # Verify the function returned a result
            assert isinstance(result, list)
            
            # The arguments should be serialized to "{}" due to circular reference
            tool_calls = result[0].get("tool_calls", [])
            args = tool_calls[0]["function"]["arguments"]
            assert args == "{}"

    def test_deeply_nested_structure_no_stack_overflow(self):
        """Test deeply nested structures don't cause stack overflow."""
        # Create a deeply nested structure (100+ levels)
        depth = 150
        nested = {"level": depth}
        for i in range(depth - 1, 0, -1):
            nested = {"level": i, "nested": nested}
        
        tool_call = {
            "id": "call_deep",
            "type": "function",
            "function": {
                "name": "deep_func",
                "arguments": nested
            }
        }
        
        messages = [{"role": "assistant", "tool_calls": [tool_call]}]
        
        # Should not raise RecursionError or cause stack overflow
        result = _serialize_tool_arguments_for_headroom(messages)
        
        assert isinstance(result, list)
        assert len(result) == 1
        
        # Verify the nested structure was serialized to valid JSON
        tool_calls = result[0].get("tool_calls", [])
        args_str = tool_calls[0]["function"]["arguments"]
        
        # Should be a valid JSON string that can be parsed back
        parsed = json.loads(args_str)
        assert "level" in parsed  # Outermost level exists
        assert isinstance(parsed["nested"], dict)  # Nested structure preserved


# =============================================================================
# Deeply Nested Structures Tests
# =============================================================================

@pytest.mark.serialization
class TestDeepNesting:
    """Tests for deeply nested structure handling."""

    def test_extreme_nesting_depth(self):
        """Test inputs with extreme nesting depth (100+ levels)."""
        depth = 200
        nested = {"value": "deep"}
        for i in range(depth - 1, 0, -1):
            nested = {"level": i, "child": nested}
        
        tool_call = {
            "id": "call_extreme",
            "type": "function",
            "function": {
                "name": "extreme_func",
                "arguments": nested
            }
        }
        
        messages = [{"role": "user", "tool_calls": [tool_call]}]
        
        result = _serialize_tool_arguments_for_headroom(messages)
        
        assert isinstance(result, list)
        args_str = result[0]["tool_calls"][0]["function"]["arguments"]
        parsed = json.loads(args_str)
        # Verify structure was serialized: outermost level=1, innermost has "value"
        assert parsed["level"] == 1  # Outermost is always 1 (last iteration value)
        assert "child" in parsed  # Nested structure preserved

    def test_large_json_string_efficiently(self):
        """Test handling of valid but large JSON strings efficiently."""
        # Create a moderately large but valid structure
        large_data = {
            "items": [{"id": i, "data": "x" * 100} for i in range(100)]
        }
        
        tool_call = {
            "id": "call_large",
            "type": "function",
            "function": {
                "name": "large_func",
                "arguments": large_data
            }
        }
        
        messages = [{"role": "assistant", "tool_calls": [tool_call]}]
        
        result = _serialize_tool_arguments_for_headroom(messages)
        
        assert isinstance(result, list)
        args_str = result[0]["tool_calls"][0]["function"]["arguments"]
        
        # Verify it's valid JSON and can be parsed back
        parsed = json.loads(args_str)
        assert len(parsed["items"]) == 100


# =============================================================================
# Malformed/Invalid JSON Strings Tests
# =============================================================================

@pytest.mark.serialization
class TestMalformedJSON:
    """Tests for malformed JSON handling in deserialization."""

    def test_unclosed_json_string(self):
        """Test deserialization of unclosed JSON string."""
        messages = [{
            "role": "assistant",
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "test_func",
                    "arguments": '{"unclosed": true'  # Missing closing brace
                }
            }]
        }]
        
        with mock.patch("router_headroom.logger") as mock_logger:
            result = _deserialize_tool_arguments(messages)
            
            # Should log a warning
            mock_logger.warning.assert_called()
            
            # Arguments should remain as string (not parsed) on failure
            args = result[0]["tool_calls"][0]["function"]["arguments"]
            assert isinstance(args, str)

    def test_random_text_not_json(self):
        """Test deserialization of random text that's not JSON."""
        messages = [{
            "role": "assistant",
            "tool_calls": [{
                "id": "call_2",
                "type": "function",
                "function": {
                    "name": "test_func",
                    "arguments": "not json at all"
                }
            }]
        }]
        
        with mock.patch("router_headroom.logger") as mock_logger:
            result = _deserialize_tool_arguments(messages)
            
            # Should log a warning
            mock_logger.warning.assert_called()
            
            # Arguments should remain as string on failure
            args = result[0]["tool_calls"][0]["function"]["arguments"]
            assert isinstance(args, str)

    def test_null_value(self):
        """Test deserialization of null JSON value."""
        messages = [{
            "role": "assistant",
            "tool_calls": [{
                "id": "call_3",
                "type": "function",
                "function": {
                    "name": "test_func",
                    "arguments": None
                }
            }]
        }]
        
        result = _deserialize_tool_arguments(messages)
        
        # Arguments should be handled gracefully (None stays as None or becomes {})
        args = result[0]["tool_calls"][0]["function"]["arguments"]
        assert args is None or args == {}

    def test_broken_array(self):
        """Test deserialization of broken array JSON."""
        messages = [{
            "role": "assistant",
            "tool_calls": [{
                "id": "call_4",
                "type": "function",
                "function": {
                    "name": "test_func",
                    "arguments": "[1, 2, {broken}"  # Invalid JSON
                }
            }]
        }]
        
        with mock.patch("router_headroom.logger") as mock_logger:
            result = _deserialize_tool_arguments(messages)
            
            # Should log a warning
            mock_logger.warning.assert_called()
            
            # Arguments should remain as string on failure
            args = result[0]["tool_calls"][0]["function"]["arguments"]
            assert isinstance(args, str)

    def test_empty_string(self):
        """Test deserialization of empty string."""
        messages = [{
            "role": "assistant",
            "tool_calls": [{
                "id": "call_5",
                "type": "function",
                "function": {
                    "name": "test_func",
                    "arguments": ""
                }
            }]
        }]
        
        with mock.patch("router_headroom.logger") as mock_logger:
            result = _deserialize_tool_arguments(messages)
            
            # Should handle gracefully without crashing
            assert isinstance(result, list)

    def test_whitespace_only_string(self):
        """Test deserialization of whitespace-only string."""
        messages = [{
            "role": "assistant",
            "tool_calls": [{
                "id": "call_6",
                "type": "function",
                "function": {
                    "name": "test_func",
                    "arguments": "   \n\t  "
                }
            }]
        }]
        
        with mock.patch("router_headroom.logger") as mock_logger:
            result = _deserialize_tool_arguments(messages)
            
            # Should handle gracefully without crashing
            assert isinstance(result, list)

    def test_non_string_type_argument(self):
        """Test deserialization when arguments is not a string."""
        messages = [{
            "role": "assistant",
            "tool_calls": [{
                "id": "call_7",
                "type": "function",
                "function": {
                    "name": "test_func",
                    "arguments": 12345  # Integer instead of string
                }
            }]
        }]
        
        result = _deserialize_tool_arguments(messages)
        
        # Should handle gracefully - non-string args should pass through
        args = result[0]["tool_calls"][0]["function"]["arguments"]
        assert args == 12345


# =============================================================================
# Type Boundary Conditions Tests
# =============================================================================

@pytest.mark.serialization
class TestTypeBoundaries:
    """Tests for type boundary conditions in serialization."""

    def test_integer_argument(self):
        """Test serialization of integer argument."""
        tool_call = {
            "id": "call_int",
            "type": "function",
            "function": {
                "name": "int_func",
                "arguments": 42
            }
        }
        
        messages = [{"role": "user", "tool_calls": [tool_call]}]
        
        result = _serialize_tool_arguments_for_headroom(messages)
        
        # Integer should be converted to string
        args_str = result[0]["tool_calls"][0]["function"]["arguments"]
        assert isinstance(args_str, str)

    def test_float_argument(self):
        """Test serialization of float argument."""
        tool_call = {
            "id": "call_float",
            "type": "function",
            "function": {
                "name": "float_func",
                "arguments": 3.14159
            }
        }
        
        messages = [{"role": "user", "tool_calls": [tool_call]}]
        
        result = _serialize_tool_arguments_for_headroom(messages)
        
        # Float should be converted to string
        args_str = result[0]["tool_calls"][0]["function"]["arguments"]
        assert isinstance(args_str, str)

    def test_boolean_argument(self):
        """Test serialization of boolean argument."""
        tool_call = {
            "id": "call_bool",
            "type": "function",
            "function": {
                "name": "bool_func",
                "arguments": True
            }
        }
        
        messages = [{"role": "user", "tool_calls": [tool_call]}]
        
        result = _serialize_tool_arguments_for_headroom(messages)
        
        # Boolean should be converted to string
        args_str = result[0]["tool_calls"][0]["function"]["arguments"]
        assert isinstance(args_str, str)

    def test_none_argument(self):
        """Test serialization of None argument."""
        tool_call = {
            "id": "call_none",
            "type": "function",
            "function": {
                "name": "none_func",
                "arguments": None
            }
        }
        
        messages = [{"role": "user", "tool_calls": [tool_call]}]
        
        result = _serialize_tool_arguments_for_headroom(messages)
        
        # None should be converted to "{}" string
        args_str = result[0]["tool_calls"][0]["function"]["arguments"]
        assert args_str == "{}"

    def test_mixed_type_list(self):
        """Test serialization of mixed-type list."""
        tool_call = {
            "id": "call_mixed",
            "type": "function",
            "function": {
                "name": "mixed_func",
                "arguments": [1, "string", {"dict": "value"}, None, True, 3.14]
            }
        }
        
        messages = [{"role": "user", "tool_calls": [tool_call]}]
        
        result = _serialize_tool_arguments_for_headroom(messages)
        
        # Mixed list should be serialized to JSON string
        args_str = result[0]["tool_calls"][0]["function"]["arguments"]
        assert isinstance(args_str, str)
        
        # Verify round-trip
        parsed = json.loads(args_str)
        assert len(parsed) == 6
        assert parsed[0] == 1
        assert parsed[1] == "string"
        assert parsed[2] == {"dict": "value"}


# =============================================================================
# Large Payloads & Resource Limits Tests
# =============================================================================

@pytest.mark.serialization
class TestResourceLimits:
    """Tests for large payload handling and resource limits."""

    def test_very_large_argument_string(self, large_payload):
        """Test with very large argument strings (>1MB)."""
        # Create a 2MB payload
        tool_call = large_payload(size_kb=2048)
        
        messages = [{"role": "assistant", "tool_calls": [tool_call]}]
        
        result = _serialize_tool_arguments_for_headroom(messages)
        
        assert isinstance(result, list)
        args_str = result[0]["tool_calls"][0]["function"]["arguments"]
        
        # Verify the large string was serialized
        assert len(args_str) > 1024 * 1024  # > 1MB

    def test_extremely_long_input_no_hang(self):
        """Test that extremely long inputs don't cause hanging."""
        # Create a very long but valid JSON string
        long_string = "a" * 100000  # 100KB string
        tool_call = {
            "id": "call_long",
            "type": "function",
            "function": {
                "name": "long_func",
                "arguments": {"data": long_string}
            }
        }
        
        messages = [{"role": "user", "tool_calls": [tool_call]}]
        
        # Should complete quickly without hanging
        result = _serialize_tool_arguments_for_headroom(messages)
        
        assert isinstance(result, list)
        args_str = result[0]["tool_calls"][0]["function"]["arguments"]
        assert len(args_str) > 100000


# =============================================================================
# Data Integrity Verification Tests
# =============================================================================

@pytest.mark.serialization
class TestDataIntegrity:
    """Tests for data integrity in round-trip serialization."""

    def test_round_trip_preserves_dict(self, sample_messages_with_tool_calls):
        """Test that dict → JSON string → dict preserves values."""
        original = sample_messages_with_tool_calls
        
        # Serialize
        serialized = _serialize_tool_arguments_for_headroom(original)
        
        # Deserialize
        deserialized = _deserialize_tool_arguments(serialized)
        
        # Verify the tool call arguments are preserved
        original_args = original[0]["tool_calls"][0]["function"]["arguments"]
        deserialized_args = deserialized[0]["tool_calls"][0]["function"]["arguments"]
        
        assert original_args == deserialized_args

    def test_round_trip_preserves_list(self):
        """Test that list → JSON string → list preserves values."""
        original_list = ["item1", "item2", {"nested": "value"}, 123, True]
        
        tool_call = {
            "id": "call_list",
            "type": "function",
            "function": {
                "name": "list_func",
                "arguments": original_list
            }
        }
        
        messages = [{"role": "user", "tool_calls": [tool_call]}]
        
        # Serialize
        serialized = _serialize_tool_arguments_for_headroom(messages)
        
        # Deserialize
        deserialized = _deserialize_tool_arguments(serialized)
        
        # Verify the list is preserved
        original_args = messages[0]["tool_calls"][0]["function"]["arguments"]
        deserialized_args = deserialized[0]["tool_calls"][0]["function"]["arguments"]
        
        assert original_args == deserialized_args

    def test_special_characters_unicode(self):
        """Test that Unicode characters survive serialization/deserialization."""
        unicode_data = {
            "chinese": "你好世界",
            "japanese": "こんにちは",
            "arabic": "مرحبا",
            "emoji": "🚀🌟💻",
            "math": "∑ ∫ √ ∞"
        }
        
        tool_call = {
            "id": "call_unicode",
            "type": "function",
            "function": {
                "name": "unicode_func",
                "arguments": unicode_data
            }
        }
        
        messages = [{"role": "assistant", "tool_calls": [tool_call]}]
        
        # Serialize and deserialize
        serialized = _serialize_tool_arguments_for_headroom(messages)
        deserialized = _deserialize_tool_arguments(serialized)
        
        # Verify Unicode is preserved
        result_args = deserialized[0]["tool_calls"][0]["function"]["arguments"]
        assert result_args == unicode_data

    def test_special_characters_escaped_quotes(self):
        """Test that escaped quotes are handled correctly."""
        quote_data = {
            "single": "it's a test",
            "double": "he said \"hello\"",
            "mixed": "both ' and \" quotes"
        }
        
        tool_call = {
            "id": "call_quotes",
            "type": "function",
            "function": {
                "name": "quote_func",
                "arguments": quote_data
            }
        }
        
        messages = [{"role": "user", "tool_calls": [tool_call]}]
        
        # Serialize and deserialize
        serialized = _serialize_tool_arguments_for_headroom(messages)
        deserialized = _deserialize_tool_arguments(serialized)
        
        # Verify quotes are preserved
        result_args = deserialized[0]["tool_calls"][0]["function"]["arguments"]
        assert result_args == quote_data

    def test_nested_structures_preserved(self):
        """Test that deeply nested structures are preserved."""
        nested_data = {
            "level1": {
                "level2": {
                    "level3": {
                        "level4": {
                            "value": "deep"
                        }
                    }
                }
            }
        }
        
        tool_call = {
            "id": "call_nested",
            "type": "function",
            "function": {
                "name": "nested_func",
                "arguments": nested_data
            }
        }
        
        messages = [{"role": "assistant", "tool_calls": [tool_call]}]
        
        # Serialize and deserialize
        serialized = _serialize_tool_arguments_for_headroom(messages)
        deserialized = _deserialize_tool_arguments(serialized)
        
        # Verify nested structure is preserved
        result_args = deserialized[0]["tool_calls"][0]["function"]["arguments"]
        assert result_args == nested_data

    def test_empty_dict_and_list(self):
        """Test that empty dict and list are handled correctly."""
        empty_cases = [
            {"empty_dict": {}},
            {"empty_list": []},
            {"both_empty": {"dict": {}, "list": []}}
        ]
        
        for case in empty_cases:
            tool_call = {
                "id": f"call_{hash(str(case))}",
                "type": "function",
                "function": {
                    "name": "empty_func",
                    "arguments": case
                }
            }
            
            messages = [{"role": "user", "tool_calls": [tool_call]}]
            
            # Serialize and deserialize
            serialized = _serialize_tool_arguments_for_headroom(messages)
            deserialized = _deserialize_tool_arguments(serialized)
            
            # Verify empty structures are preserved
            result_args = deserialized[0]["tool_calls"][0]["function"]["arguments"]
            assert result_args == case


# =============================================================================
# Additional Edge Cases
# =============================================================================

@pytest.mark.serialization
class TestAdditionalEdgeCases:
    """Additional edge case tests."""

    def test_message_without_tool_calls(self):
        """Test handling of messages without tool_calls field."""
        messages = [{"role": "user", "content": "Hello"}]
        
        result = _serialize_tool_arguments_for_headroom(messages)
        
        assert len(result) == 1
        assert result[0] == messages[0]

    def test_empty_messages_list(self):
        """Test handling of empty messages list."""
        result = _serialize_tool_arguments_for_headroom([])
        
        assert result == []

    def test_non_dict_message_in_list(self):
        """Test handling of non-dict items in messages list."""
        messages = [
            "not a dict",
            {"role": "user", "content": "valid"},
            123,
            None
        ]
        
        result = _serialize_tool_arguments_for_headroom(messages)
        
        # Should handle gracefully and return valid messages
        assert isinstance(result, list)

    def test_tool_call_without_function(self):
        """Test handling of tool call without function field."""
        messages = [{
            "role": "assistant",
            "tool_calls": [{
                "id": "call_no_fn",
                "type": "function"
                # Missing "function" field
            }]
        }]
        
        result = _serialize_tool_arguments_for_headroom(messages)
        
        assert isinstance(result, list)

    def test_multiple_tool_calls(self):
        """Test handling of multiple tool calls in one message."""
        messages = [{
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "func1", "arguments": {"a": 1}}
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "func2", "arguments": {"b": 2}}
                }
            ]
        }]
        
        result = _serialize_tool_arguments_for_headroom(messages)
        
        assert len(result[0]["tool_calls"]) == 2
        
        # Verify both are serialized correctly
        args1 = result[0]["tool_calls"][0]["function"]["arguments"]
        args2 = result[0]["tool_calls"][1]["function"]["arguments"]
        
        assert json.loads(args1) == {"a": 1}
        assert json.loads(args2) == {"b": 2}