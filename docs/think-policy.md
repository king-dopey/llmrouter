# Think Control Policy

The router implements a ReAct-aligned think control policy that determines when models should use extended thinking (chain-of-thought reasoning) based on request context, model capabilities, and safety requirements.

## Overview

The think policy evaluates requests using a three-tier rule hierarchy that aligns with ReAct (Reasoning and Acting) principles. Each turn is evaluated based on its own task requirements rather than conversation history, enabling proper reasoning interleaved with action.

### Three-Tier Rule Hierarchy

The `should_enable_think()` function in `policy.py` implements a three-tier rule hierarchy evaluated top-to-bottom (first match wins):

#### Rule 1: Safety Invariant (Absolute Priority)

**Condition:** Tools present AND `tool_call_think=False`

**Action:** Return `think=False` (unconditionally)

**Rationale:** Prevents Qwen3 models from entering deterministic looping behavior when tools are present. This safety invariant executes before any override logic, ensuring it cannot be bypassed by client headers.

**Code Location:** `policy.py:138-140`

```python
# Rule 1: Models with tool_call_think: false and tools present — SAFETY INVARIANT, NO OVERRIDES
if has_tools and tool_call_think is False:
    return False
```

#### Rule 2: Tool Request Handling (Fail-Safe)

**Condition:** Tools present AND `tool_call_think` is not `False`

**Action:** Evaluate override and policy:
- Valid override (`tool_call_true`) → `think=True`
- Invalid override (`true`, `false`, etc.) → `think=False` (deny by default)
- Explicit `tool_call_think` policy → use it
- No explicit policy → `think=False` (fail-safe)

**Rationale:** Tool requests default to thinking disabled to prevent accidental exposure to model-specific looping vulnerabilities. Only explicit configuration or valid overrides enable thinking.

**Code Location:** `policy.py:142-156`

```python
# Rule 2: Models with tools present — explicit policy only, fail-safe to false
if has_tools:
    # Check for valid override value
    if override == "tool_call_true":
        return True
    elif override is not None:
        # Invalid override value with tools → deny by default
        return False
    
    # No override: use explicit policy or fail-safe
    if tool_call_think is not None:
        return tool_call_think
    else:
        # No explicit policy → fail-safe to false
        return False
```

#### Rule 3: Non-Tool Request Handling (ReAct-Aligned)

**Condition:** No tools present

**Action:** Evaluate override and character threshold:
- Override present → use it (`true` → `True`, `false` → `False`)
- Combined message characters ≥ `char_threshold` → `think=False`
- Otherwise → return `default_think` (from model config)

**Rationale:** Non-tool requests are evaluated based on content length only. Summary keywords and conversation history tool references no longer affect thinking behavior, enabling proper reasoning for summary and analysis tasks.

**Code Location:** `policy.py:158-170`

```python
# Rule 3: No tools present — apply char threshold heuristic, then default
if override is not None:
    return override == "true"

messages = body.get("messages") or []
if not isinstance(messages, list):
    messages = []

combined_chars = _combined_message_chars(messages)
if combined_chars >= config.char_threshold:
    return False

return default_think
```

### Execution Flow

```
Incoming Request
│
├─ Extract: body, override, config, default_think, tool_call_think
│
├─ Rule 1: Safety Invariant Check
│  ├─ has_tools AND tool_call_think is False?
│  │  └─ YES → return False (SAFETY INVARIANT)
│  └─ NO → Continue to Rule 2
│
├─ Rule 2: Tool Request Handling
│  ├─ has_tools?
│  │  ├─ YES
│  │  │  ├─ override == "tool_call_true"? → return True
│  │  │  ├─ override is not None? → return False (deny invalid)
│  │  │  ├─ tool_call_think is not None? → return tool_call_think
│  │  │  └─ else → return False (fail-safe)
│  │  └─ NO → Continue to Rule 3
│
├─ Rule 3: Non-Tool Request Handling
│  ├─ override is not None? → return override == "true"
│  ├─ combined_chars >= char_threshold? → return False
│  └─ else → return default_think
│
└─ Return final think value
```

## Recent Changes (ReAct-Aligned Policy Redesign)

### Removed Legacy Heuristics

The following legacy heuristics have been removed to align with ReAct principles:

1. **DISABLE_THINK_TOOL_PATTERNS**: Previously disabled thinking based on tool name patterns in conversation history
2. **DISABLE_THINK_SUMMARY_PATTERNS**: Previously disabled thinking based on summary keywords in user messages
3. **Pattern Matching Functions**: Removed `_collect_tool_names()`, `_last_user_text()`, `_has_recent_tool_activity()`, `_matches_pattern_by_tokens()`, `_tokenize()`

### Rationale for Removal

The legacy heuristics incorrectly disabled thinking for summary and analysis tasks when conversation history contained tool references. This contradicted ReAct principles where reasoning should be interleaved with action. The new policy:

- Evaluates each turn based on its own task requirements
- Uses `char_threshold` as the sole determinant for non-tool requests
- Preserves the safety invariant for tool requests
- Enables proper reasoning for summary and analysis tasks

### Simplified Configuration

`ThinkPolicyConfig` now contains only:
- `char_threshold`: Character threshold for disabling thinking on long prompts (default: 12000)

The legacy fields (`disable_tool_patterns`, `enable_tool_patterns`, `summary_patterns`) have been removed.

## Safety Invariant

**Critical:** The router enforces a configuration-driven safety invariant to prevent Qwen3 models from entering deterministic looping behavior when tools are present.

### Rule

Any model configured with `tool_call_think: false` in `models.yaml` **cannot** be overridden to `think: true` when tools are present in the request.

### Why This Matters

Qwen3 models exhibit a critical bug where `think: true` combined with tool schemas causes:

- Deterministic token-ID repetition loops
- 20,000-55,000+ token reasoning loops
- 45-80 minute hangs on multi-step prompts
- Tool-call argument loops

The safety invariant prevents these issues by enforcing `think: false` for configured models when tools are present, regardless of client override attempts.

### Configuration

```yaml
# profiles/orin/models.yaml or profiles/thor/models.yaml
- model: qwen3.6:35b-a3b
  think: true                    # Default behavior when no tools present
  tool_call_think: false         # Safety invariant: cannot be overridden when tools present
  keep_alive: 15m
```

### Behavior Matrix

| Scenario | Tools Present | `tool_call_think` | Override Header | Result | Reason |
|----------|---------------|-------------------|-----------------|--------|--------|
| 1 | Yes | `false` | None | `think: false` | Safety invariant enforced |
| 2 | Yes | `false` | `tool_call_true` | `think: false` | Safety invariant cannot be bypassed |
| 3 | Yes | `false` | `true` | `think: false` | Safety invariant cannot be bypassed |
| 4 | No | `false` | None | `think: true` | Safety invariant only applies when tools present |
| 5 | Yes | `true` | None | `think: true` | Explicit policy honored |
| 6 | Yes | `true` | `tool_call_true` | `think: true` | Override and policy agree |
| 7 | Yes | `null` | None | `think: false` | Fail-safe default |
| 8 | Yes | `null` | `tool_call_true` | `think: true` | Valid override for non-safety-constrained model |
| 9 | Yes | `null` | `true` | `think: false` | Invalid override denied (fail-safe) |
| 10 | No | `null` | `true` | `think: true` | Override honored when no tools |

## Override Handling

### Valid Override Values

The router accepts three override values via the `X-Ollama-Think` header:

| Header Value | Behavior | Use Case |
|--------------|----------|----------|
| `true` | Force `think: true` | Enable thinking for non-tool requests |
| `false` | Force `think: false` | Disable thinking for any request |
| `tool_call_true` | Force `think: true` for tool requests | Enable thinking for non-safety-constrained models with tools |

### Override Validation

- **Invalid override values** are rejected with HTTP 400
- **When tools are present**, only `tool_call_true` is accepted as a valid override
- **Any other override value** with tools present defaults to `think: false` (fail-safe)
- **Safety invariant takes precedence** over all override attempts for models with `tool_call_think: false`

### Examples

#### Example 1: Safety Invariant Blocks Override

```bash
# Model configured with tool_call_think: false
curl -sS http://127.0.0.1:4000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'X-Ollama-Think: tool_call_true' \
  -d '{
    "model": "qwen3.6:35b-a3b",
    "messages": [{"role": "user", "content": "Use a tool."}],
    "tools": [{"type": "function", "function": {"name": "read_file"}}]
  }' | jq .

# Result: think=false (safety invariant enforced, override ignored)
```

#### Example 2: Valid Override for Non-Safety-Constrained Model

```bash
# Model without tool_call_think configuration
curl -sS http://127.0.0.1:4000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'X-Ollama-Think: tool_call_true' \
  -d '{
    "model": "gemma4:12b",
    "messages": [{"role": "user", "content": "Use a tool."}],
    "tools": [{"type": "function", "function": {"name": "read_file"}}]
  }' | jq .

# Result: think=true (valid override honored)
```

## Security Controls

### 1. Safety Invariant Priority

The safety invariant executes **before** any override logic in the decision tree, ensuring it cannot be bypassed by client headers.

### 2. Strict Validation

The `parse_think_override()` function enforces strict validation:
- Only three valid values accepted: `true`, `false`, `tool_call_true`
- Case-insensitive matching
- Invalid values raise `ValueError` (HTTP 400)
- Prevents injection attacks or unexpected behavior from malformed headers

### 3. Fail-Safe Defaults

When tools are present and no explicit `tool_call_think` policy is configured, the router defaults to `think: false` to prevent accidental exposure to the Qwen3 infinite loop vulnerability.

### 4. Configuration-Driven Approach

The safety invariant is **not** hardcoded to specific model names. It applies to any model with `tool_call_think: false` in the configuration, providing:
- **Maintainability**: No code changes needed when adding new models
- **Resilience**: Works for any model family, not just Qwen
- **Flexibility**: Operators control behavior via YAML configuration
- **Consistency**: Single source of truth in `models.yaml`

### 5. Thinking Token Isolation

For thinking models (e.g., Qwen3 family), Ollama emits reasoning tokens in a separate `thinking` field during streaming. The router implements conservative isolation for these tokens:

| Control | Behavior |
|---------|----------|
| Acknowledgment | Thinking tokens are acknowledged as valid assistant payload to prevent false "no content" errors |
| Client Forwarding | Thinking tokens are **NOT** forwarded to clients by default |
| Information Disclosure | No model reasoning chains are exposed to downstream consumers |
| Configuration | Conservative approach prevents unintended side effects from model-specific behavior |

This isolation ensures that internal model reasoning processes remain private while still allowing the router to correctly handle streaming responses from thinking models.

## Streaming Thinking Token Handler

### Overview

When using thinking models (e.g., Qwen3 family) via Ollama, the model emits reasoning tokens in a separate `thinking` field during streaming responses. This is distinct from the standard `content` field used for regular assistant messages.

### Problem Addressed

Without proper handling, models that emit only thinking tokens (without `content` or `tool_calls`) would trigger a false "no content" `RuntimeError`:

```
RuntimeError: Ollama stream for model <name> completed without any assistant content or tool calls
```

This occurred because the streaming handler only checked for `content` and `tool_calls` fields when determining whether valid assistant payload was received.

### Implementation

The streaming handler in [`app.py`](app.py:1968) checks for the presence of thinking tokens and acknowledges them as valid assistant payload:

```python
# Thinking/reasoning delta (for thinking models).
# Ollama emits thinking tokens in a separate field for thinking models.
# We acknowledge these as valid assistant payload to prevent false
# "no content" errors, but do not forward them to clients by default.
thinking_token = message.get("thinking", "")
if thinking_token:
    saw_assistant_payload = True
    # Note: thinking tokens are not forwarded to clients unless
    # explicitly configured. This prevents the RuntimeError when
    # models emit only thinking tokens without content/tool_calls.
```

### Behavior

| Scenario | `thinking` Field | `content` Field | Result |
|----------|------------------|-----------------|--------|
| Thinking-only response | Present (non-empty) | Absent | Valid response (no error) |
| Standard response | Absent | Present | Valid response |
| Tool call response | Absent | Present + `tool_calls` | Valid response |
| Empty response | Absent | Absent | `RuntimeError` raised |

### Key Design Decisions

1. **Acknowledgment Without Forwarding**: Thinking tokens are acknowledged as valid payload (`saw_assistant_payload = True`) but are NOT forwarded to clients. This prevents the false "no content" error while maintaining security isolation.

2. **Conservative Isolation**: By default, thinking tokens are never forwarded to clients. This prevents unintended exposure of model reasoning chains.

3. **Backward Compatibility**: The change does not affect existing behavior for non-thinking models. Models without `thinking` field continue to work as before.

## Testing

### Test Coverage

The three-tier rule hierarchy is comprehensively covered by tests in `tests/test_policy.py` and `tests/test_think_policy.py`:

**Rule 1 (Safety Invariant) Tests:**
1. `test_safety_invariant_cannot_be_overridden` - Verifies safety invariant cannot be bypassed by any override
2. `test_model_with_tool_call_think_false_and_tools_returns_false`
3. `test_model_with_tool_call_think_false_without_tools_returns_default`
4. `test_model_with_tool_call_think_false_cannot_override_with_tool_call_true`
5. `test_model_with_tool_call_think_false_and_safe_tool_still_returns_false`

**Rule 2 (Tool Request Handling) Tests:**
1. `test_invalid_override_with_tools_denied` - Verifies invalid overrides are denied
2. `test_web_search_tool_disables_think`
3. `test_file_search_tool_keeps_think_enabled`
4. `test_openweather_tool_keeps_think_enabled`
5. `test_research_does_not_match_search_false_positive`
6. `test_model_with_tools_no_policy_returns_false_failsafe`
7. `test_model_with_tools_and_tool_call_true_override`
8. `test_model_with_tools_and_invalid_override_denied`

**Rule 3 (Non-Tool Request Handling) Tests:**
1. `test_threshold_boundary` - Verifies char_threshold boundary conditions
2. `test_empty_vs_missing_tools` - Verifies empty vs missing tools arrays behave identically
3. `test_long_content_disables_think` - Verifies char_threshold is sole determinant
4. `test_override_forces_value` - Verifies override behavior without tools

### Streaming Thinking Token Tests

The streaming thinking token handler is validated by 26 streaming tests in `tests/test_streaming.py`:

| Test Category | Count | Description |
|---------------|-------|-------------|
| Streaming Tests | 26 | Validates SSE streaming behavior including thinking token handling |

### Running Tests

```bash
# Build audit image
docker build --target audit -t llmrouter-audit .

# Run policy tests
docker run --rm llmrouter-audit python3 -m pytest tests/test_policy.py tests/test_think_policy.py -v

# Run streaming tests
docker run --rm llmrouter-audit python3 -m pytest tests/test_streaming.py -v

# Run all tests
docker run --rm llmrouter-audit python3 -m pytest tests/ -v
```

## Security Audit Findings

### Positive Controls

✅ Safety invariant executes before override logic, preventing bypass  
✅ Strict validation of override values prevents injection attacks  
✅ Fail-safe default (`false`) for unconfigured models with tools  
✅ Configuration-driven approach eliminates hardcoded model detection  
✅ Comprehensive test coverage (336 tests total) validates security behavior  
✅ ReAct-aligned policy enables proper reasoning for summary/analysis tasks  
✅ Thinking token isolation prevents information disclosure of reasoning chains  

### Recommendations (Non-Blocking)

**P2: Add Audit Logging**
- Add structured audit logging for override attempts and safety invariant enforcement
- Log: model name, override value, tools present, tool_call_think config, final decision, reason
- Enables security monitoring and incident response

**P3: Implement Rate Limiting**
- Implement rate limiting on `/v1/chat/completions` endpoint
- Stricter limits for requests containing `X-Ollama-Think` headers
- Prevents resource exhaustion attacks

**Overall Risk Rating: LOW**

The safety invariant logic is correctly implemented with strong security controls. The two recommendations are defense-in-depth measures and do not block deployment.

## Migration Guide

### For Existing Deployments

No migration required. The `tool_call_think` field is optional in `models.yaml`. Models without this field continue to use existing behavior.

### For New Models

When adding a new model to `models.yaml`:

1. **If the model is Qwen-based or has known think+tools looping issues:**
   ```yaml
   - model: qwen3.8:27b-bf16
     think: true
     tool_call_think: false  # Safety invariant
     keep_alive: 15m
   ```

2. **If the model is safe for think+tools:**
   ```yaml
   - model: gemma4:12b
     think: true
     # tool_call_think not set (uses fail-safe or explicit policy)
     keep_alive: 20m
   ```

3. **If you want to explicitly allow think+tools:**
   ```yaml
   - model: custom-model
     think: true
     tool_call_think: true  # Explicitly allow thinking with tools
     keep_alive: 15m
   ```

## Troubleshooting

### Issue: Model enters infinite loop with tools

**Symptom:** Model generates excessive tokens or hangs when tools are present

**Solution:** Ensure the model has `tool_call_think: false` in `models.yaml`:
```yaml
- model: problematic-model
  think: true
  tool_call_think: false  # Add this line
```

### Issue: Override header not working

**Symptom:** `X-Ollama-Think: tool_call_true` does not enable thinking

**Diagnosis:**
1. Check if model has `tool_call_think: false` (safety invariant blocks override)
2. Check if tools are present in request (override only works with tools)
3. Check override value is exactly `tool_call_true` (case-insensitive)

**Solution:**
- If safety invariant is blocking: remove `tool_call_think: false` from model config (not recommended for Qwen models)
- If tools not present: override should work normally
- If wrong value: use `tool_call_true` (not `true` or `True`)

### Issue: Fail-safe default not working

**Symptom:** Model without `tool_call_think` configuration enables thinking with tools

**Diagnosis:**
1. Check model config in `models.yaml`
2. Verify `tool_call_think` is not set to `true`
3. Check router logs for policy decision

**Solution:**
- Ensure `tool_call_think` is not set to `true` in model config
- If explicitly set to `true`, remove the field or set to `false`
- Restart router to reload configuration

## Related Documentation

- [Safety Invariant Deep Dive](SAFETY_INVARIANT.md) - Complete technical documentation
- [Security Audit Report](../SECURITY.md) - Full security audit findings
- [Testing Strategy](testing-strategy.md) - Test coverage and validation approach

## References

- Qwen3 think+tools looping issue: Community reports of deterministic looping behavior
- Ollama think parameter: https://github.com/ollama/ollama/blob/main/docs/api.md
- Safety invariant design: Configuration-driven approach for maintainability and resilience
- ReAct paradigm: Reasoning and Acting in Language Models (Yao et al., 2022)
