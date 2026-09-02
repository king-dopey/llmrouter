# Safety Invariant for Tool-Call Thinking

## Overview

The router enforces a **configuration-driven safety invariant** to prevent infinite loops when models with thinking enabled receive tool schemas. This is particularly critical for Qwen3 models, which enter deterministic looping behavior when `think: true` is combined with tool definitions.

## Safety Invariant Rule

**Any model configured with `tool_call_think: false` in `models.yaml` cannot be overridden to `think: true` when tools are present in the request.**

This safety invariant:
- Executes **before** any override logic, ensuring it cannot be bypassed
- Applies to **any model** with `tool_call_think: false`, not just Qwen models
- Is configuration-driven, not hardcoded to specific model names
- Provides fail-safe defaults for models without explicit configuration

## Configuration

### Model Configuration Example

```yaml
# profiles/orin/models.yaml or profiles/thor/models.yaml
- model: qwen3.6:35b-a3b
  think: true                    # Default behavior when no tools present
  tool_call_think: false         # Safety invariant: cannot be overridden when tools present
  keep_alive: 15m
  options:
    num_ctx: 65536
    num_batch: 256
    temperature: 0.18
```

### Configuration Fields

| Field | Type | Description |
|-------|------|-------------|
| `think` | boolean | Default thinking behavior when no tools are present |
| `tool_call_think` | boolean \| null | Thinking behavior when tools are present. `false` enforces safety invariant. `null` uses fail-safe default. |

## Behavior Matrix

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

### Override Examples

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

#### Example 3: Invalid Override Denied

```bash
# Invalid override value with tools present
curl -sS http://127.0.0.1:4000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'X-Ollama-Think: true' \
  -d '{
    "model": "gemma4:12b",
    "messages": [{"role": "user", "content": "Use a tool."}],
    "tools": [{"type": "function", "function": {"name": "read_file"}}]
  }' | jq .

# Result: think=false (invalid override denied, fail-safe applied)
```

## Security Controls

### 1. Safety Invariant Priority

The safety invariant executes **before** any override logic in the decision tree:

```python
# Rule 1: Safety invariant (policy.py:221)
if has_tools and tool_call_think is False:
    return False  # Cannot be overridden

# Rule 2: Override handling (policy.py:225-238)
if has_tools:
    if override == "tool_call_true":
        return True
    elif override is not None:
        return False  # Invalid override denied
    # ... explicit policy or fail-safe
```

This ensures that models configured with `tool_call_think: false` cannot be forced into thinking mode when tools are present, regardless of client override attempts.

### 2. Strict Validation

The `parse_think_override()` function (policy.py:67-85) enforces strict validation:

```python
def parse_think_override(header_value: str | None) -> str | None:
    if header_value is None:
        return None
    
    value = header_value.strip().lower()
    if value in {"true", "false", "tool_call_true"}:
        return value
    raise ValueError("X-Ollama-Think must be 'true', 'false', or 'tool_call_true'")
```

- Only three valid values accepted
- Case-insensitive matching
- Invalid values raise `ValueError` (HTTP 400)
- Prevents injection attacks or unexpected behavior from malformed headers

### 3. Fail-Safe Defaults

When tools are present and no explicit `tool_call_think` policy is configured:

```python
# Rule 2d: Fail-safe (policy.py:237-238)
if tool_call_think is not None:
    return tool_call_think
else:
    return False  # Fail-safe default
```

This prevents accidental exposure to the Qwen3 infinite loop vulnerability for models that haven't been explicitly configured.

### 4. Configuration-Driven Approach

The safety invariant is **not** hardcoded to specific model names:

```python
# Configuration-driven, not hardcoded
if has_tools and tool_call_think is False:
    return False
```

Benefits:
- **Maintainable**: No code changes needed when adding new models
- **Resilient**: Works for any model family, not just Qwen
- **Flexible**: Operators control behavior via YAML configuration
- **Consistent**: Single source of truth in `models.yaml`

## Decision Tree

```
Incoming request?
├── Has tools in body?
│   ├── No → apply existing heuristics (char threshold, summary patterns, default)
│   └── Yes
│       ├── Model has tool_call_think: false?
│       │   └── YES → think: false (SAFETY INVARIANT — NO OVERRIDES ALLOWED)
│       └── Model has tool_call_think: true or null?
│           ├── X-Ollama-Think header present?
│           │   ├── "tool_call_true" → think: true
│           │   └── Any other value → think: false (deny by default)
│           └── No override?
│               ├── Explicit tool_call_think policy exists? → use it
│               └── No explicit policy? → think: false (FAIL-SAFE)
```

## Testing

### Test Coverage

The safety invariant is covered by 7 comprehensive tests in `tests/test_think_policy.py`:

1. `test_model_with_tool_call_think_false_and_tools_returns_false`
   - Verifies safety invariant enforces `think: false` when tools present

2. `test_model_with_tool_call_think_false_without_tools_returns_default`
   - Verifies safety invariant only applies when tools present

3. `test_model_with_tool_call_think_false_cannot_override_with_tool_call_true`
   - Verifies safety invariant cannot be bypassed by override

4. `test_model_with_tools_no_policy_returns_false_failsafe`
   - Verifies fail-safe default when no explicit policy

5. `test_model_with_tools_and_tool_call_true_override`
   - Verifies valid override for non-safety-constrained models

6. `test_model_with_tools_and_invalid_override_denied`
   - Verifies invalid override is denied

7. `test_model_with_tool_call_think_false_and_safe_tool_still_returns_false`
   - Verifies safety invariant applies regardless of tool type

### Running Tests

```bash
# Build audit image
docker build --target audit -t llmrouter-audit .

# Run policy tests
docker run --rm llmrouter-audit python3 -m pytest tests/test_think_policy.py -v

# Run all tests
docker run --rm llmrouter-audit python3 -m pytest tests/ -v
```

## Security Audit Findings

### Positive Controls

✅ Safety invariant executes before override logic, preventing bypass  
✅ Strict validation of override values prevents injection attacks  
✅ Fail-safe default (`false`) for unconfigured models with tools  
✅ Configuration-driven approach eliminates hardcoded model detection  
✅ Comprehensive test coverage (43 policy tests) validates security behavior  

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

- [Think Control Policy](../README.md#think-control-policy-router---external-llm-backend) - Overview in README
- [Security Audit Report](SECURITY.md) - Full security audit findings
- [Testing Strategy](testing-strategy.md) - Test coverage and validation approach

## References

- Qwen3 think+tools looping issue: Community reports of deterministic looping behavior
- Ollama think parameter: https://github.com/ollama/ollama/blob/main/docs/api.md
- Safety invariant design: Configuration-driven approach for maintainability and resilience
