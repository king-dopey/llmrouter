import importlib
import json
import os
from pathlib import Path
import sys
from unittest import mock

ROUTER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROUTER_DIR))


def _reload_modules_with_approx_map(model_name: str):
    os.environ["TOKENIZER_MAP"] = json.dumps({model_name: "approximate"})
    import tokenizer
    importlib.reload(tokenizer)
    import router_headroom
    importlib.reload(router_headroom)
    return router_headroom


def test_calculate_usable_prompt_budget():
    headroom = _reload_modules_with_approx_map("test-model")
    policy = headroom.HeadroomPolicy(
        model="test-model",
        max_num_ctx=1000,
        reserved_output_tokens=100,
        safety_headroom_tokens=50,
        trim_strategy="drop_oldest",
    )
    assert headroom.calculate_usable_prompt(policy) == 850


def test_trim_drops_tool_before_history():
    model = "test-model"
    headroom = _reload_modules_with_approx_map(model)
    policy_entry = {
        "options": {"num_ctx": 36},
        "reserved_output_tokens": 8,
        "safety_headroom_tokens": 8,
        "trim_strategy": "drop_oldest",
    }
    # Budget = 20 tokens. Approx token count uses len(text)//4.
    # Headroom compresses into a short replacement message list.
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "tool", "content": "x" * 80},
        {"role": "user", "content": "keep me"},
    ]

    with mock.patch.object(headroom, "_headroom_compress", return_value=[{"role": "system", "content": "tiny"}]):
        result = headroom.check_and_trim(messages, model, policy_entry)
    assert result.rejected is False
    assert result.trimmed is True
    assert result.messages == [{"role": "system", "content": "tiny"}]


def test_reject_when_cannot_fit_after_trim():
    model = "test-model"
    headroom = _reload_modules_with_approx_map(model)
    policy_entry = {
        "options": {"num_ctx": 20},
        "reserved_output_tokens": 8,
        "safety_headroom_tokens": 8,
        "trim_strategy": "drop_oldest",
    }
    # Budget = 4 tokens. No effective compression means this still exceeds budget.
    messages = [{"role": "system", "content": "z" * 40}]

    with mock.patch.object(headroom, "_headroom_compress", return_value=messages):
        result = headroom.check_and_trim(messages, model, policy_entry)
    assert result.rejected is True
    assert result.error_response is not None
    details = result.error_response["error"]["details"]
    assert details["model"] == model
    assert details["usable_prompt_budget"] == 4
    assert details["final_prompt_tokens"] > details["usable_prompt_budget"]


def test_headroom_disabled_passthrough(monkeypatch):
    model = "test-model"
    headroom = _reload_modules_with_approx_map(model)
    policy_entry = {
        "options": {"num_ctx": 24},
        "reserved_output_tokens": 8,
        "safety_headroom_tokens": 8,
        "trim_strategy": "drop_oldest",
    }
    messages = [{"role": "user", "content": "a" * 40}]
    monkeypatch.setenv("HEADROOM_ENABLED", "0")
    with mock.patch.object(headroom, "_headroom_compress", return_value=[{"role": "user", "content": "tiny"}]):
        result = headroom.check_and_trim(messages, model, policy_entry)
    assert result.rejected is True
