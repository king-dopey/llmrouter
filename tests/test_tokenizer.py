import importlib
import json
import os


def _reload_tokenizer_with_map(mapping: dict[str, str]):
    os.environ["TOKENIZER_MAP"] = json.dumps(mapping)
    # import here to ensure module loads after env is set
    import router.tokenizer as tokenizer
    importlib.reload(tokenizer)
    return tokenizer


def test_approximate_prompt_token_count(monkeypatch):
    tokenizer = _reload_tokenizer_with_map({"mymodel": "approximate"})
    messages = [{"role": "user", "content": "hello world"}]
    expected = sum(len(m["content"]) for m in messages) // 4
    assert tokenizer.count_prompt_tokens(messages, "mymodel") == expected


def test_approximate_completion_token_count(monkeypatch):
    tokenizer = _reload_tokenizer_with_map({"mymodel": "approximate"})
    text = "abcde" * 10
    expected = len(text) // 4
    assert tokenizer.count_completion_tokens(text, "mymodel") == expected
