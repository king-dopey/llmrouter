"""Tests for Thor profile model policy."""
import os
from pathlib import Path
import sys
import unittest

ROUTER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROUTER_DIR))


class TestThorPolicy(unittest.TestCase):
    """Test Thor profile policy file."""

    def test_model_policy_thor_includes_all_served_models(self):
        """Test Thor profile policy file."""
        import yaml
        
        policy_file = str(ROUTER_DIR / "profiles" / "thor" / "models.yaml")
        
        with open(policy_file, "r", encoding="utf-8") as f:
            parsed = yaml.safe_load(f) or {}
        
        models = parsed.get("models", [])
        model_table = {}
        for item in models:
            model_name = item.get("model")
            if not model_name:
                continue
            keep_alive = item.get("keep_alive")
            think = item.get("think")
            options = item.get("options") or {}
            model_table[model_name] = {
                "keep_alive": keep_alive,
                "think": bool(think) if think is not None else True,
                "options": options,
                "warmup": bool(item.get("warmup", False)),
            }

        # Thor profile models (from profiles/thor/models.yaml)
        coder = model_table["qwen3-coder-next:q4_K_M"]
        self.assertEqual(coder["keep_alive"], "45m")
        self.assertEqual(coder["think"], False)
        self.assertEqual(coder["options"]["num_ctx"], 131072)
        self.assertFalse(coder["warmup"])

        thinker = model_table["qwen3.6:35b-a3b-q8_0"]
        self.assertEqual(thinker["keep_alive"], -1)
        self.assertEqual(thinker["think"], True)
        self.assertEqual(thinker["options"]["num_ctx"], 262144)
        self.assertEqual(thinker["warmup"], True)

        chat_small = model_table["qwen3:4b"]
        self.assertEqual(chat_small["keep_alive"], "30m")
        self.assertEqual(chat_small["think"], True)
        self.assertEqual(chat_small["options"]["num_ctx"], 40960)
        self.assertEqual(chat_small["warmup"], False)


if __name__ == "__main__":
    unittest.main()
