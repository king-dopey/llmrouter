"""Tests for Orin profile model policy."""
import os
from pathlib import Path
import sys
import unittest

ROUTER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROUTER_DIR))


class TestOrinPolicy(unittest.TestCase):
    """Test Orin profile policy file."""

    def test_model_policy_orin_includes_all_served_models(self):
        """Test Orin profile policy file."""
        import yaml
        
        policy_file = str(ROUTER_DIR / "profiles" / "orin" / "models.yaml")
        
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

        # Orin profile models (from profiles/orin/models.yaml)
        coder = model_table["qwen3-coder:30b"]
        self.assertEqual(coder["keep_alive"], -1)
        self.assertEqual(coder["think"], False)
        self.assertEqual(coder["options"]["num_ctx"], 65536)
        self.assertTrue(coder["warmup"])

        thinker = model_table["qwen3.6:35b-a3b"]
        self.assertEqual(thinker["keep_alive"], "10m")
        self.assertEqual(thinker["think"], True)
        self.assertEqual(thinker["options"]["num_ctx"], 32768)
        self.assertEqual(thinker["warmup"], False)

        chat_small = model_table["qwen3:4b"]
        self.assertEqual(chat_small["keep_alive"], "10m")
        self.assertEqual(chat_small["think"], True)
        self.assertEqual(chat_small["options"]["num_ctx"], 65536)
        self.assertEqual(chat_small["warmup"], False)


if __name__ == "__main__":
    unittest.main()
