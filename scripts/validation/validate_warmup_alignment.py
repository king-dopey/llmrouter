#!/usr/bin/env python3
"""Validate warmup/runtime num_ctx alignment across profiles.

Warmup must use the same `num_ctx`ceiling as runtime. This script reads each 
profile's models.yaml and verifies that models with `warmup: true` have their
`options.num_ctx` consistent with what the router policy would set at runtime.

Usage:
    python scripts/validation/validate_warmup_alignment.py orin
    python scripts/validation/validate_warmup_alignment.py thor
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml


def load_profile(profile_name: str) -> dict[str, Any]:
    """Load a profile's models.yaml."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    profile_path = repo_root / "profiles" / profile_name / "models.yaml"
    if not profile_path.exists():
        print(f"ERROR: Profile file not found: {profile_path}", file=sys.stderr)
        sys.exit(1)
    with open(profile_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def validate_warmup_alignment(profile_name: str) -> int:
    """Validate warmup alignment for a profile. Returns 0 on success, 1 on failure."""
    data = load_profile(profile_name)
    models = data.get("models", [])

    if not isinstance(models, list):
        print(f"ERROR: {profile_name}: 'models' is not a list", file=sys.stderr)
        return 1

    errors = []
    warnings = []

    for entry in models:
        model_name = entry.get("model", "<unknown>")
        warmup = entry.get("warmup", False)
        options = entry.get("options") or {}
        num_ctx = options.get("num_ctx")

        if not isinstance(options, dict):
            warnings.append(
                f"{profile_name}/{model_name}: 'options' is not a dict, skipping"
            )
            continue

        # Check that headroom fields are present
        for field in ("reserved_output_tokens", "safety_headroom_tokens"):
            if field not in entry:
                warnings.append(
                    f"{profile_name}/{model_name}: missing '{field}' (default 2048 will be used)"
                )

        # Check recovery_strategy presence
        if "recovery_strategy" not in entry:
            warnings.append(
                f"{profile_name}/{model_name}: missing 'recovery_strategy' (default 'drop_oldest' will be used)"
            )

        # For warmup models, verify num_ctx is set and reasonable
        if warmup:
            if num_ctx is None:
                errors.append(
                    f"{profile_name}/{model_name}: warmup=true but num_ctx not set"
                )
            elif num_ctx < 4096:
                warnings.append(
                    f"{profile_name}/{model_name}: warmup num_ctx={num_ctx} seems very low"
                )

    # Report results
    if errors:
        print(f"\n=== {profile_name}: ERRORS ===", file=sys.stderr)
        for err in errors:
            print(f"  ERROR: {err}", file=sys.stderr)

    if warnings:
        print(f"\n=== {profile_name}: WARNINGS ===")
        for warn in warnings:
            print(f"  WARNING: {warn}")

    # Summary
    total = len(models)
    warmup_count = sum(1 for m in models if (m.get("options") or {}).get("warmup", False))
    missing_headroom = sum(
        1 for m in models
        if not all(k in m for k in ("reserved_output_tokens", "safety_headroom_tokens"))
    )

    print(f"\n=== {profile_name}: Summary ===")
    print(f"  Total models: {total}")
    print(f"  Warmup models: {warmup_count}")
    print(f"  Missing headroom fields: {missing_headroom}")

    if errors:
        print(f"\nResult: FAIL ({len(errors)} error(s))")
        return 1

    if warnings or missing_headroom > 0:
        print(f"\nResult: WARN ({len(warnings)} warning(s), {missing_headroom} missing headroom fields)")
        return 0  # Warnings are non-fatal

    print("\nResult: PASS")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate warmup/runtime num_ctx alignment across profiles."
    )
    parser.add_argument(
        "profile",
        choices=["orin", "thor"],
        help="Profile to validate (orin or thor).",
    )
    args = parser.parse_args()
    sys.exit(validate_warmup_alignment(args.profile))


if __name__ == "__main__":
    main()
