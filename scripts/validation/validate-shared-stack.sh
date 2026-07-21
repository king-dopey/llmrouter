#!/usr/bin/env bash
set -euo pipefail

PROFILE_NAME="${1:-${PROFILE:-orin}}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

assert_contains() {
  local haystack=$1
  local needle=$2
  local label=$3
  if ! grep -Fq "$needle" <<<"$haystack"; then
    fail "missing ${label}: expected to find '${needle}'"
  fi
}

[ -f "${ROOT_DIR}/profiles/${PROFILE_NAME}/models.yaml" ] || fail "missing profile models file for ${PROFILE_NAME}"

main_cfg="$(cd "${ROOT_DIR}" && PROFILE="${PROFILE_NAME}" docker compose config)"

assert_contains "$main_cfg" "source: ${ROOT_DIR}/profiles/${PROFILE_NAME}/models.yaml" "profile models.yaml source"
assert_contains "$main_cfg" "target: /app/model_policy.yml" "model policy target"

printf 'Shared-stack validation passed for profile=%s\n' "${PROFILE_NAME}"
