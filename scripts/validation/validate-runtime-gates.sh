#!/usr/bin/env bash
set -euo pipefail

PROFILE_NAME="${1:-${PROFILE:-orin}}"
OLLAMA_URL="${OLLAMA_URL:-http://127.0.0.1:11434}"

curl -fsS "${OLLAMA_URL}/api/tags" >/dev/null
curl -fsS "${OLLAMA_URL}/api/ps" | jq . >/dev/null

if [ "${PROFILE_NAME}" = "orin" ]; then
  curl -fsS "${OLLAMA_URL}/api/ps" | jq -e '.models[] | select(.name=="qwen3-coder:30b")' >/dev/null
  printf 'Orin runtime gate: coder warm residency present\n'
else
  curl -fsS "${OLLAMA_URL}/api/ps" | jq -e '.models[] | select(.name=="qwen3-coder-next:q4_K_M")' >/dev/null
  curl -fsS "${OLLAMA_URL}/api/ps" | jq -e '.models[] | select(.name=="qwen3.6:35b-a3b-q8_0")' >/dev/null
  printf 'Thor runtime gate: dual warm residency present\n'
fi
