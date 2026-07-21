#!/bin/sh
set -eu

OLLAMA_URL=${OLLAMA_URL:-http://ollama:11434}
WARMUP_MODELS=${WARMUP_MODELS:-qwen3-coder:30b@16384 qwen3.6:35b-a3b@32768}
WARMUP_DEFAULT_NUM_CTX=${WARMUP_DEFAULT_NUM_CTX:-16384}
KV_CACHE_TYPE=${KV_CACHE_TYPE:-q8_0}
PULL_MAX_RETRIES=${PULL_MAX_RETRIES:-3}
PULL_BACKOFF_SEC=${PULL_BACKOFF_SEC:-10}

log() {
  printf '[warmup] %s\n' "$*"
}

fetch_tags() {
  curl -fsS "${OLLAMA_URL}/api/tags"
}

fetch_ps() {
  curl -fsS "${OLLAMA_URL}/api/ps"
}

model_in_tags() {
  model_name=$1
  tags_json=$2
  compact=$(printf '%s' "${tags_json}" | tr -d '\n\r\t ')
  printf '%s' "${compact}" | grep -F "\"name\":\"${model_name}\"" >/dev/null 2>&1
}

model_ps_entry() {
  model_name=$1
  ps_json=$2
  compact=$(printf '%s' "${ps_json}" | tr -d '\n\r\t ')
  printf '%s' "${compact}" |
    sed 's/},{/}\n{/g' |
    grep -F "\"name\":\"${model_name}\"" |
    head -n 1
}

model_expires_at() {
  entry=$1
  printf '%s\n' "${entry}" | sed -n 's/.*"expires_at":"\([^"]*\)".*/\1/p'
}

model_size_vram() {
  entry=$1
  printf '%s\n' "${entry}" | sed -n 's/.*"size_vram":\([0-9][0-9]*\).*/\1/p'
}

model_context_length() {
  entry=$1
  printf '%s\n' "${entry}" | sed -n 's/.*"context_length":\([0-9][0-9]*\).*/\1/p'
}

has_date_dash_d() {
  if date -d '1970-01-01T00:00:01Z' +%s >/dev/null 2>&1; then
    return 0
  fi
  return 1
}

model_is_resident() {
  model_name=$1
  ps_json=$2
  entry=$(model_ps_entry "${model_name}" "${ps_json}" || printf '')
  if [ -z "${entry}" ]; then
    return 1
  fi

  expires_at=$(model_expires_at "${entry}")
  if [ -z "${expires_at}" ]; then
    return 1
  fi

  if has_date_dash_d; then
    now_epoch=$(date +%s)
    expires_epoch=$(date -d "${expires_at}" +%s 2>/dev/null || printf '0')
    if [ "${expires_epoch}" -gt "${now_epoch}" ]; then
      return 0
    fi
    return 1
  fi

  size_vram=$(model_size_vram "${entry}")
  if [ -z "${size_vram}" ]; then
    return 1
  fi

  case "${expires_at}" in
    1970-*)
      return 1
      ;;
  esac

  if [ "${size_vram}" -gt 0 ]; then
    return 0
  fi

  return 1
}

confirm_resident_after_warm() {
  model_name=$1
  requested_ctx=$2
  tries=1
  saw_wrong_ctx=0

  while [ "${tries}" -le 5 ]; do
    ps_json=$(fetch_ps || printf '')
    if [ -n "${ps_json}" ] && model_is_resident "${model_name}" "${ps_json}"; then
      entry=$(model_ps_entry "${model_name}" "${ps_json}" || printf '')
      resident_ctx=$(model_context_length "${entry}")
      if [ -n "${resident_ctx}" ] && [ "${resident_ctx}" = "${requested_ctx}" ]; then
        return 0
      fi
      saw_wrong_ctx=1
    fi
    if [ "${tries}" -lt 5 ]; then
      sleep 1
    fi
    tries=$((tries + 1))
  done

  if [ "${saw_wrong_ctx}" -eq 1 ]; then
    return 2
  fi

  return 1
}

emit_final_status() {
  status=$1
  model_name=$2
  detail=${3:-}
  if [ -n "${detail}" ]; then
    log "${status} ${model_name} ${detail}"
  else
    log "${status} ${model_name}"
  fi
}

random_0_to() {
  max=$1
  if [ "${max}" -le 0 ]; then
    printf '0\n'
    return
  fi

  rand_raw=$(od -An -N2 -tu2 /dev/urandom 2>/dev/null | tr -d ' ')
  if [ -z "${rand_raw}" ]; then
    rand_raw=$(date +%s)
  fi
  printf '%s\n' $((rand_raw % (max + 1)))
}

pull_stream_once() {
  model_name=$1
  fifo=$(mktemp -u)
  mkfifo "${fifo}"

  curl -fsS -N -X POST "${OLLAMA_URL}/api/pull" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"${model_name}\",\"stream\":true}" >"${fifo}" &
  curl_pid=$!

  saw_success=0
  last_log=0

  while IFS= read -r line; do
    case "${line}" in
      *'"status":"success"'*)
        saw_success=1
        ;;
    esac

    now=$(date +%s)
    if [ $((now - last_log)) -ge 5 ]; then
      status=$(printf '%s\n' "${line}" | sed -n 's/.*"status":"\([^"]*\)".*/\1/p')
      digest=$(printf '%s\n' "${line}" | sed -n 's/.*"digest":"\([^"]*\)".*/\1/p')
      total=$(printf '%s\n' "${line}" | sed -n 's/.*"total":\([0-9][0-9]*\).*/\1/p')
      completed=$(printf '%s\n' "${line}" | sed -n 's/.*"completed":\([0-9][0-9]*\).*/\1/p')

      msg="pulling ${model_name}"
      if [ -n "${status}" ]; then
        msg="${msg} status=${status}"
      fi
      if [ -n "${completed}" ] && [ -n "${total}" ]; then
        msg="${msg} progress=${completed}/${total}"
      fi
      if [ -n "${digest}" ]; then
        msg="${msg} digest=${digest}"
      fi

      log "${msg}"
      last_log=${now}
    fi
  done <"${fifo}"

  rm -f "${fifo}"

  if ! wait "${curl_pid}"; then
    return 1
  fi

  if [ "${saw_success}" -eq 1 ]; then
    return 0
  fi

  return 1
}

pull_and_confirm_with_retries() {
  model_name=$1
  attempt=1

  while [ "${attempt}" -le "${PULL_MAX_RETRIES}" ]; do
    if [ "${attempt}" -gt 1 ]; then
      log "retry ${attempt}/${PULL_MAX_RETRIES} ${model_name}"
    fi

    if pull_stream_once "${model_name}"; then
      tags_after=$(fetch_tags || printf '')
      if [ -n "${tags_after}" ] && model_in_tags "${model_name}" "${tags_after}"; then
        return 0
      fi
      last_reason=post-pull-missing
    else
      last_reason=pull-failed
    fi

    if [ "${attempt}" -lt "${PULL_MAX_RETRIES}" ]; then
      exp=$((attempt - 1))
      delay_cap=$((PULL_BACKOFF_SEC * (1 << exp)))
      sleep_for=$(random_0_to "${delay_cap}")
      sleep "${sleep_for}"
    fi
    attempt=$((attempt + 1))
  done

  if [ "${last_reason}" = "post-pull-missing" ]; then
    return 2
  fi
  return 1
}

warm_model() {
  model_name=$1
  requested_ctx=$2
  http_code=$(curl -fsS -o /dev/null -w '%{http_code}' "${OLLAMA_URL}/api/generate" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"${model_name}\",\"prompt\":\"ok\",\"stream\":false,\"keep_alive\":-1,\"options\":{\"num_ctx\":${requested_ctx},\"cache_type_k\":\"${KV_CACHE_TYPE}\",\"cache_type_v\":\"${KV_CACHE_TYPE}\",\"num_keep\":256,\"num_predict\":1}}" \
    || printf '000')

  case "${http_code}" in
    2??)
      return 0
      ;;
  esac

  return 1
}

unload_model() {
  model_name=$1
  curl -fsS -o /dev/null -X POST "${OLLAMA_URL}/api/generate" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"${model_name}\",\"prompt\":\"\",\"stream\":false,\"keep_alive\":0}" \
    >/dev/null 2>&1 || return 1
  return 0
}

log "waiting for ${OLLAMA_URL}/api/tags"
ready_tries=0
while [ "${ready_tries}" -lt 60 ]; do
  if fetch_tags >/dev/null 2>&1; then
    break
  fi
  ready_tries=$((ready_tries + 1))
  sleep 2
done

if [ "${ready_tries}" -ge 60 ]; then
  log 'ERROR: Ollama not ready after 120s'
  exit 1
fi

failed_models=0
summary_lines=''
for entry in ${WARMUP_MODELS}; do
  m=${entry%@*}
  requested_ctx=${entry##*@}
  if [ "${requested_ctx}" = "${entry}" ] || [ -z "${requested_ctx}" ]; then
    requested_ctx=${WARMUP_DEFAULT_NUM_CTX}
  fi

  status=''
  detail=''

  ps_before=$(fetch_ps || printf '')
  if [ -n "${ps_before}" ] && model_is_resident "${m}" "${ps_before}"; then
    resident_entry=$(model_ps_entry "${m}" "${ps_before}" || printf '')
    resident_ctx=$(model_context_length "${resident_entry}")
    if [ -n "${resident_ctx}" ] && [ "${resident_ctx}" = "${requested_ctx}" ]; then
      status='already-warm'
      emit_final_status "${status}" "${m}"
      summary_lines="${summary_lines}${status} ${m}\n"
      continue
    fi

    if [ -n "${resident_ctx}" ]; then
      log "reloading ${m} ctx=${resident_ctx}->ctx=${requested_ctx}"
    else
      log "reloading ${m} ctx=unknown->ctx=${requested_ctx}"
    fi
    unload_model "${m}" || true
  fi

  initial_tags=$(fetch_tags || printf '')
  can_warm=0
  had_tag_before=0

  if [ -n "${initial_tags}" ] && model_in_tags "${m}" "${initial_tags}"; then
    can_warm=1
    had_tag_before=1
  else
    if pull_and_confirm_with_retries "${m}"; then
      can_warm=1
    else
      pull_rc=$?
      if [ "${pull_rc}" -eq 2 ]; then
        status='post-pull-missing'
      else
        status='pull-failed'
      fi
    fi
  fi

  if [ "${can_warm}" -eq 1 ]; then
    confirmed_tags=$(fetch_tags || printf '')
    if [ -z "${confirmed_tags}" ] || ! model_in_tags "${m}" "${confirmed_tags}"; then
      status='post-pull-missing'
    elif warm_model "${m}" "${requested_ctx}"; then
      if confirm_resident_after_warm "${m}" "${requested_ctx}"; then
        if [ "${had_tag_before}" -eq 1 ]; then
          status='already-pulled-warmed'
        else
          status='pulled-warmed'
        fi
      else
        warm_confirm_rc=$?
        if [ "${warm_confirm_rc}" -eq 2 ]; then
          status='wrong-ctx'
        else
          status='not-resident'
        fi
      fi
    else
      status='warm-failed'
      detail="http=${http_code}"
    fi
  fi

  emit_final_status "${status}" "${m}" "${detail}"
  summary_lines="${summary_lines}${status} ${m}\n"

  case "${status}" in
    already-warm|pulled-warmed|already-pulled-warmed)
      ;;
    *)
      failed_models=$((failed_models + 1))
      ;;
  esac
done

log '=== summary ==='
printf '%b' "${summary_lines}" | while IFS=' ' read -r sname mname; do
  if [ -n "${sname}" ] && [ -n "${mname}" ]; then
    printf '[warmup]   %s  %s\n' "${sname}" "${mname}"
  fi
done
log '==============='

if [ "${failed_models}" -eq 0 ]; then
  exit 0
fi

exit 1
