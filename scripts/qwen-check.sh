#!/usr/bin/env sh
set -eu

BASE_URL="${DOCUSCOPE_AI_BASE_URL:-http://localhost:8000/v1}"
MODEL="${DOCUSCOPE_AI_MODEL:-${DOCUSCOPE_QWEN_MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}}"
MODELS_URL="${BASE_URL%/}/models"
# Large models can take several minutes to load, so poll instead of failing immediately.
WAIT_SECONDS="${DOCUSCOPE_QWEN_CHECK_WAIT_SECONDS:-300}"
POLL_INTERVAL_SECONDS=5

printf '%s\n' "Checking Qwen-compatible endpoint: ${MODELS_URL}"
printf '%s\n' "Expected model: ${MODEL}"
printf '%s\n' "Waiting up to ${WAIT_SECONDS}s for the model to finish loading..."

elapsed=0
while [ "${elapsed}" -lt "${WAIT_SECONDS}" ]; do
  if RESPONSE="$(curl -fsS "${MODELS_URL}" 2>/dev/null)" && printf '%s\n' "${RESPONSE}" | grep -F "${MODEL}" >/dev/null; then
    printf '%s\n' "Qwen endpoint is ready and exposes ${MODEL}."
    exit 0
  fi
  sleep "${POLL_INTERVAL_SECONDS}"
  elapsed=$((elapsed + POLL_INTERVAL_SECONDS))
done

printf 'ERROR: Qwen endpoint did not become ready with model %s within %ss.\n' "${MODEL}" "${WAIT_SECONDS}" >&2
printf 'Check logs with: docker compose logs qwen_model\n' >&2
exit 1