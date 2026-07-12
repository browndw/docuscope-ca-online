#!/usr/bin/env sh
set -eu

BASE_URL="${DOCUSCOPE_AI_BASE_URL:-http://localhost:8000/v1}"
MODEL="${DOCUSCOPE_AI_MODEL:-${DOCUSCOPE_QWEN_MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}}"
MODELS_URL="${BASE_URL%/}/models"

printf '%s\n' "Checking Qwen-compatible endpoint: ${MODELS_URL}"
printf '%s\n' "Expected model: ${MODEL}"

RESPONSE="$(curl -fsS "${MODELS_URL}")"
printf '%s\n' "${RESPONSE}" | grep -F "${MODEL}" >/dev/null

printf '%s\n' "Qwen endpoint is ready and exposes ${MODEL}."