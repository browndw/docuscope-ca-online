#!/usr/bin/env sh
set -eu

MODEL="${DOCUSCOPE_QWEN_MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}"

printf '%s\n' "Starting optional Qwen model service for: ${MODEL}"
printf '%s\n' "DocuScope app containers remain usable without this service unless DOCUSCOPE_AI_PROVIDER is enabled."

DOCUSCOPE_QWEN_MODEL="${MODEL}" docker compose --profile qwen up -d qwen_model

printf '%s\n' "Qwen model service requested."
printf '%s\n' "Set DocuScope AI env vars when enabling Plotbot local mode:"
printf '%s\n' "  DOCUSCOPE_AI_PROVIDER=local"
printf '%s\n' "  DOCUSCOPE_AI_BASE_URL=http://qwen_model:8000/v1"
printf '%s\n' "  DOCUSCOPE_AI_MODEL=${MODEL}"
printf '%s\n' "Check readiness with: scripts/qwen-check.sh"