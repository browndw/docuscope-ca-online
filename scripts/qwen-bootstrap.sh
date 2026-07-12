#!/usr/bin/env sh
set -eu

MODEL="${DOCUSCOPE_QWEN_MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}"

printf '%s\n' "Preparing Qwen model cache for: ${MODEL}"
printf '%s\n' "This uses the optional Docker Compose qwen profile and persistent qwen_model_cache volume."

DOCUSCOPE_QWEN_MODEL="${MODEL}" docker compose --profile qwen run --rm qwen_bootstrap

printf '%s\n' "Qwen model cache is prepared."
printf '%s\n' "Next: scripts/qwen-up.sh"