#!/usr/bin/env sh
set -eu

REQUEST_KEY="${1:-compose-rq-smoke-$(date +%Y%m%d%H%M%S)}"
WAIT_SECONDS="${DOCUSCOPE_RQ_SMOKE_WAIT_SECONDS:-20}"

printf '%s\n' "Running Redis/RQ smoke inside the streamlit_app container."
printf '%s\n' "Request key: ${REQUEST_KEY}"
printf '%s\n' "Wait seconds: ${WAIT_SECONDS}"

docker compose ps postgres redis streamlit_app rq_worker

docker compose exec -T streamlit_app \
  python -m webapp.queue.smoke_test \
    --request-key "${REQUEST_KEY}" \
    --wait-seconds "${WAIT_SECONDS}"

printf '%s\n' "Compose Redis/RQ smoke completed."