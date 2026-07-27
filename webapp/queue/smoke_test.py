"""CLI helper to exercise the Redis/RQ smoke-test architecture."""

from __future__ import annotations

import argparse
import json
import time

from webapp.persistence import initialize_database_schema, registry_service
from webapp.queue import enqueue_registry_smoke_test


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a Redis/RQ smoke test job.")
    parser.add_argument("--request-key", default="local-smoke")
    parser.add_argument("--requester", default=None)
    parser.add_argument("--wait-seconds", type=float, default=10.0)
    parser.add_argument("--poll-interval", type=float, default=0.5)
    return parser


def main() -> None:
    """Submit a small queue-backed job and optionally wait for completion."""

    args = _build_parser().parse_args()
    initialize_database_schema()

    result = enqueue_registry_smoke_test(
        request_key=args.request_key,
        requester_principal_id=args.requester,
    )
    print(
        json.dumps(
            {
                "state": result.state,
                "control_plane_job_id": result.control_plane_job_id,
                "rq_job_id": result.rq_job_id,
                "artifact_id": result.artifact_id,
            },
            indent=2,
            sort_keys=True,
        )
    )

    if result.control_plane_job_id is None:
        return

    deadline = time.monotonic() + args.wait_seconds
    while time.monotonic() < deadline:
        job = registry_service.get_job_by_id(result.control_plane_job_id)
        if job is None:
            raise SystemExit("Control-plane job disappeared before completion.")

        if job.status == "completed":
            artifact = registry_service.get_artifact_by_id(job.artifact_id)
            payload = registry_service.load_json_artifact(artifact)
            print(json.dumps(payload, indent=2, sort_keys=True))
            return

        if job.status == "failed":
            raise SystemExit(f"Smoke test job failed: {job.failure_reason}")

        time.sleep(args.poll_interval)

    raise SystemExit(
        f"Timed out waiting for control-plane job {result.control_plane_job_id} to complete."
    )


if __name__ == "__main__":
    main()