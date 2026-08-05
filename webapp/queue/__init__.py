"""Redis/RQ queue helpers for background-job experiments.

This module also guards against stdlib shadowing when Streamlit is launched via
`streamlit run webapp/index.py`. In that invocation mode, `/.../webapp` can end
up first on `sys.path`, causing `import queue` (from the Python stdlib) to
resolve to this package directory. When imported as top-level `queue`, we load
and re-export stdlib `queue.py` symbols instead of queue client helpers.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sysconfig


def _load_stdlib_queue_symbols() -> dict[str, object]:
    """Load symbols from the stdlib `queue.py` module by file path."""

    stdlib_dir = Path(sysconfig.get_paths()["stdlib"])
    queue_py = stdlib_dir / "queue.py"
    spec = importlib.util.spec_from_file_location("_stdlib_queue", queue_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load stdlib queue module from {queue_py}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    symbols: dict[str, object] = {
        name: getattr(module, name)
        for name in dir(module)
        if name not in {"__builtins__", "__cached__", "__file__", "__loader__", "__spec__"}
    }
    return symbols


if __name__ == "queue":
    globals().update(_load_stdlib_queue_symbols())
else:
    from .client import (
        QueueCollocationEnqueueResult,
        QueueInternalTargetEnqueueResult,
        QueueKeynessEnqueueResult,
        QueueKeynessPartsEnqueueResult,
        QueueNgramEnqueueResult,
        QueuePlotbotEnqueueResult,
        QueueSmokeEnqueueResult,
        build_internal_target_identity,
        build_queue_smoke_identity,
        enqueue_collocation_preparation,
        enqueue_keyness_preparation,
        enqueue_keyness_parts_preparation,
        enqueue_ngram_preparation,
        enqueue_plotbot_generation,
        enqueue_internal_target_preparation,
        enqueue_registry_smoke_test,
        get_queue,
        get_redis_connection,
    )
    from .config import RedisQueueConfig, get_redis_queue_config

    __all__ = [
        "QueueCollocationEnqueueResult",
        "QueueInternalTargetEnqueueResult",
        "QueueKeynessEnqueueResult",
        "QueueKeynessPartsEnqueueResult",
        "QueueNgramEnqueueResult",
        "QueuePlotbotEnqueueResult",
        "QueueSmokeEnqueueResult",
        "RedisQueueConfig",
        "build_internal_target_identity",
        "build_queue_smoke_identity",
        "enqueue_collocation_preparation",
        "enqueue_keyness_preparation",
        "enqueue_keyness_parts_preparation",
        "enqueue_ngram_preparation",
        "enqueue_plotbot_generation",
        "enqueue_internal_target_preparation",
        "enqueue_registry_smoke_test",
        "get_queue",
        "get_redis_queue_config",
        "get_redis_connection",
    ]
