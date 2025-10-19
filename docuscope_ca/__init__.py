"""Public API surface for DocuScope CA headless usage.

This thin layer provides a stable contract (semantic versioned indirectly with the
project) for programmatic access to corpus processing without invoking Streamlit.

Only the symbols re-exported here are considered *stable* for minor / patch
releases. Internal modules in ``webapp`` may change without notice.
"""
from .api import (
    process_corpus,
    CorpusResult,
    CorpusProcessingError,
    CorpusLoadError,
    ModelLoadError,
    ArtifactWriteError,
)

__all__ = [
    "process_corpus",
    "CorpusResult",
    "CorpusProcessingError",
    "CorpusLoadError",
    "ModelLoadError",
    "ArtifactWriteError",
]


def __getattr__(name):  # pragma: no cover - defensive
    if name == "__version__":
        try:
            from importlib.metadata import version
            return version("docuscope-ca-online")
        except Exception:  # fallback if not installed as a package
            return "0.0.0+local"
    raise AttributeError(name)
