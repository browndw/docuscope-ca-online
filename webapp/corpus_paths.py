"""Helpers for portable built-in corpus path references."""

from __future__ import annotations

from pathlib import Path

from webapp.config.unified import config


BUILTIN_CORPUS_REF_PREFIX = "builtin:"


def _builtin_corpus_root() -> Path:
    """Return the runtime-local built-in corpus root."""

    return Path(config.corpus_dir_path).resolve(strict=False)


def is_builtin_corpus_ref(value: str) -> bool:
    """Return whether a value is a portable built-in corpus reference."""

    return value.startswith(BUILTIN_CORPUS_REF_PREFIX)


def make_portable_corpus_path(path_value: str) -> str:
    """Convert runtime-local built-in corpus paths to portable references."""

    if not path_value:
        return path_value
    if is_builtin_corpus_ref(path_value):
        return path_value

    path = Path(path_value).resolve(strict=False)
    try:
        relative_path = path.relative_to(_builtin_corpus_root())
    except ValueError:
        return path_value

    return f"{BUILTIN_CORPUS_REF_PREFIX}{relative_path.as_posix()}"


def resolve_corpus_path(path_value: str) -> str:
    """Resolve a portable corpus reference to the current runtime-local path."""

    if not path_value or not is_builtin_corpus_ref(path_value):
        return path_value
    relative_path = path_value.removeprefix(BUILTIN_CORPUS_REF_PREFIX)
    return str(_builtin_corpus_root() / relative_path)


def corpus_display_name(path_value: str) -> str:
    """Return the corpus directory name for either portable refs or paths."""

    return Path(resolve_corpus_path(path_value)).name