"""
Corpus Data Manager for memory-efficient lazy loading and data access.

This module provides a centralized system for managing corpus data with:
- Lazy loading of derived data structures
- Session-scoped caching with automatic cleanup
- Unified data access across all corpus types (Internal, External, New)
- Backward compatibility with existing session state patterns
"""

import gc
import gzip
import os
import pickle
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Lock
from time import perf_counter
from typing import Dict, Iterator, Optional, Any
import polars as pl
import streamlit as st
import docuscospacy as ds

from webapp.utilities.memory import DataFrameCache, lazy_computation
from webapp.utilities.configuration.logging_config import get_logger
from webapp.persistence import (
    SharedArtifactWorkflow,
    build_shared_frequency_identity,
    registry_service,
)
from webapp.utilities.session.session_persistence import auto_persist_session
from webapp.utilities.session.session_persistence import mark_session_dirty
from webapp.corpus_paths import resolve_corpus_path

logger = get_logger()
shared_artifact_workflow = SharedArtifactWorkflow(registry_service, logger)
ARTIFACT_REF_KEY = "_artifact_refs"
SLOW_CORPUS_OPERATION_MS = 50
ARTIFACT_FRAME_CACHE_MAX_ITEMS = 32
ArtifactFrameCacheOwner = int | str
_artifact_frame_cache: OrderedDict[
    tuple[ArtifactFrameCacheOwner, str],
    pl.DataFrame,
] = OrderedDict()


@dataclass
class _ArtifactFrameLockEntry:
    lock: Lock
    users: int = 0


_artifact_frame_cache_locks: dict[
    tuple[ArtifactFrameCacheOwner, str],
    _ArtifactFrameLockEntry,
] = {}
_artifact_frame_cache_state_lock = Lock()


def _get_cached_artifact_frame(
    cache_owner: ArtifactFrameCacheOwner,
    key: str,
) -> Optional[pl.DataFrame]:
    """Return an ephemeral in-process cache hit for an artifact-backed frame."""

    cache_key = (cache_owner, key)
    with _artifact_frame_cache_state_lock:
        cached = _artifact_frame_cache.get(cache_key)
        if cached is None:
            return None

        _artifact_frame_cache.move_to_end(cache_key)
        return cached


def _set_cached_artifact_frame(
    cache_owner: ArtifactFrameCacheOwner,
    key: str,
    value: pl.DataFrame,
) -> None:
    """Store an artifact-backed frame in the ephemeral in-process cache."""

    cache_key = (cache_owner, key)
    with _artifact_frame_cache_state_lock:
        _artifact_frame_cache[cache_key] = value
        _artifact_frame_cache.move_to_end(cache_key)

        while len(_artifact_frame_cache) > ARTIFACT_FRAME_CACHE_MAX_ITEMS:
            _artifact_frame_cache.popitem(last=False)


@contextmanager
def _artifact_frame_load_lock(
    cache_owner: ArtifactFrameCacheOwner,
    key: str,
) -> Iterator[None]:
    """Serialize one frame load and discard the lock after its final user."""

    cache_key = (cache_owner, key)
    with _artifact_frame_cache_state_lock:
        entry = _artifact_frame_cache_locks.get(cache_key)
        if entry is None:
            entry = _ArtifactFrameLockEntry(lock=Lock())
            _artifact_frame_cache_locks[cache_key] = entry
        entry.users += 1

    try:
        with entry.lock:
            yield
    finally:
        with _artifact_frame_cache_state_lock:
            entry.users -= 1
            if entry.users == 0:
                _artifact_frame_cache_locks.pop(cache_key, None)


def _build_shared_frequency_cache_owner(identity) -> str:
    """Build a stable in-process cache owner key for shared frequency identities."""

    return (
        "shared-frequency:"
        f"{identity.selector_hash}:"
        f"{identity.parameter_hash}:"
        f"{identity.pipeline_version}:"
        f"{identity.model_version}"
    )


def _build_file_backed_cache_owner(path: str) -> str:
    """Build a stable cache owner key for a file-backed artifact path."""

    try:
        stat_result = os.stat(path)
        return (
            "file-backed:"
            f"{path}:"
            f"{stat_result.st_mtime_ns}:"
            f"{stat_result.st_size}"
        )
    except OSError:
        return f"file-backed:{path}"


def _build_session_frame_cache_owner(user_session_id: str, corpus_type: str) -> str:
    """Build a process-local cache owner for per-session ephemeral frame aliases."""

    return f"session-frame:{user_session_id}:{corpus_type}"


def _clear_cached_artifact_frame(
    cache_owner: ArtifactFrameCacheOwner,
    key: str,
) -> None:
    """Clear one cached frame entry without disturbing other cached frames."""

    with _artifact_frame_cache_state_lock:
        _artifact_frame_cache.pop((cache_owner, key), None)


class CorpusDataManager:
    """
    Session-scoped corpus data manager with smart caching and lazy loading.

    This manager provides a unified interface for accessing corpus data while
    implementing memory-efficient patterns like lazy loading and intelligent caching.
    """

    def __init__(
        self,
        user_session_id: str,
        corpus_type: str = "target",
        session_manager=None
    ):
        """
        Initialize corpus data manager for a specific user session and corpus type.

        Parameters
        ----------
        user_session_id : str
            The user session identifier
        corpus_type : str
            Type of corpus ('target' or 'reference')
        session_manager : optional
            Session manager instance for dependency injection
        """
        self.user_session_id = user_session_id
        self.corpus_type = corpus_type
        self.cache = DataFrameCache(
            user_session_id,
            max_size=15,
            session_manager=session_manager
        )

        # Core data keys (always loaded immediately)
        self.core_keys = ["ds_tokens"]

        # Derived data keys (computed on-demand)
        self.derived_keys = [
            "dtm_ds", "dtm_pos", "ft_ds", "ft_pos", "ft_pos_general", "tt_ds", "tt_pos"
        ]

        # Additional data keys (generated/stored independently)
        self.additional_keys = [
            "collocations"
        ]

        # All expected data keys
        self.all_keys = self.core_keys + self.derived_keys + self.additional_keys

    @property
    def session_corpus_data(self) -> Dict:
        """Get the corpus data dictionary from session state."""
        if self.user_session_id not in st.session_state:
            st.session_state[self.user_session_id] = {}

        if self.corpus_type not in st.session_state[self.user_session_id]:
            st.session_state[self.user_session_id][self.corpus_type] = {}

        return st.session_state[self.user_session_id][self.corpus_type]

    def has_core_data(self) -> bool:
        """Check if core data (ds_tokens) is available."""
        return (
            "ds_tokens" in self.session_corpus_data or
            self.has_artifact_ref("ds_tokens")
        )

    def has_data_key(self, key: str) -> bool:
        """Check if a specific data key exists in session or can be generated."""
        if key in self.session_corpus_data:
            return True
        if self.has_artifact_ref(key):
            return True
        if key == "ft_pos_general":
            return (
                "ft_pos" in self.session_corpus_data or
                self.has_artifact_ref("ft_pos") or
                self.has_core_data()
            )
        if key in self.derived_keys and self.has_core_data():
            return True
        # Additional keys are only available if explicitly stored
        if key in self.additional_keys:
            return key in self.session_corpus_data
        return False

    def get_core_data(self) -> Optional[pl.DataFrame]:
        """Get the core ds_tokens DataFrame."""
        core_data = self.session_corpus_data.get("ds_tokens")
        if core_data is not None:
            return core_data
        return self._load_artifact_backed_data("ds_tokens")

    def set_core_data(self, ds_tokens: pl.DataFrame, persist: bool = True) -> None:
        """Set the core ds_tokens DataFrame."""
        self.session_corpus_data["ds_tokens"] = ds_tokens
        self._clear_artifact_ref("ds_tokens")
        # Clear any cached derived data since core data changed
        self._invalidate_derived_cache()
        mark_session_dirty(self.user_session_id)
        if persist:
            # Persist the session with new core data
            auto_persist_session(self.user_session_id)

    def _invalidate_derived_cache(self) -> None:
        """Clear cached derived data when core data changes."""
        cache_keys_to_clear = [
            f"{self.corpus_type}_{key}" for key in self.derived_keys
        ]
        for cache_key in cache_keys_to_clear:
            # Remove from both session state and cache
            if cache_key in self.session_corpus_data:
                del self.session_corpus_data[cache_key]

        # Clear cache entries
        cache = self.cache._get_cache()
        for key in list(cache.keys()):
            if any(derived_key in key for derived_key in self.derived_keys):
                del cache[key]
        self.cache._set_cache(cache)

        self._clear_session_ephemeral_data(self.derived_keys)

    def _get_session_value(self, key: str, default=None):
        """Read a key from the session flags container without importing session helpers."""

        session_raw = st.session_state.get(self.user_session_id, {}).get("session", {})
        if hasattr(session_raw, "to_dict") and hasattr(session_raw, "columns"):
            session_dict = session_raw.to_dict(as_series=False)
            value = session_dict.get(key, default)
            if isinstance(value, list) and value:
                return value[0]
            return default if isinstance(value, list) else value

        if isinstance(session_raw, dict):
            value = session_raw.get(key, default)
            if isinstance(value, list) and value:
                return value[0]
            return default if isinstance(value, list) else value

        return default

    def _get_artifact_refs(self) -> Dict[str, Dict[str, Any]]:
        """Get or initialize artifact references for this corpus."""

        refs = self.session_corpus_data.get(ARTIFACT_REF_KEY)
        if not isinstance(refs, dict):
            refs = {}
            self.session_corpus_data[ARTIFACT_REF_KEY] = refs
        return refs

    def has_artifact_ref(self, key: str) -> bool:
        """Check whether a data key is backed by an artifact reference."""

        return key in self._get_artifact_refs()

    def set_artifact_refs(
        self,
        artifact_type: str,
        artifact_id: int,
        data_keys: list[str],
    ) -> None:
        """Store lightweight artifact references instead of large DataFrames."""

        refs = self._get_artifact_refs()
        for key in data_keys:
            refs[key] = {
                "artifact_type": artifact_type,
                "artifact_id": artifact_id,
            }
            self.session_corpus_data.pop(key, None)
        self._clear_session_ephemeral_data(data_keys)

    def set_file_refs(self, file_map: Dict[str, str]) -> None:
        """Store lightweight file references for precomputed corpus artifacts."""

        refs = self._get_artifact_refs()
        for key, path in file_map.items():
            refs[key] = {
                "storage_type": "gzip_pickle",
                "path": path,
            }
            self.session_corpus_data.pop(key, None)
        self._clear_session_ephemeral_data(list(file_map.keys()))

    def _clear_artifact_ref(self, key: str) -> None:
        """Remove any artifact reference for a specific key."""

        self._get_artifact_refs().pop(key, None)

    def _get_session_ephemeral_data(self, key: str) -> Optional[pl.DataFrame]:
        """Return a process-local per-session alias for a derived frame."""

        cache_owner = _build_session_frame_cache_owner(
            self.user_session_id,
            self.corpus_type,
        )
        return _get_cached_artifact_frame(cache_owner, key)

    def _set_session_ephemeral_data(
        self,
        key: str,
        data: Optional[pl.DataFrame],
    ) -> None:
        """Store a process-local per-session alias for a derived frame."""

        if data is None:
            return

        cache_owner = _build_session_frame_cache_owner(
            self.user_session_id,
            self.corpus_type,
        )
        _set_cached_artifact_frame(cache_owner, key, data)

    def _clear_session_ephemeral_data(self, keys: list[str]) -> None:
        """Clear process-local per-session aliases for one or more data keys."""

        cache_owner = _build_session_frame_cache_owner(
            self.user_session_id,
            self.corpus_type,
        )
        for key in keys:
            _clear_cached_artifact_frame(cache_owner, key)

    def _load_file_backed_data(self, key: str) -> Optional[pl.DataFrame]:
        """Load a DataFrame from a file-backed artifact reference."""

        ref = self._get_artifact_refs().get(key)
        if ref is None or ref.get("storage_type") != "gzip_pickle":
            return None

        path = ref.get("path")
        if not path:
            return None
        path = resolve_corpus_path(path)

        cache_owner = _build_file_backed_cache_owner(path)
        cached = _get_cached_artifact_frame(cache_owner, key)
        if cached is not None:
            return cached

        with _artifact_frame_load_lock(cache_owner, key):
            cached = _get_cached_artifact_frame(cache_owner, key)
            if cached is not None:
                return cached

            try:
                with gzip.open(path, "rb") as file_handle:
                    loaded = pickle.load(file_handle)
            except Exception as exc:
                logger.warning(f"File-backed data load failed for {key}: {exc}")
                return None

            if isinstance(loaded, pl.DataFrame):
                _set_cached_artifact_frame(cache_owner, key, loaded)
            return loaded

    def _load_artifact_backed_data(self, key: str) -> Optional[pl.DataFrame]:
        """Load a DataFrame from a stored artifact reference."""

        operation_start = perf_counter()
        cache_stage = "ref_miss"
        lock_wait_ms = 0.0
        registry_lookup_ms = 0.0
        payload_load_ms = 0.0

        ref = self._get_artifact_refs().get(key)
        if ref is None:
            return None

        if ref.get("storage_type") == "gzip_pickle":
            return self._load_file_backed_data(key)

        artifact_id = ref.get("artifact_id")
        if artifact_id is None:
            return None

        cached = _get_cached_artifact_frame(artifact_id, key)
        if cached is not None:
            cache_stage = "cache_hit"
            return cached

        lock_wait_start = perf_counter()
        with _artifact_frame_load_lock(artifact_id, key):
            lock_wait_ms = (perf_counter() - lock_wait_start) * 1000
            cached = _get_cached_artifact_frame(artifact_id, key)
            if cached is not None:
                cache_stage = "post_lock_cache_hit"
                return cached

            cache_stage = "payload_load"
            registry_lookup_start = perf_counter()
            artifact = registry_service.get_public_artifact_by_id(artifact_id)
            registry_lookup_ms = (perf_counter() - registry_lookup_start) * 1000
            if artifact is None or artifact.status != "ready":
                return None

            try:
                payload_load_start = perf_counter()
                payload = registry_service.load_artifact_payload(artifact)
                payload_load_ms = (perf_counter() - payload_load_start) * 1000
            except Exception as exc:
                logger.warning(f"Artifact-backed data load failed for {key}: {exc}")
                return None

            for payload_key, payload_value in payload.items():
                if isinstance(payload_value, pl.DataFrame):
                    _set_cached_artifact_frame(artifact_id, payload_key, payload_value)

        elapsed_ms = (perf_counter() - operation_start) * 1000
        if elapsed_ms >= SLOW_CORPUS_OPERATION_MS:
            logger.warning(
                (
                    "Slow corpus artifact load session={} corpus={} key={} "
                    "stage={} duration_ms={:.2f} lock_wait_ms={:.2f} "
                    "registry_lookup_ms={:.2f} payload_load_ms={:.2f}"
                ),
                self.user_session_id,
                self.corpus_type,
                key,
                cache_stage,
                elapsed_ms,
                lock_wait_ms,
                registry_lookup_ms,
                payload_load_ms,
            )

        return payload.get(key)

    def _get_shared_frequency_identity(self):
        """Return a shared frequency identity for built-in target corpora."""

        if self.corpus_type != "target":
            return None

        target_db = self._get_session_value("target_db", "")
        if not target_db:
            return None

        try:
            return build_shared_frequency_identity(target_source=target_db)
        except ValueError:
            return None

    def _load_cached_frequency_tables(self) -> tuple[pl.DataFrame, pl.DataFrame] | None:
        """Load shared built-in frequency tables from the artifact registry."""

        operation_start = perf_counter()

        identity = self._get_shared_frequency_identity()
        loaded = shared_artifact_workflow.load_ready(
            identity,
            registry_service.load_frequency_bundle,
            cache_name="frequency",
        )
        if loaded is None:
            return None

        artifact, frequency_frames = loaded
        self.set_artifact_refs(
            artifact.artifact_type,
            artifact.artifact_id,
            ["ft_pos", "ft_ds"],
        )
        elapsed_ms = (perf_counter() - operation_start) * 1000
        if elapsed_ms >= SLOW_CORPUS_OPERATION_MS:
            logger.warning(
                "Slow shared frequency cache load session={} duration_ms={:.2f}",
                self.user_session_id,
                elapsed_ms,
            )
        return frequency_frames["ft_pos"], frequency_frames["ft_ds"]

    def _reserve_shared_frequency_artifact(self):
        """Reserve the shared frequency artifact or defer to an in-flight build."""

        identity = self._get_shared_frequency_identity()
        return shared_artifact_workflow.reserve(
            identity,
            cache_name="frequency",
            ready_loader=self._load_cached_frequency_tables,
            poll_attempts=20,
            poll_interval_seconds=0.25,
        )

    def _store_cached_frequency_tables(
        self,
        job_id: int | None,
        ft_pos: pl.DataFrame,
        ft_ds: pl.DataFrame,
    ) -> None:
        """Store shared built-in frequency tables in the artifact registry."""

        identity = self._get_shared_frequency_identity()
        artifact = shared_artifact_workflow.store(
            identity,
            job_id,
            cache_name="frequency",
            store_func=lambda artifact_identity: registry_service.store_frequency_bundle(
                artifact_identity,
                {
                    "ft_pos": ft_pos,
                    "ft_ds": ft_ds,
                },
            ),
        )
        if artifact is not None:
            self.set_artifact_refs(
                artifact.artifact_type,
                artifact.artifact_id,
                ["ft_pos", "ft_ds"],
            )

    def warm_shared_frequency_data(self, prime_general: bool = True) -> str:
        """Warm shared built-in frequency artifacts to avoid first-wave contention."""

        if self.corpus_type != "target":
            return "not_target"

        if self._get_shared_frequency_identity() is None:
            return "not_shared"

        if self.has_artifact_ref("ft_pos") and self.has_artifact_ref("ft_ds"):
            return "precomputed_file_refs"

        cached = self._load_cached_frequency_tables()
        if cached is not None:
            if prime_general:
                self.get_data("ft_pos_general")
            return "cache_hit"

        reservation = self._reserve_shared_frequency_artifact()
        if reservation is not None and reservation.state == "ready":
            if prime_general:
                self.get_data("ft_pos_general")
            return "cache_hit"
        if reservation is not None and reservation.state == "pending":
            return "pending"

        job_id = reservation.job_id if reservation is not None else None
        ds_tokens = self.get_core_data()
        if ds_tokens is None:
            return "missing_core"

        try:
            ft_pos, ft_ds = ds.frequency_table(ds_tokens, count_by="both")
        except Exception as exc:
            if job_id is not None:
                registry_service.mark_job_failed(job_id, str(exc))
            logger.warning(
                f"Shared frequency warm failed for {self.user_session_id}: {exc}"
            )
            return "error"

        self._store_cached_frequency_tables(job_id, ft_pos, ft_ds)
        if prime_general:
            self.get_data("ft_pos_general")
        return "generated"

    def _generate_frequency_tables(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Generate frequency tables from core data."""
        cached = self._load_cached_frequency_tables()
        if cached is not None:
            return cached

        ds_tokens = self.get_core_data()
        if ds_tokens is None:
            raise ValueError(
                "Core data (ds_tokens) not available for frequency table generation"
            )

        reservation = self._reserve_shared_frequency_artifact()
        if reservation is not None and reservation.state == "ready":
            return reservation.payload
        if reservation is not None and reservation.state == "pending":
            raise ValueError(
                "Shared token frequencies are currently being generated. "
                "Please retry shortly."
            )

        job_id = reservation.job_id if reservation is not None else None

        # Generate frequency tables - no logging needed for normal operations
        try:
            ft_pos, ft_ds = ds.frequency_table(ds_tokens, count_by="both")
        except Exception as exc:
            if job_id is not None:
                registry_service.mark_job_failed(job_id, str(exc))
            raise

        self._store_cached_frequency_tables(job_id, ft_pos, ft_ds)
        return ft_pos, ft_ds

    def _generate_tags_tables(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Generate tags tables from core data."""
        ds_tokens = self.get_core_data()
        if ds_tokens is None:
            raise ValueError(
                "Core data (ds_tokens) not available for tags table generation"
            )

        # Generate tags tables - no logging needed for normal operations
        return ds.tags_table(ds_tokens, count_by="both")

    def _generate_dtm_tables(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Generate document-term matrices from core data."""
        ds_tokens = self.get_core_data()
        if ds_tokens is None:
            raise ValueError("Core data (ds_tokens) not available for DTM generation")

        # Generate DTM tables - no logging needed for normal operations
        return ds.tags_dtm(ds_tokens, count_by="both")

    def get_data(self, key: str, force_refresh: bool = False) -> Optional[pl.DataFrame]:
        """
        Get data by key with lazy loading for derived data.

        Parameters
        ----------
        key : str
            Data key to retrieve
        force_refresh : bool
            Whether to force regeneration of derived data

        Returns
        -------
        Optional[pl.DataFrame]
            The requested data or None if not available
        """
        operation_start = perf_counter()

        def finalize(result: Optional[pl.DataFrame]) -> Optional[pl.DataFrame]:
            elapsed_ms = (perf_counter() - operation_start) * 1000
            if elapsed_ms >= SLOW_CORPUS_OPERATION_MS:
                logger.warning(
                    (
                        "Slow corpus get_data session={} corpus={} key={} "
                        "duration_ms={:.2f} force_refresh={}"
                    ),
                    self.user_session_id,
                    self.corpus_type,
                    key,
                    elapsed_ms,
                    force_refresh,
                )
            return result

        if not force_refresh and key == "ft_pos_general":
            cached_session_frame = self._get_session_ephemeral_data(key)
            if cached_session_frame is not None:
                return finalize(cached_session_frame)

        # Check if data exists in session state first
        if not force_refresh and key in self.session_corpus_data:
            return finalize(self.session_corpus_data[key])

        if not force_refresh:
            artifact_data = self._load_artifact_backed_data(key)
            if artifact_data is not None:
                return finalize(artifact_data)

        # If it's core data and still not found (no in-memory or artifact-backed
        # value), return None rather than falling through to derived-data logic.
        if key in self.core_keys:
            return finalize(self.session_corpus_data.get(key))

        # Handle derived data with lazy loading
        if key == "ft_pos_general":
            if (
                "ft_pos" in self.session_corpus_data or
                self.has_artifact_ref("ft_pos") or
                self.has_core_data()
            ):
                return finalize(self._get_derived_data(key, force_refresh))

        if key in self.derived_keys and self.has_core_data():
            return finalize(self._get_derived_data(key, force_refresh))

        # Handle additional keys (stored independently)
        if key in self.additional_keys:
            return finalize(self.session_corpus_data.get(key))

        return finalize(None)

    def _get_derived_data(
        self, key: str, force_refresh: bool = False
    ) -> Optional[pl.DataFrame]:
        """Get derived data with caching."""
        if key in ["ft_pos", "ft_ds"] and self._get_shared_frequency_identity() is not None:
            ft_pos, ft_ds = self._generate_frequency_tables()
            return ft_ds if key == "ft_ds" else ft_pos

        def compute_derived_data():
            if key == "ft_pos_general":
                from webapp.utilities.analysis import freq_simplify_pl

                stage = "session_compute"
                lock_wait_ms = 0.0
                ft_pos_ms = 0.0
                simplify_ms = 0.0
                derive_start = perf_counter()

                def finalize_general(
                    result: Optional[pl.DataFrame],
                ) -> Optional[pl.DataFrame]:
                    elapsed_ms = (perf_counter() - derive_start) * 1000
                    if elapsed_ms >= SLOW_CORPUS_OPERATION_MS:
                        logger.warning(
                            (
                                "Slow corpus derived get_data session={} corpus={} key={} "
                                "stage={} duration_ms={:.2f} lock_wait_ms={:.2f} "
                                "ft_pos_ms={:.2f} simplify_ms={:.2f}"
                            ),
                            self.user_session_id,
                            self.corpus_type,
                            key,
                            stage,
                            elapsed_ms,
                            lock_wait_ms,
                            ft_pos_ms,
                            simplify_ms,
                        )
                    return result

                artifact_ref = self._get_artifact_refs().get("ft_pos")
                cache_owner: ArtifactFrameCacheOwner | None = None
                if isinstance(artifact_ref, dict):
                    cache_owner = artifact_ref.get("artifact_id")

                if cache_owner is None:
                    shared_frequency_identity = self._get_shared_frequency_identity()
                    if shared_frequency_identity is not None:
                        cache_owner = _build_shared_frequency_cache_owner(
                            shared_frequency_identity
                        )

                if cache_owner is not None:
                    if not force_refresh:
                        cached = _get_cached_artifact_frame(cache_owner, key)
                        if cached is not None:
                            stage = "shared_cache_hit"
                            self._set_session_ephemeral_data(key, cached)
                            return finalize_general(cached)

                    lock_wait_start = perf_counter()
                    with _artifact_frame_load_lock(cache_owner, key):
                        lock_wait_ms = (perf_counter() - lock_wait_start) * 1000
                        if not force_refresh:
                            cached = _get_cached_artifact_frame(cache_owner, key)
                            if cached is not None:
                                stage = "shared_post_lock_cache_hit"
                                self._set_session_ephemeral_data(key, cached)
                                return finalize_general(cached)

                        stage = "shared_compute"
                        ft_pos_start = perf_counter()
                        ft_pos = self.get_data("ft_pos", force_refresh=force_refresh)
                        ft_pos_ms = (perf_counter() - ft_pos_start) * 1000
                        if ft_pos is None:
                            return finalize_general(None)

                        simplify_start = perf_counter()
                        simplified = freq_simplify_pl(ft_pos)
                        simplify_ms = (perf_counter() - simplify_start) * 1000
                        _set_cached_artifact_frame(cache_owner, key, simplified)
                        self._set_session_ephemeral_data(key, simplified)
                        return finalize_general(simplified)

                ft_pos_start = perf_counter()
                ft_pos = self.get_data("ft_pos", force_refresh=force_refresh)
                ft_pos_ms = (perf_counter() - ft_pos_start) * 1000
                if ft_pos is None:
                    return finalize_general(None)

                simplify_start = perf_counter()
                simplified = freq_simplify_pl(ft_pos)
                simplify_ms = (perf_counter() - simplify_start) * 1000
                return finalize_general(simplified)

            if key in ["ft_pos", "ft_ds"]:
                ft_pos, ft_ds = self._generate_frequency_tables()

                if self._get_shared_frequency_identity() is not None:
                    # Built-in shared frequency tables are kept in the artifact store
                    # and referenced from session state rather than persisted inline.
                    return ft_ds if key == "ft_ds" else ft_pos

                # Cache both tables
                self.session_corpus_data["ft_pos"] = ft_pos
                self.session_corpus_data["ft_ds"] = ft_ds
                mark_session_dirty(self.user_session_id)
                # Persist session with new cached data
                auto_persist_session(self.user_session_id)
                return ft_ds if key == "ft_ds" else ft_pos

            if key in ["tt_pos", "tt_ds"]:
                tt_pos, tt_ds = self._generate_tags_tables()
                # Cache both tables
                self.session_corpus_data["tt_pos"] = tt_pos
                self.session_corpus_data["tt_ds"] = tt_ds
                mark_session_dirty(self.user_session_id)
                # Persist session with new cached data
                auto_persist_session(self.user_session_id)
                return tt_ds if key == "tt_ds" else tt_pos

            if key in ["dtm_pos", "dtm_ds"]:
                dtm_pos, dtm_ds = self._generate_dtm_tables()
                # Cache both tables
                self.session_corpus_data["dtm_pos"] = dtm_pos
                self.session_corpus_data["dtm_ds"] = dtm_ds
                mark_session_dirty(self.user_session_id)
                # Persist session with new cached data
                auto_persist_session(self.user_session_id)
                return dtm_ds if key == "dtm_ds" else dtm_pos

            return None

        if key == "ft_pos_general":
            artifact_ref = self._get_artifact_refs().get("ft_pos")
            has_shared_cache_owner = False
            if isinstance(artifact_ref, dict) and artifact_ref.get("artifact_id") is not None:  # noqa: E501
                has_shared_cache_owner = True
            elif self._get_shared_frequency_identity() is not None:
                has_shared_cache_owner = True

            if has_shared_cache_owner:
                # Shared/general frequency tables are already cached above.
                # Avoid also copying the simplified table into per-session cache state.
                return compute_derived_data()

        cache_key = f"{self.corpus_type}_{key}_{self.user_session_id}"
        return lazy_computation(
            cache_key=cache_key,
            computation_func=compute_derived_data,
            user_session_id=self.user_session_id,
            force_refresh=force_refresh,
        )

    def set_data(self, key: str, data: pl.DataFrame) -> None:
        """
        Set data by key in session state.

        Parameters
        ----------
        key : str
            Data key to set
        data : pl.DataFrame
            Data to store
        """
        self.session_corpus_data[key] = data
        self._clear_artifact_ref(key)

        # If setting core data, invalidate derived cache
        if key in self.core_keys:
            self._invalidate_derived_cache()

        mark_session_dirty(self.user_session_id)

        # Persist the session with new data
        auto_persist_session(self.user_session_id)

    def load_all_data(self, data_dict: Dict[str, pl.DataFrame]) -> None:
        """
        Load all data at once (for legacy compatibility).

        Parameters
        ----------
        data_dict : Dict[str, pl.DataFrame]
            Dictionary of all data to load
        """
        for key, data in data_dict.items():
            self.session_corpus_data[key] = data

        mark_session_dirty(self.user_session_id)

        # Persist the session with all loaded data
        auto_persist_session(self.user_session_id)

    def get_available_keys(self) -> list[str]:
        """Get list of available data keys."""
        available = list(self.session_corpus_data.keys())
        for key in self._get_artifact_refs().keys():
            if key not in available:
                available.append(key)

        # Add derived keys that can be generated
        if self.has_core_data():
            for key in self.derived_keys:
                if key not in available:
                    available.append(key)

        return available

    def is_ready(self) -> bool:
        """Check if corpus has minimum required data."""
        return self.has_core_data()

    def clear_data(self) -> None:
        """Clear all corpus data and cache."""
        # Clear session state data
        self.session_corpus_data.clear()

        # Clear cache
        self.cache.clear()

        # Force garbage collection
        gc.collect()

    def get_memory_usage_info(self) -> Dict[str, Any]:
        """Get information about memory usage."""
        info = {
            "corpus_type": self.corpus_type,
            "user_session_id": self.user_session_id,
            "available_keys": self.get_available_keys(),
            "core_data_loaded": self.has_core_data(),
            "session_data_keys": list(self.session_corpus_data.keys()),
            "cache_size": len(self.cache._get_cache())
        }

        # Add size information if possible
        if self.has_core_data():
            core_data = self.get_core_data()
            if core_data is not None:
                info["core_data_shape"] = core_data.shape
                info["core_data_memory_mb"] = core_data.estimated_size() / (1024 * 1024)

        return info


def get_corpus_manager(
    user_session_id: str, corpus_type: str = "target"
) -> CorpusDataManager:
    """
    Get or create a corpus data manager instance.

    This function creates a new manager instance each time to avoid serialization
    issues with storing complex objects in session state. The managers access the
    same underlying data through session state, ensuring consistency.

    Parameters
    ----------
    user_session_id : str
        The user session identifier
    corpus_type : str
        Type of corpus ('target' or 'reference')

    Returns
    -------
    CorpusDataManager
        A new corpus data manager instance
    """
    # Always create a new manager instance to avoid session state serialization issues
    # The manager will access the same underlying data through session state
    return CorpusDataManager(user_session_id, corpus_type)
