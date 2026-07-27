"""
Session persistence layer for integrating SQLite backend with Streamlit session state.

This module provides automatic session persistence by hooking into Streamlit's
session state management and syncing with the SQLite session backend.
"""

import polars as pl
import streamlit as st
from typing import Dict, Any
from datetime import datetime, timezone
import hashlib
import json
from time import perf_counter

from webapp.utilities.configuration.logging_config import get_logger
from webapp.utilities.state.session_keys import (
    CorpusPersistencePolicy,
    MetadataKeys,
    SessionKeys,
)

logger = get_logger()
SLOW_PERSISTENCE_OPERATION_MS = 25
MIN_PERSIST_INTERVAL_SECONDS = 0.5
ARTIFACT_REF_KEY = "_artifact_refs"
PERSISTED_CORPUS_KEYS = ("target", "reference")
CORPUS_POLICY_KEYS = {
    "target": SessionKeys.TARGET_PERSISTENCE_POLICY,
    "reference": SessionKeys.REFERENCE_PERSISTENCE_POLICY,
}
PERSISTED_SESSION_KEYS = (
    SessionKeys.CORPUS_PERSISTENCE_POLICY,
    SessionKeys.TARGET_PERSISTENCE_POLICY,
    SessionKeys.REFERENCE_PERSISTENCE_POLICY,
    SessionKeys.HAS_TARGET,
    SessionKeys.TARGET_DB,
    SessionKeys.HAS_META,
    SessionKeys.HAS_REFERENCE,
    SessionKeys.REFERENCE_DB,
)
PERSISTED_METADATA_KEYS = (
    MetadataKeys.TOKENS_POS,
    MetadataKeys.TOKENS_DS,
    MetadataKeys.NDOCS,
    MetadataKeys.MODEL,
    MetadataKeys.DOCIDS,
    MetadataKeys.TAGS_DS,
    MetadataKeys.TAGS_POS,
    MetadataKeys.DOCCATS,
)
NON_DURABLE_PERSISTENCE_POLICIES = {
    CorpusPersistencePolicy.TEMPORARY_SESSION_ONLY,
    CorpusPersistencePolicy.LOCAL_EXPORT_ONLY,
}
SUPPORTED_PERSISTENCE_POLICIES = {
    CorpusPersistencePolicy.SERVER_SAVED,
    CorpusPersistencePolicy.TEMPORARY_SESSION_ONLY,
    CorpusPersistencePolicy.LOCAL_EXPORT_ONLY,
}


def _normalize_session_mapping(raw_value: Any) -> Dict[str, Any]:
    """Normalize DataFrame-backed session payloads into plain dictionaries."""

    if raw_value is None:
        return {}

    if hasattr(raw_value, "to_dict") and hasattr(raw_value, "columns"):
        return raw_value.to_dict(as_series=False)

    if isinstance(raw_value, dict):
        return raw_value.copy()

    return {}


def _normalize_scalar(value: Any) -> Any:
    """Collapse DataFrame-style single-item lists into scalar values."""

    if isinstance(value, list):
        return value[0] if value else None
    return value


def _normalize_persistence_policy(value: Any) -> str:
    """Return a supported persistence policy or the default durable mode."""

    if value in SUPPORTED_PERSISTENCE_POLICIES:
        return value
    return CorpusPersistencePolicy.SERVER_SAVED


def _get_corpus_policy_from_mapping(
    normalized: Dict[str, Any],
    corpus_type: str,
) -> str:
    """Resolve a corpus policy from specific keys with legacy fallback."""

    policy_key = CORPUS_POLICY_KEYS[corpus_type]
    if policy_key in normalized:
        return _normalize_persistence_policy(
            _normalize_scalar(normalized.get(policy_key))
        )

    return _normalize_persistence_policy(
        _normalize_scalar(normalized.get(SessionKeys.CORPUS_PERSISTENCE_POLICY))
    )


def _get_legacy_session_policy_value(
    target_policy: str,
    reference_policy: str,
) -> str:
    """Collapse corpus-specific policies into a conservative legacy value."""

    if target_policy == reference_policy:
        return target_policy

    if (
        target_policy in NON_DURABLE_PERSISTENCE_POLICIES and
        reference_policy in NON_DURABLE_PERSISTENCE_POLICIES
    ):
        return CorpusPersistencePolicy.TEMPORARY_SESSION_ONLY

    return CorpusPersistencePolicy.SERVER_SAVED


def _corpus_policy_allows_persistence(session_raw: Any, corpus_type: str) -> bool:
    """Return whether a specific corpus may write durable state."""

    normalized = _normalize_session_mapping(session_raw)
    policy = _get_corpus_policy_from_mapping(normalized, corpus_type)
    return policy not in NON_DURABLE_PERSISTENCE_POLICIES


def _session_has_durable_persistence(session_raw: Any) -> bool:
    """Return whether any corpus in the session may write durable state."""

    return any(
        _corpus_policy_allows_persistence(session_raw, corpus_type)
        for corpus_type in CORPUS_POLICY_KEYS
    )


def _project_session_flags(session_raw: Any) -> Dict[str, Any]:
    """Extract only the durable session flags needed across reruns/restarts."""

    normalized = _normalize_session_mapping(session_raw)
    target_policy = _get_corpus_policy_from_mapping(normalized, "target")
    reference_policy = _get_corpus_policy_from_mapping(normalized, "reference")

    projected = {
        SessionKeys.CORPUS_PERSISTENCE_POLICY: _get_legacy_session_policy_value(
            target_policy,
            reference_policy,
        ),
        SessionKeys.TARGET_PERSISTENCE_POLICY: target_policy,
        SessionKeys.REFERENCE_PERSISTENCE_POLICY: reference_policy,
    }

    if _corpus_policy_allows_persistence(session_raw, "target"):
        for key in (
            SessionKeys.HAS_TARGET,
            SessionKeys.TARGET_DB,
            SessionKeys.HAS_META,
        ):
            if key in normalized:
                projected[key] = _normalize_scalar(normalized.get(key))

    if _corpus_policy_allows_persistence(session_raw, "reference"):
        for key in (
            SessionKeys.HAS_REFERENCE,
            SessionKeys.REFERENCE_DB,
        ):
            if key in normalized:
                projected[key] = _normalize_scalar(normalized.get(key))

    return projected


def _project_metadata(metadata_raw: Any) -> Dict[str, Any]:
    """Keep only lightweight metadata descriptors, never derived analysis payloads."""

    normalized = _normalize_session_mapping(metadata_raw)
    return {
        key: normalized[key]
        for key in PERSISTED_METADATA_KEYS
        if key in normalized
    }


def _project_corpus_state(corpus_raw: Any) -> Dict[str, Any]:
    """Persist only lightweight artifact/file references for corpus state."""

    if not isinstance(corpus_raw, dict):
        return {}

    refs = corpus_raw.get(ARTIFACT_REF_KEY)
    if not isinstance(refs, dict):
        return {}

    return {
        ARTIFACT_REF_KEY: {
            key: value.copy() if isinstance(value, dict) else value
            for key, value in refs.items()
        }
    }


def build_persistable_session_data(session_data: Dict[str, Any]) -> Dict[str, Any]:
    """Project Streamlit session state into the durable SQLite contract."""

    session_raw = session_data.get(SessionKeys.SESSION_DATAFRAME)

    projected: Dict[str, Any] = {
        SessionKeys.SESSION_DATAFRAME: _project_session_flags(session_raw)
    }

    target_metadata = _project_metadata(session_data.get(SessionKeys.METADATA_TARGET))
    if target_metadata and _corpus_policy_allows_persistence(session_raw, "target"):
        projected[SessionKeys.METADATA_TARGET] = target_metadata

    reference_metadata = _project_metadata(
        session_data.get(SessionKeys.METADATA_REFERENCE)
    )
    if (
        reference_metadata and
        _corpus_policy_allows_persistence(session_raw, "reference")
    ):
        projected[SessionKeys.METADATA_REFERENCE] = reference_metadata

    for corpus_key in PERSISTED_CORPUS_KEYS:
        if not _corpus_policy_allows_persistence(session_raw, corpus_key):
            continue
        corpus_state = _project_corpus_state(session_data.get(corpus_key))
        if corpus_state:
            projected[corpus_key] = corpus_state

    return projected


def restore_persisted_session_data(session_data: Dict[str, Any]) -> Dict[str, Any]:
    """Rebuild the in-memory session shape from the lightweight SQLite payload."""

    restored: Dict[str, Any] = {}

    session_flags = _project_session_flags(session_data.get(SessionKeys.SESSION_DATAFRAME))
    restored[SessionKeys.SESSION_DATAFRAME] = pl.from_dict(session_flags)

    target_metadata = _project_metadata(session_data.get(SessionKeys.METADATA_TARGET))
    if target_metadata:
        restored[SessionKeys.METADATA_TARGET] = target_metadata

    reference_metadata = _project_metadata(
        session_data.get(SessionKeys.METADATA_REFERENCE)
    )
    if reference_metadata:
        restored[SessionKeys.METADATA_REFERENCE] = reference_metadata

    for corpus_key in PERSISTED_CORPUS_KEYS:
        corpus_state = _project_corpus_state(session_data.get(corpus_key))
        if corpus_state:
            restored[corpus_key] = corpus_state

    return restored


def _update_session_value_without_persistence(
    session_id: str,
    key: str,
    value: Any,
) -> None:
    """Update a session value in memory without triggering persistence."""

    if session_id not in st.session_state:
        st.session_state[session_id] = {}

    session_container = st.session_state[session_id]
    session_raw = session_container.get("session")

    if session_raw is None:
        session_container["session"] = {key: value}
        return

    if hasattr(session_raw, "to_dict") and hasattr(session_raw, "columns"):
        if session_raw.height == 1:
            session_container["session"] = session_raw.with_columns(
                pl.lit(value).alias(key)
            )
            return

        session_data = session_raw.to_dict(as_series=False)
        session_data[key] = value
        session_container["session"] = pl.from_dict(session_data)
        return

    if isinstance(session_raw, dict):
        session_data = session_raw.copy()
        session_data[key] = value
        session_container["session"] = session_data
        return

    session_container["session"] = {key: value}


def update_session_value_without_persistence(
    session_id: str,
    key: str,
    value: Any,
) -> None:
    """Update a session value in memory without triggering persistence."""

    _update_session_value_without_persistence(session_id, key, value)


def get_session_persistence_policy(
    session_id: str,
    corpus_type: str | None = None,
) -> str:
    """Return the active corpus persistence policy for a session."""

    session_container = st.session_state.get(session_id, {})
    session_raw = session_container.get("session")

    if session_raw is None:
        return CorpusPersistencePolicy.SERVER_SAVED

    if corpus_type is not None and corpus_type not in CORPUS_POLICY_KEYS:
        raise ValueError(f"Unsupported corpus type for persistence policy: {corpus_type}")

    normalized = _normalize_session_mapping(session_raw)

    if corpus_type is not None:
        return _get_corpus_policy_from_mapping(normalized, corpus_type)

    if SessionKeys.CORPUS_PERSISTENCE_POLICY in normalized:
        return _normalize_persistence_policy(
            _normalize_scalar(normalized.get(SessionKeys.CORPUS_PERSISTENCE_POLICY))
        )

    target_policy = _get_corpus_policy_from_mapping(normalized, "target")
    reference_policy = _get_corpus_policy_from_mapping(normalized, "reference")
    return _get_legacy_session_policy_value(target_policy, reference_policy)


def set_session_persistence_policy(
    session_id: str,
    policy: str,
    corpus_type: str | None = None,
) -> None:
    """Store a session-scoped corpus persistence policy in memory."""

    if policy not in SUPPORTED_PERSISTENCE_POLICIES:
        raise ValueError(f"Unsupported corpus persistence policy: {policy}")

    if corpus_type is None:
        for policy_key in CORPUS_POLICY_KEYS.values():
            _update_session_value_without_persistence(
                session_id,
                policy_key,
                policy,
            )
        _update_session_value_without_persistence(
            session_id,
            SessionKeys.CORPUS_PERSISTENCE_POLICY,
            policy,
        )
        return

    if corpus_type not in CORPUS_POLICY_KEYS:
        raise ValueError(f"Unsupported corpus type for persistence policy: {corpus_type}")

    for candidate_corpus_type, policy_key in CORPUS_POLICY_KEYS.items():
        if candidate_corpus_type == corpus_type:
            continue
        _update_session_value_without_persistence(
            session_id,
            policy_key,
            get_session_persistence_policy(
                session_id,
                corpus_type=candidate_corpus_type,
            ),
        )

    _update_session_value_without_persistence(
        session_id,
        CORPUS_POLICY_KEYS[corpus_type],
        policy,
    )

    target_policy = get_session_persistence_policy(session_id, corpus_type="target")
    reference_policy = get_session_persistence_policy(
        session_id,
        corpus_type="reference",
    )
    _update_session_value_without_persistence(
        session_id,
        SessionKeys.CORPUS_PERSISTENCE_POLICY,
        _get_legacy_session_policy_value(target_policy, reference_policy),
    )


def session_allows_persistence(session_id: str) -> bool:
    """Return whether the session is allowed to write durable state."""

    session_container = st.session_state.get(session_id, {})
    return _session_has_durable_persistence(session_container.get("session"))


def _get_session_backend():
    """Get the session backend using factory pattern."""
    # Import backend factory only when needed to avoid circular imports
    from webapp.utilities.storage.backend_factory import get_session_backend
    return get_session_backend()


class SessionPersistenceManager:
    """
    Manages automatic session persistence between Streamlit session state and SQLite.
    """

    def __init__(self):
        """Initialize the session persistence manager."""
        self._backend = None
        self._session_cache = {}
        self._last_sync = {}
        self._dirty_sessions = set()

    def mark_dirty(self, session_id: str) -> None:
        """Mark a session as modified without hashing the full payload."""

        self._dirty_sessions.add(session_id)

    def _log_slow_operation(
        self,
        operation: str,
        session_id: str,
        start_time: float,
        detail: str | None = None,
    ) -> None:
        """Log slow persistence operations for session-path profiling."""

        elapsed_ms = (perf_counter() - start_time) * 1000
        if elapsed_ms >= SLOW_PERSISTENCE_OPERATION_MS:
            detail_suffix = f" {detail}" if detail else ""
            logger.warning(
                "Slow session persistence op={} session={}{} duration_ms={:.2f}",
                operation,
                session_id,
                detail_suffix,
                elapsed_ms,
            )

    @property
    def backend(self):
        """Get the session backend, initializing if needed."""
        if self._backend is None:
            try:
                self._backend = _get_session_backend()
            except Exception as e:
                logger.error(f"Failed to initialize session backend: {e}")
                return None
        return self._backend

    def get_user_id(self) -> str:
        """
        Get user ID from Streamlit context or generate anonymous ID.

        Returns
        -------
        str
            User identifier
        """
        try:
            # Try to get user email from Streamlit
            if hasattr(st, 'user') and st.user and hasattr(st.user, 'email'):
                return st.user.email

            # Fallback to session-based anonymous ID
            if 'anonymous_user_id' not in st.session_state:
                # Generate consistent anonymous ID based on session
                session_info = str(st.session_state.get('session_id', 'anonymous'))
                st.session_state['anonymous_user_id'] = hashlib.md5(
                    session_info.encode()
                ).hexdigest()[:16]

            return f"anon_{st.session_state['anonymous_user_id']}"

        except Exception:
            return "anonymous_user"

    def load_session(self, session_id: str) -> bool:
        """
        Load session data from SQLite into Streamlit session state.

        Parameters
        ----------
        session_id : str
            Session identifier

        Returns
        -------
        bool
            True if session was loaded successfully
        """
        if self.backend is None:
            return False

        try:
            # Check if already loaded and current
            if (session_id in self._session_cache and
                    session_id in st.session_state and
                    self._is_session_current(session_id)):
                return True

            # Load from SQLite
            session_data = self.backend.load_session(session_id)
            if session_data:
                restored_data = restore_persisted_session_data(session_data)
                # Restore to Streamlit session state
                st.session_state[session_id] = restored_data
                self._session_cache[session_id] = self._hash_session_data(restored_data)
                self._last_sync[session_id] = datetime.now(timezone.utc)
                self._dirty_sessions.discard(session_id)

                logger.info(f"Loaded session {session_id} from SQLite")
                return True

        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")

        return False

    def save_session(self, session_id: str, force: bool = False) -> bool:
        """
        Save session data from Streamlit session state to SQLite.

        Parameters
        ----------
        session_id : str
            Session identifier
        force : bool
            Force save even if data hasn't changed

        Returns
        -------
        bool
            True if session was saved successfully
        """
        if self.backend is None:
            return False

        operation_start = perf_counter()
        backend_ms = 0.0
        cache_hash_ms = 0.0
        try:
            # Check if session exists in Streamlit
            if session_id not in st.session_state:
                return False

            session_data = st.session_state[session_id]
            persistable_data = build_persistable_session_data(session_data)
            persistable_hash = self._hash_session_data(persistable_data)

            # Check if data has changed (unless forced)
            if (
                not force and
                session_id not in self._dirty_sessions and
                self._is_session_current(session_id)
            ):
                return True

            if (
                not force and
                persistable_hash == self._session_cache.get(session_id)
            ):
                self._last_sync[session_id] = datetime.now(timezone.utc)
                self._dirty_sessions.discard(session_id)
                return True

            # Get user ID
            user_id = self.get_user_id()

            # Save to SQLite
            backend_start = perf_counter()
            success = self.backend.save_session(session_id, persistable_data, user_id)
            backend_ms = (perf_counter() - backend_start) * 1000

            if success:
                self._session_cache[session_id] = persistable_hash
                self._last_sync[session_id] = datetime.now(timezone.utc)
                self._dirty_sessions.discard(session_id)
                self._log_slow_operation(
                    "save_session",
                    session_id,
                    operation_start,
                    detail=(
                        f"backend_ms={backend_ms:.2f} "
                        f"cache_hash_ms={cache_hash_ms:.2f}"
                    ),
                )
                return True

        except Exception as e:
            logger.error(f"Failed to save session {session_id}: {e}")

        self._log_slow_operation(
            "save_session",
            session_id,
            operation_start,
            detail=(
                f"backend_ms={backend_ms:.2f} "
                f"cache_hash_ms={cache_hash_ms:.2f}"
            ),
        )
        return False

    def auto_save_session(self, session_id: str) -> bool:
        """
        Automatically save session if it has been modified.

        Parameters
        ----------
        session_id : str
            Session identifier

        Returns
        -------
        bool
            True if session was saved or is current
        """
        if session_id not in st.session_state:
            return False

        operation_start = perf_counter()

        if not session_allows_persistence(session_id):
            self._log_slow_operation(
                "auto_save_session",
                session_id,
                operation_start,
                detail="branch=policy_skip",
            )
            return True

        now = datetime.now(timezone.utc)

        # Coalesce repeated persistence calls in the same short request burst.
        last_sync = self._last_sync.get(session_id)
        if (
            last_sync is not None and
            (now - last_sync).total_seconds() < MIN_PERSIST_INTERVAL_SECONDS
        ):
            self._log_slow_operation(
                "auto_save_session",
                session_id,
                operation_start,
                detail="branch=debounced",
            )
            return True

        # Only save if data has changed
        if session_id in self._dirty_sessions:
            result = self.save_session(session_id)
            self._log_slow_operation(
                "auto_save_session",
                session_id,
                operation_start,
                detail=f"branch=dirty_save result={result}",
            )
            return result

        if not self._is_session_current(session_id):
            result = self.save_session(session_id)
            self._log_slow_operation(
                "auto_save_session",
                session_id,
                operation_start,
                detail=f"branch=hash_save result={result}",
            )
            return result

        self._log_slow_operation(
            "auto_save_session",
            session_id,
            operation_start,
            detail="branch=current",
        )
        return True

    def delete_session(self, session_id: str) -> bool:
        """
        Delete session from both SQLite and Streamlit state.

        Parameters
        ----------
        session_id : str
            Session identifier

        Returns
        -------
        bool
            True if session was deleted successfully
        """
        success = True

        try:
            # Remove from SQLite
            if self.backend:
                self.backend.delete_session(session_id)

            # Remove from Streamlit state
            if session_id in st.session_state:
                del st.session_state[session_id]

            # Clean up cache
            self._session_cache.pop(session_id, None)
            self._last_sync.pop(session_id, None)
            self._dirty_sessions.discard(session_id)

            logger.info(f"Deleted session {session_id}")

        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            success = False

        return success

    def _hash_session_data(self, data: Dict[str, Any]) -> str:
        """
        Generate hash of session data for change detection.

        Parameters
        ----------
        data : Dict[str, Any]
            Session data

        Returns
        -------
        str
            Hash of the data
        """
        try:
            data = build_persistable_session_data(data)
            # Convert data to JSON string (sorted for consistency)
            json_str = json.dumps(data, sort_keys=True, default=str)
            return hashlib.md5(json_str.encode()).hexdigest()
        except Exception:
            # Fallback to timestamp if hashing fails
            return str(datetime.now(timezone.utc).timestamp())

    def _is_session_current(self, session_id: str) -> bool:
        """
        Check if session data in memory matches cached version.

        Parameters
        ----------
        session_id : str
            Session identifier

        Returns
        -------
        bool
            True if session is current (no changes)
        """
        if session_id not in st.session_state:
            return False

        if session_id not in self._session_cache:
            return False

        operation_start = perf_counter()
        current_hash = self._hash_session_data(st.session_state[session_id])
        is_current = current_hash == self._session_cache[session_id]
        self._log_slow_operation("is_session_current", session_id, operation_start)
        return is_current


# Global persistence manager instance
_persistence_manager = None


def get_persistence_manager() -> SessionPersistenceManager:
    """Get the global session persistence manager."""
    global _persistence_manager
    if _persistence_manager is None:
        _persistence_manager = SessionPersistenceManager()
    return _persistence_manager


def mark_session_dirty(session_id: str) -> None:
    """Record that a session was mutated and needs persistence."""

    get_persistence_manager().mark_dirty(session_id)


# Public API functions

def load_persistent_session(session_id: str) -> bool:
    """
    Load session from persistent storage.

    Parameters
    ----------
    session_id : str
        Session identifier

    Returns
    -------
    bool
        True if loaded successfully
    """
    return get_persistence_manager().load_session(session_id)


def save_persistent_session(session_id: str, force: bool = False) -> bool:
    """
    Save session to persistent storage.

    Parameters
    ----------
    session_id : str
        Session identifier
    force : bool
        Force save even if unchanged

    Returns
    -------
    bool
        True if saved successfully
    """
    return get_persistence_manager().save_session(session_id, force)


def auto_persist_session(session_id: str) -> bool:
    """
    Automatically persist session if changed.

    Parameters
    ----------
    session_id : str
        Session identifier

    Returns
    -------
    bool
        True if persisted or current
    """
    return get_persistence_manager().auto_save_session(session_id)


def delete_persistent_session(session_id: str) -> bool:
    """
    Delete session from persistent storage.

    Parameters
    ----------
    session_id : str
        Session identifier

    Returns
    -------
    bool
        True if deleted successfully
    """
    return get_persistence_manager().delete_session(session_id)
