"""
Enhanced session state validation with detailed health checks and auto-repair.

This module provides comprehensive session validation with automatic repair
capabilities for common session state corruption issues.
"""

from typing import Dict, List, Any, Tuple
import streamlit as st
from datetime import datetime, timezone

from webapp.utilities.configuration.logging_config import get_logger
from webapp.utilities.state import SessionKeys, CorpusKeys, TargetKeys
# Import safe_session_get for consistent session access
from webapp.utilities.session.session_core import safe_session_get

logger = get_logger()


def safe_clear_session_state(user_session_id: str) -> bool:
    """
    Safely clear session state with error handling.

    Parameters
    ----------
    user_session_id : str
        The user session ID to clear

    Returns
    -------
    bool
        True if cleared successfully, False otherwise
    """
    try:
        if user_session_id in st.session_state:
            del st.session_state[user_session_id]
            return True
        return True  # Already cleared
    except Exception as e:
        logger.error(f"Failed to clear session state for {user_session_id}: {e}")
        return False


class SessionHealthChecker:
    """
    Comprehensive session health checker with auto-repair capabilities.
    """

    def __init__(self):
        """Initialize the health checker."""
        self.last_check_time = None
        self.repair_attempts = {}
        self.max_repair_attempts = 3

    def perform_health_check(self, user_session_id: str) -> Dict[str, Any]:
        """
        Perform comprehensive health check of session state.

        Parameters
        ----------
        user_session_id : str
            The user session ID to check

        Returns
        -------
        Dict[str, Any]
            Health check report with status and recommendations
        """
        report = {
            'overall_health': 'healthy',
            'issues': [],
            'repairs_applied': [],
            'warnings': [],
            'metadata': {
                'check_time': datetime.now(timezone.utc).isoformat(),
                'session_id': user_session_id
            }
        }

        try:
            # Check if session exists
            if user_session_id not in st.session_state:
                report['overall_health'] = 'critical'
                report['issues'].append('Session does not exist')
                return report

            session = st.session_state[user_session_id]

            # Check session structure integrity
            structure_issues = self._check_session_structure(session)
            report['issues'].extend(structure_issues)

            # Check corpus data consistency
            corpus_issues = self._check_corpus_consistency(session)
            report['issues'].extend(corpus_issues)

            # Check memory usage patterns
            memory_warnings = self._check_memory_patterns(session)
            report['warnings'].extend(memory_warnings)

            # Check for orphaned keys
            orphaned_keys = self._check_orphaned_keys(session)
            report['warnings'].extend(orphaned_keys)

            # Auto-repair minor issues
            repairs = self._attempt_auto_repair(user_session_id, session, report['issues'])
            report['repairs_applied'].extend(repairs)

            # Update overall health based on remaining issues
            critical_issues = [
                issue for issue in report['issues']
                if issue.get('severity') == 'critical'
            ]
            if len(critical_issues) > 0:
                report['overall_health'] = 'critical'
            elif len(report['issues']) > 0:
                report['overall_health'] = 'degraded'
            elif len(report['warnings']) > 0:
                report['overall_health'] = 'warning'

            self.last_check_time = datetime.now(timezone.utc)

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            report['overall_health'] = 'error'
            report['issues'].append({
                'type': 'health_check_error',
                'message': f"Health check failed: {str(e)}",
                'severity': 'critical'
            })

        return report

    def _check_session_structure(self, session: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check basic session structure integrity."""
        issues = []

        required_keys = [
            SessionKeys.HAS_TARGET,
            SessionKeys.HAS_REF,
            SessionKeys.HAS_META
        ]

        for key in required_keys:
            if key not in session:
                issues.append({
                    'type': 'missing_required_key',
                    'key': key,
                    'message': f"Required session key '{key}' is missing",
                    'severity': 'high',
                    'repairable': True
                })
            elif not isinstance(session[key], list) or len(session[key]) == 0:
                issues.append({
                    'type': 'invalid_key_format',
                    'key': key,
                    'message': f"Session key '{key}' has invalid format",
                    'severity': 'medium',
                    'repairable': True
                })

        return issues

    def _check_corpus_consistency(self, session: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check corpus data consistency."""
        issues = []

        # Check target corpus consistency
        has_target = safe_session_get(session, SessionKeys.HAS_TARGET, False)
        target_data = session.get(CorpusKeys.TARGET, {})

        if has_target and not target_data:
            issues.append({
                'type': 'corpus_data_mismatch',
                'message': "Session indicates target corpus exists but no data found",
                'severity': 'high',
                'repairable': False
            })

        # Check reference corpus consistency
        has_ref = safe_session_get(session, SessionKeys.HAS_REF, False)
        ref_data = session.get(CorpusKeys.REFERENCE, {})

        if has_ref and not ref_data:
            issues.append({
                'type': 'corpus_data_mismatch',
                'message': "Session indicates reference corpus exists but no data found",
                'severity': 'medium',
                'repairable': False
            })

        # Check for missing essential target data
        if has_target and target_data:
            essential_keys = [TargetKeys.TOKENS, TargetKeys.TAGS]
            for key in essential_keys:
                if key not in target_data:
                    issues.append({
                        'type': 'missing_corpus_component',
                        'key': key,
                        'message': f"Essential corpus component '{key}' is missing",
                        'severity': 'high',
                        'repairable': False
                    })

        return issues

    def _check_memory_patterns(self, session: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for memory usage patterns that might indicate issues."""
        warnings = []

        # Check for large objects that might cause memory issues
        large_objects = []
        for key, value in session.items():
            try:
                # Estimate size for different types
                if hasattr(value, '__len__'):
                    if len(value) > 10000:  # Arbitrary threshold
                        large_objects.append((key, len(value)))
                elif hasattr(value, 'shape'):  # DataFrame-like objects
                    if value.shape[0] > 100000:  # Large DataFrame
                        large_objects.append((key, f"DataFrame {value.shape}"))
            except Exception:
                continue

        if large_objects:
            warnings.append({
                'type': 'large_objects_detected',
                'objects': large_objects,
                'message': (
                    f"Found {len(large_objects)} potentially large objects in session"
                ),
                'recommendation': "Consider data cleanup or pagination"
            })

        return warnings

    def _check_orphaned_keys(self, session: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for orphaned or unused keys."""
        warnings = []

        # Define known key patterns
        known_patterns = [
            'has_', 'target_', 'reference_', 'metadata_',
            'boxplot_', 'scatterplot_', 'pca_', 'keyness_',
            'pandasai', 'plotbot'
        ]

        orphaned_keys = []
        for key in session.keys():
            if not any(key.startswith(pattern) for pattern in known_patterns):
                if not key.startswith('_'):  # System keys often start with _
                    orphaned_keys.append(key)

        if orphaned_keys:
            warnings.append({
                'type': 'orphaned_keys_detected',
                'keys': orphaned_keys,
                'message': f"Found {len(orphaned_keys)} potentially orphaned keys",
                'recommendation': "Review and clean up unused session keys"
            })

        return warnings

    def _attempt_auto_repair(self, user_session_id: str, session: Dict[str, Any],
                             issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Attempt to auto-repair minor issues."""
        repairs = []

        for issue in issues:
            if not issue.get('repairable', False):
                continue

            repair_key = f"{user_session_id}_{issue['type']}_{issue.get('key', 'unknown')}"
            if self.repair_attempts.get(repair_key, 0) >= self.max_repair_attempts:
                continue

            try:
                if issue['type'] == 'missing_required_key':
                    key = issue['key']
                    required_keys = [
                        SessionKeys.HAS_TARGET, SessionKeys.HAS_REF, SessionKeys.HAS_META
                    ]
                    default_value = [False] if key in required_keys else []
                    session[key] = default_value
                    repairs.append({
                        'type': 'missing_key_repaired',
                        'key': key,
                        'message': f"Added missing key '{key}' with default value"
                    })

                elif issue['type'] == 'invalid_key_format':
                    key = issue['key']
                    required_keys = [
                        SessionKeys.HAS_TARGET, SessionKeys.HAS_REF, SessionKeys.HAS_META
                    ]
                    if key in required_keys:
                        session[key] = [False]
                        repairs.append({
                            'type': 'format_repaired',
                            'key': key,
                            'message': f"Repaired format for key '{key}'"
                        })

                # Track repair attempts
                repair_count = self.repair_attempts.get(repair_key, 0) + 1
                self.repair_attempts[repair_key] = repair_count

            except Exception as e:
                logger.warning(f"Auto-repair failed for {issue['type']}: {e}")

        return repairs


class SessionStateValidator:
    """
    Enhanced session state validator with comprehensive checks.
    """

    def __init__(self):
        """Initialize the validator."""
        self.health_checker = SessionHealthChecker()

    def validate_session_with_report(
        self, user_session_id: str
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate session and return detailed report.

        Parameters
        ----------
        user_session_id : str
            The user session ID to validate

        Returns
        -------
        Tuple[bool, Dict[str, Any]]
            (is_valid, health_report)
        """
        health_report = self.health_checker.perform_health_check(user_session_id)

        # Determine if session is valid based on health report
        is_valid = health_report['overall_health'] in ['healthy', 'warning']

        return is_valid, health_report

    def validate_and_repair_session(self, user_session_id: str) -> bool:
        """
        Validate session and attempt repairs if needed.

        Parameters
        ----------
        user_session_id : str
            The user session ID to validate and repair

        Returns
        -------
        bool
            True if session is valid after any repairs
        """
        is_valid, health_report = self.validate_session_with_report(user_session_id)

        if not is_valid:
            logger.warning(
                f"Session validation failed for {user_session_id}: {health_report}"
            )

            # If critical issues exist, clear the session
            if health_report['overall_health'] == 'critical':
                safe_clear_session_state(user_session_id)
                return False

        return is_valid

    def get_session_diagnostics(self, user_session_id: str) -> Dict[str, Any]:
        """
        Get comprehensive session diagnostics for debugging.

        Parameters
        ----------
        user_session_id : str
            The user session ID to diagnose

        Returns
        -------
        Dict[str, Any]
            Comprehensive diagnostics report
        """
        _, health_report = self.validate_session_with_report(user_session_id)

        # Add additional diagnostic information
        diagnostics = {
            'health_report': health_report,
            'session_keys': [],
            'memory_estimate': 0,
            'data_types': {}
        }

        if user_session_id in st.session_state:
            session = st.session_state[user_session_id]
            diagnostics['session_keys'] = list(session.keys())

            # Analyze data types
            for key, value in session.items():
                value_type = type(value).__name__
                if value_type in diagnostics['data_types']:
                    diagnostics['data_types'][value_type] += 1
                else:
                    diagnostics['data_types'][value_type] = 1

        return diagnostics


# Global instances
enhanced_health_checker = SessionHealthChecker()
enhanced_validator = SessionStateValidator()
