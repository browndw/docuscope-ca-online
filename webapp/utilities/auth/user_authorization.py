"""
User Authorization System

This module provides role-based access control for the corpus analysis application.
It manages user permissions through the Postgres control plane and integrates with
Google OAuth authentication.

Features:
- Role-based access control (admin, instructor, user)
- Permission management
- Session-based authorization caching
- Audit trail for access attempts
- Automatic control-plane initialization
"""

from datetime import datetime, timezone
from functools import wraps
from typing import List, Dict, Any, Optional, Callable

import streamlit as st

from webapp.config.unified import get_config
from webapp.persistence.authorization_service import AuthorizationService
from webapp.utilities.configuration.logging_config import get_logger

logger = get_logger()

# Default roles and permissions
DEFAULT_ROLES = {
    'admin': {
        'description': 'Full system access including health monitoring and user management',
        'permissions': [
            'health_monitor', 'user_management', 'system_config',
            'instructor_dashboard', 'all_analytics'
        ]
    },
    'instructor': {
        'description': 'Teaching features and student monitoring',
        'permissions': [
            'instructor_dashboard', 'student_analytics', 'assignment_management'
        ]
    },
    'user': {
        'description': 'Standard corpus analysis features',
        'permissions': ['corpus_analysis', 'basic_analytics']
    }
}

# Cache for authorization checks during session
_auth_cache = {}
_authorization_service: AuthorizationService | None = None


def _get_authorization_service() -> AuthorizationService:
    """Return the shared Postgres-backed authorization service."""

    global _authorization_service
    if _authorization_service is None:
        _authorization_service = AuthorizationService()
    return _authorization_service


def is_authorization_enabled() -> bool:
    """Check if user authorization is enabled based on configuration."""
    # Authorization is disabled in desktop mode and test mode.
    desktop_mode = get_config('desktop_mode', 'global')
    test_mode = get_config('test_mode', 'global', False)
    if desktop_mode or test_mode:
        return False

    # Check if explicitly enabled in config
    return get_config('enable_user_authorization', 'authorization', True)


def initialize_authorization_db() -> None:
    """Initialize the authorization tables with required defaults."""
    if not is_authorization_enabled():
        return

    try:
        default_admin = get_config('default_admin_email', 'authorization', None)
        _get_authorization_service().initialize_defaults(DEFAULT_ROLES, default_admin)

    except Exception as e:
        logger.error(f"Failed to initialize authorization database: {e}")
        raise


def _normalize_email(email: str) -> str:
    """Normalize email address to lowercase for consistent storage."""
    return email.lower().strip() if email else ""


def _get_session_cache_key(email: str) -> str:
    """Get cache key for session-based authorization caching."""
    return f"auth_cache_{_normalize_email(email)}"


def get_user_role(email: str) -> Optional[str]:
    """Get the role of a user by email."""
    if not is_authorization_enabled():
        return 'admin'  # In desktop mode, everyone is admin

    if not email:
        return None

    email = _normalize_email(email)

    # Check session cache first
    cache_key = _get_session_cache_key(email)
    if cache_key in st.session_state:
        cached_data = st.session_state[cache_key]
        if cached_data and cached_data.get('role'):
            return cached_data['role']

    try:
        role = _get_authorization_service().get_user_role(email)

        # Cache the result
        st.session_state[cache_key] = {
            'role': role,
            'cached_at': datetime.now(timezone.utc)
        }

        return role

    except Exception as e:
        logger.error(f"Failed to get user role for {email}: {e}")
        return None


def get_user_permissions(email: str) -> List[str]:
    """Get the permissions for a user."""
    if not is_authorization_enabled():
        return list(DEFAULT_ROLES['admin']['permissions'])

    role = get_user_role(email)
    if not role:
        return []

    try:
        return _get_authorization_service().get_user_permissions(email)

    except Exception as e:
        logger.error(f"Failed to get permissions for {email}: {e}")
        return []


def is_user_authorized(email: str, required_role: str = 'user',
                       required_permission: str = None) -> bool:
    """
    Check if a user is authorized for a given role or permission.

    Parameters
    ----------
    email : str
        User email address
    required_role : str
        Minimum required role ('user', 'instructor', 'admin')
    required_permission : str, optional
        Specific permission required

    Returns
    -------
    bool
        True if user is authorized, False otherwise
    """
    if not is_authorization_enabled():
        return True  # In desktop mode, everyone is authorized

    if not email:
        return False

    email = _normalize_email(email)

    # Update last accessed time
    _update_last_accessed(email)

    # Check specific permission if provided
    if required_permission:
        permissions = get_user_permissions(email)
        authorized = required_permission in permissions
    else:
        # Check role hierarchy: admin > instructor > user
        user_role = get_user_role(email)
        if not user_role:
            authorized = False
        else:
            role_hierarchy = {'user': 1, 'instructor': 2, 'admin': 3}
            user_level = role_hierarchy.get(user_role, 0)
            required_level = role_hierarchy.get(required_role, 0)
            authorized = user_level >= required_level

    # Log access attempt
    _log_access_attempt(email, required_role, required_permission, authorized)

    return authorized


def add_authorized_user(email: str, role: str = 'user',
                        added_by: str = None,
                        custom_permissions: List[str] = None) -> bool:
    """Add a new authorized user."""
    if not is_authorization_enabled():
        return True

    if not email or role not in DEFAULT_ROLES:
        return False

    email = _normalize_email(email)

    try:
        added = _get_authorization_service().add_authorized_user(
            email,
            role,
            added_by,
            custom_permissions,
        )

        # Clear cache
        cache_key = _get_session_cache_key(email)
        if cache_key in st.session_state:
            del st.session_state[cache_key]
        return added

    except Exception as e:
        logger.error(f"Failed to add authorized user {email}: {e}")
        return False


def remove_authorized_user(email: str) -> bool:
    """Remove a user's authorization."""
    if not is_authorization_enabled():
        return True

    if not email:
        return False

    email = _normalize_email(email)

    try:
        removed = _get_authorization_service().remove_authorized_user(email)

        # Clear cache
        cache_key = _get_session_cache_key(email)
        if cache_key in st.session_state:
            del st.session_state[cache_key]
        return removed

    except Exception as e:
        logger.error(f"Failed to remove authorized user {email}: {e}")
        return False


def update_user_role(email: str, new_role: str, updated_by: str = None) -> bool:
    """Update a user's role."""
    if not is_authorization_enabled():
        return True

    if not email or new_role not in DEFAULT_ROLES:
        return False

    email = _normalize_email(email)

    try:
        updated = _get_authorization_service().update_user_role(
            email,
            new_role,
            updated_by,
        )

        # Clear cache
        cache_key = _get_session_cache_key(email)
        if cache_key in st.session_state:
            del st.session_state[cache_key]

        if updated:

            logger.info(f"Updated user {email} role to: {new_role}")
        return updated

    except Exception as e:
        logger.error(f"Failed to update user role for {email}: {e}")
        return False


def list_authorized_users() -> List[Dict[str, Any]]:
    """List all authorized users."""
    if not is_authorization_enabled():
        return []

    try:
        return _get_authorization_service().list_authorized_users()

    except Exception as e:
        logger.error(f"Failed to list authorized users: {e}")
        return []


def require_authorization(
        required_role: str = 'admin',
        required_permission: str = None
) -> Callable:
    """
    Decorator to require authorization for a function or page.

    Usage:
    @require_authorization('admin')
    def admin_function():
        pass

    @require_authorization(required_permission='health_monitor')
    def health_monitor():
        pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Skip authorization in desktop mode
            if not is_authorization_enabled():
                return func(*args, **kwargs)

            # Check if user is logged in
            if not (hasattr(st, "user") and getattr(st.user, "is_logged_in", False)):
                st.error(
                    "Please log in to access this feature.",
                    icon=":material/login:"
                )
                st.stop()

            user_email = st.user.email

            # Check authorization
            if not is_user_authorized(user_email, required_role, required_permission):
                st.error(
                    f"Access denied. {required_role.title()} privileges required.",
                    icon=":material/block:"
                )

                # Show helpful message based on current role
                current_role = get_user_role(user_email)
                if current_role:
                    st.info(
                        f"Your current role is: **{current_role}**. "
                        "Contact an administrator to request elevated privileges."
                    )
                else:
                    st.info(
                        "You are not authorized to use this application. "
                        "Contact an administrator to request access."
                    )
                st.switch_page("index.py")  # <-- adjust path if needed
                st.stop()

            return func(*args, **kwargs)

        return wrapper
    return decorator


def set_current_user(email: str) -> None:
    """Set the current user email in session state for audit tracking."""
    if email:
        st.session_state['user_email'] = _normalize_email(email)


def get_current_user() -> str:
    """Get the current user email from session state."""
    return st.session_state.get('user_email', 'unknown')


def _update_last_accessed(email: str) -> None:
    """Update the last accessed timestamp for a user."""
    if not is_authorization_enabled():
        return

    try:
        _get_authorization_service().update_last_accessed(email)

    except Exception as e:
        logger.debug(f"Failed to update last accessed for {email}: {e}")


def _log_access_attempt(
        email: str,
        required_role: str,
        required_permission: str,
        granted: bool
) -> None:
    """Log an access attempt for auditing."""
    if not is_authorization_enabled():
        return

    audit_enabled = get_config('audit_access_attempts', 'authorization', True)
    if not audit_enabled:
        return

    try:
        page = getattr(
            st.runtime.scriptrunner_utils.script_run_context.get_script_run_ctx(),
            'page_script_hash',
            'unknown',
        )
        _get_authorization_service().log_access_attempt(
            email,
            page,
            required_role,
            required_permission,
            granted,
        )

    except Exception as e:
        logger.debug(f"Failed to log access attempt: {e}")


# Initialize the database when module is imported
try:
    initialize_authorization_db()
except Exception as e:
    logger.warning(f"Failed to initialize authorization database: {e}")
