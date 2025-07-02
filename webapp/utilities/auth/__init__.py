"""
User authorization utilities for controlling access to administrative features.

This package provides role-based access control for the corpus analysis application,
including user management, role assignment, and permission checking.
"""

from webapp.utilities.auth.user_authorization import (
    is_user_authorized,
    require_authorization,
    get_user_permissions,
    add_authorized_user,
    remove_authorized_user,
    update_user_role,
    get_user_role,
    list_authorized_users,
    initialize_authorization_db,
    is_authorization_enabled,
    set_current_user,
    get_current_user
)

__all__ = [
    'is_user_authorized',
    'require_authorization',
    'get_user_permissions',
    'add_authorized_user',
    'remove_authorized_user',
    'update_user_role',
    'get_user_role',
    'list_authorized_users',
    'initialize_authorization_db',
    'is_authorization_enabled',
    'set_current_user',
    'get_current_user'
]
