"""
User Management and System Administration

This page provides comprehensive user authorization management and system
administration controls for enterprise deployments.

Access is restricted to users with admin role.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timezone

from webapp.menu import menu
from webapp.config.unified import get_config
from webapp.config.config_utils import get_runtime_config
from webapp.utilities.configuration.logging_config import get_logger
from webapp.utilities.auth import (
    require_authorization,
    add_authorized_user,
    remove_authorized_user,
    update_user_role,
    list_authorized_users,
    get_user_role,
    is_authorization_enabled
)
from webapp.utilities.storage.backend_factory import get_session_backend

logger = get_logger()


st.set_page_config(
    page_title="User Management & Administration",
    page_icon=":material/admin_panel_settings:",
    layout="wide"
)


def get_quota_statistics():
    """Get system-wide quota usage statistics across ALL users."""
    # Return empty stats in desktop mode
    desktop_mode = get_config('desktop_mode', 'global', True)
    if desktop_mode:
        return {
            'total_queries_24h': 0,
            'total_users_with_queries': 0,
            'avg_queries_per_user': 0,
            'community_api_load': 0,
            'error': None
        }

    try:
        backend = get_session_backend()

        # Get total usage across ALL users (not just authorized ones)
        # This gives us the true picture of community API key usage
        total_queries = 0
        unique_users = set()

        # Query all shards to get complete usage data
        if (
            hasattr(backend, 'shard_manager') and
            hasattr(backend.shard_manager, 'shard_count')
        ):
            # Sharded backend - check all shards
            shard_count = backend.shard_manager.shard_count
            for shard_id in range(shard_count):
                try:
                    # Use the shard manager to get connections
                    session_pool = backend.shard_manager.session_pools[shard_id]
                    with session_pool.get_connection() as conn:
                        cursor = conn.cursor()

                        # Get total queries in last 24 hours from this shard
                        cursor.execute("""
                            SELECT COUNT(*), COUNT(DISTINCT user_id)
                            FROM user_queries
                            WHERE query_timestamp >= datetime('now', '-24 hours')
                        """)

                        result = cursor.fetchone()
                        if result and result[0]:
                            shard_queries = result[0]

                            total_queries += shard_queries

                            # Get unique user IDs from this shard
                            cursor.execute("""
                                SELECT DISTINCT user_id
                                FROM user_queries
                                WHERE query_timestamp >= datetime('now', '-24 hours')
                            """)

                            for row in cursor.fetchall():
                                unique_users.add(row[0])

                except Exception as e:
                    logger.error(f"Error reading shard {shard_id}: {e}")
                    continue
        else:
            # Single database backend - use the session manager methods
            try:
                # For non-sharded backends, we need to access the database differently
                # Let's use the backend's methods if available
                logger.warning("Non-sharded backend detected - using fallback method")
                # For now, return zero stats as this is primarily for enterprise mode
                pass

            except Exception as e:
                logger.error(f"Error reading single database: {e}")

        # Calculate statistics
        total_users_with_queries = len(unique_users)
        avg_queries_per_user = (
            round(total_queries / total_users_with_queries, 2)
            if total_users_with_queries > 0 else 0
        )

        return {
            'total_queries_24h': total_queries,
            'total_users_with_queries': total_users_with_queries,
            'avg_queries_per_user': avg_queries_per_user,
            'community_api_load': total_queries,  # Same as total for clarity
            'error': None
        }

    except Exception as e:
        logger.error(f"Failed to get system-wide quota statistics: {e}")
        return {
            'total_queries_24h': 0,
            'total_users_with_queries': 0,
            'avg_queries_per_user': 0,
            'community_api_load': 0,
            'error': str(e)
        }


def render_user_management_tab():
    """Render the user management interface."""
    st.subheader(":material/manage_accounts: User Authorization Management")

    if not is_authorization_enabled():
        st.warning(
            "Authorization is disabled (likely running in desktop mode). "
            "User management is only available in enterprise mode."
        )
        return

    st.markdown(
        "Manage user access and roles for the DocuScope Corpus Analysis system. "
        "Only users with admin role can modify user permissions."
    )

    # Add new user section
    with st.expander(
        label="Add New User",
        icon=":material/person_add:",
        expanded=False
    ):
        col1, col2 = st.columns(2, gap="large")

        with col1:
            new_email = st.text_input(
                "Email Address",
                placeholder="user@example.com",
                help="Enter the user's email address"
            )

        with col2:
            new_role = st.selectbox(
                "Role",
                options=["user", "instructor", "admin"],
                help="Select the user's role and permissions"
            )

        if st.button("Add User", type="primary"):
            if new_email:
                if add_authorized_user(new_email, new_role):
                    st.success(f"Added {new_email} with role '{new_role}'")
                    st.rerun()
                else:
                    st.error("Failed to add user. User may already exist.")
            else:
                st.error("Please enter an email address")

    # Current users table
    st.markdown("#### Current Authorized Users")

    users = list_authorized_users()

    if not users:
        st.info("No authorized users found.")
        return

    # Convert to DataFrame for better display
    df_data = []
    for user in users:
        df_data.append({
            "Email": user['email'],
            "Role": user['role'],
            "Added By": user.get('added_by', 'Unknown'),
            "Added At": user.get('added_at', 'Unknown'),
            "Last Accessed": user.get('last_accessed', 'Never'),
            "Active": "✅" if user.get('active', True) else "❌"
        })

    df = pd.DataFrame(df_data)

    # Display table
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Email": st.column_config.TextColumn("Email", width="medium"),
            "Role": st.column_config.TextColumn("Role", width="small"),
            "Added By": st.column_config.TextColumn("Added By", width="medium"),
            "Added At": st.column_config.TextColumn("Added At", width="medium"),
            "Last Accessed": st.column_config.TextColumn("Last Accessed", width="medium"),
            "Active": st.column_config.TextColumn("Status", width="small")
        }
    )

    # User management actions
    st.markdown("---")
    st.markdown("#### User Actions")
    st.markdown("Edit user roles or remove users from the system.")

    with st.expander(
        label="Update User Role",
        icon=":material/person_edit:",
        expanded=False
    ):
        # Columns for layout
        col1, col2 = st.columns(2, gap="large")

        with col1:
            user_emails = [user['email'] for user in users if user.get('active', True)]

            if user_emails:
                selected_user = st.selectbox(
                    "Select User",
                    options=user_emails,
                    index=None,
                    placeholder="Select a user to update",
                    key="role_update_user"
                )

                current_role = get_user_role(selected_user) if selected_user else None

        with col2:
            new_role_options = ["user", "instructor", "admin"]
            if current_role in new_role_options:
                current_index = new_role_options.index(current_role)
            else:
                current_index = 0

            updated_role = st.selectbox(
                "New Role",
                options=new_role_options,
                index=current_index,
                key="new_role_select"
            )

        if st.button("Update Role", type="primary", key="update_role_btn"):
            if selected_user and updated_role != current_role:
                admin_user = st.session_state.get('user_email', 'admin')
                if update_user_role(selected_user, updated_role, admin_user):
                    st.success(f"Updated {selected_user} to role '{updated_role}'")
                    st.rerun()
                else:
                    st.error("Failed to update user role")
            elif updated_role == current_role:
                st.info("Role is already set to the selected value")
        else:
            st.info("No active users available for role updates")

    with st.expander(
        label="Remove User",
        icon=":material/person_remove:",
        expanded=False
    ):
        if user_emails:
            user_to_remove = st.selectbox(
                "Select User to Remove",
                options=user_emails,
                index=None,
                placeholder="Select a user to remove",
                key="remove_user_select"
            )

            if st.button("Remove User", type="primary", key="remove_user_btn"):
                if user_to_remove:
                    if remove_authorized_user(user_to_remove):
                        st.success(f"Removed access for {user_to_remove}")
                        st.rerun()
                    else:
                        st.error("Failed to remove user")
        else:
            st.info("No active users available for removal")


def render_system_config_tab():
    """Render system configuration controls."""
    st.subheader(":material/build_circle: Runtime Configuration")

    desktop_mode = get_config('desktop_mode', 'global', True)

    if desktop_mode:
        st.warning(
            "Running in desktop mode. Runtime configuration controls are disabled "
            "for lightweight operation."
        )
        return

    # AI Quota Usage Statistics
    st.markdown("---")
    st.markdown("#### :material/smart_toy: Community API Key Usage Analytics")

    quota_stats = get_quota_statistics()

    if quota_stats['error']:
        st.error(f"Failed to load quota statistics: {quota_stats['error']}")
    else:
        col_stat1, col_stat2, col_stat3 = st.columns(3)

        with col_stat1:
            st.metric(
                "Total AI Queries (24h)",
                quota_stats['total_queries_24h'],
                border=True,
                help="Total AI queries from all users (authorized and unauthorized) in the last 24 hours"  # noqa: E501
            )

        with col_stat2:
            st.metric(
                "Active Users (24h)",
                quota_stats['total_users_with_queries'],
                border=True,
                help="Number of unique users who made AI queries in the last 24 hours"
            )

        with col_stat3:
            st.metric(
                "Avg Queries/User",
                quota_stats['avg_queries_per_user'],
                border=True,
                help="Average number of AI queries per active user in the last 24 hours"
            )

        # System health assessment
        st.markdown("#### :material/health_metrics: AI Usage Health Assessment")

        # Get current quota limit safely for desktop mode
        try:
            current_quota_limit = get_runtime_config().get_config_value(
                'quota',
                get_config('quota', 'llm', 10),
                'llm'
            )
        except Exception as e:
            logger.warning(f"Failed to get runtime quota limit: {e}")
            current_quota_limit = get_config('quota', 'llm', 10)

        total_queries = quota_stats['total_queries_24h']
        avg_per_user = quota_stats['avg_queries_per_user']

        # Health indicators
        col_health1, col_health2 = st.columns(2, gap="large")

        with col_health1:
            st.markdown("**System Status**")
            if total_queries == 0:
                st.metric(
                    label="System Status",
                    value="🟢",
                    border=True,
                    help="No AI usage detected"
                )
            elif avg_per_user <= current_quota_limit * 0.5:
                st.metric(
                    label="System Status",
                    value="🟢",
                    border=True,
                    help="Light usage - API key healthy"
                )
            elif avg_per_user <= current_quota_limit * 0.8:
                st.metric(
                    label="System Status",
                    value="🟡",
                    border=True,
                    help="Moderate usage - monitor closely"
                )
            else:
                st.metric(
                    label="System Status",
                    value="🔴",
                    border=True,
                    help="Heavy usage - consider quota adjustments"
                )

        with col_health2:
            # API load indicator
            # Rough estimate if current rate continues
            st.markdown(
                body="**Daily Load**",
                help="Estimated total daily queries if current 24h rate continues")
            estimated_daily_load = total_queries * 24
            st.metric(
                label="Estimated Daily Queries",
                value=f"{estimated_daily_load:,}",
                border=True
            )

    st.divider()

    # Runtime configuration overrides
    st.markdown("#### :material/settings: Runtime Configuration Overrides")

    st.markdown("**Active Configuration Overrides**")

    try:
        overrides = get_runtime_config().get_all_overrides()
    except Exception as e:
        logger.warning(f"Failed to get runtime overrides: {e}")
        overrides = {}

    if overrides:
        for key, data in overrides.items():
            with st.container():
                st.markdown(f"**{key}**: `{data['value']}`")
                st.caption(f"Updated by {data['updated_by']} at {data['updated_at']}")
                if st.button(f"Clear {key}", key=f"clear_{key}"):
                    admin_user = st.session_state.get('user_email', 'admin')
                    get_runtime_config().clear_runtime_override(key, admin_user)
                    st.success(f"Cleared override for {key}")
                    st.rerun()
    else:
        st.info("No runtime overrides active")

    st.write(" ")
    st.markdown("##### :material/expand: **Community Key Quota Limit Configuration**")

    # Get current quota limit
    try:
        current_quota_limit = get_runtime_config().get_config_value(
            'quota',
            get_config('quota', 'llm', 10),
            'llm'
        )
    except Exception as e:
        logger.warning(f"Failed to get runtime quota limit: {e}")
        current_quota_limit = get_config('quota', 'llm', 10)
    toml_quota_default = get_config('quota', 'llm', 10)

    # Show current status
    st.markdown(f"**Current Limit**: {current_quota_limit} queries/24h")
    st.markdown(f"**TOML Default**: {toml_quota_default} queries/24h")

    # Show usage recommendation based on system-wide statistics
    if not quota_stats['error'] and quota_stats['avg_queries_per_user'] > 0:
        avg_usage = quota_stats['avg_queries_per_user']
        recommended_limit = max(
            int(avg_usage * 2),  # Allow 2x average usage as buffer
            toml_quota_default
        )

        if avg_usage > current_quota_limit * 0.8:
            st.warning(
                body=(
                    f"Average usage ({avg_usage}) is approaching the quota limit "
                    f"({current_quota_limit}). Consider increasing to {recommended_limit}."
                ),
                icon="🔴"
            )
        elif avg_usage > current_quota_limit * 0.6:
            st.info(
                body=(
                    f"Average usage ({avg_usage}) is moderate. "
                    f"Current limit ({current_quota_limit}) appears adequate."
                ),
                icon="🟡"
            )
        else:
            st.success(
                body=(
                    f"Average usage ({avg_usage}) is well below the limit. "
                    f"System is operating efficiently."
                ),
                icon="🟢"
            )

    # Quota limit input
    new_quota_limit = st.number_input(
        "Set AI Quota Limit (queries per 24h)",
        min_value=1,
        max_value=1000,
        value=current_quota_limit,
        step=5,
        help="Set the maximum number of AI queries per user per 24-hour period",
        key="quota_limit_input"
    )

    if new_quota_limit != current_quota_limit:
        if st.button("Update Quota Limit", type="primary", key="update_quota_limit"):
            admin_user = st.session_state.get('user_email', 'admin')
            get_runtime_config().set_runtime_override(
                'quota', 'llm', new_quota_limit, admin_user
            )
            st.success(f"Updated AI quota limit to {new_quota_limit} queries per 24h")
            st.rerun()

    # Reset quota limit to default
    if current_quota_limit != toml_quota_default:
        if st.button("Reset Quota to TOML Default", key="reset_quota_limit"):
            admin_user = st.session_state.get('user_email', 'admin')
            get_runtime_config().clear_runtime_override(
                'quota', 'llm', admin_user
            )
            st.success(f"Reset quota limit to TOML default ({toml_quota_default})")
            st.rerun()

    st.write(" ")
    st.markdown("##### :material/backup: **Firestore Research Data Collection**")

    # Get current state
    try:
        current_state = get_runtime_config().is_firestore_enabled()
    except Exception as e:
        logger.warning(f"Failed to get firestore state: {e}")
        current_state = get_config('cache_mode', 'cache', False)
    toml_default = get_config('cache_mode', 'cache', False)

    # Show current status
    if current_state:
        st.success(
            body=(
                "Firestore collection is currently enabled. "
                "This allows research data to be stored and analyzed."
            ),
            icon="🟢"
        )
    else:
        st.error(
            body=(
                "Firestore collection is currently disabled. "
                "This prevents research data from being stored."
            ),
            icon="🔴"
        )
    st.markdown(f"**TOML Default**: {toml_default}")

    # Toggle control
    new_state = st.toggle(
        "Enable Firestore Collection",
        value=current_state,
        help="Toggle research data collection without restart",
        key="firestore_toggle"
    )

    if new_state != current_state:
        admin_user = st.session_state.get('user_email', 'admin')
        get_runtime_config().toggle_firestore_collection(new_state, admin_user)
        st.success(f"Firestore collection {'enabled' if new_state else 'disabled'}")
        st.rerun()

    # Reset to default button
    if st.button("Reset to TOML Default", key="reset_firestore"):
        admin_user = st.session_state.get('user_email', 'admin')
        get_runtime_config().clear_firestore_override(admin_user)
        st.success("Reset to TOML default")
        st.rerun()


def render_audit_log_tab():
    """Render audit log viewing interface."""
    st.subheader(":material/policy: System Audit Logs")

    desktop_mode = get_config('desktop_mode', 'global', True)

    if desktop_mode:
        st.warning(
            "Running in desktop mode. Audit logs are not available "
            "in lightweight desktop operation."
        )
        return

    # Configuration changes audit log
    st.markdown("#### Configuration Changes")

    log_limit = st.number_input(
        "Number of entries",
        min_value=5,
        max_value=100,
        value=20,
        step=5,
        help="Number of recent audit log entries to display"
    )

    if st.button("Refresh Logs", key="refresh_audit"):
        st.rerun()

    try:
        audit_log = get_runtime_config().get_audit_log(limit=log_limit)
    except Exception as e:
        logger.warning(f"Failed to get audit log: {e}")
        audit_log = []

    if audit_log:
        # Convert to DataFrame for better display
        audit_df_data = []
        for entry in audit_log:
            change_type = "UPDATE" if entry['new_value'] != 'CLEARED' else "DELETE"
            audit_df_data.append({
                "Timestamp": entry['updated_at'],
                "Action": change_type,
                "Configuration Key": entry['config_key'],
                "Old Value": entry['old_value'],
                "New Value": entry['new_value'],
                "Updated By": entry['updated_by']
            })

        audit_df = pd.DataFrame(audit_df_data)

        st.dataframe(
            audit_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Timestamp": st.column_config.TextColumn("Timestamp", width="medium"),
                "Action": st.column_config.TextColumn("Action", width="small"),
                "Configuration Key": st.column_config.TextColumn(
                    "Config Key", width="medium"
                ),
                "Old Value": st.column_config.TextColumn("Old Value", width="small"),
                "New Value": st.column_config.TextColumn("New Value", width="small"),
                "Updated By": st.column_config.TextColumn("Updated By", width="medium")
            }
        )
    else:
        st.info("No configuration changes recorded")

    # User authorization audit log (future enhancement)
    st.markdown("#### User Authorization Changes")
    st.info("User authorization audit log will be implemented in a future update")


def render_system_info_tab():
    """Render system information and status."""
    st.subheader(":material/settings_applications: Configuration")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Application Mode**")
        desktop_mode = get_config('desktop_mode', 'global', True)
        mode_text = "Desktop" if desktop_mode else "Online"
        st.metric("Mode", mode_text, border=True)

        st.markdown("**Authorization Status**")
        auth_enabled = is_authorization_enabled()
        auth_text = "Enabled" if auth_enabled else "Disabled"
        st.metric("User Authorization", auth_text, border=True)

    with col2:
        st.markdown("**Session Backend**")
        try:
            backend = get_session_backend()
            backend_type = type(backend).__name__
            # Convert class name to readable format
            if backend_type == "InMemorySessionBackend":
                backend_display = "MEMORY"
            elif backend_type == "ShardedSQLiteSessionBackend":
                backend_display = "SQLITE"
            else:
                backend_display = backend_type.upper()
        except Exception:
            # Fallback to config value
            backend_display = get_config('backend', 'session', 'sqlite').upper()

        st.metric("Session Storage", backend_display, border=True)

        st.markdown("**Current User**")
        current_user = st.user.get('email', 'Guest')
        st.metric("Logged in as", current_user, border=True)

    with col3:
        st.markdown("**System Status**")
        st.metric("Status", "Running", border=True)

        st.markdown("**Last Updated**")
        st.metric("Timestamp", datetime.now(timezone.utc).strftime("%H:%M:%S"), border=True)

    # System-wide usage overview
    st.divider()
    st.markdown("#### :material/monitoring: System-Wide Usage Overview")

    desktop_mode = get_config('desktop_mode', 'global', True)
    if desktop_mode:
        st.info("System-wide usage analytics are only available in enterprise mode.")
    else:
        try:
            # Get quota statistics for system overview
            quota_stats = get_quota_statistics()

            # Get basic system statistics
            try:
                current_quota_limit = get_runtime_config().get_config_value(
                    'quota',
                    get_config('quota', 'llm', 10),
                    'llm'
                )
            except Exception as e:
                logger.warning(f"Failed to get runtime quota limit: {e}")
                current_quota_limit = get_config('quota', 'llm', 10)

            # Get authorized user count for context
            authorized_users = list_authorized_users()
            total_authorized = len([
                u for u in authorized_users if u.get('active', True)
            ]) if authorized_users else 0

            # Display system overview metrics
            col_sys1, col_sys2, col_sys3 = st.columns(3)

            with col_sys1:
                st.metric(
                    "Authorized Users",
                    total_authorized,
                    border=True,
                    help="Number of users with system authorization"
                )

            with col_sys2:
                st.metric(
                    "Current Quota Limit",
                    f"{current_quota_limit}/user/24h",
                    border=True,
                    help="Current AI query limit per user per 24 hours"
                )

            with col_sys3:
                # Calculate quota utilization
                if quota_stats and not quota_stats['error']:
                    avg_usage = quota_stats['avg_queries_per_user']
                    utilization = (
                        (avg_usage / current_quota_limit * 100)
                        if current_quota_limit > 0 else 0
                    )
                    st.metric(
                        "Avg Quota Utilization",
                        f"{utilization:.1f}%",
                        border=True,
                        help="Average quota usage across all active users"
                    )
                else:
                    st.metric("Avg Quota Utilization", "N/A")

            # Usage pattern insights
            st.divider()
            if (
                quota_stats and not quota_stats['error'] and
                quota_stats['total_queries_24h'] > 0
            ):
                st.markdown("#### :material/query_stats: Usage Pattern Insights")

                insights = []
                insights_icons = []
                total_queries = quota_stats['total_queries_24h']
                active_users = quota_stats['total_users_with_queries']
                avg_per_user = quota_stats['avg_queries_per_user']

                # Generate insights based on usage patterns
                if active_users > total_authorized:
                    insights.append(
                        f"**High Adoption**: {active_users} active users vs "
                        f"{total_authorized} authorized (includes unauthorized usage)"
                    )
                    insights_icons.append(":material/water_full:")
                elif active_users < total_authorized * 0.5:
                    insights.append(
                        f"**Low Adoption**: Only {active_users} of "
                        f"{total_authorized} authorized users are active"
                    )
                    insights_icons.append(":material/water_loss:")
                else:
                    insights.append(
                        f"**Normal Adoption**: {active_users} of "
                        f"{total_authorized} authorized users are active"
                    )
                    insights_icons.append(":material/water_medium:")
                if avg_per_user > current_quota_limit * 0.8:
                    insights.append(
                        f"**High Usage**: Average {avg_per_user} queries/user "
                        f"approaching limit of {current_quota_limit}"
                    )
                    insights_icons.append(":material/warning:")
                elif avg_per_user < current_quota_limit * 0.3:
                    insights.append(
                        f"**Conservative Usage**: Average {avg_per_user} "
                        f"queries/user well below limit"
                    )
                    insights_icons.append(":material/check_circle:")

                if total_queries > 100:
                    insights.append(
                        f"**High Volume**: {total_queries} total queries "
                        f"indicate heavy system usage"
                    )
                    insights_icons.append(":material/traffic_jam:")
                elif total_queries < 10:
                    insights.append(
                        f"**Low Volume**: {total_queries} total queries "
                        f"indicate light system usage"
                    )
                    insights_icons.append(":material/no_crash:")

                for insight in insights:
                    st.info(
                        body=f"- {insight}",
                        icon=insights_icons[insights.index(insight)]
                    )
            else:
                st.info("💡 No usage data available for pattern analysis")

        except Exception as e:
            st.error(f"Failed to load system usage overview: {e}")

    st.markdown("### Configuration Details")

    # Detect actual session backend being used
    try:
        backend = get_session_backend()
        backend_type = type(backend).__name__
        if backend_type == "InMemorySessionBackend":
            actual_backend = "memory"
        elif backend_type == "ShardedSQLiteSessionBackend":
            actual_backend = "sqlite"
        else:
            actual_backend = backend_type.lower()
    except Exception:
        # Fallback to config value
        actual_backend = get_config('backend', 'session', 'sqlite')

    config_details = {
        "Desktop Mode": get_config('desktop_mode', 'global', True),
        "Session Backend": actual_backend,
        "Check Size": get_config('check_size', 'global', False),
        "Check Language": get_config('check_language', 'global', False),
        "Max Text Size": get_config('max_text_size', 'global', 20000000),
        "Authorization Enabled": is_authorization_enabled(),
        "Quota Limit": (
            get_runtime_config().get_config_value(
                'quota', get_config('quota', 'llm', 10), 'llm'
                )
            if not get_config('desktop_mode', 'global', True)
            else get_config('quota', 'llm', 10)
        ),
        "Firestore Collection": (
            get_runtime_config().is_firestore_enabled()
            if not get_config('desktop_mode', 'global', True)
            else get_config('cache_mode', 'cache', False)
        ),
    }

    config_df = pd.DataFrame([
        {"Setting": key, "Value": str(value)}
        for key, value in config_details.items()
    ])

    st.dataframe(
        config_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Setting": st.column_config.TextColumn("Setting", width="medium"),
            "Value": st.column_config.TextColumn("Value", width="large")
        }
    )


@require_authorization('admin')
def main():
    """Main user management and administration interface."""

    menu()

    st.markdown(
        body="# :material/admin_panel_settings: User Management & Administration",
    )

    st.markdown(
        "Comprehensive user authorization management and system administration controls."
    )

    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs([
        ":material/manage_accounts: User Management",
        ":material/build_circle: System Configuration",
        ":material/policy: Audit Logs",
        ":material/settings_applications: Application Info"
    ])

    with tab1:
        render_user_management_tab()

    with tab2:
        render_system_config_tab()

    with tab3:
        render_audit_log_tab()

    with tab4:
        render_system_info_tab()


if __name__ == "__main__":
    main()
