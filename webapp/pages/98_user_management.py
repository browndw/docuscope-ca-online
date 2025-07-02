"""
User Management and System Administration

This page provides comprehensive user authorization management and system
administration controls for enterprise deployments.

Access is restricted to users with admin role.
"""

import streamlit as st
import pandas as pd
from datetime import datetime

from webapp.menu import menu
from webapp.config.unified import get_config
from webapp.config.runtime_config import runtime_config
from webapp.utilities.auth import (
    require_authorization,
    add_authorized_user,
    remove_authorized_user,
    update_user_role,
    list_authorized_users,
    get_user_role,
    is_authorization_enabled
)


st.set_page_config(
    page_title="User Management & Administration",
    page_icon=":material/admin_panel_settings:",
    layout="wide"
)


def render_user_management_tab():
    """Render the user management interface."""
    st.subheader("👥 User Authorization Management")
    
    if not is_authorization_enabled():
        st.warning(
            "Authorization is disabled (likely running in desktop mode). "
            "User management is only available in enterprise mode."
        )
        return
    
    # Add new user section
    with st.expander("➕ Add New User", expanded=False):
        col1, col2, col3 = st.columns([2, 1, 1])
        
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
        
        with col3:
            st.write("")  # Spacing
            st.write("")  # Spacing
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
    st.markdown("### Current Authorized Users")
    
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
    st.markdown("### User Actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Update User Role**")
        user_emails = [user['email'] for user in users if user.get('active', True)]
        
        if user_emails:
            selected_user = st.selectbox(
                "Select User",
                options=user_emails,
                key="role_update_user"
            )
            
            current_role = get_user_role(selected_user) if selected_user else None
            
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
            
            if st.button("Update Role", key="update_role_btn"):
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
    
    with col2:
        st.markdown("**Remove User Access**")
        
        if user_emails:
            user_to_remove = st.selectbox(
                "Select User to Remove",
                options=user_emails,
                key="remove_user_select"
            )
            
            if st.button("Remove User", type="secondary", key="remove_user_btn"):
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
    st.subheader("🔧 Runtime Configuration")
    
    desktop_mode = get_config('desktop_mode', 'global', True)
    
    if desktop_mode:
        st.warning(
            "Running in desktop mode. Runtime configuration controls are disabled "
            "for lightweight operation."
        )
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Firestore Research Data Collection**")
        
        # Get current state
        current_state = runtime_config.is_firestore_enabled()
        toml_default = get_config('cache_mode', 'cache', False)
        
        # Show current status
        status_text = "✅ Enabled" if current_state else "❌ Disabled"
        st.markdown(f"**Current Status**: {status_text}")
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
            runtime_config.toggle_firestore_collection(new_state, admin_user)
            st.success(f"Firestore collection {'enabled' if new_state else 'disabled'}")
            st.rerun()
        
        # Reset to default button
        if st.button("Reset to TOML Default", key="reset_firestore"):
            admin_user = st.session_state.get('user_email', 'admin')
            runtime_config.clear_firestore_override(admin_user)
            st.success("Reset to TOML default")
            st.rerun()
    
    with col2:
        st.markdown("**Active Configuration Overrides**")
        
        overrides = runtime_config.get_all_overrides()
        if overrides:
            for key, data in overrides.items():
                with st.container():
                    st.markdown(f"**{key}**: `{data['value']}`")
                    st.caption(f"Updated by {data['updated_by']} at {data['updated_at']}")
                    if st.button(f"Clear {key}", key=f"clear_{key}"):
                        admin_user = st.session_state.get('user_email', 'admin')
                        runtime_config.clear_override(key, admin_user)
                        st.success(f"Cleared override for {key}")
                        st.rerun()
        else:
            st.info("No runtime overrides active")


def render_audit_log_tab():
    """Render audit log viewing interface."""
    st.subheader("📋 System Audit Logs")
    
    desktop_mode = get_config('desktop_mode', 'global', True)
    
    if desktop_mode:
        st.warning(
            "Running in desktop mode. Audit logs are not available "
            "in lightweight desktop operation."
        )
        return
    
    # Configuration changes audit log
    st.markdown("### Configuration Changes")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        log_limit = st.number_input(
            "Number of entries",
            min_value=5,
            max_value=100,
            value=20,
            step=5,
            help="Number of recent audit log entries to display"
        )
    
    with col2:
        if st.button("Refresh Logs", key="refresh_audit"):
            st.rerun()
    
    audit_log = runtime_config.get_audit_log(limit=log_limit)
    
    if audit_log:
        # Convert to DataFrame for better display
        audit_df_data = []
        for entry in audit_log:
            change_type = "🔄 Update" if entry['new_value'] != 'CLEARED' else "🗑️ Clear"
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
                "Configuration Key": st.column_config.TextColumn("Config Key", width="medium"),
                "Old Value": st.column_config.TextColumn("Old Value", width="medium"),
                "New Value": st.column_config.TextColumn("New Value", width="medium"),
                "Updated By": st.column_config.TextColumn("Updated By", width="medium")
            }
        )
    else:
        st.info("No configuration changes recorded")
    
    # User authorization audit log (future enhancement)
    st.markdown("### User Authorization Changes")
    st.info("User authorization audit log will be implemented in a future update")


def render_system_info_tab():
    """Render system information and status."""
    st.subheader("ℹ️ System Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Application Mode**")
        desktop_mode = get_config('desktop_mode', 'global', True)
        mode_text = "🖥️ Desktop" if desktop_mode else "🏢 Enterprise"
        st.metric("Mode", mode_text)
        
        st.markdown("**Authorization Status**")
        auth_enabled = is_authorization_enabled()
        auth_text = "✅ Enabled" if auth_enabled else "❌ Disabled"
        st.metric("User Authorization", auth_text)
    
    with col2:
        st.markdown("**Session Backend**")
        backend = get_config('backend', 'session', 'sqlite')
        st.metric("Session Storage", backend.upper())
        
        st.markdown("**Current User**")
        current_user = st.session_state.get('user_email', 'Not logged in')
        st.metric("Logged in as", current_user)
    
    with col3:
        st.markdown("**System Status**")
        st.metric("Status", "✅ Online")
        
        st.markdown("**Last Updated**")
        st.metric("Timestamp", datetime.now().strftime("%H:%M:%S"))
    
    # Additional system information
    st.markdown("### Configuration Details")
    
    config_details = {
        "Desktop Mode": get_config('desktop_mode', 'global', True),
        "Session Backend": get_config('backend', 'session', 'sqlite'),
        "Check Size": get_config('check_size', 'global', False),
        "Check Language": get_config('check_language', 'global', False),
        "Max Text Size": get_config('max_bytes_text', 'global', 20000000),
        "Authorization Enabled": is_authorization_enabled()
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
        "👥 User Management",
        "🔧 System Configuration", 
        "📋 Audit Logs",
        "ℹ️ System Info"
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
