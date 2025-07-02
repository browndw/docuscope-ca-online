"""
Enterprise Health Monitoring Page

This page provides system health monitoring for enterprise deployments,
supporting 99.9% uptime requirements and automatic failover detection.

Access is restricted to users with admin role.
"""

import streamlit as st
from datetime import datetime

from webapp.menu import menu
from webapp.config.unified import get_config
from webapp.utilities.auth import require_authorization
from webapp.utilities.monitoring.enterprise_health import (
    render_health_dashboard,
    render_simple_health_check,
    get_metrics_json
)

ENTERPRISE_MONITORING_AVAILABLE = True

st.set_page_config(
    page_title="System Health Monitor",
    page_icon=":material/cardiology:",
    layout="wide"
)


def render_admin_controls():
    """Render admin controls for runtime configuration."""
    st.markdown("---")
    st.subheader("🔧 Runtime Configuration")
    
    st.info(
        "💡 **Admin Controls Moved**: Runtime configuration and user management "
        "controls have been moved to the dedicated User Management page. "
        "Access it from the Admin Features section in the navigation menu."
    )
    
    st.markdown(
        "This page now focuses on system health monitoring. "
        "For administrative controls, please use:"
    )
    
    st.page_link(
        "pages/98_user_management.py",
        label="� User Management & Administration",
        icon=":material/admin_panel_settings:"
    )


def simple_fallback_health():
    """Simple fallback health check when enterprise monitoring unavailable."""

    st.warning("Enterprise monitoring not available - showing basic health check")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Status", "✅ Online", help="Basic application status")

    with col2:
        desktop_mode = st.session_state.get('desktop_mode', True)
        mode_text = "Desktop" if desktop_mode else "Enterprise"
        st.metric("Mode", mode_text)

    with col3:
        st.metric("Last Check", datetime.now().strftime("%H:%M:%S"))

    # Basic session info
    st.markdown("Session Information")
    session_keys = [k for k in st.session_state.keys()
                    if isinstance(st.session_state.get(k), dict)]
    session_count = len(session_keys)
    st.metric("Active Sessions", session_count)

    # Configuration check
    st.markdown("Configuration")
    try:
        desktop_mode = get_config('desktop_mode', 'global', True)
        backend = get_config('backend', 'session', 'sqlite')

        st.write(f"**Desktop Mode**: {desktop_mode}")
        st.write(f"**Session Backend**: {backend}")

        if not desktop_mode:
            st.info("Enterprise mode detected - full monitoring should be available")

    except Exception as e:
        st.error(f"Configuration check failed: {e}")


def simple_json_metrics():
    """Simple JSON metrics fallback."""
    session_keys = [k for k in st.session_state.keys()
                    if isinstance(st.session_state.get(k), dict)]
    metrics = {
        "status": "online",
        "timestamp": datetime.now().isoformat(),
        "mode": "basic_fallback",
        "active_sessions": len(session_keys)
    }
    st.json(metrics)


@require_authorization('admin')
def main():
    """Main health monitoring interface."""
    
    menu()

    st.markdown(
        body="# :material/cardiology: System Health Monitor",
    )

    # Check if this is a simple health check request
    query_params = st.query_params

    if not ENTERPRISE_MONITORING_AVAILABLE:
        if 'json' in query_params:
            simple_json_metrics()
        else:
            simple_fallback_health()
        return

    if 'simple' in query_params:
        render_simple_health_check()
        return

    if 'json' in query_params:
        # Return JSON metrics for external monitoring
        metrics = get_metrics_json()
        st.json(metrics)
        return

    # Render full dashboard
    render_health_dashboard()

    # Add admin controls section - now automatically authorized
    render_admin_controls()


if __name__ == "__main__":
    main()
