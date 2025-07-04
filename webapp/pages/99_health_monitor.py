"""
Enterprise Health Monitoring Page

This page provides system health monitoring for enterprise deployments,
supporting 99.9% uptime requirements and automatic failover detection.

Access is restricted to users with admin role.
"""

import streamlit as st
from datetime import datetime, timezone

from webapp.menu import menu
from webapp.config.unified import get_config
from webapp.utilities.auth import require_authorization
from webapp.utilities.monitoring.enterprise_health import (
    render_health_dashboard,
    render_simple_health_check,
    get_metrics_json
)

# Enterprise monitoring available only in non-desktop mode
ENTERPRISE_MONITORING_AVAILABLE = not get_config('desktop_mode', 'global', True)

st.set_page_config(
    page_title="System Health Monitor",
    page_icon=":material/cardiology:",
    layout="wide"
)


def simple_fallback_health():
    """Simple fallback health check when enterprise monitoring unavailable."""

    st.warning("Enterprise monitoring not available - showing basic health check")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Status",
            value="Running",
            border=True,
            help="Basic application status"
            )

    with col2:
        desktop_mode = get_config('desktop_mode', 'global', True)
        mode_text = "Desktop" if desktop_mode else "Enterprise"
        st.metric(
            label="Mode",
            value=mode_text,
            border=True
            )

    with col3:
        st.metric(
            label="Last Check",
            value=datetime.now(timezone.utc).strftime("%H:%M:%S"),
            border=True
            )

    # Basic session info
    st.markdown("### Session Information")
    session_keys = [k for k in st.session_state.keys()]
    session_count = 1 if len(session_keys) > 0 else 0
    st.metric(
        label="Active Sessions",
        value=session_count,
        border=True
        )

    st.write(session_keys)

    # Configuration check
    st.markdown("### Configuration")
    try:
        # Detect actual session backend being used
        try:
            from webapp.utilities.storage.backend_factory import get_session_backend
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

        st.write(f"**Desktop Mode**: {desktop_mode}")
        st.write(f"**Session Backend**: {actual_backend}")

        if not desktop_mode:
            st.info("Enterprise mode detected - full monitoring should be available")
        else:
            st.success("Desktop mode - simplified monitoring active")

    except Exception as e:
        st.error(f"Configuration check failed: {e}")


def simple_json_metrics():
    """Simple JSON metrics fallback."""
    session_keys = [k for k in st.session_state.keys()
                    if isinstance(st.session_state.get(k), dict)]

    # Detect actual backend
    try:
        from webapp.utilities.storage.backend_factory import get_session_backend
        backend = get_session_backend()
        backend_type = type(backend).__name__
        if backend_type == "InMemorySessionBackend":
            actual_backend = "memory"
        elif backend_type == "ShardedSQLiteSessionBackend":
            actual_backend = "sqlite"
        else:
            actual_backend = backend_type.lower()
    except Exception:
        actual_backend = get_config('backend', 'session', 'sqlite')

    desktop_mode = get_config('desktop_mode', 'global', True)

    metrics = {
        "status": "online",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "desktop" if desktop_mode else "enterprise",
        "monitoring_level": "basic_fallback",
        "session_backend": actual_backend,
        "active_sessions": len(session_keys),
        "desktop_mode": desktop_mode
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


if __name__ == "__main__":
    main()
