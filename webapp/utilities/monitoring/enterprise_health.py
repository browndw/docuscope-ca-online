"""
Enterprise Health and Monitoring Endpoints

This module provides health checking and monitoring endpoints for enterprise
deployment monitoring, supporting 99.9% uptime requirements.
"""

import time
import threading
from datetime import datetime, timezone
from typing import Dict, Any, List

import streamlit as st

# Try to import psutil, but handle gracefully if not available
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

from webapp.config.unified import get_config
from webapp.utilities.configuration.logging_config import get_logger
from webapp.utilities.storage.backend_factory import get_session_backend
from webapp.utilities.ai.enterprise_router import get_ai_router
from webapp.utilities.ai.enterprise_circuit_breaker import get_circuit_breaker_manager

logger = get_logger()


class HealthMonitor:
    """
    Enterprise health monitoring system for production deployments.

    Provides comprehensive health checks, performance monitoring,
    and alerting capabilities for 500+ concurrent users.
    """

    def __init__(self):
        """Initialize the health monitor."""
        self.last_comprehensive_check = datetime.now(timezone.utc)
        self.check_interval = get_config('health_check_interval', 'monitoring', 30)
        self.performance_thresholds = self._load_performance_thresholds()
        self.alert_history = []
        self._lock = threading.RLock()

    def _load_performance_thresholds(self) -> Dict[str, float]:
        """Load performance thresholds from configuration."""
        return {
            'max_response_time_ms': get_config(
                'max_response_time_ms', 'monitoring', 2000
            ),
            'max_memory_usage_percent': get_config(
                'max_memory_usage_percent', 'monitoring', 85
            ),
            'max_cpu_usage_percent': get_config(
                'max_cpu_usage_percent', 'monitoring', 80
            ),
            'min_available_disk_percent': get_config(
                'min_available_disk_percent', 'monitoring', 20
            ),
            'max_error_rate_percent': get_config(
                'max_error_rate_percent', 'monitoring', 5
            ),
            'max_concurrent_sessions': get_config(
                'max_concurrent_sessions', 'monitoring', 500
            )
        }

    def get_health_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system health status.

        Returns
        -------
        Dict[str, Any]
            Complete health status including all subsystems
        """
        start_time = time.time()

        try:
            health_status = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'overall_status': 'healthy',
                'response_time_ms': 0,
                'uptime_seconds': self._get_uptime(),
                'subsystems': {},
                'performance_metrics': {},
                'alerts': []
            }

            # Check database health
            db_health = self._check_database_health()
            health_status['subsystems']['database'] = db_health

            # Check system resources
            system_health = self._check_system_resources()
            health_status['subsystems']['system'] = system_health

            # Check session management
            session_health = self._check_session_health()
            health_status['subsystems']['sessions'] = session_health

            # Check cache performance
            cache_health = self._check_cache_health()
            health_status['subsystems']['cache'] = cache_health

            # Check circuit breaker status
            circuit_breaker_health = self._check_circuit_breaker_health()
            health_status['subsystems']['circuit_breaker'] = circuit_breaker_health

            # Check request router status
            request_router_health = self._check_request_router_health()
            health_status['subsystems']['request_router'] = request_router_health

            # Aggregate overall status
            all_healthy = all(
                subsystem.get('status') == 'healthy'
                for subsystem in health_status['subsystems'].values()
            )

            if not all_healthy:
                health_status['overall_status'] = 'degraded'

            # Check for critical failures
            critical_failures = [
                subsystem for subsystem in health_status['subsystems'].values()
                if subsystem.get('status') == 'critical'
            ]

            if critical_failures:
                health_status['overall_status'] = 'critical'

            # Calculate response time
            health_status['response_time_ms'] = (time.time() - start_time) * 1000

            # Generate alerts if needed
            health_status['alerts'] = self._generate_alerts(health_status)

            return health_status

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'overall_status': 'critical',
                'error': str(e),
                'response_time_ms': (time.time() - start_time) * 1000
            }

    def _check_database_health(self) -> Dict[str, Any]:
        """Check database subsystem health."""
        try:
            start_time = time.time()
            backend = get_session_backend()

            # Test database connectivity and performance
            if hasattr(backend, 'health_check'):
                # Use sharded backend health check
                db_stats = backend.health_check()
                response_time = (time.time() - start_time) * 1000

                # Determine status based on response time and shard health
                if response_time > self.performance_thresholds['max_response_time_ms']:
                    status = 'degraded'
                elif not db_stats.get('overall_healthy', True):
                    status = 'critical'
                else:
                    status = 'healthy'

                return {
                    'status': status,
                    'response_time_ms': response_time,
                    'details': db_stats
                }
            else:
                # Standard SQLite backend
                stats = backend.get_session_stats()
                response_time = (time.time() - start_time) * 1000

                status = 'healthy'
                if response_time > self.performance_thresholds['max_response_time_ms']:
                    status = 'degraded'

                return {
                    'status': status,
                    'response_time_ms': response_time,
                    'active_sessions': stats.get('active_sessions', 0),
                    'database_size_mb': stats.get('database_size_bytes', 0) / 1024 / 1024
                }

        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                'status': 'critical',
                'error': str(e)
            }

    def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resource usage."""
        try:
            if not PSUTIL_AVAILABLE:
                return {
                    'status': 'healthy',
                    'message': 'System monitoring unavailable (psutil not installed)',
                    'memory_percent': 0,
                    'cpu_percent': 0,
                    'disk_free_percent': 100
                }

            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)

            # Disk usage
            disk = psutil.disk_usage('/')
            disk_free_percent = (disk.free / disk.total) * 100

            # Load average (Unix-like systems)
            load_avg = None
            try:
                load_avg = psutil.getloadavg()
            except AttributeError:
                # Windows doesn't have load average
                pass

            # Determine status
            status = 'healthy'
            issues = []

            if memory_percent > self.performance_thresholds['max_memory_usage_percent']:
                status = 'degraded'
                issues.append(f"High memory usage: {memory_percent:.1f}%")

            if cpu_percent > self.performance_thresholds['max_cpu_usage_percent']:
                status = 'degraded'
                issues.append(f"High CPU usage: {cpu_percent:.1f}%")

            min_disk_threshold = self.performance_thresholds['min_available_disk_percent']
            if disk_free_percent < min_disk_threshold:
                status = 'critical'
                issues.append(f"Low disk space: {disk_free_percent:.1f}% free")

            return {
                'status': status,
                'memory_usage_percent': memory_percent,
                'memory_available_gb': memory.available / 1024 / 1024 / 1024,
                'cpu_usage_percent': cpu_percent,
                'disk_free_percent': disk_free_percent,
                'load_average': load_avg,
                'issues': issues
            }

        except Exception as e:
            logger.error(f"System resource check failed: {e}")
            return {
                'status': 'critical',
                'error': str(e)
            }

    def _check_session_health(self) -> Dict[str, Any]:
        """Check session management health."""
        try:
            backend = get_session_backend()
            stats = backend.get_session_stats()

            active_sessions = stats.get('active_sessions', 0)
            max_sessions = self.performance_thresholds['max_concurrent_sessions']

            status = 'healthy'
            issues = []

            if active_sessions > max_sessions * 0.9:  # 90% capacity
                status = 'degraded'
                issues.append(f"High session load: {active_sessions}/{max_sessions}")
            elif active_sessions > max_sessions:
                status = 'critical'
                issues.append(
                    f"Session capacity exceeded: {active_sessions}/{max_sessions}"
                )

            return {
                'status': status,
                'active_sessions': active_sessions,
                'max_sessions': max_sessions,
                'session_utilization_percent': (active_sessions / max_sessions) * 100,
                'queries_last_24h': stats.get('queries_last_24h', 0),
                'avg_session_size_kb': stats.get('avg_session_size_bytes', 0) / 1024,
                'issues': issues
            }

        except Exception as e:
            logger.error(f"Session health check failed: {e}")
            return {
                'status': 'critical',
                'error': str(e)
            }

    def _check_cache_health(self) -> Dict[str, Any]:
        """Check cache subsystem health."""
        try:
            backend = get_session_backend()

            # Test cache operations if available
            if hasattr(backend, 'cache_set') and hasattr(backend, 'cache_get'):
                start_time = time.time()
                test_key = f"health_check_{int(time.time())}"
                test_value = {'test': True, 'timestamp': time.time()}

                # Test cache set/get
                set_success = backend.cache_set(test_key, test_value, 60)
                get_result = backend.cache_get(test_key)
                cleanup_success = backend.cache_delete(test_key)

                response_time = (time.time() - start_time) * 1000

                status = 'healthy'
                if not (set_success and get_result and cleanup_success):
                    status = 'degraded'
                elif response_time > 500:  # Cache should be fast
                    status = 'degraded'

                return {
                    'status': status,
                    'response_time_ms': response_time,
                    'operations_successful': set_success and get_result and cleanup_success
                }
            else:
                return {
                    'status': 'healthy',
                    'message': 'Cache not implemented in current backend'
                }

        except Exception as e:
            logger.error(f"Cache health check failed: {e}")
            return {
                'status': 'critical',
                'error': str(e)
            }

    def _check_circuit_breaker_health(self) -> Dict[str, Any]:
        """Check circuit breaker status."""
        try:

            cb_manager = get_circuit_breaker_manager()
            all_metrics = cb_manager.get_all_metrics()

            # Check status of both circuit breakers
            community_metrics = all_metrics.get('community', {})
            individual_metrics = all_metrics.get('individual', {})

            community_state = community_metrics.get('state', 'unknown')
            individual_state = individual_metrics.get('state', 'unknown')

            # Determine overall status
            if community_state == 'open' or individual_state == 'open':
                overall_status = 'critical'
            elif community_state == 'half_open' or individual_state == 'half_open':
                overall_status = 'degraded'
            else:
                overall_status = 'healthy'

            return {
                'status': overall_status,
                'community_state': community_state,
                'individual_state': individual_state,
                'community_metrics': community_metrics,
                'individual_metrics': individual_metrics
            }
        except Exception as e:
            logger.error(f"Circuit breaker health check failed: {e}")
            return {
                'status': 'critical',
                'error': str(e)
            }

    def _check_request_router_health(self) -> Dict[str, Any]:
        """Check request router status."""
        try:
            router = get_ai_router()
            stats = router.get_router_stats()

            active_requests = stats.get('total_active_requests', 0)
            status = 'healthy' if active_requests < 50 else 'degraded'

            return {
                'status': status,
                'router_stats': stats
            }
        except Exception as e:
            logger.error(f"Request router health check failed: {e}")
            return {
                'status': 'critical',
                'error': str(e)
            }

    def _get_uptime(self) -> float:
        """Get system uptime in seconds."""
        try:
            if PSUTIL_AVAILABLE:
                boot_time = psutil.boot_time()
                return time.time() - boot_time
            else:
                return 0  # Cannot determine uptime without psutil
        except Exception:
            return 0

    def _generate_alerts(self, health_status: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate alerts based on health status."""
        alerts = []

        # High response time alert
        response_time = health_status.get('response_time_ms', 0)
        if response_time > self.performance_thresholds['max_response_time_ms']:
            alerts.append({
                'level': 'warning',
                'message': f"High response time: {response_time:.1f}ms",
                'threshold': self.performance_thresholds['max_response_time_ms'],
                'actual': response_time
            })

        # System resource alerts
        system_health = health_status.get('subsystems', {}).get('system', {})
        for issue in system_health.get('issues', []):
            alerts.append({
                'level': 'critical' if 'disk space' in issue else 'warning',
                'message': issue,
                'subsystem': 'system'
            })

        # Database alerts
        db_health = health_status.get('subsystems', {}).get('database', {})
        if db_health.get('status') != 'healthy':
            alerts.append({
                'level': 'critical' if db_health.get('status') == 'critical' else 'warning',
                'message': f"Database health: {db_health.get('status')}",
                'subsystem': 'database'
            })

        # Session capacity alerts
        session_health = health_status.get('subsystems', {}).get('sessions', {})
        for issue in session_health.get('issues', []):
            alerts.append({
                'level': 'critical' if 'exceeded' in issue else 'warning',
                'message': issue,
                'subsystem': 'sessions'
            })

        # Circuit breaker alerts
        circuit_breaker_health = health_status.get(
            'subsystems', {}).get('circuit_breaker', {}
                                  )
        if circuit_breaker_health.get('status') != 'healthy':
            alerts.append({
                'level': 'warning',
                'message': f"Circuit breaker status: {circuit_breaker_health.get('status')}",  # noqa: E501
                'subsystem': 'circuit_breaker'
            })

        # Request router alerts
        request_router_health = health_status.get(
            'subsystems', {}).get('request_router', {}
                                  )
        if request_router_health.get('status') != 'healthy':
            alerts.append({
                'level': 'warning',
                'message': f"Request router status: {request_router_health.get('status')}",
                'subsystem': 'request_router'
            })

        return alerts

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics for monitoring dashboards."""
        try:
            backend = get_session_backend()
            stats = backend.get_session_stats()

            # System metrics
            if PSUTIL_AVAILABLE:
                memory = psutil.virtual_memory()
                cpu_percent = psutil.cpu_percent()
                memory_percent = memory.percent
                memory_used_mb = memory.used / 1024 / 1024
                memory_total_mb = memory.total / 1024 / 1024
            else:
                cpu_percent = 0
                memory_percent = 0
                memory_used_mb = 0
                memory_total_mb = 0

            metrics = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'response_metrics': {
                    'avg_response_time_ms': 0,  # Would need to track this
                    'p95_response_time_ms': 0,
                    'p99_response_time_ms': 0,
                },
                'session_metrics': {
                    'active_sessions': stats.get('active_sessions', 0),
                    'total_sessions': stats.get('total_sessions', 0),
                    'queries_per_hour': stats.get('queries_last_24h', 0) / 24,
                    'avg_session_size_kb': stats.get('avg_session_size_bytes', 0) / 1024,
                    'max_session_size_kb': stats.get('max_session_size_bytes', 0) / 1024
                },
                'system_metrics': {
                    'memory_usage_percent': memory_percent,
                    'memory_available_gb': (memory_total_mb - memory_used_mb) / 1024,
                    'cpu_usage_percent': cpu_percent,
                    'uptime_hours': self._get_uptime() / 3600
                },
                'error_metrics': {
                    'error_rate_percent': 0,  # Would need error tracking
                    'total_errors_24h': 0
                }
            }

            # Add shard-specific metrics if using sharded backend
            if hasattr(backend, 'shard_manager'):
                shard_stats = backend.get_session_stats()
                metrics['shard_metrics'] = {
                    'shard_count': shard_stats.get('shard_count', 0),
                    'healthy_shards': sum(
                        1 for shard in shard_stats.get('shards', {}).values()
                        if shard.get('healthy', False)
                    ),
                    'total_size_mb': shard_stats.get('total_size_bytes', 0) / 1024 / 1024,
                    'shards': shard_stats.get('shards', {})
                }

            return metrics

        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}")
            return {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'error': str(e)
            }

    def get_readiness_status(self) -> Dict[str, Any]:
        """
        Get readiness status for load balancer health checks.

        Returns simple pass/fail for automated systems.
        """
        try:
            # Quick checks for readiness
            backend = get_session_backend()

            # Test database connection
            start_time = time.time()
            stats = backend.get_session_stats()
            response_time = (time.time() - start_time) * 1000

            # Check basic thresholds
            ready = True
            issues = []

            if response_time > 5000:  # 5 second timeout
                ready = False
                issues.append("Database response timeout")

            active_sessions = stats.get('active_sessions', 0)
            max_sessions = self.performance_thresholds['max_concurrent_sessions']

            if active_sessions > max_sessions:
                ready = False
                issues.append("Session capacity exceeded")

            return {
                'ready': ready,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'response_time_ms': response_time,
                'issues': issues
            }

        except Exception as e:
            logger.error(f"Readiness check failed: {e}")
            return {
                'ready': False,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'error': str(e)
            }

    def get_liveness_status(self) -> Dict[str, Any]:
        """
        Get liveness status for container orchestration.

        Simple check to ensure the application is running.
        """
        return {
            'alive': True,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'uptime_seconds': self._get_uptime()
        }


# Global health monitor instance
_health_monitor = None
_monitor_lock = threading.Lock()


def get_health_monitor() -> HealthMonitor:
    """Get the global health monitor instance (singleton)."""
    global _health_monitor

    if _health_monitor is None:
        with _monitor_lock:
            if _health_monitor is None:
                _health_monitor = HealthMonitor()

    return _health_monitor


# Streamlit health endpoints for monitoring

def render_health_dashboard():
    """Render comprehensive health dashboard for operations team."""

    monitor = get_health_monitor()

    # Auto-refresh
    if st.button(
        label="Refresh Status",
        icon=":material/refresh:",
        key="refresh_health"
    ):
        st.rerun()

    # Get health status
    health_status = monitor.get_health_status()

    # Overall status
    overall_status = health_status.get('overall_status', 'unknown')
    status_color = {
        'healthy': '🟢',
        'degraded': '🟡',
        'critical': '🔴',
        'unknown': '⚪'
    }.get(overall_status, '⚪')

    st.markdown(f"## {status_color} Overall Status: {overall_status.upper()}")

    # Response time metric
    response_time = health_status.get('response_time_ms', 0)
    st.metric(
        "Health Check Response Time",
        f"{response_time:.1f}ms",
        border=True
        )

    # Subsystem status
    st.markdown("### :material/readiness_score: Subsystem Status")

    subsystems = health_status.get('subsystems', {})
    cols = st.columns(len(subsystems))

    for i, (name, status) in enumerate(subsystems.items()):
        with cols[i]:
            subsystem_status = status.get('status', 'unknown')
            subsystem_color = {
                'healthy': '🟢',
                'degraded': '🟡',
                'critical': '🔴',
                'unknown': '⚪'
            }.get(subsystem_status, '⚪')

            st.markdown(f"**{subsystem_color} {name.title()}**")
            st.write(f"Status: {subsystem_status}")

            if 'response_time_ms' in status:
                st.write(f"Response: {status['response_time_ms']:.1f}ms")

            if 'issues' in status and status['issues']:
                st.warning("Issues: " + "; ".join(status['issues']))

    # AI Service Status (Circuit Breakers and Request Router)
    st.divider()
    render_ai_service_status()

    # Alerts
    alerts = health_status.get('alerts', [])
    if alerts:
        st.subheader("⚠️ Active Alerts")
        for alert in alerts:
            level = alert.get('level', 'info')
            message = alert.get('message', 'Unknown alert')

            if level == 'critical':
                st.error(f"🔴 CRITICAL: {message}")
            elif level == 'warning':
                st.warning(f"🟡 WARNING: {message}")
            else:
                st.info(f"ℹ️ INFO: {message}")

    # Performance metrics
    st.markdown("### :material/analytics: Performance Metrics")
    metrics = monitor.get_performance_metrics()

    # Session metrics
    session_metrics = metrics.get('session_metrics', {})
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Active Sessions",
            session_metrics.get('active_sessions', 0),
            border=True
            )
    with col2:
        st.metric(
            "Queries/Hour",
            f"{session_metrics.get('queries_per_hour', 0):.1f}",
            border=True
            )
    with col3:
        st.metric(
            "Avg Session Size",
            f"{session_metrics.get('avg_session_size_kb', 0):.1f}KB",
            border=True
            )
    with col4:
        st.metric(
            "System Memory",
            f"{metrics.get('system_metrics', {}).get('memory_usage_percent', 0):.1f}%",
            border=True
            )

    # Shard information (if available)
    if 'shard_metrics' in metrics:
        st.subheader(":material/database: Database Shards")
        shard_metrics = metrics['shard_metrics']

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Total Shards",
                shard_metrics.get('shard_count', 0),
                border=True
                )
        with col2:
            st.metric(
                "Healthy Shards",
                shard_metrics.get('healthy_shards', 0),
                border=True
                )
        with col3:
            st.metric(
                "Total DB Size",
                f"{shard_metrics.get('total_size_mb', 0):.1f}MB",
                border=True
                )

        # Shard details
        shards = shard_metrics.get('shards', {})
        if shards:
            shard_data = []
            for shard_id, shard_info in shards.items():
                shard_data.append({
                    'Shard ID': shard_id,
                    'Status': '🟢 Healthy' if shard_info.get('healthy', False) else '🔴 Unhealthy',  # noqa: E501
                    'Active Sessions': shard_info.get('active_sessions', 0),
                    'Size (MB)': f"{shard_info.get('size_bytes', 0) / 1024 / 1024:.1f}",
                    'Queries (24h)': shard_info.get('queries_24h', 0)
                })

            st.dataframe(shard_data, use_container_width=True)

    # System details
    with st.expander(
        label="System Details",
        icon=":material/computer:",
        expanded=False
    ):
        system_metrics = metrics.get('system_metrics', {})
        st.json(system_metrics)

    # Raw health data
    with st.expander(
        label="Raw Health Data",
        icon=":material/data_object:",
        expanded=False
    ):
        st.json(health_status)


def render_simple_health_check():
    """Simple health check endpoint for load balancers."""
    monitor = get_health_monitor()

    # Get readiness status
    readiness = monitor.get_readiness_status()

    if readiness.get('ready', False):
        st.success("✅ System Ready")
        st.json({
            'status': 'ready',
            'timestamp': readiness.get('timestamp'),
            'response_time_ms': readiness.get('response_time_ms')
        })
    else:
        st.error("❌ System Not Ready")
        st.json(readiness)

        # Set proper HTTP status for load balancer
        st.markdown(
            """
            <script>
                // Signal unhealthy to load balancer
                fetch('/health', {method: 'POST', body: JSON.stringify({status: 'unhealthy'})});
            </script>
            """,  # noqa: E501
            unsafe_allow_html=True
        )


def get_metrics_json() -> Dict[str, Any]:
    """Get metrics in JSON format for external monitoring systems."""
    monitor = get_health_monitor()

    return {
        'health': monitor.get_health_status(),
        'metrics': monitor.get_performance_metrics(),
        'readiness': monitor.get_readiness_status(),
        'liveness': monitor.get_liveness_status()
    }


def get_circuit_breaker_health() -> Dict[str, Any]:
    """Get circuit breaker health status for integration with existing monitoring."""
    try:
        cb_manager = get_circuit_breaker_manager()

        # Get individual circuit breakers
        community_breaker = cb_manager.get_breaker("community")
        individual_breaker = cb_manager.get_breaker("individual")

        community_status = community_breaker.state.value
        individual_status = individual_breaker.state.value

        # Determine overall circuit breaker health
        if community_status == "open" or individual_status == "open":
            overall_status = "critical"
        elif community_status == "half_open" or individual_status == "half_open":
            overall_status = "degraded"
        else:
            overall_status = "healthy"

        return {
            "status": overall_status,
            "community": {
                "state": community_status,
                "failure_count": community_breaker.metrics.current_failures,
                "success_count": community_breaker.metrics.current_successes,
                "total_requests": community_breaker.metrics.total_requests
            },
            "individual": {
                "state": individual_status,
                "failure_count": individual_breaker.metrics.current_failures,
                "success_count": individual_breaker.metrics.current_successes,
                "total_requests": individual_breaker.metrics.total_requests
            }
        }
    except Exception as e:
        logger.error(f"Failed to get circuit breaker health: {e}")
        return {"status": "unknown", "error": str(e)}


def get_request_router_health() -> Dict[str, Any]:
    """Get request router health status for integration with existing monitoring."""
    try:
        router = get_ai_router()
        stats = router.get_router_stats()

        # Get current metrics
        total_active = stats.get('total_active_requests', 0)
        community_queue = stats.get('community_queue_size', 0)
        individual_queue = stats.get('individual_queue_size', 0)
        total_queue = community_queue + individual_queue

        # Determine health based on queue size
        if total_queue > 500:
            status = "critical"
        elif total_queue > 100:
            status = "degraded"
        else:
            status = "healthy"

        return {
            "status": status,
            "queue_size": total_queue,
            "active_requests": total_active,
            "deduplication_saves": stats.get('deduplication_hits', 0)
        }
    except Exception as e:
        logger.error(f"Failed to get request router health: {e}")
        return {"status": "unknown", "error": str(e)}


def render_ai_service_status():
    """Render AI service status including circuit breakers and request routing."""
    st.subheader(":material/smart_toy: AI Service Status")

    try:
        cb_health = get_circuit_breaker_health()
        router_health = get_request_router_health()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### Community Circuit Breakers")
            # Community key status
            community_state = cb_health.get("community", {}).get("state", "unknown")
            if community_state == "closed":
                st.success(
                    body="Community Key: Available",
                    icon="🟢"
                    )
            elif community_state == "half_open":
                st.warning(
                    body="Community Key: Testing",
                    icon="🟡"
                    )
            else:
                st.error(
                    body="Community Key: Protected",
                    icon="🔴"
                    )
        with col2:
            st.markdown("##### Individual Circuit Breakers")
            # Individual key status
            individual_state = cb_health.get("individual", {}).get("state", "unknown")
            if individual_state == "closed":
                st.success(
                    body="Individual Key: Available",
                    icon="🟢"
                    )
            elif individual_state == "half_open":
                st.warning(
                    body="Individual Key: Testing",
                    icon="🟡"
                    )
            else:
                st.error(
                    body="Individual Key: Protected",
                    icon="🔴"
                    )

        with col3:
            st.markdown("##### Request Router")

            router_status = router_health.get("status", "unknown")
            queue_size = router_health.get("queue_size", 0)
            active_requests = router_health.get("active_requests", 0)

            if router_status == "healthy":
                st.success(
                    body=f"Router: Healthy ({queue_size} queued)",
                    icon="🟢"
                    )
            elif router_status == "degraded":
                st.warning(
                    body=f"Router: Degraded ({queue_size} queued)",
                    icon="🟡"
                    )
            else:
                st.error(
                    body=f"Router: Critical ({queue_size} queued)",
                    icon="🔴"
                    )
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label="Active Requests",
                value=active_requests,
                border=True
                )
        with col2:
            st.metric(
                label="Deduplication Saves",
                value=router_health.get("deduplication_saves", 0),
                border=True
                )

    except Exception as e:
        st.error(f"Failed to load AI service status: {e}")
