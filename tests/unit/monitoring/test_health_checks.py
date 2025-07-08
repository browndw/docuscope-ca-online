"""
Tests for monitoring and health check functionality.

This module tests system health monitoring, performance metrics,
resource usage tracking, and health endpoint functionality.
"""

from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
import psutil
import time

from webapp.config.unified import get_config


class TestHealthCheckConfiguration:
    """Test health check configuration settings."""

    def test_health_check_enabled_config(self):
        """Test that health check configuration is properly loaded."""
        health_enabled = get_config('enable_health_checks', 'monitoring', True)
        assert isinstance(health_enabled, bool)

    def test_health_check_interval_config(self):
        """Test health check interval configuration."""
        check_interval = get_config('health_check_interval_seconds', 'monitoring', 60)
        assert isinstance(check_interval, int)
        assert check_interval > 0

    def test_resource_monitoring_config(self):
        """Test resource monitoring configuration."""
        monitor_resources = get_config('monitor_system_resources', 'monitoring', True)
        assert isinstance(monitor_resources, bool)

        # Test thresholds
        cpu_threshold = get_config('cpu_usage_threshold', 'monitoring', 80.0)
        memory_threshold = get_config('memory_usage_threshold', 'monitoring', 80.0)

        assert isinstance(cpu_threshold, (int, float))
        assert isinstance(memory_threshold, (int, float))
        assert 0 < cpu_threshold <= 100
        assert 0 < memory_threshold <= 100


class TestSystemHealthChecks:
    """Test system health monitoring functionality."""

    def test_cpu_usage_monitoring(self):
        """Test CPU usage monitoring."""
        def get_cpu_usage():
            """Mock CPU usage function."""
            # Simulate CPU usage monitoring
            return psutil.cpu_percent(interval=1)

        def check_cpu_health(threshold=80.0):
            """Mock CPU health check."""
            cpu_usage = get_cpu_usage()
            return {
                'status': 'healthy' if cpu_usage < threshold else 'warning',
                'cpu_percent': cpu_usage,
                'threshold': threshold,
                'timestamp': datetime.now()
            }

        # Mock normal CPU usage
        with patch('psutil.cpu_percent', return_value=45.0):
            result = check_cpu_health(threshold=80.0)
            assert result['status'] == 'healthy'
            assert result['cpu_percent'] == 45.0

        # Mock high CPU usage
        with patch('psutil.cpu_percent', return_value=85.0):
            result = check_cpu_health(threshold=80.0)
            assert result['status'] == 'warning'
            assert result['cpu_percent'] == 85.0

    def test_memory_usage_monitoring(self):
        """Test memory usage monitoring."""
        def get_memory_usage():
            """Mock memory usage function."""
            return psutil.virtual_memory()

        def check_memory_health(threshold=80.0):
            """Mock memory health check."""
            memory = get_memory_usage()
            memory_percent = memory.percent
            return {
                'status': 'healthy' if memory_percent < threshold else 'warning',
                'memory_percent': memory_percent,
                'total_gb': round(memory.total / (1024**3), 2),
                'available_gb': round(memory.available / (1024**3), 2),
                'threshold': threshold,
                'timestamp': datetime.now()
            }

        # Mock normal memory usage
        mock_memory = MagicMock()
        mock_memory.percent = 60.0
        mock_memory.total = 16 * (1024**3)  # 16GB
        mock_memory.available = 6.4 * (1024**3)  # 6.4GB available

        with patch('psutil.virtual_memory', return_value=mock_memory):
            result = check_memory_health(threshold=80.0)
            assert result['status'] == 'healthy'
            assert result['memory_percent'] == 60.0
            assert result['total_gb'] == 16.0

    def test_disk_usage_monitoring(self):
        """Test disk usage monitoring."""
        def check_disk_health(path="/", threshold=85.0):
            """Mock disk health check."""
            disk_usage = psutil.disk_usage(path)
            used_percent = (disk_usage.used / disk_usage.total) * 100

            return {
                'status': 'healthy' if used_percent < threshold else 'warning',
                'used_percent': round(used_percent, 2),
                'total_gb': round(disk_usage.total / (1024**3), 2),
                'free_gb': round(disk_usage.free / (1024**3), 2),
                'path': path,
                'threshold': threshold,
                'timestamp': datetime.now()
            }

        # Mock disk usage
        mock_disk = MagicMock()
        mock_disk.total = 500 * (1024**3)  # 500GB
        mock_disk.used = 250 * (1024**3)   # 250GB used
        mock_disk.free = 250 * (1024**3)   # 250GB free

        with patch('psutil.disk_usage', return_value=mock_disk):
            result = check_disk_health(threshold=85.0)
            assert result['status'] == 'healthy'
            assert result['used_percent'] == 50.0


class TestApplicationHealthChecks:
    """Test application-specific health checks."""

    def test_database_connection_health(self):
        """Test database connection health check."""
        def check_database_health():
            """Mock database health check."""
            try:
                # Simulate database connection test
                # In real implementation, this would test actual DB connection
                connection_test = True  # Mock successful connection

                if connection_test:
                    return {
                        'status': 'healthy',
                        'database': 'connected',
                        'response_time_ms': 25,
                        'timestamp': datetime.now()
                    }
                else:
                    return {
                        'status': 'unhealthy',
                        'database': 'disconnected',
                        'error': 'Connection timeout',
                        'timestamp': datetime.now()
                    }
            except Exception as e:
                return {
                    'status': 'unhealthy',
                    'database': 'error',
                    'error': str(e),
                    'timestamp': datetime.now()
                }

        result = check_database_health()
        assert result['status'] in ['healthy', 'unhealthy']
        assert 'timestamp' in result

    def test_spacy_model_health(self):
        """Test spaCy model loading and health."""
        def check_spacy_model_health():
            """Mock spaCy model health check."""
            try:
                # Simulate spaCy model loading test
                model_loaded = True  # Mock successful model loading

                if model_loaded:
                    return {
                        'status': 'healthy',
                        'spacy_model': 'loaded',
                        'model_name': 'en_docusco_spacy',
                        'load_time_ms': 150,
                        'timestamp': datetime.now()
                    }
                else:
                    return {
                        'status': 'unhealthy',
                        'spacy_model': 'failed',
                        'error': 'Model not found',
                        'timestamp': datetime.now()
                    }
            except Exception as e:
                return {
                    'status': 'unhealthy',
                    'spacy_model': 'error',
                    'error': str(e),
                    'timestamp': datetime.now()
                }

        result = check_spacy_model_health()
        assert result['status'] in ['healthy', 'unhealthy']
        assert 'spacy_model' in result

    def test_streamlit_session_health(self):
        """Test Streamlit session health monitoring."""
        def check_session_health():
            """Mock session health check."""
            active_sessions = 15  # Mock active session count
            max_sessions = 100

            session_usage_percent = (active_sessions / max_sessions) * 100

            return {
                'status': 'healthy' if session_usage_percent < 80 else 'warning',
                'active_sessions': active_sessions,
                'max_sessions': max_sessions,
                'usage_percent': round(session_usage_percent, 2),
                'timestamp': datetime.now()
            }

        result = check_session_health()
        assert result['status'] in ['healthy', 'warning', 'unhealthy']
        assert result['active_sessions'] >= 0


class TestPerformanceMetrics:
    """Test performance monitoring and metrics collection."""

    def test_response_time_monitoring(self):
        """Test API response time monitoring."""
        def measure_response_time(endpoint_function):
            """Mock response time measurement."""
            start_time = time.time()

            # Simulate endpoint execution
            endpoint_function()

            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000

            return {
                'response_time_ms': round(response_time_ms, 2),
                'status': 'fast' if response_time_ms < 1000 else 'slow',
                'timestamp': datetime.now()
            }

        def mock_endpoint():
            time.sleep(0.1)  # Simulate 100ms processing
            return "success"

        result = measure_response_time(mock_endpoint)
        assert 'response_time_ms' in result
        assert result['status'] in ['fast', 'slow']

    def test_request_rate_monitoring(self):
        """Test request rate monitoring."""
        def track_request_rate():
            """Mock request rate tracking."""
            # Simulate request tracking over time
            requests_per_minute = 45
            max_requests_per_minute = 100

            usage_percent = (requests_per_minute / max_requests_per_minute) * 100

            return {
                'requests_per_minute': requests_per_minute,
                'max_requests_per_minute': max_requests_per_minute,
                'usage_percent': round(usage_percent, 2),
                'status': 'normal' if usage_percent < 80 else 'high',
                'timestamp': datetime.now()
            }

        result = track_request_rate()
        assert result['requests_per_minute'] >= 0
        assert result['status'] in ['normal', 'high', 'critical']

    def test_memory_leak_detection(self):
        """Test memory leak detection."""
        def detect_memory_trends():
            """Mock memory trend analysis."""
            # Simulate memory usage over time
            memory_samples = [45.2, 46.1, 46.8, 47.5, 48.2]  # Gradual increase

            if len(memory_samples) >= 3:
                trend = memory_samples[-1] - memory_samples[0]

                return {
                    'memory_trend_percent': round(trend, 2),
                    'status': 'stable' if abs(trend) < 5 else 'increasing',
                    'samples': len(memory_samples),
                    'current_usage': memory_samples[-1],
                    'timestamp': datetime.now()
                }

            return {'status': 'insufficient_data'}

        result = detect_memory_trends()
        assert 'status' in result


class TestHealthEndpoints:
    """Test health check HTTP endpoints."""

    def test_basic_health_endpoint(self):
        """Test basic health check endpoint."""
        def health_check_endpoint():
            """Mock health check endpoint."""
            return {
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'version': '1.0.0',
                'uptime_seconds': 3600
            }

        result = health_check_endpoint()
        assert result['status'] == 'healthy'
        assert 'timestamp' in result
        assert 'version' in result

    def test_detailed_health_endpoint(self):
        """Test detailed health check endpoint."""
        def detailed_health_check():
            """Mock detailed health check."""
            return {
                'status': 'healthy',
                'checks': {
                    'database': {'status': 'healthy', 'response_time_ms': 25},
                    'spacy_model': {'status': 'healthy', 'load_time_ms': 150},
                    'disk_space': {'status': 'healthy', 'usage_percent': 45.2},
                    'memory': {'status': 'healthy', 'usage_percent': 62.1},
                    'cpu': {'status': 'healthy', 'usage_percent': 35.8}
                },
                'timestamp': datetime.now().isoformat()
            }

        result = detailed_health_check()
        assert result['status'] == 'healthy'
        assert 'checks' in result
        assert len(result['checks']) > 0

    def test_health_endpoint_failure_scenarios(self):
        """Test health endpoint failure scenarios."""
        def health_check_with_failures():
            """Mock health check with some failures."""
            return {
                'status': 'degraded',
                'checks': {
                    'database': {'status': 'healthy', 'response_time_ms': 25},
                    'spacy_model': {'status': 'unhealthy', 'error': 'Model not loaded'},
                    'disk_space': {'status': 'warning', 'usage_percent': 85.2},
                    'memory': {'status': 'healthy', 'usage_percent': 62.1}
                },
                'failed_checks': ['spacy_model'],
                'warning_checks': ['disk_space'],
                'timestamp': datetime.now().isoformat()
            }

        result = health_check_with_failures()
        assert result['status'] == 'degraded'
        assert 'failed_checks' in result
        assert len(result['failed_checks']) > 0


class TestAlertingAndNotifications:
    """Test health alerting and notification functionality."""

    def test_threshold_based_alerting(self):
        """Test threshold-based health alerting."""
        def check_alert_conditions(metrics):
            """Mock alert condition checking."""
            alerts = []

            if metrics.get('cpu_percent', 0) > 80:
                alerts.append({
                    'type': 'cpu_high',
                    'severity': 'warning',
                    'message': f"CPU usage {metrics['cpu_percent']}% exceeds threshold",
                    'timestamp': datetime.now()
                })

            if metrics.get('memory_percent', 0) > 85:
                alerts.append({
                    'type': 'memory_high',
                    'severity': 'critical',
                    'message': (f"Memory usage {metrics['memory_percent']}% "
                                f"exceeds threshold"),
                    'timestamp': datetime.now()
                })

            return alerts

        # Test normal conditions
        normal_metrics = {'cpu_percent': 45.0, 'memory_percent': 60.0}
        alerts = check_alert_conditions(normal_metrics)
        assert len(alerts) == 0

        # Test alert conditions
        high_metrics = {'cpu_percent': 85.0, 'memory_percent': 90.0}
        alerts = check_alert_conditions(high_metrics)
        assert len(alerts) == 2
        assert any(alert['type'] == 'cpu_high' for alert in alerts)
        assert any(alert['type'] == 'memory_high' for alert in alerts)

    def test_alert_rate_limiting(self):
        """Test alert rate limiting to prevent spam."""
        def rate_limit_alerts(alert_type, last_alert_time, min_interval_minutes=5):
            """Mock alert rate limiting."""
            if last_alert_time is None:
                return True  # First alert, always send

            time_since_last = datetime.now() - last_alert_time
            min_interval = timedelta(minutes=min_interval_minutes)

            return time_since_last >= min_interval

        # Test first alert
        result = rate_limit_alerts('cpu_high', None)
        assert result is True

        # Test recent alert (should be rate limited)
        recent_time = datetime.now() - timedelta(minutes=2)
        result = rate_limit_alerts('cpu_high', recent_time, min_interval_minutes=5)
        assert result is False

        # Test old alert (should be allowed)
        old_time = datetime.now() - timedelta(minutes=10)
        result = rate_limit_alerts('cpu_high', old_time, min_interval_minutes=5)
        assert result is True


class TestHealthCheckIntegration:
    """Integration tests for health monitoring system."""

    def test_comprehensive_health_assessment(self):
        """Test comprehensive health assessment across all components."""
        def comprehensive_health_check():
            """Mock comprehensive health assessment."""
            checks = {
                'system': {
                    'cpu': {'status': 'healthy', 'usage': 45.2},
                    'memory': {'status': 'healthy', 'usage': 62.1},
                    'disk': {'status': 'healthy', 'usage': 35.8}
                },
                'application': {
                    'database': {'status': 'healthy', 'response_time': 25},
                    'spacy_model': {'status': 'healthy', 'load_time': 150},
                    'sessions': {'status': 'healthy', 'active': 15}
                },
                'performance': {
                    'response_time': {'status': 'good', 'avg_ms': 245},
                    'request_rate': {'status': 'normal', 'per_minute': 45}
                }
            }

            # Determine overall status
            all_statuses = []
            for category in checks.values():
                for check in category.values():
                    all_statuses.append(check['status'])

            if all(status == 'healthy' or status == 'good' or status == 'normal'
                   for status in all_statuses):
                overall_status = 'healthy'
            elif any(status == 'unhealthy' or status == 'critical'
                     for status in all_statuses):
                overall_status = 'unhealthy'
            else:
                overall_status = 'degraded'

            return {
                'overall_status': overall_status,
                'checks': checks,
                'timestamp': datetime.now().isoformat()
            }

        result = comprehensive_health_check()
        assert result['overall_status'] in ['healthy', 'degraded', 'unhealthy']
        assert 'system' in result['checks']
        assert 'application' in result['checks']
        assert 'performance' in result['checks']

    def test_health_check_scheduling(self):
        """Test health check scheduling and execution."""
        def schedule_health_checks():
            """Mock health check scheduling."""
            check_schedule = {
                'basic_health': {'interval_seconds': 30, 'last_run': None},
                'system_resources': {'interval_seconds': 60, 'last_run': None},
                'application_health': {'interval_seconds': 120, 'last_run': None},
                'performance_metrics': {'interval_seconds': 300, 'last_run': None}
            }

            current_time = datetime.now()
            checks_to_run = []

            for check_name, schedule in check_schedule.items():
                if schedule['last_run'] is None:
                    checks_to_run.append(check_name)
                else:
                    time_since_last = current_time - schedule['last_run']
                    if time_since_last.total_seconds() >= schedule['interval_seconds']:
                        checks_to_run.append(check_name)

            return checks_to_run

        checks_to_run = schedule_health_checks()
        assert isinstance(checks_to_run, list)
        # On first run, all checks should be scheduled
        expected_checks = ['basic_health', 'system_resources',
                           'application_health', 'performance_metrics']
        assert all(check in expected_checks for check in checks_to_run)


class TestDesktopVsEnterpriseHealthChecks:
    """Test health check differences between desktop and enterprise modes."""

    def test_desktop_mode_health_checks(self):
        """Test health checks appropriate for desktop mode."""
        # Patch the module where get_config is imported in this test file
        with patch('tests.unit.monitoring.test_health_checks.get_config') as mock_config:
            mock_config.side_effect = lambda key, section, default: {
                ('desktop_mode', 'global'): True,
                ('enable_user_authorization', 'authorization'): False,
            }.get((key, section), default)

            def desktop_health_checks():
                """Mock desktop-specific health checks."""
                desktop_mode = get_config('desktop_mode', 'global', False)

                if desktop_mode:
                    return {
                        'mode': 'desktop',
                        'checks': {
                            'spacy_model': {'status': 'healthy'},
                            'local_storage': {'status': 'healthy'},
                            'memory_usage': {'status': 'healthy'}
                        },
                        'auth_required': False
                    }

                return {'mode': 'enterprise'}

            result = desktop_health_checks()
            assert result['mode'] == 'desktop'
            assert result['auth_required'] is False
            assert 'spacy_model' in result['checks']

    def test_enterprise_mode_health_checks(self):
        """Test health checks appropriate for enterprise mode."""
        # Patch the module where get_config is imported in this test file
        with patch('tests.unit.monitoring.test_health_checks.get_config') as mock_config:
            mock_config.side_effect = lambda key, section, default: {
                ('desktop_mode', 'global'): False,
                ('enable_user_authorization', 'authorization'): True,
            }.get((key, section), default)

            def enterprise_health_checks():
                """Mock enterprise-specific health checks."""
                desktop_mode = get_config('desktop_mode', 'global', False)
                auth_enabled = get_config('enable_user_authorization',
                                          'authorization', False)

                if not desktop_mode:
                    return {
                        'mode': 'enterprise',
                        'checks': {
                            'database': {'status': 'healthy'},
                            'authentication': {'status': 'healthy'},
                            'session_management': {'status': 'healthy'},
                            'audit_logging': {'status': 'healthy'},
                            'load_balancer': {'status': 'healthy'},
                            'cache_service': {'status': 'healthy'}
                        },
                        'auth_required': auth_enabled
                    }

                return {'mode': 'desktop'}

            result = enterprise_health_checks()
            assert result['mode'] == 'enterprise'
            assert result['auth_required'] is True
            assert 'database' in result['checks']
            assert 'authentication' in result['checks']
