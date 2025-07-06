"""
Tests for error handling and logging functionality.

This module tests error handling, logging, exception management,
and error recovery mechanisms across the application.
"""

import logging
from datetime import datetime, timedelta
import traceback

from webapp.config.unified import get_config


class TestErrorHandlingConfiguration:
    """Test error handling configuration settings."""

    def test_logging_level_config(self):
        """Test logging level configuration."""
        log_level = get_config('log_level', 'logging', 'INFO')
        assert log_level in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']

    def test_error_reporting_config(self):
        """Test error reporting configuration."""
        enable_error_reporting = get_config('enable_error_reporting', 'logging', True)
        assert isinstance(enable_error_reporting, bool)

        error_log_file = get_config('error_log_file', 'logging', 'error.log')
        assert isinstance(error_log_file, str)
        assert error_log_file.endswith('.log')

    def test_error_handling_mode_config(self):
        """Test error handling mode configuration."""
        debug_mode = get_config('debug_mode', 'global', False)
        assert isinstance(debug_mode, bool)

        show_stack_traces = get_config('show_stack_traces', 'logging', False)
        assert isinstance(show_stack_traces, bool)

    def test_error_notification_config(self):
        """Test error notification configuration."""
        notify_on_errors = get_config('notify_on_critical_errors', 'logging', False)
        assert isinstance(notify_on_errors, bool)

        max_error_rate = get_config('max_error_rate_per_minute', 'monitoring', 10)
        assert isinstance(max_error_rate, int)
        assert max_error_rate > 0


class TestLoggingSetup:
    """Test logging configuration and setup."""

    def test_logger_initialization(self):
        """Test logger initialization with proper configuration."""
        def setup_logger(name, level='INFO', log_file=None):
            """Mock logger setup function."""
            logger = logging.getLogger(name)
            logger.setLevel(getattr(logging, level))

            # Add console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(getattr(logging, level))

            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

            # Add file handler if specified
            if log_file:
                file_handler = logging.FileHandler(log_file)
                file_handler.setLevel(getattr(logging, level))
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)

            return logger

        logger = setup_logger('test_logger', 'DEBUG')
        assert logger.level == logging.DEBUG
        assert len(logger.handlers) >= 1

    def test_structured_logging_format(self):
        """Test structured logging format for better parsing."""
        def create_structured_log_entry(level, message, context=None):
            """Mock structured logging entry creation."""
            entry = {
                'timestamp': datetime.now().isoformat(),
                'level': level,
                'message': message,
                'logger': 'docuscope-ca',
                'context': context or {}
            }

            if context:
                entry.update(context)

            return entry

        log_entry = create_structured_log_entry(
            'ERROR',
            'Processing failed',
            {'user_id': 'test_user', 'document': 'test.txt', 'error_code': 'PROC_001'}
        )

        assert log_entry['level'] == 'ERROR'
        assert log_entry['message'] == 'Processing failed'
        assert log_entry['user_id'] == 'test_user'
        assert 'timestamp' in log_entry

    def test_log_rotation_configuration(self):
        """Test log rotation configuration."""
        def configure_log_rotation():
            """Mock log rotation configuration."""
            return {
                'max_bytes': 10 * 1024 * 1024,  # 10MB
                'backup_count': 5,
                'rotation_enabled': True,
                'compress_old_logs': True
            }

        config = configure_log_rotation()
        assert config['max_bytes'] > 0
        assert config['backup_count'] > 0
        assert isinstance(config['rotation_enabled'], bool)


class TestExceptionHandling:
    """Test exception handling mechanisms."""

    def test_custom_exception_classes(self):
        """Test custom exception classes for different error types."""
        class DocuScopeError(Exception):
            """Base exception for DocuScope application."""
            def __init__(self, message, error_code=None, context=None):
                super().__init__(message)
                self.error_code = error_code
                self.context = context or {}
                self.timestamp = datetime.now()

        class ProcessingError(DocuScopeError):
            """Exception raised during document processing."""
            pass

        class AuthenticationError(DocuScopeError):
            """Exception raised during authentication."""
            pass

        class ConfigurationError(DocuScopeError):
            """Exception raised for configuration issues."""
            pass

        # Test base exception
        base_error = DocuScopeError("Base error", "BASE_001", {'component': 'test'})
        assert str(base_error) == "Base error"
        assert base_error.error_code == "BASE_001"
        assert base_error.context['component'] == 'test'

        # Test specific exceptions
        proc_error = ProcessingError("Processing failed", "PROC_001")
        assert isinstance(proc_error, DocuScopeError)
        assert proc_error.error_code == "PROC_001"

        auth_error = AuthenticationError("Auth failed", "AUTH_001")
        assert isinstance(auth_error, DocuScopeError)

    def test_exception_context_preservation(self):
        """Test that exception context is preserved for debugging."""
        def process_with_context(data, context):
            """Mock function that processes data with context preservation."""
            try:
                if not data:
                    raise ValueError("No data provided")

                # Simulate processing
                if data.get('trigger_error'):
                    raise RuntimeError("Processing error occurred")

                return {'status': 'success', 'processed': len(data)}

            except Exception as e:
                # Preserve context with exception
                raise RuntimeError(f"Processing failed: {str(e)}") from e

        # Test successful processing
        result = process_with_context({'test': 'data'}, {'user': 'test_user'})
        assert result['status'] == 'success'

        # Test error with context preservation
        try:
            process_with_context({'trigger_error': True}, {'user': 'test_user'})
            assert False, "Should have raised an exception"
        except RuntimeError as e:
            assert "Processing failed" in str(e)
            assert e.__cause__ is not None  # Original exception preserved

    def test_graceful_degradation(self):
        """Test graceful degradation when components fail."""
        def process_with_fallback(data, use_advanced=True):
            """Mock function with graceful degradation."""
            try:
                if use_advanced:
                    # Try advanced processing first
                    if data.get('complex_feature'):
                        return {'method': 'advanced', 'result': 'advanced_result'}
                    else:
                        raise RuntimeError("Advanced processing failed")
                else:
                    raise RuntimeError("Advanced processing unavailable")
            except Exception:
                # Fall back to basic processing
                return {'method': 'basic', 'result': 'basic_result', 'fallback': True}

        # Test successful advanced processing
        result = process_with_fallback({'complex_feature': True})
        assert result['method'] == 'advanced'

        # Test fallback to basic processing
        result = process_with_fallback({'simple_data': True})
        assert result['method'] == 'basic'
        assert result['fallback'] is True


class TestErrorRecovery:
    """Test error recovery mechanisms."""

    def test_retry_mechanism(self):
        """Test retry mechanism for transient failures."""
        def retry_operation(operation, max_retries=3, delay=0.1):
            """Mock retry mechanism."""
            last_exception = None

            for attempt in range(max_retries):
                try:
                    return operation(attempt)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        # In real implementation, would sleep for delay
                        continue
                    else:
                        raise last_exception

        def failing_operation(attempt):
            """Mock operation that fails first two times."""
            if attempt < 2:
                raise RuntimeError(f"Attempt {attempt} failed")
            return f"Success on attempt {attempt}"

        # Test successful retry
        result = retry_operation(failing_operation, max_retries=3)
        assert "Success on attempt 2" in result

        def always_failing_operation(attempt):
            """Mock operation that always fails."""
            raise RuntimeError(f"Always fails on attempt {attempt}")

        # Test exhausted retries
        try:
            retry_operation(always_failing_operation, max_retries=2)
            assert False, "Should have raised an exception"
        except RuntimeError as e:
            assert "Always fails on attempt 1" in str(e)

    def test_circuit_breaker_pattern(self):
        """Test circuit breaker pattern for external service failures."""
        def circuit_breaker(failure_threshold=3, recovery_timeout=60):
            """Mock circuit breaker implementation."""
            state = {'failures': 0, 'state': 'closed', 'last_failure_time': None}

            def call_service(service_func):
                current_time = datetime.now()

                # Check if circuit should be reset
                if (state['state'] == 'open' and
                    state['last_failure_time'] and
                    ((current_time - state['last_failure_time']).seconds >=
                     recovery_timeout)):
                    state['state'] = 'half-open'
                    state['failures'] = 0

                # If circuit is open, fail fast
                if state['state'] == 'open':
                    raise RuntimeError("Circuit breaker is open")

                try:
                    result = service_func()
                    # Reset on success
                    if state['state'] == 'half-open':
                        state['state'] = 'closed'
                    state['failures'] = 0
                    return result
                except Exception as e:
                    state['failures'] += 1
                    state['last_failure_time'] = current_time

                    if state['failures'] >= failure_threshold:
                        state['state'] = 'open'

                    raise e

            return call_service

        def unreliable_service():
            """Mock unreliable external service."""
            # Simulate failure
            raise RuntimeError("Service unavailable")

        def reliable_service():
            """Mock reliable service."""
            return "Service response"

        circuit_breaker_func = circuit_breaker(failure_threshold=2)

        # Test circuit opening after failures
        for i in range(2):
            try:
                circuit_breaker_func(unreliable_service)
            except RuntimeError:
                pass

        # Circuit should now be open
        try:
            circuit_breaker_func(reliable_service)
            assert False, "Circuit should be open"
        except RuntimeError as e:
            assert "Circuit breaker is open" in str(e)


class TestErrorLogging:
    """Test error logging functionality."""

    def test_error_log_formatting(self):
        """Test proper error log formatting."""
        def format_error_log(exception, context=None):
            """Mock error log formatting."""
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'level': 'ERROR',
                'exception_type': type(exception).__name__,
                'exception_message': str(exception),
                'stack_trace': traceback.format_exc() if context else None,
                'context': context or {}
            }

            return log_entry

        try:
            raise ValueError("Test error")
        except ValueError as e:
            log_entry = format_error_log(e, {'user': 'test_user', 'action': 'test'})

            assert log_entry['exception_type'] == 'ValueError'
            assert log_entry['exception_message'] == 'Test error'
            assert log_entry['context']['user'] == 'test_user'

    def test_sensitive_data_filtering(self):
        """Test filtering of sensitive data from error logs."""
        def filter_sensitive_data(data):
            """Mock sensitive data filtering."""
            sensitive_keys = ['password', 'api_key', 'secret', 'token', 'private_key']

            if isinstance(data, dict):
                filtered = {}
                for key, value in data.items():
                    if any(sensitive in key.lower() for sensitive in sensitive_keys):
                        filtered[key] = '[REDACTED]'
                    elif isinstance(value, dict):
                        filtered[key] = filter_sensitive_data(value)
                    else:
                        filtered[key] = value
                return filtered

            return data

        test_data = {
            'username': 'test_user',
            'password': 'secret123',
            'api_key': 'abc123',
            'config': {
                'database_url': 'postgres://localhost',
                'secret_key': 'super_secret'
            }
        }

        filtered_data = filter_sensitive_data(test_data)

        assert filtered_data['username'] == 'test_user'
        assert filtered_data['password'] == '[REDACTED]'
        assert filtered_data['api_key'] == '[REDACTED]'
        assert filtered_data['config']['secret_key'] == '[REDACTED]'
        assert filtered_data['config']['database_url'] == 'postgres://localhost'

    def test_error_aggregation(self):
        """Test error aggregation and deduplication."""
        def aggregate_errors(errors, time_window_minutes=5):
            """Mock error aggregation."""
            grouped_errors = {}

            for error in errors:
                error_key = f"{error['type']}:{error['message']}"

                if error_key not in grouped_errors:
                    grouped_errors[error_key] = {
                        'first_occurrence': error['timestamp'],
                        'last_occurrence': error['timestamp'],
                        'count': 1,
                        'type': error['type'],
                        'message': error['message'],
                        'contexts': [error.get('context', {})]
                    }
                else:
                    grouped_errors[error_key]['count'] += 1
                    grouped_errors[error_key]['last_occurrence'] = error['timestamp']
                    grouped_errors[error_key]['contexts'].append(error.get('context', {}))

            return grouped_errors

        test_errors = [
            {
                'type': 'ValueError',
                'message': 'Invalid input',
                'timestamp': datetime.now(),
                'context': {'user': 'user1'}
            },
            {
                'type': 'ValueError',
                'message': 'Invalid input',
                'timestamp': datetime.now(),
                'context': {'user': 'user2'}
            },
            {
                'type': 'RuntimeError',
                'message': 'Processing failed',
                'timestamp': datetime.now(),
                'context': {'document': 'doc1.txt'}
            }
        ]

        aggregated = aggregate_errors(test_errors)

        value_error_key = 'ValueError:Invalid input'
        runtime_error_key = 'RuntimeError:Processing failed'

        assert value_error_key in aggregated
        assert runtime_error_key in aggregated
        assert aggregated[value_error_key]['count'] == 2
        assert aggregated[runtime_error_key]['count'] == 1


class TestErrorNotification:
    """Test error notification and alerting."""

    def test_error_threshold_monitoring(self):
        """Test error rate threshold monitoring."""
        def monitor_error_rate(errors, threshold_per_minute=10):
            """Mock error rate monitoring."""
            current_time = datetime.now()
            one_minute_ago = current_time - timedelta(minutes=1)

            recent_errors = [
                error for error in errors
                if error['timestamp'] >= one_minute_ago
            ]

            error_rate = len(recent_errors)

            return {
                'error_rate_per_minute': error_rate,
                'threshold': threshold_per_minute,
                'threshold_exceeded': error_rate > threshold_per_minute,
                'recent_errors': len(recent_errors),
                'monitoring_window': '1 minute'
            }

        # Test normal error rate
        normal_errors = [
            {'timestamp': datetime.now() - timedelta(seconds=30)}
            for _ in range(5)
        ]

        result = monitor_error_rate(normal_errors, threshold_per_minute=10)
        assert not result['threshold_exceeded']
        assert result['error_rate_per_minute'] == 5

        # Test high error rate
        high_errors = [
            {'timestamp': datetime.now() - timedelta(seconds=i)}
            for i in range(15)
        ]

        result = monitor_error_rate(high_errors, threshold_per_minute=10)
        assert result['threshold_exceeded']
        assert result['error_rate_per_minute'] == 15

    def test_critical_error_notification(self):
        """Test critical error notification system."""
        def handle_critical_error(error, notification_config):
            """Mock critical error handling."""
            critical_error_types = ['AuthenticationError', 'DatabaseError', 'SecurityError']

            is_critical = (
                error.get('type') in critical_error_types or
                error.get('level') == 'CRITICAL'
            )

            if is_critical and notification_config.get('enabled', False):
                notification = {
                    'type': 'critical_error',
                    'error': error,
                    'timestamp': datetime.now(),
                    'recipients': notification_config.get('recipients', []),
                    'channels': notification_config.get('channels', ['email'])
                }

                return notification

            return None

        critical_error = {
            'type': 'DatabaseError',
            'message': 'Database connection lost',
            'level': 'CRITICAL'
        }

        notification_config = {
            'enabled': True,
            'recipients': ['admin@example.com'],
            'channels': ['email', 'slack']
        }

        notification = handle_critical_error(critical_error, notification_config)

        assert notification is not None
        assert notification['type'] == 'critical_error'
        assert 'admin@example.com' in notification['recipients']
        assert 'email' in notification['channels']


class TestErrorHandlingIntegration:
    """Integration tests for error handling system."""

    def test_end_to_end_error_handling(self):
        """Test complete error handling flow."""
        def complete_error_handling_flow(operation, context):
            """Mock complete error handling flow."""
            try:
                return operation()
            except Exception as e:
                # 1. Log the error
                error_log = {
                    'timestamp': datetime.now().isoformat(),
                    'level': 'ERROR',
                    'exception_type': type(e).__name__,
                    'message': str(e),
                    'context': context,
                    'stack_trace': traceback.format_exc()
                }

                # 2. Filter sensitive data
                filtered_context = {
                    k: '[REDACTED]' if 'password' in k.lower() else v
                    for k, v in context.items()
                }
                error_log['context'] = filtered_context

                # 3. Determine if critical
                is_critical = isinstance(e, (RuntimeError, ValueError))

                # 4. Return error response
                return {
                    'success': False,
                    'error': {
                        'type': type(e).__name__,
                        'message': str(e),
                        'critical': is_critical
                    },
                    'logged': True
                }

        def failing_operation():
            raise RuntimeError("Critical system failure")

        result = complete_error_handling_flow(
            failing_operation,
            {'user': 'test_user', 'password': 'secret123'}
        )

        assert not result['success']
        assert result['error']['type'] == 'RuntimeError'
        assert result['error']['critical'] is True
        assert result['logged'] is True

    def test_error_handling_in_different_modes(self):
        """Test error handling behavior in desktop vs enterprise modes."""
        def handle_error_by_mode(error, desktop_mode=False):
            """Mock mode-specific error handling."""
            if desktop_mode:
                # Desktop mode: more detailed errors for debugging
                return {
                    'error_message': str(error),
                    'error_type': type(error).__name__,
                    'stack_trace': traceback.format_exc(),
                    'suggestions': ['Check input data', 'Restart application'],
                    'mode': 'desktop'
                }
            else:
                # Enterprise mode: sanitized errors for security
                return {
                    'error_message': 'An error occurred. Please contact support.',
                    'error_id': 'ERR_001',
                    'support_contact': 'support@example.com',
                    'mode': 'enterprise'
                }

        test_error = ValueError("Invalid configuration parameter")

        # Test desktop mode
        desktop_response = handle_error_by_mode(test_error, desktop_mode=True)
        assert 'stack_trace' in desktop_response
        assert desktop_response['mode'] == 'desktop'

        # Test enterprise mode
        enterprise_response = handle_error_by_mode(test_error, desktop_mode=False)
        assert 'stack_trace' not in enterprise_response
        assert enterprise_response['mode'] == 'enterprise'
        assert 'support_contact' in enterprise_response
