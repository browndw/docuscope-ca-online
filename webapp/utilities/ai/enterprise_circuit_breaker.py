"""
Enterprise Circuit Breaker for AI API calls.

This module provides circuit breaker patterns specifically designed for enterprise
scale AI operations with dual-key support (community vs individual keys).
Focuses on protecting the community key resource during high-load classroom scenarios.
"""

import threading
from contextlib import contextmanager
from typing import Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta

from webapp.utilities.core import safe_config_value

# Import centralized logging configuration and logger
from webapp.utilities.configuration.logging_config import get_logger

logger = get_logger()


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, blocking requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerMetrics:
    """Metrics for circuit breaker monitoring."""
    total_requests: int = 0
    failed_requests: int = 0
    successful_requests: int = 0
    circuit_opens: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    current_failures: int = 0
    current_successes: int = 0


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""
    pass


class EnterpriseCircuitBreaker:
    """
    Enterprise-grade circuit breaker with dual-key support.

    Provides different policies for community keys (high protection)
    vs individual keys (more permissive).
    """

    def __init__(self, key_type: str = "individual"):
        """
        Initialize circuit breaker for specific key type.

        Parameters
        ----------
        key_type : str
            Either "community" or "individual"
        """
        self.key_type = key_type
        self.state = CircuitBreakerState.CLOSED
        self.metrics = CircuitBreakerMetrics()
        self._lock = threading.RLock()

        # Load configuration based on key type
        self.failure_threshold = safe_config_value(
            f'{key_type}_circuit_breaker_failure_threshold',
            default=3 if key_type == "community" else 5,
            config_type='llm.enterprise'
        )
        self.recovery_timeout = safe_config_value(
            f'{key_type}_circuit_breaker_recovery_timeout',
            default=30 if key_type == "community" else 60,
            config_type='llm.enterprise'
        )
        self.success_threshold = safe_config_value(
            f'{key_type}_circuit_breaker_success_threshold',
            default=2 if key_type == "community" else 3,
            config_type='llm.enterprise'
        )
        self.request_timeout = safe_config_value(
            f'{key_type}_request_timeout_seconds',
            default=10 if key_type == "community" else 20,
            config_type='llm.enterprise'
        )

    @contextmanager
    def call(self):
        """
        Context manager for circuit breaker protected calls.

        Usage:
            with circuit_breaker.call():
                result = api_call()

        Raises
        ------
        CircuitBreakerError
            When circuit is open
        """
        with self._lock:
            self._check_state()

            if self.state == CircuitBreakerState.OPEN:
                logger.warning(f"{self.key_type} circuit breaker is OPEN")
                raise CircuitBreakerError(
                    f"Circuit breaker is open for {self.key_type} key"
                )

            self.metrics.total_requests += 1

        try:
            yield
            self._on_success()
        except Exception as e:
            self._on_failure(e)
            raise

    def _check_state(self):
        """Check and update circuit breaker state."""
        current_time = datetime.now(timezone.utc)

        if self.state == CircuitBreakerState.OPEN:
            if (self.metrics.last_failure_time and
                    current_time - self.metrics.last_failure_time >
                    timedelta(seconds=self.recovery_timeout)):

                logger.info(f"Moving {self.key_type} circuit breaker to HALF_OPEN")
                self.state = CircuitBreakerState.HALF_OPEN
                self.metrics.current_successes = 0

    def _on_success(self):
        """Handle successful API call."""
        with self._lock:
            self.metrics.successful_requests += 1
            self.metrics.last_success_time = datetime.now(timezone.utc)

            if self.state == CircuitBreakerState.HALF_OPEN:
                self.metrics.current_successes += 1
                if self.metrics.current_successes >= self.success_threshold:
                    logger.info(f"Moving {self.key_type} circuit breaker to CLOSED")
                    self.state = CircuitBreakerState.CLOSED
                    self.metrics.current_failures = 0
            else:
                # Reset failure count on success in CLOSED state
                self.metrics.current_failures = 0

    def _on_failure(self, error: Exception):
        """Handle failed API call."""
        with self._lock:
            self.metrics.failed_requests += 1
            self.metrics.current_failures += 1
            self.metrics.last_failure_time = datetime.now(timezone.utc)

            logger.warning(
                f"{self.key_type} API call failed: {error}. "
                f"Failures: {self.metrics.current_failures}/{self.failure_threshold}"
            )

            if self.metrics.current_failures >= self.failure_threshold:
                if self.state != CircuitBreakerState.OPEN:
                    logger.error(f"Opening {self.key_type} circuit breaker")
                    self.state = CircuitBreakerState.OPEN
                    self.metrics.circuit_opens += 1

    def get_metrics(self) -> Dict[str, Any]:
        """Get current circuit breaker metrics."""
        with self._lock:
            return {
                "key_type": self.key_type,
                "state": self.state.value,
                "total_requests": self.metrics.total_requests,
                "failed_requests": self.metrics.failed_requests,
                "successful_requests": self.metrics.successful_requests,
                "circuit_opens": self.metrics.circuit_opens,
                "failure_rate": (
                    self.metrics.failed_requests / self.metrics.total_requests
                    if self.metrics.total_requests > 0 else 0
                ),
                "current_failures": self.metrics.current_failures,
                "last_failure_time": (
                    self.metrics.last_failure_time.isoformat()
                    if self.metrics.last_failure_time else None
                ),
                "last_success_time": (
                    self.metrics.last_success_time.isoformat()
                    if self.metrics.last_success_time else None
                )
            }

    def reset(self):
        """Reset circuit breaker to CLOSED state."""
        with self._lock:
            logger.info(f"Resetting {self.key_type} circuit breaker")
            self.state = CircuitBreakerState.CLOSED
            self.metrics.current_failures = 0
            self.metrics.current_successes = 0

    def is_available(self) -> bool:
        """Check if circuit breaker allows requests."""
        with self._lock:
            self._check_state()
            return self.state != CircuitBreakerState.OPEN


class CircuitBreakerManager:
    """
    Manager for multiple circuit breakers (community vs individual keys).

    Provides a centralized interface for managing both key types.
    """

    def __init__(self):
        """Initialize circuit breaker manager."""
        self._breakers: Dict[str, EnterpriseCircuitBreaker] = {}
        self._lock = threading.RLock()

        # Initialize breakers for both key types
        self._breakers["community"] = EnterpriseCircuitBreaker("community")
        self._breakers["individual"] = EnterpriseCircuitBreaker("individual")

    def get_breaker(self, key_type: str) -> EnterpriseCircuitBreaker:
        """
        Get circuit breaker for specific key type.

        Parameters
        ----------
        key_type : str
            Either "community" or "individual"

        Returns
        -------
        EnterpriseCircuitBreaker
            Circuit breaker instance for the key type
        """
        if key_type not in self._breakers:
            raise ValueError(f"Unknown key type: {key_type}")

        return self._breakers[key_type]

    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all circuit breakers."""
        with self._lock:
            return {
                key_type: breaker.get_metrics()
                for key_type, breaker in self._breakers.items()
            }

    def reset_all(self):
        """Reset all circuit breakers."""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()
            logger.info("All circuit breakers reset")

    def is_key_type_available(self, key_type: str) -> bool:
        """Check if specific key type is available."""
        return self.get_breaker(key_type).is_available()


# Global circuit breaker manager instance
_circuit_breaker_manager: Optional[CircuitBreakerManager] = None
_manager_lock = threading.Lock()


def get_circuit_breaker_manager() -> CircuitBreakerManager:
    """Get or create the global circuit breaker manager."""
    global _circuit_breaker_manager

    if _circuit_breaker_manager is None:
        with _manager_lock:
            if _circuit_breaker_manager is None:
                _circuit_breaker_manager = CircuitBreakerManager()

    return _circuit_breaker_manager


def get_circuit_breaker(key_type: str) -> EnterpriseCircuitBreaker:
    """
    Convenience function to get circuit breaker for key type.

    Parameters
    ----------
    key_type : str
        Either "community" or "individual"

    Returns
    -------
    EnterpriseCircuitBreaker
        Circuit breaker instance
    """
    return get_circuit_breaker_manager().get_breaker(key_type)


def circuit_breaker_protected_call(key_type: str):
    """
    Decorator for circuit breaker protected API calls.

    Parameters
    ----------
    key_type : str
        Either "community" or "individual"

    Usage
    ------
    @circuit_breaker_protected_call("community")
    def make_api_call():
        return openai_api_call()
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            breaker = get_circuit_breaker(key_type)
            with breaker.call():
                return func(*args, **kwargs)
        return wrapper
    return decorator
