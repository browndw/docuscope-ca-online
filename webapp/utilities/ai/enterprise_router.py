"""
Enterprise AI Request Router for dual-key API management.

This module provides intelligent routing of AI requests between community
and individual keys, with queue management and load balancing.
"""

import asyncio
import hashlib
from typing import Dict, Any, Optional, Callable, Awaitable
from dataclasses import dataclass
from datetime import datetime, timezone
from queue import Queue
import threading

from webapp.utilities.core import safe_config_value
from webapp.utilities.ai.enterprise_circuit_breaker import (
    get_circuit_breaker,
    CircuitBreakerError
)

# Import centralized logging configuration and logger
from webapp.utilities.configuration.logging_config import get_logger

logger = get_logger()


@dataclass
class AIRequest:
    """Represents an AI API request."""
    request_id: str
    user_id: str
    session_id: str
    key_type: str  # "community" or "individual"
    priority: int
    request_func: Callable
    args: tuple
    kwargs: dict
    timestamp: datetime
    timeout: Optional[float] = None


class RequestDeduplicator:
    """
    Deduplicates identical AI requests to reduce API calls.

    Particularly important for community key protection during classroom scenarios
    where multiple students might ask similar questions.
    """

    def __init__(self, cache_ttl: int = 300):
        """
        Initialize request deduplicator.

        Parameters
        ----------
        cache_ttl : int
            Time-to-live for cached responses in seconds
        """
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._pending_requests: Dict[str, asyncio.Future] = {}
        self._lock = threading.RLock()

    def _generate_request_hash(self, request_data: str, key_type: str) -> str:
        """Generate hash for request deduplication."""
        # Include key_type to separate community vs individual caches
        combined = f"{key_type}:{request_data}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    async def deduplicate_request(
        self,
        request_data: str,
        key_type: str,
        request_func: Callable[..., Awaitable[Any]],
        *args,
        **kwargs
    ) -> Any:
        """
        Execute request with deduplication.

        Parameters
        ----------
        request_data : str
            Serialized request data for hashing
        key_type : str
            Type of API key being used
        request_func : Callable
            Async function to execute
        *args, **kwargs
            Arguments for the request function

        Returns
        -------
        Any
            Result from the API call
        """
        request_hash = self._generate_request_hash(request_data, key_type)

        with self._lock:
            # Check cache first
            if request_hash in self._cache:
                cache_entry = self._cache[request_hash]
                cache_age = (datetime.now(timezone.utc) - cache_entry["timestamp"]).total_seconds()  # noqa: E501
                if cache_age < self.cache_ttl:
                    logger.debug(f"Cache hit for {key_type} request: {request_hash}")
                    return cache_entry["result"]
                else:
                    # Cache expired
                    del self._cache[request_hash]

            # Check if request is already pending
            if request_hash in self._pending_requests:
                logger.debug(f"Deduplicating {key_type} request: {request_hash}")
                return await self._pending_requests[request_hash]

        # Create new request
        future = asyncio.create_task(request_func(*args, **kwargs))

        with self._lock:
            self._pending_requests[request_hash] = future

        try:
            result = await future

            # Cache successful results
            with self._lock:
                self._cache[request_hash] = {
                    "result": result,
                    "timestamp": datetime.now(timezone.utc)
                }

            logger.debug(f"Cached new {key_type} result: {request_hash}")
            return result

        except Exception as e:
            logger.error(f"Request failed for {key_type}: {e}")
            raise
        finally:
            # Clean up pending request
            with self._lock:
                self._pending_requests.pop(request_hash, None)

    def clear_cache(self):
        """Clear all cached responses."""
        with self._lock:
            self._cache.clear()
            logger.info("Request cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                "cached_responses": len(self._cache),
                "pending_requests": len(self._pending_requests),
                "cache_ttl": self.cache_ttl
            }


class EnterpriseAIRouter:
    """
    Enterprise AI request router with queue management and load balancing.

    Handles routing between community and individual keys with appropriate
    protection for the shared community resource.
    """

    def __init__(self):
        """Initialize the AI router."""
        self.deduplicator = RequestDeduplicator()
        self._request_queues: Dict[str, Queue] = {
            "community": Queue(maxsize=safe_config_value(
                'max_queue_size', 1000, 'llm.enterprise'
            )),
            "individual": Queue(maxsize=safe_config_value(
                'max_queue_size', 1000, 'llm.enterprise'
            ))
        }
        self._concurrent_limits = {
            "community": safe_config_value(
                'community_max_concurrent_requests', 5, 'llm.enterprise'
            ),
            "individual": safe_config_value(
                'individual_max_concurrent_requests', 10, 'llm.enterprise'
            )
        }
        self._active_requests = {"community": 0, "individual": 0}
        self._lock = threading.RLock()

        logger.info(f"AI Router initialized with limits: {self._concurrent_limits}")

    def determine_key_type(
        self,
        desktop_mode: bool,
        has_community_key: bool,
        has_individual_key: bool,
        quota_remaining: int
    ) -> str:
        """
        Determine which key type to use for a request.

        Parameters
        ----------
        desktop_mode : bool
            Whether in desktop mode
        has_community_key : bool
            Whether community key is available
        has_individual_key : bool
            Whether user has individual key
        quota_remaining : int
            Remaining quota for community key

        Returns
        -------
        str
            Key type to use: "community" or "individual"
        """
        # Desktop mode always uses individual key
        if desktop_mode:
            return "individual"

        # No community key available
        if not has_community_key:
            return "individual"

        # Community quota exhausted, fallback to individual if available
        if quota_remaining <= 0:
            return "individual" if has_individual_key else "community"

        # Check circuit breaker states
        community_available = get_circuit_breaker("community").is_available()
        individual_available = get_circuit_breaker("individual").is_available()

        # Prefer community key if available and quota exists
        if community_available and quota_remaining > 0:
            return "community"

        # Fallback to individual key
        if individual_available and has_individual_key:
            return "individual"

        # Last resort - use community even if circuit breaker is open
        # This will raise CircuitBreakerError but allows for proper error handling
        return "community"

    async def route_request(
        self,
        user_id: str,
        session_id: str,
        request_data: str,
        request_func: Callable[..., Awaitable[Any]],
        key_type: str,
        priority: int = 1,
        enable_deduplication: bool = True,
        *args,
        **kwargs
    ) -> Any:
        """
        Route and execute an AI request with appropriate protections.

        Parameters
        ----------
        user_id : str
            User identifier
        session_id : str
            Session identifier
        request_data : str
            Serialized request data for deduplication
        request_func : Callable
            Async function to execute
        key_type : str
            Type of key to use ("community" or "individual")
        priority : int
            Request priority (higher = more important)
        enable_deduplication : bool
            Whether to enable request deduplication
        *args, **kwargs
            Arguments for the request function

        Returns
        -------
        Any
            Result from the API call
        """
        # Check circuit breaker first
        circuit_breaker = get_circuit_breaker(key_type)
        if not circuit_breaker.is_available():
            raise CircuitBreakerError(f"Circuit breaker open for {key_type} key")

        # Apply concurrency limits
        with self._lock:
            if self._active_requests[key_type] >= self._concurrent_limits[key_type]:
                raise Exception(f"Concurrent limit reached for {key_type} key")

            self._active_requests[key_type] += 1

        try:
            # Use deduplication for community keys or when explicitly enabled
            deduplication_enabled = safe_config_value(
                'enable_request_deduplication', True, 'llm.enterprise'
            )
            if enable_deduplication and (key_type == "community" or deduplication_enabled):

                result = await self.deduplicator.deduplicate_request(
                    request_data, key_type, request_func, *args, **kwargs
                )
            else:
                # Direct execution for individual keys or when deduplication disabled
                result = await request_func(*args, **kwargs)

            return result

        finally:
            with self._lock:
                self._active_requests[key_type] -= 1

    def get_router_stats(self) -> Dict[str, Any]:
        """Get router statistics."""
        with self._lock:
            return {
                "active_requests": self._active_requests.copy(),
                "concurrent_limits": self._concurrent_limits.copy(),
                "queue_sizes": {
                    key: queue.qsize() for key, queue in self._request_queues.items()
                },
                "deduplication_stats": self.deduplicator.get_cache_stats()
            }


# Global router instance
_ai_router: Optional[EnterpriseAIRouter] = None
_router_lock = threading.Lock()


def get_ai_router() -> EnterpriseAIRouter:
    """Get or create the global AI router."""
    global _ai_router

    if _ai_router is None:
        with _router_lock:
            if _ai_router is None:
                _ai_router = EnterpriseAIRouter()

    return _ai_router


async def route_ai_request(
    user_id: str,
    session_id: str,
    request_data: str,
    request_func: Callable[..., Awaitable[Any]],
    key_type: str,
    *args,
    **kwargs
) -> Any:
    """
    Convenience function for routing AI requests.

    Parameters
    ----------
    user_id : str
        User identifier
    session_id : str
        Session identifier
    request_data : str
        Serialized request data
    request_func : Callable
        Async function to execute
    key_type : str
        Key type ("community" or "individual")
    *args, **kwargs
        Arguments for request function

    Returns
    -------
    Any
        Result from API call
    """
    router = get_ai_router()
    return await router.route_request(
        user_id, session_id, request_data, request_func, key_type, *args, **kwargs
    )
