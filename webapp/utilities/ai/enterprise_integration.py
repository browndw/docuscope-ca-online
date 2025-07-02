"""
Enterprise AI integration helpers.

This module provides helper functions to integrate the circuit breaker and routing
system with existing AI functions while maintaining backward compatibility.
"""

import asyncio
import json
from typing import Any, Dict

from webapp.utilities.ai.enterprise_circuit_breaker import (
    get_circuit_breaker,
    CircuitBreakerError
)
from webapp.utilities.ai.enterprise_router import get_ai_router
from webapp.utilities.configuration.logging_config import get_logger

logger = get_logger()


def determine_api_key_type(
    desktop_mode: bool,
    api_key: str,
    community_key_available: bool = True
) -> str:
    """
    Determine the type of API key being used.

    Parameters
    ----------
    desktop_mode : bool
        Whether in desktop mode
    api_key : str
        The API key being used
    community_key_available : bool
        Whether community key is available in secrets

    Returns
    -------
    str
        Key type: "community" or "individual"
    """
    if desktop_mode:
        return "individual"

    if not community_key_available:
        return "individual"

    try:
        import streamlit as st
        if "openai" in st.secrets and "api_key" in st.secrets["openai"]:
            community_key = st.secrets["openai"]["api_key"]
            if api_key == community_key:
                return "community"
    except Exception:
        pass

    return "individual"


def create_request_hash(prompt: str, model_params: Dict[str, Any]) -> str:
    """
    Create a hash for request deduplication.

    Parameters
    ----------
    prompt : str
        The user prompt
    model_params : Dict[str, Any]
        Model parameters

    Returns
    -------
    str
        Request hash for deduplication
    """
    request_data = {
        "prompt": prompt,
        "model": model_params.get("model", ""),
        "temperature": model_params.get("temperature", 0.1),
        "max_tokens": model_params.get("max_tokens", 3000)
    }
    return json.dumps(request_data, sort_keys=True)


async def protected_openai_call(
    api_key: str,
    key_type: str,
    user_id: str,
    session_id: str,
    request_data: str,
    openai_func: callable,
    *args,
    **kwargs
) -> Any:
    """
    Execute OpenAI API call with circuit breaker protection.

    Parameters
    ----------
    api_key : str
        OpenAI API key
    key_type : str
        Type of key ("community" or "individual")
    user_id : str
        User identifier
    session_id : str
        Session identifier
    request_data : str
        Serialized request data for deduplication
    openai_func : callable
        OpenAI function to call
    *args, **kwargs
        Arguments for the OpenAI function

    Returns
    -------
    Any
        Result from OpenAI API

    Raises
    ------
    CircuitBreakerError
        When circuit breaker is open
    """
    circuit_breaker = get_circuit_breaker(key_type)

    # Check if circuit breaker allows requests
    if not circuit_breaker.is_available():
        logger.warning(f"Circuit breaker open for {key_type} key")
        raise CircuitBreakerError(f"Circuit breaker is open for {key_type} key")

    # Use circuit breaker protection
    with circuit_breaker.call():
        # For community keys, use router with deduplication
        if key_type == "community":
            router = get_ai_router()

            # Convert sync function to async if needed
            if not asyncio.iscoroutinefunction(openai_func):
                async def async_openai_func(*args, **kwargs):
                    return openai_func(*args, **kwargs)
                actual_func = async_openai_func
            else:
                actual_func = openai_func

            return await router.route_request(
                user_id=user_id,
                session_id=session_id,
                request_data=request_data,
                request_func=actual_func,
                key_type=key_type,
                *args,
                **kwargs
            )
        else:
            # Individual keys - direct call with circuit breaker protection
            return openai_func(*args, **kwargs)


def with_circuit_breaker_protection(
    api_key: str,
    desktop_mode: bool,
    user_id: str,
    session_id: str,
    prompt: str,
    model_params: Dict[str, Any]
):
    """
    Decorator factory for adding circuit breaker protection to AI functions.

    Parameters
    ----------
    api_key : str
        OpenAI API key
    desktop_mode : bool
        Whether in desktop mode
    user_id : str
        User identifier
    session_id : str
        Session identifier
    prompt : str
        User prompt
    model_params : Dict[str, Any]
        Model parameters

    Returns
    -------
    callable
        Decorator function
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            try:
                # Determine key type
                key_type = determine_api_key_type(desktop_mode, api_key)

                # Create request hash for deduplication
                request_hash = create_request_hash(prompt, model_params)

                logger.debug(f"Executing {func.__name__} with {key_type} key")

                # Execute with protection
                return await protected_openai_call(
                    api_key=api_key,
                    key_type=key_type,
                    user_id=user_id,
                    session_id=session_id,
                    request_data=request_hash,
                    openai_func=func,
                    *args,
                    **kwargs
                )

            except CircuitBreakerError as e:
                logger.error(f"Circuit breaker blocked {func.__name__}: {e}")

                # Return appropriate error based on key type
                if key_type == "community":
                    error_msg = (
                        "Community API is temporarily unavailable due to high load. "
                        "Please try again in a moment or provide your own OpenAI API key."
                    )
                else:
                    error_msg = (
                        "API temporarily unavailable. Please check your API key "
                        "and try again in a moment."
                    )

                return {"type": "error", "value": error_msg}

            except Exception as e:
                logger.error(f"Error in {func.__name__}: {e}")
                return {"type": "error", "value": f"An error occurred: {str(e)}"}

        return wrapper
    return decorator


def get_circuit_breaker_status() -> Dict[str, Any]:
    """
    Get status of all circuit breakers for monitoring.

    Returns
    -------
    Dict[str, Any]
        Status information for all circuit breakers
    """
    try:
        from webapp.utilities.ai.enterprise_circuit_breaker import (
            get_circuit_breaker_manager
        )

        manager = get_circuit_breaker_manager()
        metrics = manager.get_all_metrics()

        # Add router stats if available
        try:
            router = get_ai_router()
            router_stats = router.get_router_stats()
            metrics["router"] = router_stats
        except Exception as e:
            logger.warning(f"Could not get router stats: {e}")
            metrics["router"] = {"error": str(e)}

        return metrics

    except Exception as e:
        logger.error(f"Error getting circuit breaker status: {e}")
        return {"error": str(e)}


def reset_circuit_breakers():
    """Reset all circuit breakers (for administrative use)."""
    try:
        from webapp.utilities.ai.enterprise_circuit_breaker import (
            get_circuit_breaker_manager
        )

        manager = get_circuit_breaker_manager()
        manager.reset_all()

        logger.info("All circuit breakers reset")

    except Exception as e:
        logger.error(f"Error resetting circuit breakers: {e}")
        raise


def make_protected_openai_call(
    api_key: str,
    request_params: Dict[str, Any],
    request_type: str = "chat_completion",
    cache_key: str = None
) -> Any:
    """
    Synchronous wrapper for protected OpenAI calls with circuit breaker protection.

    This function provides circuit breaker protection and request routing for
    direct OpenAI API calls in a synchronous context.

    Parameters
    ----------
    api_key : str
        OpenAI API key
    request_params : Dict[str, Any]
        Parameters for the OpenAI API call
    request_type : str
        Type of request (e.g., "chat_completion")
    cache_key : str, optional
        Cache key for request deduplication

    Returns
    -------
    Any
        OpenAI API response or error dict
    """
    import openai
    from webapp.config.unified import get_config
    from webapp.utilities.ai.enterprise_circuit_breaker import CircuitBreakerManager

    try:
        # Get configuration
        desktop_mode = get_config('desktop_mode', 'global', True)

        # Determine key type
        key_type = determine_api_key_type(desktop_mode, api_key)

        # Get circuit breaker manager
        cb_manager = CircuitBreakerManager()

        # Check circuit breaker status
        breaker = cb_manager.get_breaker(key_type)
        breaker_status = breaker.state.value
        if breaker_status == "open":
            return {
                "type": "error",
                "value": (f"Circuit breaker is open for {key_type} key - "
                          "service temporarily unavailable")
            }

        # Make the OpenAI API call
        openai.api_key = api_key
        # Get circuit breaker for protection
        breaker = cb_manager.get_breaker(key_type)

        # Execute with circuit breaker protection
        with breaker.call():
            # Execute the request based on type
            if request_type == "chat_completion":
                response = openai.chat.completions.create(**request_params)
            else:
                raise ValueError(f"Unsupported request type: {request_type}")

        return response

    except Exception as e:
        # Return error response
        error_msg = str(e)
        if "rate_limit" in error_msg.lower():
            return {
                "type": "error",
                "value": f"Rate limit exceeded for {key_type} key"
            }
        elif "quota" in error_msg.lower():
            return {
                "type": "error",
                "value": f"Quota exceeded for {key_type} key"
            }
        else:
            return {
                "type": "error",
                "value": f"API call failed: {error_msg}"
            }


def enterprise_ai_call(function_name: str):
    """
    Decorator for AI functions to add enterprise-grade protection.

    This decorator adds circuit breaker protection, request routing,
    and error handling to AI function calls like pandabot_user_query.

    Parameters
    ----------
    function_name : str
        Name of the function for logging and monitoring

    Returns
    -------
    callable
        Decorated function with enterprise protection
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                # Import here to avoid circular imports
                from webapp.config.unified import get_config
                from webapp.utilities.ai.enterprise_circuit_breaker import (
                    CircuitBreakerManager
                )
                from webapp.utilities.configuration.logging_config import get_logger

                logger = get_logger()

                # Extract common parameters
                api_key = kwargs.get('api_key') or (
                    args[1] if len(args) > 1 else None
                )

                if not api_key:
                    raise ValueError("API key is required for enterprise AI calls")

                # Get configuration and determine key type
                desktop_mode = get_config('desktop_mode', 'global', True)
                key_type = determine_api_key_type(desktop_mode, api_key)

                # Get circuit breaker manager
                cb_manager = CircuitBreakerManager()

                # Check circuit breaker status before proceeding
                breaker = cb_manager.get_breaker(key_type)
                breaker_status = breaker.state.value
                if breaker_status == "open":
                    # Circuit breaker is open - fail fast
                    logger.warning(
                        f"Circuit breaker open for {key_type} key in {function_name}"
                    )
                    raise Exception(f"Circuit breaker is open - {key_type} key protected")

                # Execute the original function with circuit breaker protection
                breaker = cb_manager.get_breaker(key_type)
                with breaker.call():
                    result = func(*args, **kwargs)

                return result

            except Exception as e:
                # Handle circuit breaker and other enterprise-level errors
                error_msg = str(e).lower()
                if "circuit breaker" in error_msg:
                    logger.warning(
                        f"Circuit breaker protection triggered for {function_name}"
                    )
                elif "rate limit" in error_msg:
                    logger.warning(f"Rate limit hit for {function_name}")
                elif "quota" in error_msg:
                    logger.info(f"Quota exceeded for {function_name}")
                else:
                    logger.error(f"Enterprise AI call failed for {function_name}: {e}")

                # Re-raise the exception to be handled by the calling function
                raise

        return wrapper
    return decorator
