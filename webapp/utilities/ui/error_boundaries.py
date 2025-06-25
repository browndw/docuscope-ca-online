"""
Error boundary utilities for graceful degradation and asset failure handling.

This module provides utilities to handle UI failures, asset loading errors,
and other runtime issues with graceful fallbacks.
"""

import functools
import traceback
from typing import Any, Callable, Dict, Optional
import streamlit as st


class UIErrorBoundary:
    """
    Error boundary for UI components with graceful degradation.
    """

    def __init__(self, fallback_message: str = "Component temporarily unavailable",
                 show_error_details: bool = False):
        """
        Initialize the error boundary.

        Parameters
        ----------
        fallback_message : str
            Message to show when component fails
        show_error_details : bool
            Whether to show detailed error information
        """
        self.fallback_message = fallback_message
        self.show_error_details = show_error_details

    def __call__(self, func: Callable) -> Callable:
        """
        Decorator to wrap functions with error boundary.

        Parameters
        ----------
        func : Callable
            Function to wrap

        Returns
        -------
        Callable
            Wrapped function with error handling
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                self._handle_error(e, func.__name__)
                return None
        return wrapper

    def _handle_error(self, error: Exception, component_name: str):
        """Handle error with appropriate UI feedback."""
        if self.show_error_details:
            st.error(
                f"Error in {component_name}: {str(error)}\n\n"
                f"Details: {traceback.format_exc()}",
                icon=":material/error:"
            )
        else:
            st.warning(
                f"{self.fallback_message} ({component_name})",
                icon=":material/warning:"
            )


def with_fallback(fallback_value: Any = None,
                  error_message: str = "Operation failed"):
    """
    Decorator to provide fallback value on function failure.

    Parameters
    ----------
    fallback_value : Any
        Value to return on failure
    error_message : str
        Error message to display

    Returns
    -------
    callable
        Decorated function with fallback
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                st.warning(f"{error_message}: {str(e)}", icon=":material/warning:")
                return fallback_value
        return wrapper
    return decorator


def safe_asset_loader(asset_path: str, asset_type: str = 'image',
                      fallback_content: Optional[str] = None) -> Optional[str]:
    """
    Safely load assets with fallback handling.

    Parameters
    ----------
    asset_path : str
        Path to the asset
    asset_type : str
        Type of asset ('image', 'css', 'js', etc.)
    fallback_content : Optional[str]
        Fallback content if asset fails to load

    Returns
    -------
    Optional[str]
        Asset content or fallback
    """
    try:
        if asset_type == 'image':
            # For images, just return path if file exists
            import os
            if os.path.exists(asset_path):
                return asset_path
            else:
                st.warning(
                    f"Image not found: {asset_path}",
                    icon=":material/image_not_supported:"
                )
                return fallback_content
        elif asset_type in ['css', 'js', 'html']:
            # For text-based assets, read content
            with open(asset_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            # Generic file reading
            with open(asset_path, 'r', encoding='utf-8') as f:
                return f.read()
    except Exception as e:
        st.warning(
            f"Failed to load {asset_type} asset: {asset_path}. Error: {str(e)}",
            icon=":material/error:"
        )
        return fallback_content


class SafeComponentRenderer:
    """
    Safe component renderer with error boundaries and fallbacks.
    """

    @staticmethod
    def safe_markdown(content: str, fallback: str = "Content unavailable"):
        """Safely render markdown with fallback."""
        try:
            if content:
                st.markdown(content)
            else:
                st.info(fallback)
        except Exception as e:
            st.warning(f"Failed to render content: {str(e)}")

    @staticmethod
    def safe_dataframe(df, fallback_message: str = "Data unavailable", **kwargs):
        """Safely render dataframe with fallback."""
        try:
            if df is not None and (hasattr(df, 'empty') and not df.empty):
                st.dataframe(df, **kwargs)
            else:
                st.info(fallback_message)
        except Exception as e:
            st.error(f"Failed to render data: {str(e)}")

    @staticmethod
    def safe_plotly_chart(fig, fallback_message: str = "Chart unavailable", **kwargs):
        """Safely render plotly chart with fallback."""
        try:
            if fig is not None:
                # Set default use_container_width if not specified
                if 'use_container_width' not in kwargs:
                    kwargs['use_container_width'] = True
                st.plotly_chart(fig, **kwargs)
            else:
                st.info(fallback_message)
        except Exception as e:
            st.error(f"Failed to render chart: {str(e)}")

    @staticmethod
    def safe_image(image_path: str, fallback_message: str = "Image unavailable", **kwargs):
        """Safely display image with fallback."""
        try:
            import os
            if os.path.exists(image_path):
                st.image(image_path, **kwargs)
            else:
                st.info(fallback_message)
        except Exception as e:
            st.warning(f"Failed to load image: {str(e)}")


def graceful_component(component_name: str, show_errors: bool = False):
    """
    Decorator for graceful component rendering.

    Parameters
    ----------
    component_name : str
        Name of the component for error messages
    show_errors : bool
        Whether to show detailed error messages

    Returns
    -------
    callable
        Decorated function with graceful error handling
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if show_errors:
                    st.error(
                        f"Error in {component_name}: {str(e)}",
                        icon=":material/error:"
                    )
                else:
                    st.warning(
                        f"{component_name} temporarily unavailable",
                        icon=":material/warning:"
                    )
                return None
        return wrapper
    return decorator


def safe_session_operation(func: Callable) -> Callable:
    """
    Decorator for safe session state operations.

    Parameters
    ----------
    func : Callable
        Function that operates on session state

    Returns
    -------
    Callable
        Wrapped function with session safety
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.warning(
                f"Session operation failed: {str(e)}. "
                "Please refresh the page if issues persist.",
                icon=":material/warning:"
            )
            return None
    return wrapper


class AssetManager:
    """
    Manager for safe asset loading with caching and fallbacks.
    """

    def __init__(self):
        self._asset_cache: Dict[str, Any] = {}
        self._failed_assets: set = set()

    def load_asset(self, asset_path: str, asset_type: str = 'auto',
                   cache: bool = True) -> Optional[Any]:
        """
        Load an asset with caching and error handling.

        Parameters
        ----------
        asset_path : str
            Path to the asset
        asset_type : str
            Type of asset or 'auto' to detect
        cache : bool
            Whether to cache the asset

        Returns
        -------
        Optional[Any]
            Loaded asset or None if failed
        """
        # Check if asset previously failed
        if asset_path in self._failed_assets:
            return None

        # Check cache
        if cache and asset_path in self._asset_cache:
            return self._asset_cache[asset_path]

        try:
            # Detect asset type if auto
            if asset_type == 'auto':
                if asset_path.endswith(('.png', '.jpg', '.jpeg', '.gif', '.svg')):
                    asset_type = 'image'
                elif asset_path.endswith('.css'):
                    asset_type = 'css'
                elif asset_path.endswith('.js'):
                    asset_type = 'js'
                else:
                    asset_type = 'text'

            # Load asset
            content = safe_asset_loader(asset_path, asset_type)

            # Cache if requested and successful
            if cache and content is not None:
                self._asset_cache[asset_path] = content

            return content

        except Exception as e:
            self._failed_assets.add(asset_path)
            st.warning(f"Failed to load asset {asset_path}: {str(e)}")
            return None

    def clear_cache(self):
        """Clear the asset cache."""
        self._asset_cache.clear()
        self._failed_assets.clear()


# Global asset manager instance
asset_manager = AssetManager()
