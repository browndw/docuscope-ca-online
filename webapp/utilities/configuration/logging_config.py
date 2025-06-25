"""
Centralized logging configuration for the corpus analysis application.

This module provides standardized logging setup that can be imported and used
across all parts of the application, ensuring consistent error reporting,
log formatting, and file management for both desktop and online versions.
"""

import pathlib
from typing import Optional, Dict, Any
from loguru import logger


class LoggingConfig:
    """
    Centralized logging configuration manager.

    Provides standardized logging setup with consistent formatting,
    rotation, and retention policies across the entire application.
    """

    # Default configuration
    DEFAULT_ROTATION = "10 MB"
    DEFAULT_RETENTION = "10 days"
    DEFAULT_LEVEL = "ERROR"
    DEFAULT_FORMAT = ("{time:YYYY-MM-DD HH:mm:ss} | {level} | "
                      "{name}:{function}:{line} | {message}")

    def __init__(self, base_log_dir: Optional[pathlib.Path] = None):
        """
        Initialize logging configuration.

        Parameters
        ----------
        base_log_dir : pathlib.Path, optional
            Base directory for log files. If None, uses webapp/logs/
        """
        if base_log_dir is None:
            # Default to webapp/logs from the utilities directory
            self.log_dir = pathlib.Path(__file__).parents[2] / "logs"
        else:
            self.log_dir = base_log_dir

        # Ensure log directory exists
        self.log_dir.mkdir(exist_ok=True)

        # Track added handlers to avoid duplicates
        self._added_handlers: Dict[str, Any] = {}

    def setup_logger(self,
                     log_name: str,
                     level: str = None,
                     rotation: str = None,
                     retention: str = None,
                     format_string: str = None) -> None:
        """
        Set up a logger with the specified configuration.

        Parameters
        ----------
        log_name : str
            Name of the log file (without .log extension)
        level : str, optional
            Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        rotation : str, optional
            When to rotate log files (e.g., "10 MB", "1 day")
        retention : str, optional
            How long to keep old log files (e.g., "10 days", "1 week")
        format_string : str, optional
            Custom log format string
        """
        # Use defaults if not specified
        level = level or self.DEFAULT_LEVEL
        rotation = rotation or self.DEFAULT_ROTATION
        retention = retention or self.DEFAULT_RETENTION
        format_string = format_string or self.DEFAULT_FORMAT

        log_file = self.log_dir / f"{log_name}.log"

        # Check if this logger is already configured
        handler_key = str(log_file)
        if handler_key in self._added_handlers:
            return

        # Add the logger
        handler_id = logger.add(
            log_file,
            rotation=rotation,
            retention=retention,
            level=level,
            format=format_string
        )

        # Track this handler
        self._added_handlers[handler_key] = handler_id

    def setup_module_logging(self, module_type: str, module_name: str) -> None:
        """
        Set up logging for any module type with consistent naming.

        Parameters
        ----------
        module_type : str
            Type of module ('ai', 'page', 'utility', 'debug')
        module_name : str
            Name of the specific module
        """
        if module_type == "ai":
            self.setup_logger(f"{module_name}_error")
        elif module_type == "page":
            self.setup_logger(f"page_{module_name}_error")
        elif module_type == "utility":
            self.setup_logger(f"utility_{module_name}_error")
        elif module_type == "debug":
            self.setup_logger(
                f"debug_{module_name}",
                level="DEBUG",
                rotation="5 MB",
                retention="3 days"
            )
        else:
            raise ValueError(f"Unknown module_type: {module_type}")

    def setup_ai_logging(self) -> None:
        """Set up logging for AI modules (plotbot and pandabot)."""
        self.setup_module_logging("ai", "plotbot")
        self.setup_module_logging("ai", "pandabot")

    def setup_page_logging(self, page_name: str) -> None:
        """
        Set up logging for a specific page.

        Parameters
        ----------
        page_name : str
            Name of the page (e.g., "corpus_loading", "plotting")
        """
        self.setup_module_logging("page", page_name)

    def setup_utility_logging(self, utility_name: str) -> None:
        """
        Set up logging for a specific utility module.

        Parameters
        ----------
        utility_name : str
            Name of the utility (e.g., "analysis", "storage", "exports")
        """
        self.setup_module_logging("utility", utility_name)

    def setup_debug_logging(self, module_name: str) -> None:
        """
        Set up debug-level logging for development.

        Parameters
        ----------
        module_name : str
            Name of the module for debug logging
        """
        self.setup_module_logging("debug", module_name)

    def get_log_directory(self) -> pathlib.Path:
        """Get the log directory path."""
        return self.log_dir

    def list_log_files(self) -> list[pathlib.Path]:
        """List all log files in the log directory."""
        return list(self.log_dir.glob("*.log"))

    def cleanup_old_logs(self) -> None:
        """
        Manual cleanup of very old log files.

        Note: Automatic cleanup is handled by loguru's retention setting,
        but this can be used for manual cleanup if needed.
        """
        # This is primarily handled by loguru's retention settings
        # but could be extended for custom cleanup logic
        pass


# Global logging configuration instance
_global_logging_config: Optional[LoggingConfig] = None


def get_logging_config() -> LoggingConfig:
    """
    Get the global logging configuration instance.

    Returns
    -------
    LoggingConfig
        The global logging configuration instance
    """
    global _global_logging_config
    if _global_logging_config is None:
        _global_logging_config = LoggingConfig()
    return _global_logging_config


def setup_module_logging(module_type: str, module_name: str) -> None:
    """
    Convenience function to set up module logging with unified interface.

    Parameters
    ----------
    module_type : str
        Type of module ('ai', 'page', 'utility', 'debug')
    module_name : str
        Name of the specific module
    """
    config = get_logging_config()
    config.setup_module_logging(module_type, module_name)


def setup_ai_logging() -> None:
    """Convenience function to set up AI module logging."""
    config = get_logging_config()
    config.setup_ai_logging()


def setup_page_logging(page_name: str) -> None:
    """
    Convenience function to set up page-specific logging.

    Parameters
    ----------
    page_name : str
        Name of the page
    """
    config = get_logging_config()
    config.setup_page_logging(page_name)


def setup_utility_logging(utility_name: str) -> None:
    """
    Convenience function to set up utility-specific logging.

    Parameters
    ----------
    utility_name : str
        Name of the utility module
    """
    config = get_logging_config()
    config.setup_utility_logging(utility_name)


def setup_debug_logging(module_name: str) -> None:
    """
    Convenience function to set up debug logging.

    Parameters
    ----------
    module_name : str
        Name of the module for debug logging
    """
    config = get_logging_config()
    config.setup_debug_logging(module_name)


def get_log_directory() -> pathlib.Path:
    """Get the log directory path."""
    config = get_logging_config()
    return config.get_log_directory()


def get_logger():
    """
    Get the configured logger instance.

    Returns
    -------
    loguru.Logger
        The configured logger instance
    """
    return logger


# Pre-configure AI logging when module is imported
setup_ai_logging()
