"""
Asynchronous storage utilities for non-blocking Firestore operations.

This module provides background processing for analytics and research data
that should not block the user experience.
"""


import atexit
import queue
import threading
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timezone

from webapp.utilities.storage import add_message, add_plot
from webapp.config.static_config import get_static_value

# Import centralized logging configuration and logger
from webapp.utilities.configuration.logging_config import get_logger

logger = get_logger()


@dataclass
class StorageTask:
    """Represents a storage task to be processed asynchronously."""
    task_type: str  # 'message', 'plot'
    data: Dict[str, Any]
    timestamp: datetime
    retry_count: int = 0


class AsyncStorageManager:
    """
    Manages asynchronous storage operations for Firestore.

    This class handles queuing and background processing of storage operations
    that are important for analytics but should not block user interactions.
    """

    def __init__(self, max_queue_size: int = 1000, max_retries: int = 3):
        self.storage_queue = queue.Queue(maxsize=max_queue_size)
        self.max_retries = max_retries
        self.worker_thread = None
        self.shutdown_event = threading.Event()
        self.is_running = False

    def start(self):
        """Start the background worker thread."""
        if not self.is_running:
            self.is_running = True
            self.shutdown_event.clear()
            self.worker_thread = threading.Thread(
                target=self._worker_loop,
                daemon=True,
                name="AsyncStorageWorker"
            )
            self.worker_thread.start()

    def stop(self):
        """Stop the background worker thread gracefully."""
        if self.is_running:
            self.shutdown_event.set()
            self.is_running = False
            if self.worker_thread:
                self.worker_thread.join(timeout=5.0)

    def _worker_loop(self):
        """Main worker loop that processes storage tasks."""
        while not self.shutdown_event.is_set():
            try:
                # Get task with timeout to allow periodic shutdown checks
                task = self.storage_queue.get(timeout=1.0)
                self._process_task(task)
                self.storage_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in storage worker loop: {e}")

    def _process_task(self, task: StorageTask):
        """Process a single storage task."""
        try:
            if task.task_type == 'message':
                add_message(**task.data)
            elif task.task_type == 'plot':
                add_plot(**task.data)
            else:
                return

        except Exception as e:
            logger.warning(f"Failed to process {task.task_type} storage task: {e}")

            # Retry logic
            if task.retry_count < self.max_retries:
                task.retry_count += 1
                try:
                    # Re-queue with exponential backoff
                    time.sleep(2 ** task.retry_count)
                    self.storage_queue.put_nowait(task)
                except queue.Full:
                    logger.error(f"Queue full, dropping {task.task_type} task after retry")
            else:
                logger.error(f"Max retries exceeded for {task.task_type} task, dropping")

    def queue_message_storage(self, **kwargs):
        """Queue a message storage operation."""
        self._queue_task('message', kwargs)

    def queue_plot_storage(self, **kwargs):
        """Queue a plot storage operation."""
        self._queue_task('plot', kwargs)

    def _queue_task(self, task_type: str, data: Dict[str, Any]):
        """Queue a storage task for background processing."""
        if not self.is_running:
            logger.warning(f"AsyncStorageManager not running, dropping {task_type} task")
            return

        task = StorageTask(
            task_type=task_type,
            data=data,
            timestamp=datetime.now(timezone.utc)
        )

        try:
            self.storage_queue.put_nowait(task)
        except queue.Full:
            logger.warning(f"Storage queue full, dropping {task_type} task")

    def get_queue_size(self) -> int:
        """Get current queue size."""
        return self.storage_queue.qsize()

    def flush_queue(self, timeout: float = 10.0):
        """Wait for all queued tasks to complete."""
        try:
            self.storage_queue.join()
        except Exception as e:
            logger.warning(f"Error flushing storage queue: {e}")


# Global instance
_storage_manager: Optional[AsyncStorageManager] = None


def get_storage_manager() -> AsyncStorageManager:
    """Get or create the global AsyncStorageManager instance."""
    global _storage_manager

    if _storage_manager is None:
        _storage_manager = AsyncStorageManager()
        _storage_manager.start()

        # Register cleanup on app shutdown
        atexit.register(_storage_manager.stop)

    return _storage_manager


def async_add_message(**kwargs):
    """Asynchronously add message data to storage."""
    manager = get_storage_manager()
    manager.queue_message_storage(**kwargs)


def async_add_plot(**kwargs):
    """Asynchronously add plot data to storage."""
    manager = get_storage_manager()
    manager.queue_plot_storage(**kwargs)


def is_desktop_mode() -> bool:
    """Check if the application is running in desktop mode."""
    return get_static_value('desktop_mode', 'global', True)


def should_use_async_storage() -> bool:
    """
    Check if async storage should be used.

    Returns True only if:
    1. Not in desktop mode
    2. Firestore collection is enabled (static config only)
    """
    if is_desktop_mode():
        return False

    # Check runtime configuration with fallback
    try:
        from webapp.config.config_utils import get_runtime_setting
        return get_runtime_setting('cache_mode', False, 'cache')
    except ImportError:
        # Fallback to static config if runtime config not available
        return get_static_value('cache_mode', 'cache', False)


def conditional_async_add_message(enable_firestore=None, **kwargs):
    """Add message data, using async storage only when enabled."""
    # If enable_firestore is explicitly provided, use it directly
    if enable_firestore is not None:
        should_store = enable_firestore
    else:
        # Fall back to runtime configuration
        should_store = should_use_async_storage()

    if should_store:
        async_add_message(**kwargs)
    else:
        pass


def conditional_async_add_plot(enable_firestore=None, **kwargs):
    """Add plot data, using async storage only when enabled."""
    # If enable_firestore is explicitly provided, use it directly
    if enable_firestore is not None:
        should_store = enable_firestore
    else:
        # Fall back to runtime configuration
        should_store = should_use_async_storage()

    if should_store:
        async_add_plot(**kwargs)
    else:
        pass
