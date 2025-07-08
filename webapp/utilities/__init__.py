
"""
Modular utilities package for corpus analysis application.

This package contains specialized modules for different aspects of the application:
- ai: AI and LLM integration
- analysis: Corpus analysis and statistical functions
- config: Configuration management
- core: Core integration and application orchestration (import separately)
- exports: Data export functionality
- plotting: Visualization and plotting
- processing: Data processing
- session: Session management
- state: Widget state management
- storage: Storage and caching
- ui: User interface utilities
"""

# Import the modular utilities
from webapp.utilities import ai
from webapp.utilities import analysis
from webapp.utilities import configuration
from webapp.utilities import exports
from webapp.utilities import plotting
from webapp.utilities import processing
from webapp.utilities import session
from webapp.utilities import state
from webapp.utilities import storage
from webapp.utilities import ui

# Note: core module should be imported separately to avoid circular imports
# from webapp.utilities.core import app_core

__all__ = [
    "ai",
    "analysis",
    "configuration",
    "data",
    "exports",
    "plotting",
    "processing",
    "session",
    "state",
    "storage",
    "ui"
]
