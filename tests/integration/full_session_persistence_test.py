#!/usr/bin/env python3
"""
Full Session Persistence Integration Test

This test verifies that the complete session management system works with
SQLite persistence, including:
1. Session initialization and loading from storage
2. Session state updates and automatic persistence
3. Corpus data management with persistence
4. Metadata initialization with persistence
5. AI assistant state with persistence
6. Cross-session continuity after app restart simulation

This test validates the comprehensive session storage solution.
"""

import os
import sys
import tempfile
import shutil
import sqlite3
import polars as pl
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the webapp directory to the Python path
webapp_root = Path(__file__).parent.parent.parent / "webapp"
sys.path.insert(0, str(webapp_root))

# Use Streamlit session state mocking for backend persistence testing
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    # Fallback to comprehensive mocking if Streamlit not available
    sys.modules['streamlit'] = MagicMock()
    STREAMLIT_AVAILABLE = False


class MockSessionState:
    """Mock session state that behaves like Streamlit's session state."""

    def __init__(self):
        self._data = {}

    def __contains__(self, key):
        return key in self._data

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def __delitem__(self, key):
        del self._data[key]

    def get(self, key, default=None):
        return self._data.get(key, default)

    def setdefault(self, key, default=None):
        return self._data.setdefault(key, default)

    def clear(self):
        self._data.clear()

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()


# If we have Streamlit, patch its session_state, otherwise create our own st module
if STREAMLIT_AVAILABLE:
    # Store original for restoration if needed
    _original_session_state = st.session_state
else:
    # Create mock st module
    st = MagicMock()

# Always use our mock session state for this test
st.session_state = MockSessionState()

from webapp.utilities.storage.sqlite_session_backend import SQLiteSessionBackend  # noqa: E402, E501
from webapp.utilities.session.session_persistence import (  # noqa: E402
    load_persistent_session
)
from webapp.utilities.session.session_core import (  # noqa: E402
    init_session, update_session
)
from webapp.utilities.session.session_management import (  # noqa: E402
    init_ai_assist, generate_temp, ensure_session_loaded, persist_session_changes
)
from webapp.utilities.corpus.data_manager import CorpusDataManager  # noqa: E402


class TestFullSessionPersistence:
    """Test comprehensive session persistence functionality."""

    def setup_method(self):
        """Set up test environment for each test."""
        # Create temporary directory for test database
        self.test_dir = tempfile.mkdtemp()
        # Backend will create sessions.db in the provided directory
        self.db_path = os.path.join(self.test_dir, "sessions.db")

        # Mock configuration to use test database
        self.mock_config = MagicMock()
        self.mock_config.get_session_settings.return_value = {
            'backend': 'sqlite',
            'sqlite_db_path': self.db_path,
            'connection_pool_size': 2,
            'cleanup_hours': 24,
            'auto_persist': True
        }

        # Reset session state to clean mock state
        st.session_state = MockSessionState()

        # Patch configuration - patch the actual config function used
        self.config_patcher = patch('webapp.config.unified.get_config')
        self.mock_get_config = self.config_patcher.start()

        # Configure the mock to return appropriate values based on the key
        def mock_config_side_effect(key, section=None, default=None):
            if key == 'sqlite_db_path' and section == 'session':
                return self.db_path
            elif key == 'connection_pool_size' and section == 'session':
                return 2
            elif key == 'storage_dir' and section == 'core':
                return self.test_dir
            elif key == 'cleanup_hours' and section == 'session':
                return 24
            elif key == 'desktop_mode' and section == 'global':
                return False  # Force non-desktop mode
            elif key == 'backend' and section == 'session':
                return 'sqlite'
            else:
                return default

        self.mock_get_config.side_effect = mock_config_side_effect

        # Initialize backend with explicit path
        self.backend = SQLiteSessionBackend(self.test_dir)

        # Patch the backend factory to always return our SQLite backend for this test
        backend_factory_path = (
            'webapp.utilities.storage.backend_factory.get_session_backend'
        )
        self.backend_factory_patcher = patch(backend_factory_path)
        self.mock_backend_factory = self.backend_factory_patcher.start()
        self.mock_backend_factory.return_value = self.backend

        # Test session ID
        self.session_id = "test_session_123"

        # Clear any existing session data for this session ID
        if hasattr(self.backend, 'delete_session'):
            self.backend.delete_session(self.session_id)

        # Reset the global persistence manager to ensure it uses our mocked backend
        import webapp.utilities.session.session_persistence as sp
        sp._persistence_manager = None

        print(f"Setup: Using test database at {self.db_path}")

    def teardown_method(self):
        """Clean up after each test."""
        try:
            # Stop patches
            self.config_patcher.stop()
            self.backend_factory_patcher.stop()

            # Close backend connections if method exists
            if hasattr(self, 'backend') and hasattr(self.backend, 'close'):
                self.backend.close()
        except Exception:
            pass

        # Clean up test directory
        try:
            shutil.rmtree(self.test_dir)
        except Exception:
            pass

    def test_basic_session_initialization_and_persistence(self):
        """Test basic session initialization with persistence."""
        print("Testing basic session initialization and persistence...")

        # Initialize a new session
        init_session(self.session_id)

        # Verify session was created in memory
        assert self.session_id in st.session_state
        assert "session" in st.session_state[self.session_id]

        # Verify session was persisted to database
        session_data = self.backend.load_session(self.session_id)
        assert session_data is not None
        assert "session" in session_data

        print("✓ Basic session initialization and persistence works")

    def test_session_updates_with_persistence(self):
        """Test session updates are automatically persisted."""
        print("Testing session updates with persistence...")

        # Initialize session
        init_session(self.session_id)

        # Update session
        update_session("HAS_TARGET", True, self.session_id)
        update_session("TARGET_DB", "test_corpus", self.session_id)

        # Verify updates in memory
        session_df = st.session_state[self.session_id]["session"]
        session_dict = (session_df.to_dict(as_series=False)
                        if hasattr(session_df, 'to_dict')
                        else session_df)

        assert session_dict.get("HAS_TARGET") == [True] or session_dict.get("HAS_TARGET") is True  # noqa: E501
        assert session_dict.get("TARGET_DB") == ["test_corpus"] or session_dict.get("TARGET_DB") == "test_corpus"  # noqa: E501

        # Verify updates were persisted
        session_data = self.backend.load_session(self.session_id)
        assert session_data is not None

        print("✓ Session updates with persistence works")

    def test_corpus_data_persistence(self):
        """Test corpus data manager persistence."""
        print("Testing corpus data persistence...")

        # Initialize session
        init_session(self.session_id)

        # Create test corpus data
        test_data = pl.DataFrame({
            "doc_id": ["doc1", "doc2", "doc3"],
            "token": ["hello", "world", "test"],
            "ds_tag": ["Greeting", "Object", "Action"],
            "pos_tag": ["NOUN", "NOUN", "VERB"]
        })

        # Create data manager and set core data
        data_manager = CorpusDataManager(self.session_id, "target")
        data_manager.set_core_data(test_data)

        # Verify data in session state
        assert "target" in st.session_state[self.session_id]
        assert "ds_tokens" in st.session_state[self.session_id]["target"]

        # Verify data was persisted
        session_data = self.backend.load_session(self.session_id)
        assert session_data is not None
        assert "target" in session_data

        print("✓ Corpus data persistence works")

    def test_ai_assistant_persistence(self):
        """Test AI assistant state persistence."""
        print("Testing AI assistant state persistence...")

        # Initialize AI assistant
        init_ai_assist(self.session_id)

        # Verify AI state in memory
        assert self.session_id in st.session_state
        assert "messages" in st.session_state[self.session_id]
        assert "plot_intent" in st.session_state[self.session_id]

        # Add a test message
        st.session_state[self.session_id]["messages"].append({
            "role": "user",
            "content": "Test message"
        })

        # Trigger persistence
        persist_session_changes(self.session_id)

        # Verify persistence
        session_data = self.backend.load_session(self.session_id)
        assert session_data is not None
        assert "messages" in session_data

        print("✓ AI assistant persistence works")

    def test_session_loading_from_storage(self):
        """Test loading existing session from storage."""
        print("Testing session loading from storage...")

        # Create and persist initial session
        init_session(self.session_id)
        update_session("HAS_TARGET", True, self.session_id)
        update_session("TARGET_DB", "persisted_corpus", self.session_id)

        # Clear memory
        st.session_state.clear()

        # Load session from storage
        loaded = load_persistent_session(self.session_id)
        assert loaded is True

        # Verify session was loaded correctly
        assert self.session_id in st.session_state
        session_df = st.session_state[self.session_id]["session"]
        session_dict = session_df.to_dict(as_series=False) if hasattr(session_df, 'to_dict') else session_df  # noqa: E501

        target_value = session_dict.get("HAS_TARGET")
        if isinstance(target_value, list):
            assert target_value[0] is True
        else:
            assert target_value is True

        db_value = session_dict.get("TARGET_DB")
        if isinstance(db_value, list):
            assert db_value[0] == "persisted_corpus"
        else:
            assert db_value == "persisted_corpus"

        print("✓ Session loading from storage works")

    def test_cross_session_continuity(self):
        """Test session continuity across app restarts."""
        print("Testing cross-session continuity...")

        # Phase 1: Create and populate session
        init_session(self.session_id)
        init_ai_assist(self.session_id)

        # Add some data
        update_session("HAS_TARGET", True, self.session_id)
        update_session("TARGET_DB", "continuity_test", self.session_id)

        # Add AI messages
        st.session_state[self.session_id]["messages"].append({
            "role": "user",
            "content": "Session continuity test"
        })
        persist_session_changes(self.session_id)

        # Simulate app restart by clearing memory and creating new backend
        st.session_state.clear()
        # Note: SQLiteSessionBackend doesn't have a close() method, so we just recreate it
        self.backend = SQLiteSessionBackend()

        # Phase 2: Load session like a fresh app start
        loaded = ensure_session_loaded(self.session_id)
        assert loaded is True

        # Verify all data is preserved
        assert self.session_id in st.session_state
        assert "session" in st.session_state[self.session_id]
        assert "messages" in st.session_state[self.session_id]

        # Check specific values
        session_df = st.session_state[self.session_id]["session"]
        session_dict = (session_df.to_dict(as_series=False)
                        if hasattr(session_df, 'to_dict')
                        else session_df)

        target_value = session_dict.get("HAS_TARGET")
        if isinstance(target_value, list):
            assert target_value[0] is True
        else:
            assert target_value is True

        # Check AI messages
        messages = st.session_state[self.session_id]["messages"]
        user_messages = [msg for msg in messages if msg["role"] == "user"]
        assert len(user_messages) > 0
        assert any("continuity test" in msg["content"] for msg in user_messages)

        print("✓ Cross-session continuity works")

    def test_generate_temp_with_persistence(self):
        """Test generate_temp function with persistence."""
        print("Testing generate_temp with persistence...")

        # Initialize session
        init_session(self.session_id)

        # Use generate_temp to add temporary states
        temp_states = [
            ("test_key1", "test_value1"),
            ("test_key2", {"nested": "data"}),
            ("test_key3", [1, 2, 3])
        ]

        generate_temp(temp_states, self.session_id)

        # Verify in memory
        assert "test_key1" in st.session_state[self.session_id]
        assert "test_key2" in st.session_state[self.session_id]
        assert "test_key3" in st.session_state[self.session_id]

        # Verify persistence
        session_data = self.backend.load_session(self.session_id)
        assert session_data is not None
        assert "test_key1" in session_data
        assert "test_key2" in session_data
        assert "test_key3" in session_data

        print("✓ generate_temp with persistence works")

    def test_database_structure_and_integrity(self):
        """Test that the database structure is correct and data integrity is maintained."""
        print("Testing database structure and integrity...")

        # Initialize and populate session to trigger database creation
        init_session(self.session_id)
        update_session("test_key", "test_value", self.session_id)

        # Force a save to ensure database is created
        session_data = st.session_state[self.session_id]

        # Force table initialization by calling the private method if needed
        if hasattr(self.backend, '_initialize_database'):
            self.backend._initialize_database()

        save_result = self.backend.save_session(self.session_id, session_data, "test_user")
        assert save_result, "Failed to save session to database"

        # Check database directly
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check sessions table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='sessions'"
        )
        table_exists = cursor.fetchone() is not None

        if not table_exists:
            # List all tables to debug
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            print(f"Available tables: {tables}")

        assert table_exists, "Sessions table was not created"

        # Check session exists in database
        cursor.execute(
            "SELECT session_id, data FROM sessions WHERE session_id = ?",
            (self.session_id,)
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == self.session_id
        assert row[1] is not None  # Data should be present

        conn.close()

        print("✓ Database structure and integrity checks pass")

    def test_performance_with_large_session(self):
        """Test performance with a larger session dataset."""
        print("Testing performance with large session...")

        # Initialize session
        init_session(self.session_id)

        # Create large dataset
        large_data = pl.DataFrame({
            "doc_id": [f"doc_{i}" for i in range(1000)],
            "token": [f"token_{i}" for i in range(1000)],
            "ds_tag": [f"tag_{i % 10}" for i in range(1000)],
            "pos_tag": [f"pos_{i % 5}" for i in range(1000)]
        })

        # Time the operation
        import time
        start_time = time.time()

        # Store large data
        data_manager = CorpusDataManager(self.session_id, "target")
        data_manager.set_core_data(large_data)

        end_time = time.time()
        duration = end_time - start_time

        print(f"Large data storage took {duration:.3f} seconds")

        # Verify data persisted
        session_data = self.backend.load_session(self.session_id)
        assert session_data is not None
        assert "target" in session_data

        # Performance should be reasonable (under 5 seconds for 1000 rows)
        assert duration < 5.0, f"Performance test failed: {duration:.3f}s > 5.0s"

        print("✓ Performance with large session is acceptable")


def run_tests():
    """Run all tests."""
    print("=== Full Session Persistence Integration Test ===\n")

    test_instance = TestFullSessionPersistence()
    tests = [
        "test_basic_session_initialization_and_persistence",
        "test_session_updates_with_persistence",
        "test_corpus_data_persistence",
        "test_ai_assistant_persistence",
        "test_session_loading_from_storage",
        "test_cross_session_continuity",
        "test_generate_temp_with_persistence",
        "test_database_structure_and_integrity",
        "test_performance_with_large_session"
    ]

    passed = 0
    failed = 0

    for test_name in tests:
        try:
            print(f"\n--- {test_name} ---")
            test_instance.setup_method()

            test_method = getattr(test_instance, test_name)
            test_method()

            passed += 1
            print(f"✅ {test_name} PASSED")

        except Exception as e:
            failed += 1
            print(f"❌ {test_name} FAILED: {e}")
            import traceback
            traceback.print_exc()

        finally:
            test_instance.teardown_method()

    print("\n=== Test Results ===")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total:  {passed + failed}")

    if failed == 0:
        print("\n🎉 All tests passed! Full session persistence is working correctly.")
        return True
    else:
        print(f"\n⚠️  {failed} test(s) failed. Session persistence needs attention.")
        return False


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
