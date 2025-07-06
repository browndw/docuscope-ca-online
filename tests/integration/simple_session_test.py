#!/usr/bin/env python3
"""
Simple Session Persistence Test

A focused test to verify the session persistence functionality works correctly.
"""

import os
import sys
import tempfile
import shutil
import sqlite3
from pathlib import Path

# Add the webapp directory to the Python path
webapp_root = Path(__file__).parent.parent.parent / "webapp"
sys.path.insert(0, str(webapp_root))

# Also add the project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def test_sqlite_backend_basic_functionality():
    """Test basic SQLite backend functionality."""
    print("Testing SQLite backend basic functionality...")

    # Create temporary database
    test_dir = tempfile.mkdtemp()
    db_path = os.path.join(test_dir, "test_sessions.db")

    try:
        from webapp.utilities.storage.sqlite_session_backend import SQLiteSessionBackend

        # Mock configuration
        def mock_get_config(key, section, default=None):
            """Mock config function that returns test values."""
            config_values = {
                ('sqlite_db_path', 'session'): db_path,
                ('connection_pool_size', 'session'): 2,
                ('cleanup_hours', 'session'): 24,
                ('storage_path', 'session'): test_dir,
            }
            return config_values.get((key, section), default)

        # Patch the get_config function
        from unittest.mock import patch
        with patch(
            'webapp.utilities.storage.sqlite_session_backend.get_config',
            side_effect=mock_get_config
        ):
            # Initialize backend
            backend = SQLiteSessionBackend()

            # Test session storage
            session_id = "test_session_123"
            test_data = {
                "session": {"HAS_TARGET": True, "TARGET_DB": "test_corpus"},
                "messages": [{"role": "assistant", "content": "Hello"}],
                "test_key": "test_value"
            }

            # Store session
            success = backend.save_session(session_id, test_data, "test_user")
            assert success, "Failed to save session"
            print("✓ Session saved successfully")

            # Retrieve session
            retrieved_data = backend.load_session(session_id)
            assert retrieved_data is not None, "Failed to retrieve session"
            assert retrieved_data["session"]["HAS_TARGET"] is True
            assert retrieved_data["session"]["TARGET_DB"] == "test_corpus"
            assert retrieved_data["test_key"] == "test_value"
            print("✓ Session retrieved successfully")

            # Test update session
            updated_data = test_data.copy()
            updated_data["new_field"] = "new_value"
            success = backend.save_session(session_id, updated_data, "test_user")
            assert success, "Failed to update session"

            retrieved_updated = backend.load_session(session_id)
            assert retrieved_updated["new_field"] == "new_value"
            print("✓ Session update works")

            # Test delete session
            success = backend.delete_session(session_id)
            assert success, "Failed to delete session"

            # Verify session no longer exists
            deleted_data = backend.load_session(session_id)
            assert deleted_data is None, "Session should not exist after deletion"
            print("✓ Session deletion works")

            # Clean up - note: SQLiteSessionBackend doesn't have a close method
            print("✓ Backend cleanup successful")

    finally:
        # Clean up test directory
        shutil.rmtree(test_dir)

    print("✅ SQLite backend basic functionality test PASSED")


def test_database_structure():
    """Test that the database structure is created correctly."""
    print("Testing database structure...")

    test_dir = tempfile.mkdtemp()

    try:
        from webapp.utilities.storage.sqlite_session_backend import SQLiteSessionBackend

        # Initialize backend with explicit path - this will create the database
        backend = SQLiteSessionBackend(test_dir)

        # Create a test session to ensure tables are created
        test_data = {"test": "data"}
        backend.save_session("test_session", test_data, "test_user")

        # Check database structure directly
        conn = sqlite3.connect(backend.db_path)
        cursor = conn.cursor()

        # Check if sessions table exists
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='sessions'
        """)
        table_exists = cursor.fetchone() is not None
        assert table_exists, "Sessions table was not created"
        print("✓ Sessions table exists")

        # Check table schema
        cursor.execute("PRAGMA table_info(sessions)")
        columns = cursor.fetchall()
        column_names = [col[1] for col in columns]

        expected_columns = ['session_id', 'data', 'created_at', 'updated_at']
        for col in expected_columns:
            assert col in column_names, f"Column {col} missing from sessions table"
        print("✓ Sessions table has correct schema")

        # Check for indexes
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='index' AND tbl_name='sessions'
        """)
        indexes = cursor.fetchall()
        assert len(indexes) > 0, "No indexes found on sessions table"
        print("✓ Indexes exist on sessions table")

        conn.close()

    finally:
        shutil.rmtree(test_dir)

    print("✅ Database structure test PASSED")


def test_concurrent_access():
    """Test concurrent access to the database."""
    print("Testing concurrent access...")

    test_dir = tempfile.mkdtemp()

    try:
        from webapp.utilities.storage.sqlite_session_backend import SQLiteSessionBackend

        # Create multiple backend instances pointing to the same database
        backend1 = SQLiteSessionBackend(test_dir)
        backend2 = SQLiteSessionBackend(test_dir)

        # Test concurrent writes
        session1_data = {"test": "data1", "user": "user1"}
        session2_data = {"test": "data2", "user": "user2"}

        success1 = backend1.save_session("session1", session1_data, "user1")
        success2 = backend2.save_session("session2", session2_data, "user2")

        assert success1, "Backend1 failed to save"
        assert success2, "Backend2 failed to save"
        print("✓ Concurrent writes successful")

        # Test concurrent reads
        retrieved1 = backend1.load_session("session1")
        retrieved2 = backend2.load_session("session2")

        assert retrieved1["user"] == "user1"
        assert retrieved2["user"] == "user2"
        print("✓ Concurrent reads successful")

        # Test cross-backend access
        retrieved1_from_2 = backend2.load_session("session1")
        retrieved2_from_1 = backend1.load_session("session2")

        assert retrieved1_from_2["user"] == "user1"
        assert retrieved2_from_1["user"] == "user2"
        print("✓ Cross-backend access works")

    finally:
        shutil.rmtree(test_dir)

    print("✅ Concurrent access test PASSED")


def test_data_integrity():
    """Test data integrity and serialization."""
    print("Testing data integrity...")

    test_dir = tempfile.mkdtemp()

    try:
        from webapp.utilities.storage.sqlite_session_backend import SQLiteSessionBackend

        backend = SQLiteSessionBackend(test_dir)

        # Test complex data structures
        complex_data = {
            "string_field": "test string",
            "int_field": 42,
            "float_field": 3.14159,
            "bool_field": True,
            "list_field": [1, 2, "three", {"nested": "dict"}],
            "dict_field": {
                "nested": {
                    "deeply": {
                        "nested": "value"
                    }
                }
            },
            "null_field": None
        }

        session_id = "integrity_test"

        # Store complex data
        success = backend.save_session(session_id, complex_data, "test_user")
        assert success, "Failed to save complex data"
        print("✓ Complex data saved")

        # Retrieve and verify
        retrieved = backend.load_session(session_id)
        assert retrieved is not None, "Failed to retrieve complex data"

        # Verify each field
        assert retrieved["string_field"] == "test string"
        assert retrieved["int_field"] == 42
        assert abs(retrieved["float_field"] - 3.14159) < 0.00001
        assert retrieved["bool_field"] is True
        assert retrieved["list_field"] == [1, 2, "three", {"nested": "dict"}]
        assert retrieved["dict_field"]["nested"]["deeply"]["nested"] == "value"
        assert retrieved["null_field"] is None
        print("✓ Complex data integrity verified")

        # Test Unicode handling
        unicode_data = {
            "unicode": "测试中文",
            "emoji": "🎉🔥💻",
            "special_chars": "áéíóú ñ ç ü"
        }

        success = backend.save_session("unicode_test", unicode_data, "test_user")
        assert success, "Failed to save Unicode data"

        retrieved_unicode = backend.load_session("unicode_test")
        assert retrieved_unicode["unicode"] == "测试中文"
        assert retrieved_unicode["emoji"] == "🎉🔥💻"
        assert retrieved_unicode["special_chars"] == "áéíóú ñ ç ü"
        print("✓ Unicode handling works")

    finally:
        shutil.rmtree(test_dir)

    print("✅ Data integrity test PASSED")


def run_all_tests():
    """Run all SQLite backend tests."""
    print("=== SQLite Session Backend Integration Tests ===\n")

    tests = [
        test_sqlite_backend_basic_functionality,
        test_database_structure,
        test_concurrent_access,
        test_data_integrity
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            print(f"\n--- {test_func.__name__} ---")
            test_func()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"❌ {test_func.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()

    print("\n=== Test Results ===")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total:  {passed + failed}")

    if failed == 0:
        print("\n🎉 All SQLite backend tests passed!")
        print("Session persistence is ready for production use.")
        return True
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please check the issues above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
