"""
Integration helper for SQLite session backend testing and deployment.

This module provides utilities for testing the SQLite session backend
and migrating from the existing session management system.
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any

# Add the project root to Python path for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Import webapp modules after path setup
from webapp.utilities.storage.sqlite_session_backend import (  # noqa: E402
    SQLiteSessionBackend,
    get_session_backend
)
from webapp.utilities.storage.cache_management import get_session_analytics  # noqa: E402
from webapp.config.unified import get_config  # noqa: E402
from webapp.utilities.configuration.logging_config import get_logger  # noqa: E402

# Mock streamlit for testing if not available
try:
    import streamlit as st
except ImportError:
    class MockStreamlit:
        class session_state:
            def items():
                return []
        user = None
    st = MockStreamlit()

logger = get_logger()


def test_sqlite_session_backend():
    """Test the SQLite session backend functionality."""

    print("Testing SQLite Session Backend...")

    try:
        # Use temporary directory for testing
        temp_dir = tempfile.mkdtemp()
        test_db_path = os.path.join(temp_dir, "test_sessions.db")

        # Create backend with test path
        backend = SQLiteSessionBackend(test_db_path)
        print("✓ SQLite backend initialized successfully")

        # Test session save/load
        test_session_id = "test_session_123"
        test_data = {
            "user_id": "test_user",
            "has_target": [True],
            "metadata_target": {"test": "data"},
            "created_at": datetime.now(timezone.utc).isoformat()
        }

        # Save session
        success = backend.save_session(test_session_id, test_data, "test_user")
        assert success, "Session save failed"
        print("✓ Session save successful")

        # Load session
        loaded_data = backend.load_session(test_session_id)
        assert loaded_data is not None, "Session load failed"
        assert loaded_data.get("user_id") == "test_user", "Loaded session data incorrect"
        print("✓ Session load successful")

        # Test query logging
        success = backend.log_user_query(
            "test_user", test_session_id, "plotbot", "test query"
        )
        assert success, "Query logging failed"
        print("✓ Query logging successful")

        # Test quota checking
        count = backend.get_user_query_count_24h("test_user")
        assert count >= 0, "Quota checking failed"
        print(f"✓ Quota checking successful (count: {count})")

        # Test stats
        stats = backend.get_session_stats()
        assert stats is not None, "Statistics retrieval failed"
        assert 'active_sessions' in stats, "Statistics format incorrect"
        print("✓ Statistics retrieval successful")
        print(f"  Active sessions: {stats['active_sessions']}")
        print(f"  Database size: {stats['database_size_bytes']} bytes")

        # Cleanup test data
        backend.delete_session(test_session_id)

        # Remove temporary directory
        shutil.rmtree(temp_dir)

        print("✓ Test cleanup completed")

        print("\n🎉 All SQLite session backend tests passed!")

    except Exception as e:
        print(f"✗ SQLite session backend test failed: {e}")
        raise  # Re-raise the exception so pytest can handle it


def test_quota_integration():
    """Test the quota checking integration."""

    print("\nTesting Quota Integration...")

    try:
        # Use temporary directory for testing
        temp_dir = tempfile.mkdtemp()
        test_db_path = os.path.join(temp_dir, "test_quota.db")

        # Create backend with test path
        backend = SQLiteSessionBackend(test_db_path)

        test_user = "quota_test_user"

        # Create a test session first (required for foreign key constraint)
        test_session_data = {
            "user_id": test_user,
            "metadata": {"test": True}
        }
        test_session_id = "quota_session"
        success = backend.save_session(
            test_session_id, test_session_data, test_user
        )
        assert success, "Failed to create test session"

        # Log some test queries directly to backend
        for i in range(3):
            success = backend.log_user_query(
                test_user, test_session_id, "plotbot", f"test query {i}"
            )
            assert success, f"Failed to log query {i}"

        # Check quota count using backend
        count = backend.get_user_query_count_24h(test_user)
        assert count >= 3, f"Quota integration failed (expected >= 3, got {count})"
        print(f"✓ Quota integration successful (count: {count})")

        # Cleanup
        shutil.rmtree(temp_dir)

        print("🎉 Quota integration test passed!")

    except Exception as e:
        print(f"✗ Quota integration test failed: {e}")
        raise  # Re-raise the exception so pytest can handle it


def migrate_existing_sessions():
    """
    Migrate existing Streamlit session state to SQLite backend.

    This function can be called during application startup to preserve
    existing user sessions during the transition to SQLite.
    """

    print("Migrating existing sessions to SQLite...")

    try:
        backend = get_session_backend()
        migrated_count = 0

        # Iterate through existing Streamlit sessions
        for session_id, session_data in st.session_state.items():
            if isinstance(session_data, dict) and 'session' in session_data:
                try:
                    # Extract user ID if available
                    user_id = None
                    if hasattr(st, 'user') and st.user:
                        user_id = getattr(st.user, 'email', None)

                    # Save to SQLite
                    success = backend.save_session(session_id, session_data, user_id)
                    if success:
                        migrated_count += 1
                        logger.info(f"Migrated session {session_id} to SQLite")

                except Exception as e:
                    logger.warning(f"Failed to migrate session {session_id}: {e}")

        print(f"✓ Migrated {migrated_count} sessions to SQLite")
        return migrated_count

    except Exception as e:
        print(f"✗ Session migration failed: {e}")
        return 0


def get_deployment_readiness_report() -> Dict[str, Any]:
    """
    Generate a deployment readiness report for the SQLite session backend.

    Returns
    -------
    Dict[str, Any]
        Report with readiness status and recommendations
    """

    report = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'backend_type': 'sqlite',
        'status': 'unknown',
        'tests_passed': 0,
        'tests_total': 3,
        'issues': [],
        'recommendations': []
    }

    try:
        # Test 1: Backend initialization
        backend = get_session_backend()
        report['tests_passed'] += 1

        # Test 2: Database operations
        test_session_id = "readiness_test"
        test_data = {"test": True, "timestamp": datetime.now(timezone.utc).isoformat()}

        if (backend.save_session(test_session_id, test_data) and
                backend.load_session(test_session_id) and
                backend.delete_session(test_session_id)):
            report['tests_passed'] += 1

        # Test 3: Configuration
        storage_path = get_config('storage_path', 'session', 'webapp/_session')
        pool_size = get_config('connection_pool_size', 'session', 10)

        if storage_path and pool_size > 0:
            report['tests_passed'] += 1

        # Get analytics
        analytics = get_session_analytics()
        report['analytics'] = analytics

        # Determine status
        if report['tests_passed'] == report['tests_total']:
            report['status'] = 'ready'
        elif report['tests_passed'] >= 2:
            report['status'] = 'caution'
            report['issues'].append('Some tests failed - review configuration')
        else:
            report['status'] = 'not_ready'
            report['issues'].append('Critical tests failed - do not deploy')

        # Add recommendations
        if pool_size < 10:
            report['recommendations'].append(
                'Consider increasing connection_pool_size for classroom usage'
            )

        if analytics.get('database_stats', {}).get('active_sessions', 0) > 1000:
            report['recommendations'].append(
                'High session count detected - monitor performance'
            )

    except Exception as e:
        report['status'] = 'error'
        report['issues'].append(f'Test execution failed: {str(e)}')

    return report


def print_deployment_report():
    """Print a formatted deployment readiness report."""

    report = get_deployment_readiness_report()

    print("\n" + "="*60)
    print("SQLite Session Backend - Deployment Readiness Report")
    print("="*60)
    print(f"Status: {report['status'].upper()}")
    print(f"Tests Passed: {report['tests_passed']}/{report['tests_total']}")
    print(f"Timestamp: {report['timestamp']}")

    if report['issues']:
        print("\n⚠️  Issues:")
        for issue in report['issues']:
            print(f"  • {issue}")

    if report['recommendations']:
        print("\n💡 Recommendations:")
        for rec in report['recommendations']:
            print(f"  • {rec}")

    if 'analytics' in report:
        analytics = report['analytics']
        if 'database_stats' in analytics:
            stats = analytics['database_stats']
            print("\n📊 Current Statistics:")
            print(f"  • Active Sessions: {stats.get('active_sessions', 0)}")
            print(f"  • Database Size: {stats.get('database_size_bytes', 0)} bytes")
            print(f"  • Pool Available: {stats.get('pool_available', 0)}")

    print("="*60)


if __name__ == "__main__":
    # Run all tests
    print("SQLite Session Backend Integration Tests")
    print("="*50)

    test1 = test_sqlite_session_backend()
    test2 = test_quota_integration()

    print("\n" + "="*50)
    print("Test Summary:")
    print(f"SQLite Backend: {'✓ PASS' if test1 else '✗ FAIL'}")
    print(f"Quota Integration: {'✓ PASS' if test2 else '✗ FAIL'}")

    # Print deployment report
    print_deployment_report()
