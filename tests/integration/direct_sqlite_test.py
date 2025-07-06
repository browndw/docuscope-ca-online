#!/usr/bin/env python3
"""
Direct SQLite Session Backend Test

Test the SQLite session backend directly without complex imports.
"""

import pathlib
import os
import sys
import tempfile
import shutil
import sqlite3
import json
import threading
import time

project_root = pathlib.Path(__file__).resolve()
for _ in range(10):  # Search up to 10 levels
    if (project_root / 'webapp').exists() or (project_root / 'pyproject.toml').exists():
        break
    project_root = project_root.parent
else:
    raise RuntimeError("Could not find project root")


def test_direct_sqlite_functionality():
    """Test SQLite functionality directly."""
    print("Testing direct SQLite functionality...")

    # Create temporary database
    test_dir = tempfile.mkdtemp()
    db_path = os.path.join(test_dir, "test_direct.db")

    try:
        # Create database schema directly
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Create sessions table
        cursor.execute("""
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create indexes
        cursor.execute("""
            CREATE INDEX idx_sessions_updated_at ON sessions(updated_at)
        """)

        conn.commit()
        print("✓ Database schema created")

        # Test data insertion
        session_id = "test_session_123"
        test_data = {
            "session": {"HAS_TARGET": True, "TARGET_DB": "test_corpus"},
            "messages": [{"role": "assistant", "content": "Hello"}],
            "test_key": "test_value"
        }

        data_json = json.dumps(test_data)
        cursor.execute("""
            INSERT INTO sessions (session_id, data)
            VALUES (?, ?)
        """, (session_id, data_json))
        conn.commit()
        print("✓ Data inserted successfully")

        # Test data retrieval
        cursor.execute("""
            SELECT data FROM sessions WHERE session_id = ?
        """, (session_id,))
        row = cursor.fetchone()
        assert row is not None, "No data found"

        retrieved_data = json.loads(row[0])
        assert retrieved_data["session"]["HAS_TARGET"] is True
        assert retrieved_data["session"]["TARGET_DB"] == "test_corpus"
        assert retrieved_data["test_key"] == "test_value"
        print("✓ Data retrieved and verified")

        # Test data update
        updated_data = test_data.copy()
        updated_data["new_field"] = "new_value"
        updated_json = json.dumps(updated_data)

        cursor.execute("""
            UPDATE sessions
            SET data = ?, updated_at = CURRENT_TIMESTAMP
            WHERE session_id = ?
        """, (updated_json, session_id))
        conn.commit()
        print("✓ Data updated successfully")

        # Test concurrent access with WAL mode
        conn.execute("PRAGMA journal_mode=WAL")
        conn.commit()
        print("✓ WAL mode enabled")

        # Test concurrent writes (simple simulation)
        def write_session(session_id, data):
            try:
                conn_thread = sqlite3.connect(db_path)
                cursor_thread = conn_thread.cursor()

                cursor_thread.execute("""
                    INSERT OR REPLACE INTO sessions (session_id, data)
                    VALUES (?, ?)
                """, (session_id, json.dumps(data)))
                conn_thread.commit()
                conn_thread.close()
            except Exception as e:
                print(f"Thread error: {e}")
                return False

        # Start multiple threads
        threads = []
        for i in range(3):
            thread_data = {"thread_id": i, "data": f"thread_{i}_data"}
            thread = threading.Thread(
                target=write_session,
                args=(f"concurrent_session_{i}", thread_data)
            )
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Verify concurrent writes
        cursor.execute(
            "SELECT COUNT(*) FROM sessions WHERE session_id LIKE 'concurrent_session_%'"
            )
        count = cursor.fetchone()[0]
        assert count == 3, f"Expected 3 concurrent sessions, got {count}"
        print("✓ Concurrent writes successful")

        conn.close()

    finally:
        shutil.rmtree(test_dir)

    print("✅ Direct SQLite functionality test PASSED")


def test_session_data_persistence():
    """Test session data persistence patterns."""
    print("Testing session data persistence patterns...")

    test_dir = tempfile.mkdtemp()
    db_path = os.path.join(test_dir, "test_persistence.db")

    try:
        # Create connection
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Create sessions table
        cursor.execute("""
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()

        # Test complex session data structures
        session_id = "complex_session"
        complex_data = {
            "session": {
                "HAS_TARGET": True,
                "TARGET_DB": "complex_corpus",
                "HAS_META": False,
                "HAS_REFERENCE": True,
                "REFERENCE_DB": "ref_corpus"
            },
            "target": {
                "ds_tokens": "serialized_dataframe_data",
                "metadata": {"docs": 100, "tokens": 50000}
            },
            "reference": {
                "ds_tokens": "ref_dataframe_data",
                "metadata": {"docs": 200, "tokens": 75000}
            },
            "messages": [
                {"role": "assistant", "content": "Hello"},
                {"role": "user", "content": "Analyze my corpus"},
                {"role": "assistant", "content": "Analysis complete"}
            ],
            "plot_intent": False,
            "analysis_cache": {
                "frequency_tables": "cached_ft_data",
                "ngrams": "cached_ngram_data"
            }
        }

        # Store complex data
        data_json = json.dumps(complex_data)
        cursor.execute("""
            INSERT INTO sessions (session_id, data)
            VALUES (?, ?)
        """, (session_id, data_json))
        conn.commit()
        print("✓ Complex session data stored")

        # Retrieve and verify
        cursor.execute("""
            SELECT data FROM sessions WHERE session_id = ?
        """, (session_id,))
        row = cursor.fetchone()
        retrieved_data = json.loads(row[0])

        # Verify session state
        assert retrieved_data["session"]["HAS_TARGET"] is True
        assert retrieved_data["session"]["TARGET_DB"] == "complex_corpus"
        assert retrieved_data["session"]["HAS_REFERENCE"] is True

        # Verify corpus data
        assert retrieved_data["target"]["metadata"]["docs"] == 100
        assert retrieved_data["reference"]["metadata"]["tokens"] == 75000

        # Verify AI messages
        assert len(retrieved_data["messages"]) == 3
        assert retrieved_data["messages"][1]["role"] == "user"

        # Verify cache data
        assert "frequency_tables" in retrieved_data["analysis_cache"]
        print("✓ Complex session data integrity verified")

        # Test incremental updates (simulating session state changes)
        retrieved_data["session"]["HAS_META"] = True
        retrieved_data["messages"].append({
            "role": "user",
            "content": "Generate a plot"
        })
        retrieved_data["plot_intent"] = True

        # Update in database
        updated_json = json.dumps(retrieved_data)
        cursor.execute("""
            UPDATE sessions
            SET data = ?, updated_at = CURRENT_TIMESTAMP
            WHERE session_id = ?
        """, (updated_json, session_id))
        conn.commit()
        print("✓ Incremental session updates applied")

        # Verify incremental updates
        cursor.execute("""
            SELECT data FROM sessions WHERE session_id = ?
        """, (session_id,))
        row = cursor.fetchone()
        final_data = json.loads(row[0])

        assert final_data["session"]["HAS_META"] is True
        assert final_data["plot_intent"] is True
        assert len(final_data["messages"]) == 4
        print("✓ Incremental updates verified")

        conn.close()

    finally:
        shutil.rmtree(test_dir)

    print("✅ Session data persistence test PASSED")


def test_database_performance():
    """Test database performance with larger datasets."""
    print("Testing database performance...")

    test_dir = tempfile.mkdtemp()
    db_path = os.path.join(test_dir, "test_performance.db")

    try:
        # Create connection with optimizations
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Enable WAL mode and optimizations
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA cache_size=10000")
        cursor.execute("PRAGMA temp_store=MEMORY")

        # Create sessions table
        cursor.execute("""
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create indexes
        cursor.execute("CREATE INDEX idx_sessions_updated_at ON sessions(updated_at)")
        conn.commit()
        print("✓ Optimized database created")

        # Generate test data
        num_sessions = 100
        session_data_template = {
            "session": {"HAS_TARGET": True, "TARGET_DB": "perf_test"},
            "messages": [{"role": "assistant", "content": "Performance test message"}],
            "large_data": ["item_" + str(i) for i in range(100)]  # 100 items per session
        }

        # Measure insertion performance
        start_time = time.time()

        for i in range(num_sessions):
            session_id = f"perf_session_{i}"
            data = session_data_template.copy()
            data["session_number"] = i

            cursor.execute("""
                INSERT INTO sessions (session_id, data)
                VALUES (?, ?)
            """, (session_id, json.dumps(data)))

        conn.commit()
        insert_time = time.time() - start_time
        print(f"✓ Inserted {num_sessions} sessions in {insert_time:.3f} seconds")

        # Measure query performance
        start_time = time.time()

        for i in range(num_sessions):
            session_id = f"perf_session_{i}"
            cursor.execute("""
                SELECT data FROM sessions WHERE session_id = ?
            """, (session_id,))
            row = cursor.fetchone()
            data = json.loads(row[0])
            assert data["session_number"] == i

        query_time = time.time() - start_time
        print(f"✓ Queried {num_sessions} sessions in {query_time:.3f} seconds")

        # Measure update performance
        start_time = time.time()

        for i in range(num_sessions):
            session_id = f"perf_session_{i}"
            cursor.execute("""
                SELECT data FROM sessions WHERE session_id = ?
            """, (session_id,))
            row = cursor.fetchone()
            data = json.loads(row[0])

            # Modify data
            data["updated"] = True
            data["update_time"] = time.time()

            cursor.execute("""
                UPDATE sessions
                SET data = ?, updated_at = CURRENT_TIMESTAMP
                WHERE session_id = ?
            """, (json.dumps(data), session_id))

        conn.commit()
        update_time = time.time() - start_time
        print(f"✓ Updated {num_sessions} sessions in {update_time:.3f} seconds")

        # Performance assertions (reasonable thresholds)
        assert insert_time < 5.0, f"Insert performance too slow: {insert_time:.3f}s"
        assert query_time < 2.0, f"Query performance too slow: {query_time:.3f}s"
        assert update_time < 5.0, f"Update performance too slow: {update_time:.3f}s"

        conn.close()

    finally:
        shutil.rmtree(test_dir)

    print("✅ Database performance test PASSED")


def run_all_tests():
    """Run all direct SQLite tests."""
    print("=== Direct SQLite Session Backend Tests ===\n")

    tests = [
        test_direct_sqlite_functionality,
        test_session_data_persistence,
        test_database_performance
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            print(f"\n--- {test_func.__name__} ---")
            result = test_func()
            if result:
                passed += 1
            else:
                failed += 1
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
        print("\n🎉 All direct SQLite tests passed!")
        print("The SQLite session backend foundation is solid.")
        print("\nNEXT STEPS:")
        print("1. Start the Streamlit app: streamlit run webapp/index.py")
        print("2. Test session persistence by:")
        print("   - Loading a corpus")
        print("   - Making some changes")
        print("   - Restarting the app")
        print("   - Verifying data is restored")
        return True
    else:
        print(f"\n⚠️  {failed} test(s) failed. Check the issues above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
