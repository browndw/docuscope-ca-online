TEST_CONFIG = {
    "desktop_mode": True,  # Always test in desktop mode for consistency
    "check_size": False,
    "check_language": False,
    "max_text_size": 1000,  # Smaller limits for faster tests
    "llm_quota": 5,  # Limited for testing
    "cache_mode": False  # Disable Firestore for tests
}
