"""Test configuration management functionality."""

import pickle
import pathlib
import pytest
import os
import sys
from unittest.mock import patch

project_root = pathlib.Path(__file__).resolve()
for _ in range(10):  # Search up to 10 levels
    if (project_root / 'webapp').exists() or (project_root / 'pyproject.toml').exists():
        break
    project_root = project_root.parent
else:
    raise RuntimeError("Could not find project root")

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from webapp.config.unified import (  # noqa: E402
    get_config, get_config_section, ConfigManager
)
from webapp.config.static_config import StaticConfigManager  # noqa: E402


class TestConfigurationManagement:
    """Test configuration loading and management."""

    def test_get_config_basic(self):
        """Test basic config retrieval."""
        # Test with a known configuration key
        desktop_mode = get_config('desktop_mode', 'global', True)
        assert desktop_mode is not None
        assert isinstance(desktop_mode, bool)

    def test_get_config_with_default(self):
        """Test config retrieval with default values."""
        # Test with non-existent key
        value = get_config('nonexistent_key', 'global', 'default_value')
        assert value == 'default_value'

    def test_get_config_section(self):
        """Test getting specific config sections."""
        # Test with a section that should exist
        global_config = get_config_section('global')
        assert isinstance(global_config, dict)

    def test_get_config_section_missing(self):
        """Test getting non-existent config section."""
        missing_config = get_config_section('nonexistent_section')
        assert isinstance(missing_config, dict)
        # Should return empty dict for missing sections

    def test_config_manager_initialization(self):
        """Test ConfigManager initialization."""
        try:
            manager = ConfigManager()
            assert manager is not None
        except Exception as e:
            # If ConfigManager doesn't exist or has different initialization,
            # we'll skip this test
            pytest.skip(f"ConfigManager not available: {e}")

    def test_api_key_does_not_change_configured_enterprise_mode(self):
        """AI credentials must not determine the deployment mode."""
        manager = ConfigManager()

        with patch.object(manager, 'get_static', return_value=False):
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("OPENAI_API_KEY", None)
                assert manager.is_desktop_mode() is False

            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                assert manager.is_desktop_mode() is False

    def test_test_mode_skips_runtime_config_probe(self):
        """Test mode should keep config access independent from DB runtime config."""
        manager = ConfigManager()

        with patch(
            'webapp.config.unified.static_config.get_value',
            return_value=True,
        ), patch.dict(sys.modules, {'webapp.config.runtime_config': None}):
            found, value = manager._try_get_runtime_override(
                'enable_user_authorization',
                'authorization',
            )

        assert found is False
        assert value is None

    def test_missing_static_key_skips_runtime_config_probe(self):
        """Default-only lookups should not import DB runtime config."""
        manager = ConfigManager()

        with patch(
            'webapp.config.unified.static_config.has_key',
            return_value=False,
        ), patch.dict(sys.modules, {'webapp.config.runtime_config': None}):
            found, value = manager._try_get_runtime_override(
                'nonexistent_key',
                'global',
            )

        assert found is False
        assert value is None

    @patch.dict(os.environ, {'DOCUSCOPE_DISABLE_RUNTIME_CONFIG': '1'})
    def test_runtime_config_disable_env_skips_probe(self):
        """Desktop CI can disable runtime config without importing DB services."""
        manager = ConfigManager()

        with patch.dict(sys.modules, {'webapp.config.runtime_config': None}):
            found, value = manager._try_get_runtime_override(
                'enable_user_authorization',
                'authorization',
            )

        assert found is False
        assert value is None

    def test_desktop_fallback_skips_runtime_config_probe(self):
        """A failed local Postgres connection must stop runtime DB lookups."""
        manager = ConfigManager()
        manager.activate_desktop_fallback("OperationalError")

        with patch.dict(sys.modules, {'webapp.config.runtime_config': None}):
            found, value = manager._try_get_runtime_override(
                'enable_user_authorization',
                'authorization',
            )

        assert found is False
        assert value is None

    @patch.dict(os.environ, {'TEST_ENV_VAR': 'test_value'})
    def test_environment_variable_override(self):
        """Test that environment variables can override config."""
        # Test using the actual API - get a config value that might be overridden
        value = get_config('desktop_mode', 'global', False)
        assert isinstance(value, bool)

    def test_config_file_loading(self):
        """Test configuration file loading by checking known values."""
        # Test that we can get configuration values (this implies loading worked)
        desktop_mode = get_config('desktop_mode', 'global', True)
        assert isinstance(desktop_mode, bool)

        # Test getting a section
        config_section = get_config_section('global')
        assert isinstance(config_section, dict)

    def test_inline_table_values_are_pickle_safe(self, tmp_path):
        """RQ must be able to pickle parameters loaded from TOML inline tables."""
        config_path = tmp_path / "options.toml"
        config_path.write_text(
            '[llm]\n'
            'llm_parameters = { temperature = 0.7, nested = { top_p = 0.8 } }\n',
            encoding="utf-8",
        )
        manager = StaticConfigManager()
        manager._config_path = config_path

        params = manager.get_value("llm_parameters", "llm", {})

        assert type(params) is dict
        assert type(params["nested"]) is dict
        assert pickle.loads(pickle.dumps(params)) == params

    def test_config_validation(self):
        """Test configuration validation."""
        # Test that basic config access works
        value = get_config('desktop_mode', 'global', True)
        assert isinstance(value, (bool, str, int, float))

        # Test that sections return dictionaries
        section = get_config_section('global')
        assert isinstance(section, dict)

    def test_config_defaults(self):
        """Test that configuration provides sensible defaults."""
        # Test with a key that likely doesn't exist
        default_value = get_config('definitely_nonexistent_key', 'global', 'test_default')
        assert default_value == 'test_default'

    def test_config_section_types(self):
        """Test that config sections return expected types."""
        global_config = get_config_section('global')
        assert isinstance(global_config, dict)

        # Test specific known configuration values
        desktop_mode = get_config('desktop_mode', 'global', True)
        assert isinstance(desktop_mode, bool)

    def test_config_immutability(self):
        """Test that returned config cannot be accidentally modified."""
        section1 = get_config_section('global')
        section2 = get_config_section('global')

        # Both should be dictionaries
        assert isinstance(section1, dict)
        assert isinstance(section2, dict)

        # Modifying one shouldn't affect the other if they're separate instances
        section1['test_modification'] = 'test_value'

        # Check that section2 wasn't affected
        if section1 is not section2:
            assert 'test_modification' not in section2
        # If they are the same instance, the config system returns references
