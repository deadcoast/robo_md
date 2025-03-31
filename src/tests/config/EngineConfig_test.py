import logging
from unittest.mock import Mock, patch

import pytest

from src.config.EngineConfig import EngineConfig
from src.config.SystemConfig import SystemConfig


class TestEngineConfig:
    """Test suite for the EngineConfig class."""

    @pytest.fixture
    def system_config(self):
        """Create a mock SystemConfig instance for testing."""
        # Create a mock with better support for special methods
        mock_config = Mock(spec=SystemConfig)
        
        # Configure special methods with direct function assignments rather than Mock objects
        # This approach avoids issues with Callable mock objects
        mock_config.__str__.return_value = "MockSystemConfig"
        mock_config.__repr__.return_value = "MockSystemConfig"
        
        # Use a stable hash value for consistent behavior
        mock_config.__hash__.return_value = hash("MockSystemConfig")
        
        # Set up equality to compare by identity
        mock_config.__eq__.side_effect = lambda other: mock_config is other
        mock_config.__ne__.side_effect = lambda other: mock_config is not other
        
        return mock_config

    @pytest.fixture
    def engine_config(self, system_config):
        """Create an EngineConfig instance with mocked logger for testing."""
        with patch("src.config.EngineConfig.logging.getLogger") as mock_get_logger:
            # Create mock logger
            mock_logger = Mock(spec=logging.Logger)
            mock_get_logger.return_value = mock_logger

            # Create engine config
            config = EngineConfig(system_config)

            # Set mocked logger for easier testing
            config.logger = mock_logger

            return config

    def test_init(self, engine_config, system_config):
        """Test initialization of EngineConfig."""
        assert engine_config.config is system_config
        assert engine_config.logger is not None

        # Verify logger methods were called
        engine_config.logger.info.assert_any_call("Engine configuration initialized.")
        engine_config.logger.debug.assert_any_call(
            f"Engine configuration: {system_config}"
        )
        engine_config.logger.info.assert_any_call("Engine configuration loaded.")

    def test_load_config(self, engine_config):
        """Test _load_config method."""
        # Reset the mock to clear previous calls
        engine_config.logger.reset_mock()

        # Call the method
        engine_config._load_config()

        # Verify logger calls
        engine_config.logger.info.assert_any_call("Loading engine configuration.")
        engine_config.logger.debug.assert_called_once()
        engine_config.logger.info.assert_any_call("Engine configuration loaded.")

    def test_get_config(self, engine_config, system_config):
        """Test get_config method."""
        # Reset the mock to clear previous calls
        engine_config.logger.reset_mock()

        # Call the method
        result = engine_config.get_config()

        # Verify
        assert result is system_config
        engine_config.logger.info.assert_called_once()

    def test_update_config(self, engine_config):
        """Test update_config method."""
        # Reset the mock to clear previous calls
        engine_config.logger.reset_mock()

        # Create a new mock config
        new_config = Mock(spec=SystemConfig)

        # Call the method
        engine_config.update_config(new_config)

        # Verify
        assert engine_config.config is new_config
        engine_config.logger.info.assert_any_call("Engine configuration updated.")
        engine_config.logger.debug.assert_called_once()

    def test_str(self, engine_config, system_config):
        """Test __str__ method."""
        # Reset the mock to clear previous calls
        engine_config.logger.reset_mock()

        # Configure system_config.__str__ to return a known value
        system_config.__str__.return_value = "SystemConfig String"

        # Call the method
        result = str(engine_config)

        # Verify
        assert result == "SystemConfig String"
        engine_config.logger.info.assert_called_once()

    def test_repr(self, engine_config, system_config):
        """Test __repr__ method."""
        # Reset the mock to clear previous calls
        engine_config.logger.reset_mock()

        # Configure system_config.__str__ to return a known value
        system_config.__str__.return_value = "SystemConfig Repr"

        # Call the method
        result = repr(engine_config)

        # Verify
        assert result == "SystemConfig Repr"
        engine_config.logger.info.assert_called_once()

    def test_equality(self, engine_config, system_config):
        """Test equality methods."""
        # Reset the mock to clear previous calls
        engine_config.logger.reset_mock()

        # Create an identical config
        with patch("src.config.EngineConfig.logging.getLogger"):
            identical_config = EngineConfig(system_config)

        # Create a different config
        different_system_config = Mock(spec=SystemConfig)
        with patch("src.config.EngineConfig.logging.getLogger"):
            different_config = EngineConfig(different_system_config)

        # Test equality
        assert engine_config == identical_config
        assert engine_config != different_config
        assert engine_config != "not an engine config"

        # Test inequality
        assert engine_config == identical_config
        assert engine_config != different_config

        # Verify logger calls
        engine_config.logger.info.assert_called()

    def test_hash(self, engine_config, system_config):
        """Test __hash__ method."""
        # Reset the mock to clear previous calls
        engine_config.logger.reset_mock()

        # Configure system_config.__hash__ to return a known value
        system_config.__hash__.return_value = 12345

        # Call the method
        result = hash(engine_config)

        # Verify
        assert result == 12345
        engine_config.logger.info.assert_called_once()

    def test_copy(self, engine_config, system_config):
        """Test __copy__ method."""
        # Reset the mock to clear previous calls
        engine_config.logger.reset_mock()

        with patch("src.config.EngineConfig.__init__", return_value=None) as mock_init:
            # Call the method
            result = engine_config.__copy__()

            # Verify
            assert isinstance(result, EngineConfig)
            mock_init.assert_called_once_with(system_config)
            engine_config.logger.info.assert_called_once()

    def test_deepcopy(self, engine_config, system_config):
        """Test __deepcopy__ method."""
        # Reset the mock to clear previous calls
        engine_config.logger.reset_mock()

        memo: dict[int, object] = {}

        with patch("src.config.EngineConfig.__init__", return_value=None) as mock_init:
            # Call the method
            result = engine_config.__deepcopy__(memo)

            # Verify
            assert isinstance(result, EngineConfig)
            mock_init.assert_called_once_with(system_config)
            engine_config.logger.info.assert_called_once()

    def test_reduce(self, engine_config, system_config):
        """Test __reduce__ method."""
        # Reset the mock to clear previous calls
        engine_config.logger.reset_mock()

        # Call the method
        cls, args = engine_config.__reduce__()

        # Verify
        assert cls is EngineConfig
        assert len(args) == 1
        assert args[0] is system_config
        engine_config.logger.info.assert_called_once()

    def test_reduce_ex(self, engine_config, system_config):
        """Test __reduce_ex__ method."""
        # Reset the mock to clear previous calls
        engine_config.logger.reset_mock()

        # Call the method
        cls, args = engine_config.__reduce_ex__(4)  # protocol 4

        # Verify
        assert cls is EngineConfig
        assert len(args) == 1
        assert args[0] is system_config
        engine_config.logger.info.assert_called_once()

    def test_getstate(self, engine_config, system_config):
        """Test __getstate__ method."""
        # Reset the mock to clear previous calls
        engine_config.logger.reset_mock()

        # Call the method
        state = engine_config.__getstate__()

        # Verify
        assert isinstance(state, dict)
        assert "config" in state
        assert state["config"] is system_config
        engine_config.logger.info.assert_called_once()

    def test_setstate(self, engine_config):
        """Test __setstate__ method."""
        # Reset the mock to clear previous calls
        engine_config.logger.reset_mock()

        # Create a new mock config
        new_config = Mock(spec=SystemConfig)

        # Prepare state
        state = {"config": new_config}

        # Mock the _load_config method
        with patch.object(engine_config, "_load_config") as mock_load_config:
            # Call the method
            engine_config.__setstate__(state)

            # Verify
            assert engine_config.config is new_config
            mock_load_config.assert_called_once()
            engine_config.logger.info.assert_called_once()
