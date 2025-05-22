"""
test_config.py - Default configuration settings for Orbit Analyzer tests

This module defines default test configuration settings that can be imported
and used by test modules. It serves as a bridge between the old configuration
approach and the new ConfigManager in test_utils.py.

Configuration precedence:
1. Environment variables
2. User configuration file (test_config.json)
3. Default values defined here

For new tests, it's recommended to use the ConfigManager from test_utils.py directly.

Available configuration options:
- LOGS_DIR: Directory containing log files for testing
- OUTPUT_DIR: Directory for test output files
- TEST_DATA_DIR: Directory for test data files
- SAMPLE_FEATURE_FILE: Path to sample Gherkin feature file
- SAMPLE_LOGS_DIR: Directory containing sample logs for testing
- GHERKIN_LOG_CORRELATOR_PATH: Path to the Gherkin log correlator module
- ENHANCED_ADAPTERS_PATH: Path to enhanced adapters module
- SKIP_SLOW_TESTS: Whether to skip slow tests (bool)
- TEST_VERBOSITY: Verbosity level for test output (int, 0-3)
- FAIL_FAST: Whether to stop testing on first failure (bool)
- TEST_TIMEOUT: Maximum time in seconds for test execution (int)
- MAX_MEMORY_USAGE: Maximum memory usage in MB (int)
"""

import os
import sys
import json
import logging
from typing import Dict, Any, Union, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_config")

# Base directories
base_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(base_dir)

def get_env_value(key: str, default: Any, value_type: type = str) -> Any:
    """
    Get environment variable with proper type conversion.
    
    Args:
        key: Environment variable name
        default: Default value if not found
        value_type: Type to convert to (str, int, float, bool)
        
    Returns:
        Properly typed value from environment or default
    """
    env_value = os.environ.get(key)
    
    if env_value is None:
        return default
        
    try:
        if value_type == bool:
            return env_value.lower() in ('true', '1', 'yes', 't', 'y')
        elif value_type == int:
            return int(env_value)
        elif value_type == float:
            return float(env_value)
        else:
            return env_value
    except (ValueError, TypeError):
        logger.warning(f"Could not convert environment variable {key}={env_value} to {value_type.__name__}, using default: {default}")
        return default

# Define default configuration with proper type conversion
TEST_CONFIG = {
    # Input paths
    "LOGS_DIR": get_env_value("LOGS_DIR", os.path.join(parent_dir, "logs")),
    
    # Output paths
    "OUTPUT_DIR": get_env_value("OUTPUT_DIR", os.path.join(base_dir, "output")),
    
    # Test data directory
    "TEST_DATA_DIR": get_env_value("TEST_DATA_DIR", os.path.join(base_dir, "test_data")),
 
    # Configuration for Gherkin log correlation tests
    'SAMPLE_FEATURE_FILE': get_env_value("SAMPLE_FEATURE_FILE", ''),
    'SAMPLE_LOGS_DIR': get_env_value("SAMPLE_LOGS_DIR", ''),
    
    # Path to Gherkin-related modules
    'GHERKIN_LOG_CORRELATOR_PATH': get_env_value("GHERKIN_LOG_CORRELATOR_PATH", 
                                               os.path.join(parent_dir, 'gherkin_log_correlator.py')),
    'ENHANCED_ADAPTERS_PATH': get_env_value("ENHANCED_ADAPTERS_PATH", 
                                          os.path.join(parent_dir, 'enhanced_adapters.py')),
    
    # Test execution settings
    'SKIP_SLOW_TESTS': get_env_value("SKIP_SLOW_TESTS", False, bool),
    'TEST_VERBOSITY': get_env_value("TEST_VERBOSITY", 1, int),
    'FAIL_FAST': get_env_value("FAIL_FAST", False, bool),
    
    # Resource limits
    'TEST_TIMEOUT': get_env_value("TEST_TIMEOUT", 60, int),  # seconds
    'MAX_MEMORY_USAGE': get_env_value("MAX_MEMORY_USAGE", 512, int),  # MB
}

def load_user_config() -> Dict[str, Any]:
    """
    Load user configuration from test_config.json if it exists.
    
    Returns:
        Dictionary with user configuration or empty dict if not found/invalid
    """
    user_config_path = os.path.join(base_dir, 'test_config.json')
    if os.path.exists(user_config_path):
        try:
            with open(user_config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
                logger.info(f"Loaded user configuration from {user_config_path}")
                return user_config
        except Exception as e:
            logger.warning(f"Error loading user configuration: {e}")
    return {}

# Update config with user settings
TEST_CONFIG.update(load_user_config())

def create_directories() -> None:
    """
    Create necessary directories for testing.
    
    Creates the output, test data, and logs directories if they don't exist.
    Checks for permissions and provides helpful error messages.
    """
    directories = [
        ("OUTPUT_DIR", TEST_CONFIG["OUTPUT_DIR"]),
        ("TEST_DATA_DIR", TEST_CONFIG["TEST_DATA_DIR"]),
        ("LOGS_DIR", TEST_CONFIG["LOGS_DIR"])
    ]
    
    for name, path in directories:
        try:
            os.makedirs(path, exist_ok=True)
            if not os.path.exists(path):
                logger.warning(f"Failed to create {name} directory at {path}")
            elif not os.access(path, os.W_OK):
                logger.warning(f"No write permission for {name} directory at {path}")
        except Exception as e:
            logger.warning(f"Error creating {name} directory at {path}: {e}")

# Create the necessary directories
create_directories()

# Legacy compatibility
# Define top-level variables for backward compatibility with old test scripts
LOGS_DIR = TEST_CONFIG["LOGS_DIR"]
OUTPUT_DIR = TEST_CONFIG["OUTPUT_DIR"]
TEST_DATA_DIR = TEST_CONFIG["TEST_DATA_DIR"]
SAMPLE_FEATURE_FILE = TEST_CONFIG["SAMPLE_FEATURE_FILE"]
SAMPLE_LOGS_DIR = TEST_CONFIG["SAMPLE_LOGS_DIR"]
GHERKIN_LOG_CORRELATOR_PATH = TEST_CONFIG["GHERKIN_LOG_CORRELATOR_PATH"]
ENHANCED_ADAPTERS_PATH = TEST_CONFIG["ENHANCED_ADAPTERS_PATH"]
SKIP_SLOW_TESTS = TEST_CONFIG["SKIP_SLOW_TESTS"]
TEST_VERBOSITY = TEST_CONFIG["TEST_VERBOSITY"]
FAIL_FAST = TEST_CONFIG["FAIL_FAST"]
TEST_TIMEOUT = TEST_CONFIG["TEST_TIMEOUT"]
MAX_MEMORY_USAGE = TEST_CONFIG["MAX_MEMORY_USAGE"]

# Try to interface with the new ConfigManager if available
try:
    from test_utils import ConfigManager
    # Update ConfigManager with our settings
    if hasattr(ConfigManager, '_config'):
        ConfigManager._config.update(TEST_CONFIG)
        logger.info("Updated ConfigManager with settings from test_config.py")
    else:
        logger.warning("ConfigManager found but doesn't have expected structure")
except ImportError:
    logger.debug("ConfigManager not available - operating in standalone mode")
except Exception as e:
    logger.warning(f"Error interfacing with ConfigManager: {e}")