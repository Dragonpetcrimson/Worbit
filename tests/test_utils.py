"""
Test utilities for Orbit Analyzer tests.

This module provides common functionality used across test modules, including
configuration management, test data creation, path handling, and validation helpers.
"""

import os
import sys
import tempfile
import logging
import shutil
import json
import time
import unittest
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from functools import wraps

# Import test_config if available
try:
    from test_config import TEST_CONFIG
except ImportError:
    # Default configuration if test_config is not available
    TEST_CONFIG = {
        "TEST_DATA_DIR": os.path.join(os.path.dirname(__file__), "test_data"),
        "OUTPUT_DIR": os.path.join(os.path.dirname(__file__), "output"),
        "LOGS_DIR": os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs"),
        "LOG_LEVEL": "INFO",
        "SKIP_SLOW_TESTS": False
    }

# Configure logging
logging.basicConfig(
    level=getattr(logging, TEST_CONFIG.get("LOG_LEVEL", "INFO")),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_utils")

class ConfigManager:
    """
    Manages test configuration with proper overrides.
    
    This class provides a centralized way to access configuration values,
    with a clear precedence order: defaults -> config file -> environment.
    """
    
    _config = {}  # Cached configuration
    
    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key to retrieve
            default: Default value if key doesn't exist
            
        Returns:
            The configuration value, or default if not found
        """
        if not cls._config:
            cls._load_config()
        return cls._config.get(key, default)
    
    @classmethod
    def _load_config(cls) -> None:
        """
        Load configuration from all sources.
        
        Follows precedence: defaults -> config file -> environment variables
        """
        # Start with defaults
        cls._config = {
            "TEST_DATA_DIR": os.path.join(os.path.dirname(__file__), "test_data"),
            "OUTPUT_DIR": os.path.join(os.path.dirname(__file__), "output"),
            "LOGS_DIR": os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs"),
            "LOG_LEVEL": "INFO",
            "SKIP_SLOW_TESTS": False,
            "TEST_VERBOSITY": 1,
            "FAIL_FAST": False,
            "TEST_TIMEOUT": 60,
            "MAX_MEMORY_USAGE": 512
        }
        
        # Override from test_config.py
        try:
            from test_config import TEST_CONFIG
            cls._config.update(TEST_CONFIG)
        except ImportError:
            logger.warning("test_config.py not found, using default configuration")
        
        # Override from environment variables
        for key in cls._config:
            env_value = os.environ.get(key)
            if env_value is not None:
                # Convert to appropriate type based on default
                if isinstance(cls._config[key], bool):
                    cls._config[key] = env_value.lower() in ("true", "yes", "1", "t")
                elif isinstance(cls._config[key], int):
                    try:
                        cls._config[key] = int(env_value)
                    except ValueError:
                        logger.warning(f"Could not convert {key}={env_value} to int, using default")
                else:
                    cls._config[key] = env_value
        
        # Create necessary directories
        for dir_key in ["TEST_DATA_DIR", "OUTPUT_DIR", "LOGS_DIR"]:
            os.makedirs(cls._config[dir_key], exist_ok=True)
    
    @classmethod
    def set(cls, key: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            key: Configuration key to set
            value: Value to set
        """
        if not cls._config:
            cls._load_config()
        cls._config[key] = value

def find_suitable_log_folder() -> Optional[str]:
    """
    Find the first available SXM-* folder in logs directory.
    
    Looks for SXM-* folders in the configured logs directory to use as
    test data source.
    
    Returns:
        Path to first suitable test folder, or None if none found
    """
    base_dir = ConfigManager.get("LOGS_DIR")
    if not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)
        return None
        
    for item in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, item)
        if os.path.isdir(folder_path) and item.startswith("SXM-"):
            return folder_path
            
    return None

def create_test_data() -> str:
    """
    Creates minimal test data if no real data found.
    
    Generates a test directory with basic log files for testing when
    no suitable real test data is available.
    
    Returns:
        Path to the created test data directory
    """
    test_folder = os.path.join(ConfigManager.get("TEST_DATA_DIR"), "logs")
    os.makedirs(test_folder, exist_ok=True)
    
    # Create a simple log file
    with open(os.path.join(test_folder, "sample.log"), "w") as f:
        f.write("2025-03-26 12:00:00 INFO: Application starting\n")
        f.write("2025-03-26 12:00:05 ERROR: Database connection failed\n")
        f.write("2025-03-26 12:00:10 WARNING: Retrying operation\n")
        f.write("2025-03-26 12:00:15 ERROR: Operation timeout\n")
    
    # Create a fake app_debug.log file for component testing
    with open(os.path.join(test_folder, "app_debug.log"), "w") as f:
        f.write("2025-03-26 12:00:00 INFO: SOA application starting\n")
        f.write("2025-03-26 12:00:05 ERROR: Exception in SOA component\n")
        f.write("2025-03-26 12:00:10 WARNING: SOA network delay\n")
    
    # Create a simple mimosa log file
    with open(os.path.join(test_folder, "mimosa.log"), "w") as f:
        f.write("2025-03-26 12:00:00 INFO: Mimosa starting\n")
        f.write("2025-03-26 12:00:05 ERROR: Data feed error\n")
    
    # Create a test image for OCR testing
    create_test_image(os.path.join(test_folder, "screenshot.png"), "Test Error Screen")
    
    return test_folder

def create_test_image(path: str, text: str) -> None:
    """
    Create a simple test image with text for OCR testing.
    
    Args:
        path: Path where the image should be saved
        text: Text to include in the image
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
        
        # Create a blank image with white background
        img = Image.new('RGB', (400, 200), color='white')
        draw = ImageDraw.Draw(img)
        
        # Try to use a default font, fallback to bitmap if not available
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except IOError:
            font = ImageFont.load_default()
        
        # Draw the text
        draw.text((20, 80), text, fill='black', font=font)
        
        # Save the image
        img.save(path)
    except ImportError:
        # If PIL is not available, create an empty PNG file
        logger.warning("PIL (Pillow) is not available, creating an empty test image")
        with open(path, 'wb') as f:
            # Minimal PNG file - not a valid image but sufficient for tests
            f.write(bytes.fromhex('89504E470D0A1A0A0000000D49484452000000010000000108060000001F15C4890000000D4944415478DA63FAFFFF3F0300060600024891BE000000004945' + 
                                 '4E44AE426082'))

def get_test_folder() -> str:
    """
    Get a valid test folder, creating one if needed.
    
    Returns:
        Path to a test folder containing log files
    """
    folder = find_suitable_log_folder()
    if folder:
        return folder
    else:
        logger.warning("No SXM-* folders found in logs directory - using generated test data")
        return create_test_data()

def setup_test_output_directories(test_id: str) -> Dict[str, str]:
    """
    Set up standardized test output directories.
    
    Creates a standard directory structure for test outputs including
    base directory, json subdirectory, and debug subdirectory.
    
    Args:
        test_id: Test identifier
        
    Returns:
        Dictionary with paths to created directories
    """
    # Normalize test ID (add SXM- prefix if missing)
    if not test_id.startswith("SXM-") and not test_id.startswith("TEST-"):
        test_id = f"SXM-{test_id}"
    
    # Create base directories
    base_dir = os.path.join(ConfigManager.get("OUTPUT_DIR"), test_id)
    json_dir = os.path.join(base_dir, "json")
    debug_dir = os.path.join(base_dir, "debug")

    # Create all directories
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)

    return {
        "base": base_dir,
        "json": json_dir,
        "debug": debug_dir,
        "test_id": test_id
    }

def get_test_output_path(test_id: str, filename: str, file_type: Optional[str] = None) -> str:
    """
    Get standardized test output path.
    
    Determines the correct path for a file based on its type and ensures
    the destination directory exists.
    
    Args:
        test_id: Test identifier
        filename: Name of the file
        file_type: Optional type ('json', 'image', 'debug')
        
    Returns:
        Full path to the file in the appropriate directory
    """
    # Get output directories for the test
    dirs = setup_test_output_directories(test_id)
    
    if file_type == "json":
        return os.path.join(dirs["json"], filename)
    elif file_type == "debug":
        return os.path.join(dirs["debug"], filename)
    else:
        return os.path.join(dirs["base"], filename)

def validate_directory_structure(test_id: str) -> Dict[str, List[str]]:
    """
    Validate the test output directory structure.
    
    Checks that the directory structure follows the standards and
    files are in the correct locations.
    
    Args:
        test_id: Test identifier
        
    Returns:
        Dictionary with validation results
    """
    dirs = setup_test_output_directories(test_id)
    
    issues = {
        "json_in_images": [],  # JSON files in images directory (legacy)
        "images_in_json": [],  # Image files in JSON directory
        "json_in_base": [],    # JSON files in base directory
        "images_in_base": [],  # Image files in base directory
        "other_issues": []
    }

    # Legacy check: images directory no longer used
    images_dir = os.path.join(dirs["base"], "supporting_images")
    if os.path.isdir(images_dir):
        for filename in os.listdir(images_dir):
            if filename.endswith(".json"):
                issues["json_in_images"].append(filename)
    
    # Check files in JSON directory
    for filename in os.listdir(dirs["json"]):
        if any(filename.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".gif"]):
            issues["images_in_json"].append(filename)
    
    # Check files in base directory
    for filename in os.listdir(dirs["base"]):
        if os.path.isfile(os.path.join(dirs["base"], filename)):
            if filename.endswith(".json"):
                issues["json_in_base"].append(filename)
            elif any(filename.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".gif"]):
                issues["images_in_base"].append(filename)
    
    return issues

def validate_report_file(file_path: str, required_fields: List[str] = None) -> Tuple[bool, List[str]]:
    """
    Validate a report file exists and contains required fields.
    
    Args:
        file_path: Path to the report file
        required_fields: List of fields that should be present in JSON reports
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    # Check file exists
    if not os.path.exists(file_path):
        issues.append(f"File does not exist: {file_path}")
        return False, issues
    
    # Check file has content
    if os.path.getsize(file_path) == 0:
        issues.append(f"File is empty: {file_path}")
        return False, issues
    
    # For JSON files, check required fields
    if file_path.endswith(".json") and required_fields:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for field in required_fields:
                if field not in data:
                    issues.append(f"Missing required field: {field}")
        except json.JSONDecodeError:
            issues.append("File is not valid JSON")
        except Exception as e:
            issues.append(f"Error reading file: {str(e)}")
    
    return len(issues) == 0, issues

def validate_visualization(image_path: str, min_size: int = 1000) -> Tuple[bool, List[str]]:
    """
    Validate that a visualization image was properly created.
    
    Args:
        image_path: Path to the image file
        min_size: Minimum expected file size in bytes
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    # Check file exists
    if not os.path.exists(image_path):
        issues.append(f"Image file not created: {image_path}")
        return False, issues
    
    # Check file has reasonable size
    if os.path.getsize(image_path) < min_size:
        issues.append(f"Image file is too small: {image_path}")
    
    # Check for accidental nested directories from legacy structure
    if "supporting_images/supporting_images" in image_path.replace("\\", "/"):
        issues.append(
            f"Image path contains legacy supporting_images nesting: {image_path}"
        )
    
    return len(issues) == 0, issues

def timeit(func):
    """
    Decorator to time function execution.
    
    Args:
        func: Function to time
        
    Returns:
        Wrapped function that logs execution time
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        result = func(*args, **kwargs)
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"{func.__name__} completed in {duration:.2f} seconds")
        return result
    return wrapper

def has_required_module(module_name: str) -> bool:
    """
    Check if a required module is available.
    
    Args:
        module_name: Name of the module to check
        
    Returns:
        True if module is available, False otherwise
    """
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False

class MockLogEntry:
    """Mock log entry for testing."""
    def __init__(self, timestamp=None, is_error=False, file="test.log", line_num=1, 
                 severity="Medium", text="Test log entry", component=None):
        self.timestamp = timestamp or datetime.now()
        self.is_error = is_error
        self.file = file
        self.line_number = line_num  # LogEntry in gherkin_log_correlator uses line_number
        self.line_num = line_num     # Error objects use line_num
        self.severity = severity
        self.text = text
        self.component = component

class MockGherkinStep:
    """Mock Gherkin step for testing."""
    def __init__(self, step_number, text="Test step", keyword="Given"):
        self.step_number = step_number
        self.text = text
        self.keyword = keyword

def create_mock_errors(count: int = 5, components: List[str] = None) -> List[Dict]:
    """
    Create a list of mock errors for testing.
    
    Args:
        count: Number of errors to create
        components: List of components to use (defaults to ['soa', 'android', 'mimosa'])
        
    Returns:
        List of error dictionaries
    """
    if components is None:
        components = ['soa', 'android', 'mimosa']
    
    errors = []
    now = datetime.now()
    
    for i in range(count):
        component = components[i % len(components)]
        severity = "High" if i % 3 == 0 else "Medium" if i % 3 == 1 else "Low"
        
        errors.append({
            "file": f"{component}.log",
            "line_num": i + 1,
            "text": f"Test error {i+1} in {component}",
            "severity": severity,
            "timestamp": (now + timedelta(seconds=i*5)).isoformat(),
            "component": component
        })
    
    return errors

def create_mock_clusters(errors: List[Dict], num_clusters: int = 2) -> Dict[int, List[Dict]]:
    """
    Create mock clusters from errors for testing.
    
    Args:
        errors: List of errors to group into clusters
        num_clusters: Number of clusters to create
        
    Returns:
        Dictionary mapping cluster IDs to lists of errors
    """
    clusters = {}
    
    for i in range(num_clusters):
        # Assign errors to clusters based on index
        cluster_errors = [error for j, error in enumerate(errors) if j % num_clusters == i]
        if cluster_errors:
            clusters[i] = cluster_errors
    
    return clusters

def get_component_schema_path() -> str:
    """
    Get the path to the component schema file.
    
    Returns:
        Path to the component schema file
    """
    # Check possible locations
    possible_paths = [
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'components', 'schemas', 'component_schema.json'),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'schemas', 'component_schema.json'),
        os.path.join(os.path.dirname(__file__), 'test_data', 'component_schema.json')
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    # If not found, create a minimal schema in test_data
    schema_path = os.path.join(ConfigManager.get("TEST_DATA_DIR"), "component_schema.json")
    os.makedirs(os.path.dirname(schema_path), exist_ok=True)
    
    # Create a minimal schema if it doesn't exist
    if not os.path.exists(schema_path):
        schema = {
            "components": [
                {
                    "id": "soa",
                    "name": "SOA",
                    "description": "Test app component",
                    "type": "application",
                    "logSources": ["app_debug.log"],
                    "errorPatterns": ["error.*soa", "exception"]
                },
                {
                    "id": "mimosa",
                    "name": "Mimosa",
                    "description": "Test data provider",
                    "type": "test_data_provider",
                    "logSources": ["mimosa.log"],
                    "errorPatterns": ["error.*data", "unavailable"]
                },
                {
                    "id": "android",
                    "name": "Android",
                    "description": "Platform component",
                    "type": "platform",
                    "logSources": ["android.log", "logcat.log"],
                    "errorPatterns": ["error.*android", "crash"]
                }
            ],
            "dataFlows": [
                {
                    "source": "mimosa",
                    "target": "soa",
                    "description": "Test data flow",
                    "dataType": "test_data"
                }
            ]
        }
        
        with open(schema_path, 'w') as f:
            json.dump(schema, f, indent=2)
    
    return schema_path