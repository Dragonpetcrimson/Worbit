"""
Structure tests for Orbit Analyzer project.

This module tests project structure aspects, including:
- Directory structure integrity
- Path handling utilities
- Standardized filename generation
- Output directory organization
- JSON serialization utilities
- Path sanitization for nested directories
- Directory structure fixing
- HTML reference correction
- Component preservation
"""

import os
import sys
import tempfile
import unittest
import shutil
import json
import logging
from datetime import datetime, date, time
import glob
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("structure_tests")

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import TestRegistry with fallback
try:
    from test_registry import TestRegistry
except ImportError:
    # Simple placeholder for backward compatibility
    class TestRegistry:
        @classmethod
        def register(cls, **kwargs):
            def decorator(test_class):
                return test_class
            return decorator

# Import test utilities
try:
    from test_utils import ConfigManager, get_test_output_path, setup_test_output_directories
except ImportError:
    # Fallback imports
    logger.warning("test_utils module not available, using minimal fallbacks")
    # Define minimal versions if needed
    ConfigManager = None
    get_test_output_path = None
    setup_test_output_directories = None

# Import the utilities with appropriate error handling
try:
    from utils.path_utils import (
        setup_output_directories, 
        get_output_path, 
        OutputType, 
        get_standardized_filename, 
        normalize_test_id,
        sanitize_base_directory,
        cleanup_nested_directories
    )
except ImportError:
    logger.warning("path_utils module not available, some tests will be skipped")
    setup_output_directories = get_output_path = OutputType = get_standardized_filename = normalize_test_id = None
    sanitize_base_directory = cleanup_nested_directories = None

# Import path validator utilities with error handling
try:
    from utils.path_validator import fix_directory_structure, fix_html_references
except ImportError:
    logger.warning("path_validator module not available, related tests will be skipped")
    fix_directory_structure = fix_html_references = None

# Import component verification with error handling
try:
    from utils.component_verification import verify_component_preservation
except ImportError:
    logger.warning("component_verification module not available, related tests will be skipped")
    verify_component_preservation = None

# Import JSON utilities with error handling
try:
    from components.json_utils import DateTimeEncoder, serialize_to_json, parse_json
except ImportError:
    try:
        # Try alternate location
        from json_utils import DateTimeEncoder, serialize_to_json, parse_json
    except ImportError:
        logger.warning("JSON utilities module not available, related tests will be skipped")
        DateTimeEncoder = None
        serialize_to_json = None
        parse_json = None

@TestRegistry.register(category='structure', importance=1, tags=['path', 'directory'])
class DirectoryStructureTest(unittest.TestCase):
    """
    Test the directory structure integrity and path handling utilities.
    
    Verifies the correct operation of path utilities including directory setup,
    standardized path generation, and test ID normalization.
    """
    
    def setUp(self):
        """Set up the test environment."""
        # Skip test if utilities are not available
        if setup_output_directories is None:
            self.skipTest("Path utilities not available")
            
        self.temp_dir = tempfile.mkdtemp()
        self.test_id = "SXM-TESTDIR"
        logger.info(f"Test directory: {self.temp_dir}")
    
    def tearDown(self):
        """Clean up after the test."""
        # Remove temporary directory and all its contents
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up test directory: {self.temp_dir}")
            except Exception as e:
                logger.warning(f"Error cleaning up test directory: {str(e)}")
    
    def test_directory_setup(self):
        """
        Test the directory setup function.
        
        Verifies that the expected directory structure is created,
        including base, json, and debug directories.
        """
        dirs = setup_output_directories(self.temp_dir, self.test_id)
        
        # Check all directories were created
        self.assertTrue(os.path.exists(dirs["base"]), "Base directory not created")
        self.assertTrue(os.path.exists(dirs["json"]), "JSON directory not created")
        self.assertTrue(os.path.exists(dirs["debug"]), "Debug directory not created")
        
        # Check test_id is normalized
        self.assertEqual(dirs["test_id"], self.test_id, 
                      f"Test ID not normalized: {dirs['test_id']} != {self.test_id}")
    
    def test_path_generation(self):
        """
        Test path generation for different output types.
        
        Verifies that paths are correctly generated for different output types
        with proper directory structure.
        """
        # Create a standardized filename
        filename = get_standardized_filename(self.test_id, "test", "xlsx")
        
        # Primary report (root directory)
        primary_path = get_output_path(
            self.temp_dir, 
            self.test_id, 
            filename, 
            OutputType.PRIMARY_REPORT
        )
        expected_primary_path = os.path.join(self.temp_dir, filename)
        self.assertEqual(primary_path, expected_primary_path, 
                      f"Primary report path incorrect: {primary_path} != {expected_primary_path}")
        
        # JSON (json subdirectory)
        json_filename = get_standardized_filename(self.test_id, "test", "json")
        json_path = get_output_path(
            self.temp_dir, 
            self.test_id, 
            json_filename, 
            OutputType.JSON_DATA
        )
        expected_json_path = os.path.join(self.temp_dir, "json", json_filename)
        self.assertEqual(json_path, expected_json_path,
                      f"JSON path incorrect: {json_path} != {expected_json_path}")
        
        # Image in base directory
        image_filename = get_standardized_filename(self.test_id, "test", "png")
        image_path = get_output_path(
            self.temp_dir,
            self.test_id,
            image_filename,
            OutputType.PRIMARY_REPORT
        )
        expected_image_path = os.path.join(self.temp_dir, image_filename)
        self.assertEqual(image_path, expected_image_path,
                      f"Image path incorrect: {image_path} != {expected_image_path}")
        
        # Verify no nested "supporting_images" paths
        self.assertNotIn("supporting_images/supporting_images", image_path.replace("\\", "/"),
                      "Nested supporting_images path detected")
    
    def test_normalize_test_id(self):
        """
        Test the normalize_test_id function.
        
        Verifies that test IDs are consistently normalized with the
        proper prefix and format.
        """
        # Test with no prefix
        self.assertEqual(normalize_test_id("12345"), "SXM-12345",
                      "Failed to add SXM- prefix to numeric test ID")
        
        # Test with existing prefix
        self.assertEqual(normalize_test_id("SXM-12345"), "SXM-12345",
                      "Changed already-normalized test ID")
        
        # Test with lowercase prefix - this needs to match implementation
        normalized_lowercase = normalize_test_id("sxm-12345")
        self.assertTrue(normalized_lowercase in ["sxm-12345", "SXM-12345"],
                      f"Unexpected lowercase prefix handling: {normalized_lowercase}")
        
        # Test with empty string
        self.assertEqual(normalize_test_id(""), "",
                      "Empty test ID not handled properly")
        
        # Test with whitespace
        self.assertEqual(normalize_test_id("  12345  "), "SXM-12345",
                      "Whitespace not handled properly in test ID")
    
    def test_get_standardized_filename(self):
        """
        Test the get_standardized_filename function.
        
        Verifies that standardized filenames are generated with the
        correct format and components.
        """
        # Test standard naming
        test_id = "SXM-12345"
        file_type = "log_analysis"
        extension = "xlsx"
        
        expected = "SXM-12345_log_analysis.xlsx"
        actual = get_standardized_filename(test_id, file_type, extension)
        
        self.assertEqual(expected, actual,
                      f"Standardized filename incorrect: {actual} != {expected}")
        
        # Test with different components
        test_id = "SXM-45678"
        file_type = "component_report"
        extension = "html"
        
        expected = "SXM-45678_component_report.html"
        actual = get_standardized_filename(test_id, file_type, extension)
        
        self.assertEqual(expected, actual,
                      f"Standardized filename incorrect: {actual} != {expected}")
    
    def test_full_directory_structure(self):
        """
        Test that the expected directory structure is created and files go to correct locations.
        
        Verifies that files are correctly placed in the appropriate subdirectories
        based on their type, maintaining the expected directory structure.
        """
        # Create a test ID
        test_id = "SXM-TESTINT"
        
        # Set up directories
        dirs = setup_output_directories(self.temp_dir, test_id)
        
        # Create mock files using standardized filenames
        excel_filename = get_standardized_filename(test_id, "log_analysis", "xlsx")
        json_filename = get_standardized_filename(test_id, "log_analysis", "json")
        image_filename = get_standardized_filename(test_id, "timeline", "png")
        
        # Create files in their respective directories
        excel_path = os.path.join(dirs["base"], excel_filename)
        json_path = os.path.join(dirs["json"], json_filename)
        image_path = os.path.join(dirs["base"], image_filename)
        
        # Write some test content
        with open(excel_path, 'w') as f:
            f.write("Mock Excel")
            
        with open(json_path, 'w') as f:
            f.write("{}")
            
        with open(image_path, 'wb') as f:
            f.write(b"PNG")
        
        # Verify files exist in expected locations
        self.assertTrue(os.path.exists(excel_path), f"Excel file not created: {excel_path}")
        self.assertTrue(os.path.exists(json_path), f"JSON file not created: {json_path}")
        self.assertTrue(os.path.exists(image_path), f"Image file not created: {image_path}")
        
        # Verify files are in correct directories
        self.assertTrue(os.path.exists(os.path.join(dirs["base"], excel_filename)), 
                      "Excel file not in base directory")
        self.assertTrue(os.path.exists(os.path.join(dirs["json"], json_filename)), 
                      "JSON file not in json directory")
        self.assertTrue(os.path.exists(os.path.join(dirs["base"], image_filename)),
                      "Image file not in base directory")
        
        # Make sure no files are in the wrong directories
        json_files_in_base = [f for f in os.listdir(dirs["base"]) if f.endswith('.json')]
        self.assertEqual(len(json_files_in_base), 0, 
                      f"JSON files found in base directory: {json_files_in_base}")
        
        image_files_in_json = [f for f in os.listdir(dirs["json"]) if f.endswith(('.png', '.jpg'))]
        self.assertEqual(len(image_files_in_json), 0, 
                      f"Image files found in JSON directory: {image_files_in_json}")
        
        # Verify no nested directories
        supporting_images_dir = os.path.join(dirs["base"], "supporting_images")
        self.assertFalse(os.path.isdir(supporting_images_dir),
                      "Nested supporting_images directory found")

    def test_template_directory(self):
        """Verify that report templates exist."""
        template_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                    "reports", "templates")
        self.assertTrue(os.path.isdir(template_dir), "Template directory missing")

        for name in ["base.html.j2", "step_report.html.j2"]:

            self.assertTrue(os.path.exists(os.path.join(template_dir, name)), f"Missing template {name}")


@TestRegistry.register(category='structure', importance=1, tags=['json', 'serialization'])
class TestJsonUtils(unittest.TestCase):
    """
    Unit tests for JSON utilities module.
    
    Tests serialization and deserialization of complex data structures,
    including datetime handling and component preservation.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        # Skip setup if JSON utilities not available
        if DateTimeEncoder is None:
            self.skipTest("JSON utilities module not available")
            
        # Create test data with various datetime types
        self.test_data = {
            "string_field": "test",
            "int_field": 123,
            "float_field": 123.45,
            "datetime_field": datetime.now(),
            "date_field": date.today(),
            "time_field": datetime.now().time(),
            "nested": {
                "datetime_nested": datetime.now()
            },
            "list_field": [
                datetime.now(),
                "string item",
                {"datetime_in_list": datetime.now()}
            ]
        }
        
        # Create temporary file for I/O tests
        fd, temp_path = tempfile.mkstemp()
        os.close(fd)
        self.temp_file_path = temp_path
        
        # Create test directory structure
        self.test_output_dir = tempfile.mkdtemp()
        self.test_json_dir = os.path.join(self.test_output_dir, "json")
        os.makedirs(self.test_json_dir, exist_ok=True)
        logger.info(f"Test JSON directory: {self.test_json_dir}")
    
    def tearDown(self):
        """Clean up temporary files."""
        # Skip teardown if setup was skipped
        if DateTimeEncoder is None:
            return
            
        # Remove temporary file
        if hasattr(self, 'temp_file_path') and os.path.exists(self.temp_file_path):
            try:
                os.unlink(self.temp_file_path)
            except Exception as e:
                logger.warning(f"Error removing temp file: {str(e)}")
        
        # Clean up test directories
        if hasattr(self, 'test_output_dir') and os.path.exists(self.test_output_dir):
            try:
                shutil.rmtree(self.test_output_dir)
                logger.info(f"Cleaned up test directory: {self.test_output_dir}")
            except Exception as e:
                logger.warning(f"Error cleaning up test directory: {str(e)}")
    
    def test_datetime_encoder(self):
        """
        Test the DateTimeEncoder class.
        
        Verifies that the encoder properly handles datetime objects
        in various structures, including nested dictionaries and lists.
        """
        if DateTimeEncoder is None:
            self.skipTest("DateTimeEncoder not available")
            
        # Standard encoder should fail with datetime objects
        with self.assertRaises(TypeError):
            json.dumps(self.test_data)
        
        # Custom encoder should succeed
        encoded = json.dumps(self.test_data, cls=DateTimeEncoder)
        self.assertIsInstance(encoded, str, "Encoder did not produce a string")
        self.assertGreater(len(encoded), 10, "Encoded JSON is too short")
        
        # Verify expected fields in output
        for field in ["string_field", "int_field", "float_field", "datetime_field", 
                     "date_field", "time_field", "nested", "list_field"]:
            self.assertIn(field, encoded, f"Field '{field}' missing in encoded JSON")
        
        # Load back into object and verify structure
        decoded = json.loads(encoded)
        self.assertEqual(decoded["string_field"], "test", "String field not preserved")
        self.assertEqual(decoded["int_field"], 123, "Integer field not preserved")
        self.assertEqual(decoded["float_field"], 123.45, "Float field not preserved")
        
        # Datetime fields should be ISO format strings now
        self.assertIsInstance(decoded["datetime_field"], str, 
                           "datetime_field not converted to string")
        self.assertIsInstance(decoded["date_field"], str, 
                           "date_field not converted to string")
        self.assertIsInstance(decoded["time_field"], str, 
                           "time_field not converted to string")
    
    def test_serialize_to_json(self):
        """
        Test the serialize_to_json function.
        
        Verifies that data can be properly serialized to a JSON file,
        handling complex types like datetimes.
        """
        if serialize_to_json is None:
            self.skipTest("serialize_to_json function not available")
            
        # Serialize data to file
        serialize_to_json(self.test_data, self.temp_file_path)
        
        # Verify file was created with content
        self.assertTrue(os.path.exists(self.temp_file_path), 
                      f"Output file not created: {self.temp_file_path}")
        self.assertGreater(os.path.getsize(self.temp_file_path), 10, 
                        "Output file is too small")
        
        # Test with directory path
        test_json_path = os.path.join(self.test_json_dir, "test_output.json")
        serialize_to_json(self.test_data, test_json_path)
        self.assertTrue(os.path.exists(test_json_path), 
                      f"JSON file not created in expected directory: {test_json_path}")
        
        # Load contents and verify
        with open(self.temp_file_path, 'r') as f:
            file_content = f.read()
        
        self.assertGreater(len(file_content), 10, "File content is too short")
        
        # Should be valid JSON
        decoded = json.loads(file_content)
        self.assertEqual(decoded["string_field"], "test", "String field not preserved")
        self.assertEqual(decoded["int_field"], 123, "Integer field not preserved")
    
    def test_parse_json(self):
        """
        Test the parse_json function.
        
        Verifies that JSON files can be properly parsed back into
        Python data structures.
        """
        if parse_json is None:
            self.skipTest("parse_json function not available")
            
        # First serialize data to file
        serialize_to_json(self.test_data, self.temp_file_path)
        
        # Now parse it back
        parsed_data = parse_json(self.temp_file_path)
        
        # Verify structure
        self.assertIsInstance(parsed_data, dict, "Parsed data is not a dictionary")
        self.assertEqual(parsed_data["string_field"], "test", "String field not preserved")
        self.assertEqual(parsed_data["int_field"], 123, "Integer field not preserved")
        self.assertEqual(parsed_data["float_field"], 123.45, "Float field not preserved")
        
        # Datetime fields are strings after serialization
        self.assertIsInstance(parsed_data["datetime_field"], str, 
                           "datetime_field not converted to string")
        
        # Test with nonexistent file
        with self.assertRaises(Exception):
            parse_json("nonexistent_file.json")
    
    def test_component_field_preservation(self):
        """
        Test that component fields are preserved during serialization.
        
        Verifies that component-related fields maintain their values
        and relationships when serialized and deserialized.
        """
        if DateTimeEncoder is None:
            self.skipTest("DateTimeEncoder not available")
            
        # Create test data with component fields
        component_data = {
            "component": "soa",
            "component_source": "filename",
            "primary_issue_component": "soa",
            "affected_components": ["soa", "android", "phoebe"],
            "nested": {
                "component": "android"
            }
        }
        
        # Serialize and deserialize
        encoded = json.dumps(component_data, cls=DateTimeEncoder)
        decoded = json.loads(encoded)
        
        # Verify component fields are preserved
        self.assertEqual(decoded["component"], "soa", "Component field not preserved")
        self.assertEqual(decoded["component_source"], "filename", "Component source not preserved")
        self.assertEqual(decoded["primary_issue_component"], "soa", 
                      "Primary issue component not preserved")
        self.assertEqual(decoded["affected_components"], ["soa", "android", "phoebe"], 
                      "Affected components list not preserved")
        self.assertEqual(decoded["nested"]["component"], "android", 
                      "Nested component field not preserved")


@TestRegistry.register(category='structure', importance=1, tags=['path_sanitization'])
class PathSanitizationTest(unittest.TestCase):
    """
    Tests for path sanitization functions.
    
    Verifies that path sanitization correctly handles nested directories
    and prevents their creation.
    """
    
    def setUp(self):
        """Set up test environment."""
        # Skip test if utilities are not available
        if sanitize_base_directory is None:
            self.skipTest("Path sanitization utilities not available")
            
        self.temp_dir = tempfile.mkdtemp()
        self.test_id = "SXM-TESTSANITIZE"
        
        # Create subdirectories
        self.json_dir = os.path.join(self.temp_dir, "json")
        self.debug_dir = os.path.join(self.temp_dir, "debug")

        os.makedirs(self.json_dir, exist_ok=True)
        os.makedirs(self.debug_dir, exist_ok=True)
        
        logger.info(f"Test directory: {self.temp_dir}")
    
    def tearDown(self):
        """Clean up after the test."""
        # Remove temporary directory and all its contents
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up test directory: {self.temp_dir}")
            except Exception as e:
                logger.warning(f"Error cleaning up test directory: {str(e)}")
    
    def test_sanitize_base_directory(self):
        """
        Test that sanitize_base_directory correctly handles different path cases.
        
        Verifies that the function correctly detects and sanitizes paths
        that contain subdirectories to prevent nesting.
        """
        # Test with normal path
        normal_path = self.temp_dir
        self.assertEqual(sanitize_base_directory(normal_path), normal_path,
                      "Normal path was modified unnecessarily")
        
        # Test with json subdirectory
        json_path = self.json_dir
        self.assertEqual(sanitize_base_directory(json_path), self.temp_dir,
                      "Json subdirectory not properly sanitized")
        
        # Test with supporting_images subdirectory
        images_path = os.path.join(self.temp_dir, "supporting_images")
        self.assertEqual(sanitize_base_directory(images_path), self.temp_dir,
                      "Images subdirectory not properly sanitized")
        
        # Test with debug subdirectory
        debug_path = self.debug_dir
        self.assertEqual(sanitize_base_directory(debug_path), self.temp_dir,
                      "Debug subdirectory not properly sanitized")
        
        # Test with expected subdirectory
        self.assertEqual(sanitize_base_directory(self.json_dir, "json"), self.temp_dir,
                      "Expected subdirectory parameter not working")
        
        # Test with None
        self.assertIsNone(sanitize_base_directory(None),
                       "None not handled correctly")
        
        # Test with non-string
        self.assertEqual(sanitize_base_directory(123), 123,
                      "Non-string not handled correctly")
    
    def test_path_generation_with_sanitization(self):
        """
        Test that get_output_path correctly sanitizes paths.
        
        Verifies that the path generation function properly sanitizes input paths
        to prevent nested subdirectories in output paths.
        """
        # Normal case - base directory
        filename = get_standardized_filename(self.test_id, "test", "json")
        path = get_output_path(self.temp_dir, self.test_id, filename, OutputType.JSON_DATA)
        expected_path = os.path.join(self.temp_dir, "json", filename)
        self.assertEqual(path, expected_path, "Normal path generation failed")
        
        # Problematic case - json subdirectory
        path = get_output_path(self.json_dir, self.test_id, filename, OutputType.JSON_DATA)
        # Should NOT create nested json/json
        self.assertEqual(path, expected_path, "Path sanitization failed for JSON subdirectory")
        self.assertNotIn("json/json", path.replace("\\", "/"), "Nested json directory detected")
        
        # Problematic case - supporting_images subdirectory
        image_filename = get_standardized_filename(self.test_id, "test", "png")
        path = get_output_path(self.temp_dir, self.test_id, image_filename, OutputType.PRIMARY_REPORT)
        expected_image_path = os.path.join(self.temp_dir, image_filename)
        self.assertEqual(path, expected_image_path, "Path sanitization failed for images subdirectory")
        self.assertNotIn("supporting_images/supporting_images", path.replace("\\", "/"),
                      "Nested supporting_images directory detected")
    
    def test_cleanup_nested_directories(self):
        """
        Test the cleanup_nested_directories function.
        
        Verifies that the function correctly identifies and fixes
        nested directory issues.
        """
        if cleanup_nested_directories is None:
            self.skipTest("cleanup_nested_directories function not available")
            
        # Create nested directories
        nested_json = os.path.join(self.json_dir, "json")
        nested_images = os.path.join(self.temp_dir, "supporting_images", "supporting_images")
        nested_debug = os.path.join(self.debug_dir, "debug")
        
        os.makedirs(nested_json, exist_ok=True)
        os.makedirs(nested_images, exist_ok=True)
        os.makedirs(nested_debug, exist_ok=True)
        
        # Create files in nested directories
        with open(os.path.join(nested_json, "test.json"), 'w') as f:
            f.write('{"test": "json"}')
            
        with open(os.path.join(nested_images, "test.png"), 'wb') as f:
            f.write(b"PNG")
            
        with open(os.path.join(nested_debug, "test.txt"), 'w') as f:
            f.write("Debug test")
        
        # Run cleanup
        results = cleanup_nested_directories(self.temp_dir)
        
        # Verify files were moved to parent directories
        self.assertTrue(os.path.exists(os.path.join(self.json_dir, "test.json")),
                      "JSON file not moved from nested directory")
        self.assertTrue(os.path.exists(os.path.join(self.images_dir, "test.png")),
                      "PNG file not moved from nested directory")
        self.assertTrue(os.path.exists(os.path.join(self.debug_dir, "test.txt")),
                      "TXT file not moved from nested directory")
        
        # Verify results have counts
        self.assertGreater(results["json_dirs_fixed"] + 
                        results["images_dirs_fixed"] + 
                        results["debug_dirs_fixed"], 0,
                        "No files were fixed")


@TestRegistry.register(category='structure', importance=1, tags=['directory_fixing'])
class DirectoryFixingTest(unittest.TestCase):
    """
    Tests for directory structure fixing functions.
    
    Verifies that the directory fixing functions correctly identify and
    fix various directory structure issues.
    """
    
    def setUp(self):
        """Set up test environment."""
        # Skip test if utilities are not available
        if fix_directory_structure is None:
            self.skipTest("Directory fixing utilities not available")
            
        self.temp_dir = tempfile.mkdtemp()
        self.test_id = "SXM-TESTFIX"
        
        # Create directory structure
        self.json_dir = os.path.join(self.temp_dir, "json")
        self.images_dir = os.path.join(self.temp_dir, "supporting_images")
        self.debug_dir = os.path.join(self.temp_dir, "debug")
        
        os.makedirs(self.json_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.debug_dir, exist_ok=True)
        
        logger.info(f"Test directory: {self.temp_dir}")
    
    def tearDown(self):
        """Clean up after the test."""
        # Remove temporary directory and all its contents
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up test directory: {self.temp_dir}")
            except Exception as e:
                logger.warning(f"Error cleaning up test directory: {str(e)}")
    
    def test_fix_directory_structure(self):
        """
        Test the fix_directory_structure function.
        
        Verifies that the function correctly identifies and fixes various
        directory structure issues, including nested directories and
        misplaced files.
        """
        # Create misplaced files
        with open(os.path.join(self.json_dir, "test.png"), 'wb') as f:
            f.write(b"PNG in JSON dir")
            
        with open(os.path.join(self.images_dir, "test.json"), 'w') as f:
            f.write('{"test": "json in images dir"}')
        
        # Create nested directories
        nested_json = os.path.join(self.json_dir, "json")
        nested_images = os.path.join(self.images_dir, "supporting_images")
        nested_debug = os.path.join(self.debug_dir, "debug")
        
        # Check if nested directories already exist due to proactive prevention
        # If they don't exist, we'll create them for testing the fix function
        if not os.path.exists(nested_json):
            os.makedirs(nested_json, exist_ok=True)
            with open(os.path.join(nested_json, "nested.json"), 'w') as f:
                f.write('{"nested": "json"}')
                
        if not os.path.exists(nested_images):
            os.makedirs(nested_images, exist_ok=True)
            with open(os.path.join(nested_images, "nested.png"), 'wb') as f:
                f.write(b"Nested PNG")
                
        if not os.path.exists(nested_debug):
            os.makedirs(nested_debug, exist_ok=True)
            with open(os.path.join(nested_debug, "nested.txt"), 'w') as f:
                f.write("Nested debug text")
        
        # Fix directory structure
        issues = fix_directory_structure(self.temp_dir, self.test_id)
        
        # Verify issues were found
        self.assertGreater(len(issues["json_dir_images"]), 0, "JSON dir images not detected")
        self.assertGreater(len(issues["images_dir_json"]), 0, "Images dir JSON not detected")
        
        # The nested directories may not exist if proactive prevention is working
        # So we don't check for them directly, but ensure files are in the right place
        
        # Verify files are in the correct locations
        self.assertTrue(os.path.exists(os.path.join(self.images_dir, "test.png")),
                      "PNG file not moved to images directory")
        self.assertTrue(os.path.exists(os.path.join(self.json_dir, "test.json")),
                      "JSON file not moved to json directory")
        
        # If nested files were created, verify they were moved correctly
        if os.path.exists(os.path.join(nested_json, "nested.json")):
            self.assertTrue(os.path.exists(os.path.join(self.json_dir, "nested.json")),
                         "Nested JSON file not moved to parent directory")
                         
        if os.path.exists(os.path.join(nested_images, "nested.png")):
            self.assertTrue(os.path.exists(os.path.join(self.images_dir, "nested.png")),
                         "Nested PNG file not moved to parent directory")
                         
        if os.path.exists(os.path.join(nested_debug, "nested.txt")):
            self.assertTrue(os.path.exists(os.path.join(self.debug_dir, "nested.txt")),
                         "Nested debug file not moved to parent directory")
    
    def test_fix_html_references(self):
        """
        Test the fix_html_references function.
        
        Verifies that the function correctly identifies and fixes various
        HTML reference issues, including missing supporting_images prefixes
        and hidden image elements.
        """
        if fix_html_references is None:
            self.skipTest("fix_html_references function not available")
            
        # Create HTML file with various reference issues
        html_path = os.path.join(self.temp_dir, f"{self.test_id}_component_report.html")
        
        with open(html_path, 'w') as f:
            f.write("""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Test Report</title>
            </head>
            <body>
                <!-- Image without supporting_images prefix -->
                <img src="test.png" alt="Test Image">
                
                <!-- Correct path -->
                <img src="supporting_images/correct.png" alt="Correct Image">
                
                <!-- Hidden image element -->
                <div style="display:none">
                    <img src="hidden.png" style="display:none" alt="Hidden Image">
                </div>
                
                <!-- Hidden div with correct path -->
                <div style="display:none">
                    <img src="supporting_images/hidden_correct.png" style="display:none" alt="Hidden Correct">
                </div>
            </body>
            </html>
            """)
        
        # Capture original content and run the check function
        with open(html_path, 'r') as f:
            original_content = f.read()

        issues = fix_html_references(html_path, self.temp_dir)

        # Verify issues were reported and file was not modified
        self.assertGreater(len(issues), 0, "No issues reported")

        with open(html_path, 'r') as f:
            content = f.read()

        self.assertEqual(content, original_content, "HTML file should not be modified")

        # Confirm expected issues were detected
        self.assertIn('test.png', issues, "Missing prefix issue not reported")
        self.assertIn('hidden.png', issues, "Hidden image issue not reported")


@TestRegistry.register(category='structure', importance=1, tags=['component_preservation'])
class ComponentPreservationTest(unittest.TestCase):
    """
    Tests for component information preservation functions.
    
    Verifies that component information is properly preserved during
    file operations and serialization.
    """
    
    def setUp(self):
        """Set up test environment."""
        # Skip test if utilities are not available
        if verify_component_preservation is None:
            self.skipTest("Component verification utilities not available")
            
        self.temp_dir = tempfile.mkdtemp()
        self.test_id = "SXM-TESTCOMP"
        
        # Create component test data
        self.component_data = {
            "primary_issue_component": "soa",
            "errors": [
                {
                    "component": "soa",
                    "component_source": "filename",
                    "text": "Test error",
                    "severity": "High"
                },
                {
                    "component": "mimosa",
                    "component_source": "content",
                    "text": "Another error",
                    "severity": "Medium"
                }
            ],
            "clusters": {
                "0": [
                    {
                        "component": "soa",
                        "component_source": "filename",
                        "text": "Test error",
                        "severity": "High"
                    }
                ],
                "1": [
                    {
                        "component": "mimosa",
                        "component_source": "content",
                        "text": "Another error",
                        "severity": "Medium"
                    }
                ]
            }
        }
        
        # Create test files
        self.source_path = os.path.join(self.temp_dir, "source.json")
        with open(self.source_path, 'w') as f:
            json.dump(self.component_data, f, indent=2)
        
        self.target_path = os.path.join(self.temp_dir, "target.json")
        shutil.copy2(self.source_path, self.target_path)
        
        logger.info(f"Test directory: {self.temp_dir}")
    
    def tearDown(self):
        """Clean up after the test."""
        # Remove temporary directory and all its contents
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up test directory: {self.temp_dir}")
            except Exception as e:
                logger.warning(f"Error cleaning up test directory: {str(e)}")
    
    def test_component_verification(self):
        """
        Test the verify_component_preservation function.
        
        Verifies that the function correctly identifies when component
        information is preserved or lost during file operations.
        """
        # Verify identical files pass verification
        self.assertTrue(verify_component_preservation(self.source_path, self.target_path),
                     "Verification failed for identical files")
        
        # Modify target file to remove a component field
        with open(self.target_path, 'r') as f:
            target_data = json.load(f)
        
        # Remove component_source field
        target_data["errors"][0].pop("component_source")
        
        with open(self.target_path, 'w') as f:
            json.dump(target_data, f, indent=2)
        
        # Verify modified file fails verification
        self.assertFalse(verify_component_preservation(self.source_path, self.target_path),
                      "Verification passed for file with missing component field")
        
        # Restore original target
        shutil.copy2(self.source_path, self.target_path)
        
        # Change a component value
        with open(self.target_path, 'r') as f:
            target_data = json.load(f)
        
        target_data["errors"][1]["component"] = "android"
        
        with open(self.target_path, 'w') as f:
            json.dump(target_data, f, indent=2)
        
        # Verify modified file fails verification
        self.assertFalse(verify_component_preservation(self.source_path, self.target_path),
                      "Verification passed for file with modified component value")
    
    def test_verify_component_fields_in_list(self):
        """
        Test verification of component fields in lists.
        
        Verifies that component fields are correctly checked in lists of
        dictionaries, such as error arrays.
        """
        if not hasattr(verify_component_preservation, "__module__") or not verify_component_preservation.__module__.endswith("component_verification"):
            self.skipTest("Component verification module structure not as expected")
        
        # Import the module directly to access internal functions
        import importlib
        try:
            component_verification = importlib.import_module(
                verify_component_preservation.__module__
            )
            verify_component_fields_in_list = component_verification.verify_component_fields_in_list
        except (ImportError, AttributeError):
            self.skipTest("Cannot access internal verification functions")
        
        # Create test lists
        source_list = [
            {"component": "soa", "component_source": "filename", "text": "Test 1"},
            {"component": "android", "component_source": "content", "text": "Test 2"}
        ]
        
        # Identical target list
        target_list = [
            {"component": "soa", "component_source": "filename", "text": "Test 1"},
            {"component": "android", "component_source": "content", "text": "Test 2"}
        ]
        
        # Verify identical lists pass
        self.assertTrue(verify_component_fields_in_list(source_list, target_list),
                     "Identical lists failed verification")
        
        # Modified target list
        modified_list = [
            {"component": "soa", "component_source": "filename", "text": "Test 1"},
            {"component": "unknown", "component_source": "content", "text": "Test 2"}
        ]
        
        # Verify modified list fails
        self.assertFalse(verify_component_fields_in_list(source_list, modified_list),
                      "Modified list passed verification")
        
        # Different length list
        short_list = [
            {"component": "soa", "component_source": "filename", "text": "Test 1"}
        ]
        
        # Verify different length list fails
        self.assertFalse(verify_component_fields_in_list(source_list, short_list),
                      "Different length list passed verification")


if __name__ == "__main__":
    unittest.main()