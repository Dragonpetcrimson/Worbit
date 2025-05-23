"""
tests/utils/path_utils_test.py - Test path handling utilities
"""

import os
import sys
import unittest
import tempfile
import shutil

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the module to test
from utils.path_utils import (
    normalize_test_id,
    setup_output_directories,
    get_output_path,
    OutputType,
    get_standardized_filename
)

class TestPathUtils(unittest.TestCase):
    """Tests for the path_utils module."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        self.test_id = "SXM-123456"
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_normalize_test_id(self):
        """Test normalize_test_id function."""
        # Test with "SXM-" prefix
        self.assertEqual(normalize_test_id("SXM-123456"), "SXM-123456")
        
        # Test without prefix
        self.assertEqual(normalize_test_id("123456"), "SXM-123456")
        
        # Test with whitespace
        self.assertEqual(normalize_test_id("  123456  "), "SXM-123456")
        
        # Test with empty string
        self.assertEqual(normalize_test_id(""), "")
    
    def test_setup_output_directories(self):
        """Test setup_output_directories function."""
        # Test with SXM- prefix
        output_paths = setup_output_directories(self.temp_dir, "SXM-123456")
        
        # Check test_id is preserved
        self.assertEqual(output_paths["test_id"], "SXM-123456")
        
        # Check directories are created
        json_dir = os.path.join(self.temp_dir, "json")
        debug_dir = os.path.join(self.temp_dir, "debug")
        
        self.assertTrue(os.path.exists(json_dir), "JSON directory was not created")
        self.assertTrue(os.path.exists(debug_dir), "Debug directory was not created")
        
        # Check paths in returned dictionary
        self.assertEqual(output_paths["base"], self.temp_dir)
        self.assertEqual(output_paths["json"], json_dir)
        self.assertEqual(output_paths["debug"], debug_dir)
        
        # Test without SXM- prefix
        output_paths = setup_output_directories(self.temp_dir, "123456")
        
        # Check prefix is added
        self.assertEqual(output_paths["test_id"], "SXM-123456")
    
    def test_get_output_path(self):
        """Test get_output_path function."""
        filename = "test_file.txt"
        
        # Test PRIMARY_REPORT
        primary_path = get_output_path(self.temp_dir, self.test_id, filename, OutputType.PRIMARY_REPORT)
        self.assertEqual(primary_path, os.path.join(self.temp_dir, filename))
        self.assertTrue(os.path.exists(os.path.dirname(primary_path)))
        
        # Test JSON_DATA
        json_path = get_output_path(self.temp_dir, self.test_id, filename, OutputType.JSON_DATA)
        self.assertEqual(json_path, os.path.join(self.temp_dir, "json", filename))
        self.assertTrue(os.path.exists(os.path.dirname(json_path)))
        
        
        # Test DEBUGGING
        debug_path = get_output_path(self.temp_dir, self.test_id, filename, OutputType.DEBUGGING)
        self.assertEqual(debug_path, os.path.join(self.temp_dir, "debug", filename))
        self.assertTrue(os.path.exists(os.path.dirname(debug_path)))
        
        # Test with create_dirs=False
        non_existent = os.path.join(self.temp_dir, "non_existent")
        path = get_output_path(non_existent, self.test_id, filename, create_dirs=False)
        self.assertFalse(os.path.exists(os.path.dirname(path)))
    
    def test_get_standardized_filename(self):
        """Test get_standardized_filename function."""
        # Test with SXM- prefix
        filename = get_standardized_filename("SXM-123456", "log_analysis", "xlsx")
        self.assertEqual(filename, "SXM-123456_log_analysis.xlsx")
        
        # Test without SXM- prefix
        filename = get_standardized_filename("123456", "component_report", "html")
        self.assertEqual(filename, "SXM-123456_component_report.html")
        
        # Test with whitespace
        filename = get_standardized_filename("  123456  ", "error_graph", "json")
        self.assertEqual(filename, "SXM-123456_error_graph.json")

def test_path_utils():
    """Function-based test for path_utils module."""
    print("Testing Path Utils Module...")
    
    # Test normalize_test_id
    try:
        # Test with "SXM-" prefix
        normalized = normalize_test_id("SXM-123456")
        if normalized != "SXM-123456":
            print(f"❌ normalize_test_id failed with SXM- prefix: got {normalized}")
            return False
            
        # Test without prefix
        normalized = normalize_test_id("123456")
        if normalized != "SXM-123456":
            print(f"❌ normalize_test_id failed without prefix: got {normalized}")
            return False
            
        # Test with whitespace
        normalized = normalize_test_id("  123456  ")
        if normalized != "SXM-123456":
            print(f"❌ normalize_test_id failed with whitespace: got {normalized}")
            return False
            
        print("✅ normalize_test_id function works correctly")
    except Exception as e:
        print(f"❌ Error testing normalize_test_id: {str(e)}")
        return False
    
    # Test setup_output_directories
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test with SXM- prefix
            output_paths = setup_output_directories(temp_dir, "SXM-123456")
            
            if output_paths["test_id"] != "SXM-123456":
                print(f"❌ setup_output_directories failed to set test_id correctly: {output_paths['test_id']}")
                return False
                
            json_dir = os.path.join(temp_dir, "json")
            debug_dir = os.path.join(temp_dir, "debug")
            
            if not os.path.exists(json_dir):
                print("❌ setup_output_directories failed to create json directory")
                return False
                
                
            if not os.path.exists(debug_dir):
                print("❌ setup_output_directories failed to create debug directory")
                return False
                
            print("✅ setup_output_directories function works correctly")
    except Exception as e:
        print(f"❌ Error testing setup_output_directories: {str(e)}")
        return False
    
    # Test get_output_path
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            filename = "test_file.txt"
            test_id = "SXM-123456"
            
            # Test PRIMARY_REPORT
            primary_path = get_output_path(temp_dir, test_id, filename, OutputType.PRIMARY_REPORT)
            if primary_path != os.path.join(temp_dir, filename):
                print(f"❌ get_output_path returned incorrect path for PRIMARY_REPORT: {primary_path}")
                return False
                
            # Test JSON_DATA
            json_path = get_output_path(temp_dir, test_id, filename, OutputType.JSON_DATA)
            if json_path != os.path.join(temp_dir, "json", filename):
                print(f"❌ get_output_path returned incorrect path for JSON_DATA: {json_path}")
                return False
                
                
            # Test DEBUGGING
            debug_path = get_output_path(temp_dir, test_id, filename, OutputType.DEBUGGING)
            if debug_path != os.path.join(temp_dir, "debug", filename):
                print(f"❌ get_output_path returned incorrect path for DEBUGGING: {debug_path}")
                return False
                
            print("✅ get_output_path function works correctly")
    except Exception as e:
        print(f"❌ Error testing get_output_path: {str(e)}")
        return False
    
    # Test get_standardized_filename
    try:
        # Test with SXM- prefix
        filename = get_standardized_filename("SXM-123456", "log_analysis", "xlsx")
        if filename != "SXM-123456_log_analysis.xlsx":
            print(f"❌ get_standardized_filename failed with SXM- prefix: got {filename}")
            return False
            
        # Test without SXM- prefix
        filename = get_standardized_filename("123456", "component_report", "html")
        if filename != "SXM-123456_component_report.html":
            print(f"❌ get_standardized_filename failed without prefix: got {filename}")
            return False
            
        print("✅ get_standardized_filename function works correctly")
    except Exception as e:
        print(f"❌ Error testing get_standardized_filename: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    # Run the function-based test
    success = test_path_utils()
    if success:
        print("✅ All path_utils tests passed!")
    else:
        print("❌ path_utils tests failed!")
        sys.exit(1)

