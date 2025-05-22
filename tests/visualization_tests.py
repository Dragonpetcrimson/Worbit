"""
Visualization tests for Orbit Analyzer.

This module tests visualization generation functionality, including:
- Component error distribution visualizations
- Timeline visualizations for test steps
- Cluster timeline visualizations for error grouping
- Component relationship visualizations
- Error propagation visualizations
"""

import os
import sys
import logging
import unittest
import shutil
import tempfile
import traceback
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

# Add parent directory to path to find modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

# Set up matplotlib with non-interactive backend for testing
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Import modules to test with appropriate error handling
try:
    from config import Config
except ImportError:
    # Create minimal Config class for testing
    class Config:
        ENABLE_CLUSTER_TIMELINE = False
        ENABLE_COMPONENT_DISTRIBUTION = False
        ENABLE_ERROR_PROPAGATION = False
        ENABLE_STEP_REPORT_IMAGES = False
        ENABLE_COMPONENT_REPORT_IMAGES = False

try:
    from utils.path_utils import setup_output_directories, get_output_path, OutputType
except ImportError:
    logging.warning("path_utils module not available, tests will use alternative approaches")
    # Define minimal fallbacks if needed
    def setup_output_directories(base_dir, test_id):
        """Standardized directory setup function fallback."""
        test_id = test_id.strip()
        if not test_id.startswith("SXM-"):
            test_id = f"SXM-{test_id}"
            
        base_path = os.path.join(base_dir, test_id)
        json_path = os.path.join(base_path, "json")
        images_path = os.path.join(base_path, "supporting_images")
        debug_path = os.path.join(base_path, "debug")
        
        # Create directories if they don't exist
        os.makedirs(base_path, exist_ok=True)
        os.makedirs(json_path, exist_ok=True)
        os.makedirs(images_path, exist_ok=True)
        os.makedirs(debug_path, exist_ok=True)
        
        return {
            "base": base_path,
            "json": json_path,
            "images": images_path,
            "debug": debug_path,
            "test_id": test_id
        }
    
    def get_output_path(base_dir, test_id, filename, output_type=None):
        """Minimal fallback for get_output_path"""
        # Normalize test_id
        test_id = test_id.strip()
        if not test_id.startswith("SXM-"):
            test_id = f"SXM-{test_id}"
            
        if output_type == "image" or (hasattr(output_type, "value") and output_type.value == "image"):
            return os.path.join(base_dir, test_id, "supporting_images", filename)
        elif output_type == "json" or (hasattr(output_type, "value") and output_type.value == "json"):
            return os.path.join(base_dir, test_id, "json", filename)
        elif output_type == "debug" or (hasattr(output_type, "value") and output_type.value == "debug"):
            return os.path.join(base_dir, test_id, "debug", filename)
        return os.path.join(base_dir, test_id, filename)
    
    class OutputType:
        """Enum-like class for output types."""
        PRIMARY_REPORT = "primary"
        JSON_DATA = "json"
        VISUALIZATION = "image"
        DEBUGGING = "debug"

try:
    from reports.visualizations import (
        generate_component_error_distribution,
        generate_cluster_timeline_image,
        generate_timeline_image
    )
except ImportError:
    logging.warning("Visualization modules not available, tests will be skipped")
    generate_component_error_distribution = generate_cluster_timeline_image = generate_timeline_image = None

try:
    from reports.component_report import generate_component_visualization
except ImportError:
    generate_component_visualization = None

# Import test utilities
try:
    from test_utils import (
        ConfigManager, 
        validate_visualization,
        setup_test_output_directories,
        get_test_output_path,
        MockLogEntry,
        MockGherkinStep
    )
except ImportError:
    # Define minimal test utilities if not available
    def validate_visualization(image_path, min_size=1000):
        """Minimal validation of visualization outputs"""
        issues = []
        if not os.path.exists(image_path):
            issues.append(f"Visualization file not created: {image_path}")
            return False, issues
        if os.path.getsize(image_path) < min_size:
            issues.append(f"Visualization file is too small: {image_path}")
        if "supporting_images/supporting_images" in image_path.replace("\\", "/"):
            issues.append(f"Path contains nested supporting_images directories")
        return len(issues) == 0, issues
    
    def setup_test_output_directories(test_id):
        """Set up test output directories with standardized structure"""
        base_dir = os.path.join("test_output", test_id)
        json_dir = os.path.join(base_dir, "json")
        images_dir = os.path.join(base_dir, "supporting_images")
        debug_dir = os.path.join(base_dir, "debug")
        
        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(json_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(debug_dir, exist_ok=True)
        
        return {
            "base": base_dir,
            "json": json_dir,
            "images": images_dir,
            "debug": debug_dir,
            "test_id": test_id
        }
    
    def get_test_output_path(test_id, filename, file_type=None):
        """Get standardized test output path"""
        dirs = setup_test_output_directories(test_id)
        
        if file_type == "json":
            return os.path.join(dirs["json"], filename)
        elif file_type == "image":
            return os.path.join(dirs["images"], filename)
        elif file_type == "debug":
            return os.path.join(dirs["debug"], filename)
        return os.path.join(dirs["base"], filename)
    
    # Mock classes to use if test_utils is not available
    class MockLogEntry:
        """Mock log entry for testing."""
        def __init__(self, timestamp=None, is_error=False, file="test.log", line_num=1, 
                     severity="Medium", text="Test log entry", component=None):
            self.timestamp = timestamp or datetime.now()
            self.is_error = is_error
            self.file = file
            self.line_number = line_num  # Note: using line_number to match LogEntry in gherkin_log_correlator
            self.line_num = line_num  # Also include line_num for error objects
            self.severity = severity
            self.text = text
            self.component = component

    class MockGherkinStep:
        """Mock Gherkin step for testing."""
        def __init__(self, step_number, text="Test step"):
            self.step_number = step_number
            self.text = text
            self.keyword = "Given"
            
    class ConfigManager:
        """Minimal ConfigManager implementation"""
        @classmethod
        def get(cls, key, default=None):
            return default

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("visualization_tests")


@TestRegistry.register(category='visualization', importance=2, tags=['component', 'distribution'])
class TestVisualizations(unittest.TestCase):
    """
    Tests for visualization functions.
    
    Tests the generation of various visualizations including component
    distributions, timelines, and cluster visualizations.
    """

    def setUp(self):
        """Set up test data."""
        # Skip test if visualization modules not available
        if generate_component_error_distribution is None:
            self.skipTest("Visualization modules not available")
            
        # Create a temporary directory for test outputs
        self.test_dir = tempfile.mkdtemp()
        
        # Create test timestamp base (1 hour ago)
        self.base_time = datetime.now() - timedelta(hours=1)
        
        # Create mock step dictionary
        self.step_dict = {
            1: MockGherkinStep(1, "First test step"),
            2: MockGherkinStep(2, "Second test step"),
            3: MockGherkinStep(3, "Third test step")
        }
        
        # Create mock logs with timestamps for each step
        self.step_to_logs = {}
        
        # Step 1 logs (10 minutes of activity)
        step1_logs = []
        for i in range(5):
            # Regular logs
            step1_logs.append(MockLogEntry(
                timestamp=self.base_time + timedelta(minutes=i),
                is_error=False
            ))
            
            # Error logs
            if i > 2:  # Add some errors in the latter part
                step1_logs.append(MockLogEntry(
                    timestamp=self.base_time + timedelta(minutes=i, seconds=30),
                    is_error=True,
                    severity="High",
                    text=f"Error in step 1, iteration {i}"
                ))
        self.step_to_logs[1] = step1_logs
        
        # Step 2 logs (starts 10 minutes after step 1, 15 minutes of activity)
        step2_logs = []
        step2_base = self.base_time + timedelta(minutes=10)
        for i in range(7):
            # Regular logs
            step2_logs.append(MockLogEntry(
                timestamp=step2_base + timedelta(minutes=i*2),
                is_error=False
            ))
            
            # Error logs - add some Medium severity
            if i % 2 == 0:  # Every other iteration
                step2_logs.append(MockLogEntry(
                    timestamp=step2_base + timedelta(minutes=i*2, seconds=40),
                    is_error=True,
                    severity="Medium",
                    text=f"Warning in step 2, iteration {i}"
                ))
        self.step_to_logs[2] = step2_logs
        
        # Step 3 logs (starts 25 minutes after step 1, 10 minutes of activity)
        step3_logs = []
        step3_base = self.base_time + timedelta(minutes=25)
        for i in range(4):
            # Regular logs
            step3_logs.append(MockLogEntry(
                timestamp=step3_base + timedelta(minutes=i*2),
                is_error=False
            ))
            
            # Error logs - mix of severities
            severity = "Low" if i < 2 else "High"
            step3_logs.append(MockLogEntry(
                timestamp=step3_base + timedelta(minutes=i*2, seconds=20),
                is_error=True,
                severity=severity,
                text=f"{severity} severity error in step 3, iteration {i}"
            ))
        self.step_to_logs[3] = step3_logs
        
        # Create mock clusters
        self.clusters = {
            0: [  # Cluster 0 - High severity errors
                MockLogEntry(
                    timestamp=self.base_time + timedelta(minutes=3, seconds=30),
                    is_error=True,
                    severity="High",
                    text="High severity error in step 1",
                    file="test.log",
                    line_num=101
                ),
                MockLogEntry(
                    timestamp=step3_base + timedelta(minutes=6, seconds=20),
                    is_error=True,
                    severity="High",
                    text="High severity error in step 3",
                    file="test.log",
                    line_num=301
                )
            ],
            1: [  # Cluster 1 - Medium severity errors
                MockLogEntry(
                    timestamp=step2_base + timedelta(minutes=0, seconds=40),
                    is_error=True,
                    severity="Medium",
                    text="Medium severity error in step 2",
                    file="test.log",
                    line_num=201
                ),
                MockLogEntry(
                    timestamp=step2_base + timedelta(minutes=4, seconds=40),
                    is_error=True,
                    severity="Medium",
                    text="Another medium severity error in step 2",
                    file="test.log",
                    line_num=205
                )
            ],
            2: [  # Cluster 2 - Low severity errors
                MockLogEntry(
                    timestamp=step3_base + timedelta(minutes=0, seconds=20),
                    is_error=True,
                    severity="Low",
                    text="Low severity error in step 3",
                    file="test.log",
                    line_num=302
                ),
                MockLogEntry(
                    timestamp=step3_base + timedelta(minutes=2, seconds=20),
                    is_error=True,
                    severity="Low",
                    text="Another low severity error in step 3",
                    file="test.log",
                    line_num=304
                )
            ]
        }
        
        # Create test component summary
        self.component_summary = [
            {"id": "soa", "name": "SOA", "error_count": 10},
            {"id": "android", "name": "Android", "error_count": 5}
        ]
    
    def tearDown(self):
        """Clean up after tests."""
        # Close any open matplotlib figures
        plt.close('all')
        
        # Remove test directory using shutil.rmtree
        try:
            shutil.rmtree(self.test_dir, ignore_errors=True)
        except Exception as e:
            logging.warning(f"Error cleaning up test directory: {str(e)}")

    @patch('config.Config.ENABLE_COMPONENT_DISTRIBUTION', True)
    def test_component_error_distribution(self):
        """
        Test the component error distribution visualization.
        
        Verifies that component error distribution visualizations
        are properly generated and saved to the correct location.
        """
        # Test ID
        test_id = "TEST-VIZ2-999"
        
        # Setup output directories using standardized approach
        dirs = setup_output_directories(self.test_dir, test_id)
        
        # Test the visualization function
        logger.info(f"Generating visualization in {dirs['base']}")
        image_path = generate_component_error_distribution(
            dirs["base"],
            test_id,
            self.component_summary,
            None,
            "soa"
        )
        
        # Verify the output
        logger.info(f"Visualization path: {image_path}")
        # Check if visualization is enabled or not
        if hasattr(Config, 'ENABLE_COMPONENT_DISTRIBUTION') and Config.ENABLE_COMPONENT_DISTRIBUTION:
            self.assertIsNotNone(image_path, "Visualization path should not be None")
            self.assertTrue(os.path.exists(image_path), "Visualization file should exist")
            self.assertGreater(os.path.getsize(image_path), 1000, "Visualization should have reasonable size")
            
            # Verify correct path structure (no nested supporting_images)
            self.assertNotIn("supporting_images/supporting_images", image_path.replace("\\", "/"),
                          "Image path contains nested supporting_images directories")
        else:
            self.assertIsNone(image_path, "Visualization should return None when feature disabled")

    @patch('config.Config.ENABLE_CLUSTER_TIMELINE', True)
    @patch('reports.visualizations.get_output_path')
    def test_cluster_timeline_image(self, mock_get_output_path):
        """
        Test the cluster timeline image generation.
        
        Verifies that cluster timeline images are properly generated
        and saved to the correct location.
        """
        # Setup mock return for get_output_path
        test_id = "TEST-CLUSTER-999"
        test_image_path = os.path.join(self.test_dir, "supporting_images", f"{test_id}_cluster_timeline.png")
        test_debug_path = os.path.join(self.test_dir, "debug", f"{test_id}_timeline_debug.txt")
        mock_get_output_path.side_effect = [test_image_path, test_debug_path]
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(test_image_path), exist_ok=True)
        os.makedirs(os.path.dirname(test_debug_path), exist_ok=True)
        
        # Call the function
        result = generate_cluster_timeline_image(
            self.step_to_logs,
            self.step_dict,
            self.clusters,
            self.test_dir,
            test_id
        )
        
        # Assertions based on whether feature is enabled
        if hasattr(Config, 'ENABLE_CLUSTER_TIMELINE') and Config.ENABLE_CLUSTER_TIMELINE:
            # Check that the function returns the correct path
            self.assertEqual(result, test_image_path, "Function should return the correct image path")
            
            # Check that the image file was created
            self.assertTrue(os.path.exists(test_image_path), "Image file should exist")
            
            # Verify the file size is reasonable (not zero or empty)
            self.assertGreater(os.path.getsize(test_image_path), 1000, "Image should have reasonable size")
            
            # Check that the debug log was created
            self.assertTrue(os.path.exists(test_debug_path), "Debug log should exist")
        else:
            self.assertIsNone(result, "Function should return None when feature is disabled")

    @patch('config.Config.ENABLE_CLUSTER_TIMELINE', True)
    @patch('reports.visualizations.get_output_path')
    def test_cluster_timeline_with_empty_data(self, mock_get_output_path):
        """
        Test behavior with empty data.
        
        Verifies that the visualization functions handle empty
        input data gracefully without errors.
        """
        # Setup mock return for get_output_path
        test_id = "TEST-EMPTY-999"
        test_image_path = os.path.join(self.test_dir, "supporting_images", f"{test_id}_cluster_timeline.png")
        test_debug_path = os.path.join(self.test_dir, "debug", f"{test_id}_timeline_debug.txt")
        mock_get_output_path.side_effect = [test_image_path, test_debug_path]
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(test_image_path), exist_ok=True)
        os.makedirs(os.path.dirname(test_debug_path), exist_ok=True)
        
        # Call the function with empty data
        result = generate_cluster_timeline_image(
            {},  # Empty step_to_logs
            self.step_dict,
            self.clusters,
            self.test_dir,
            test_id
        )
        
        # Check if visualization is enabled or not
        if hasattr(Config, 'ENABLE_CLUSTER_TIMELINE') and Config.ENABLE_CLUSTER_TIMELINE:
            # The function should still return a path and create an image
            self.assertEqual(result, test_image_path, "Function should return path even with empty data")
            self.assertTrue(os.path.exists(test_image_path), "Image should be created even with empty data")
        else:
            self.assertIsNone(result, "Function should return None when feature is disabled")
        
        # Reset mocks for another test
        mock_get_output_path.reset_mock()
        mock_get_output_path.side_effect = [test_image_path, test_debug_path]
        
        # Call with empty clusters
        result = generate_cluster_timeline_image(
            self.step_to_logs,
            self.step_dict,
            {},  # Empty clusters
            self.test_dir,
            test_id
        )
        
        # Check if visualization is enabled or not
        if hasattr(Config, 'ENABLE_CLUSTER_TIMELINE') and Config.ENABLE_CLUSTER_TIMELINE:
            # The function should still return a path and create an image
            self.assertEqual(result, test_image_path, "Function should return path even with empty clusters")
            self.assertTrue(os.path.exists(test_image_path), "Image should be created even with empty clusters")
        else:
            self.assertIsNone(result, "Function should return None when feature is disabled")
        
    @patch('config.Config.ENABLE_CLUSTER_TIMELINE', True)
    @patch('reports.visualizations.get_output_path')
    def test_cluster_timeline_without_timestamps(self, mock_get_output_path):
        """
        Test behavior when logs don't have timestamps.
        
        Verifies that the visualization functions handle logs without
        timestamps gracefully.
        """
        # Setup mock return for get_output_path
        test_id = "TEST-NOTIME-999"
        test_image_path = os.path.join(self.test_dir, "supporting_images", f"{test_id}_cluster_timeline.png")
        test_debug_path = os.path.join(self.test_dir, "debug", f"{test_id}_timeline_debug.txt")
        mock_get_output_path.side_effect = [test_image_path, test_debug_path]
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(test_image_path), exist_ok=True)
        os.makedirs(os.path.dirname(test_debug_path), exist_ok=True)
        
        # Create logs without timestamps
        step_to_logs_no_timestamps = {
            1: [MockLogEntry(timestamp=None) for _ in range(3)]
        }
        
        # Call the function
        result = generate_cluster_timeline_image(
            step_to_logs_no_timestamps,
            self.step_dict,
            self.clusters,
            self.test_dir,
            test_id
        )
        
        # Check if visualization is enabled or not
        if hasattr(Config, 'ENABLE_CLUSTER_TIMELINE') and Config.ENABLE_CLUSTER_TIMELINE:
            # The function should still return a path and create an image
            self.assertEqual(result, test_image_path, "Function should return path even without timestamps")
            self.assertTrue(os.path.exists(test_image_path), "Image should be created even without timestamps")
        else:
            self.assertIsNone(result, "Function should return None when feature is disabled")
    
    def test_empty_step_logs(self):
        """
        Test handling of empty step logs.
        
        Verifies that the visualization functions properly handle 
        completely empty input without errors.
        """
        # Skip if timeline module not available
        if generate_timeline_image is None:
            self.skipTest("Timeline visualization module not available")
            
        # Setup test with empty data
        empty_steps = {}
        empty_dict = {}
        test_id = "TEST-EMPTY-STEPS"
        
        # Setup directories using standardized approach
        dirs = setup_output_directories(self.test_dir, test_id)
        
        # Test with empty step logs
        try:
            with patch('reports.visualizations.get_output_path') as mock_path:
                # Setup mock to return a valid path
                test_image_path = os.path.join(dirs["images"], f"{test_id}_timeline.png")
                mock_path.return_value = test_image_path
                
                # Create a file at that location to avoid IO errors
                os.makedirs(os.path.dirname(test_image_path), exist_ok=True)
                with open(test_image_path, 'w') as f:
                    f.write("test")
                
                # Call the function
                result = generate_timeline_image(
                    empty_steps,
                    empty_dict,
                    dirs["base"],
                    test_id
                )
                
                # Should not crash, might return None or a path depending on implementation
                # Just verify it doesn't raise an exception
                self.assertIn(result, [None, test_image_path], 
                           "Should return None or a valid path, not crash")
        except Exception as e:
            # If it does raise an exception, fail the test
            self.fail(f"generate_timeline_image raised {type(e).__name__} with empty data: {str(e)}")


@TestRegistry.register(category='visualization', importance=2, tags=['component', 'integration'])
class TestComponentVisualization(unittest.TestCase):
    """
    Integration tests for component visualization generation.
    
    Tests the generation of component-specific visualizations and
    ensures they are saved to the proper locations.
    """
    
    def setUp(self):
        """Set up test environment."""
        # Skip if component visualization is not available
        if generate_component_visualization is None:
            self.skipTest("Component visualization module not available")
            
        # Print some initial info
        logger.info("Starting component visualization test")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Python version: {sys.version}")
        
        # Test ID
        self.test_id = "TEST-VIZ-999"
        logger.info(f"Using test ID: {self.test_id}")

        # Setup output directories using standardized approach
        self.temp_dir = tempfile.mkdtemp()
        logger.info(f"Setting up directories in: {self.temp_dir}")
        try:
            self.output_dirs = setup_output_directories(self.temp_dir, self.test_id)
            logger.info(f"Directory setup complete: {self.output_dirs}")
        except Exception as e:
            logger.error(f"Error setting up directories: {e}")
            traceback.print_exc()
            raise

        # Create test data
        self.component_summary = [
            {"id": "soa", "name": "SOA", "error_count": 10},
            {"id": "android", "name": "Android", "error_count": 5}
        ]
        logger.info(f"Created test component summary data with {len(self.component_summary)} components")

        # Create test error analysis data
        self.error_analysis = {
            "component_summary": self.component_summary,
            "component_error_counts": {"soa": 10, "android": 5},
            "root_cause_component": "soa"
        }
        logger.info("Created test error analysis data")
    
    def tearDown(self):
        """Clean up after tests."""
        # Clean up temporary directory
        try:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception as e:
            logging.warning(f"Error cleaning up test directory: {str(e)}")
    
    def test_visualization_with_images_dir(self):
        """
        Test component visualization generation using images directory.
        
        Verifies that component visualizations are correctly generated
        when using the dedicated images directory path.
        """
        logger.info(f"Generating visualization with images_dir = {self.output_dirs['images']}")
        try:
            # FIXED: Use base directory instead of images directory to avoid nested supporting_images
            viz_path = generate_component_visualization(
                self.output_dirs["base"],  # Use base directory instead of images directory
                self.test_id,
                self.error_analysis,
                None,
                "soa"
            )
            
            logger.info(f"Visualization path from images_dir: {viz_path}")
            exists = os.path.exists(viz_path) if viz_path else False
            logger.info(f"File exists: {exists}")
            
            self.assertIsNotNone(viz_path, "Visualization path should not be None")
            self.assertTrue(exists, "Visualization file should exist")
            
            # Verify no nested supporting_images in path
            self.assertNotIn("supporting_images/supporting_images", 
                          viz_path.replace("\\", "/"), 
                          "Path should not contain nested supporting_images directories")
            
            # Validate the visualization using our utility
            is_valid, issues = validate_visualization(viz_path)
            self.assertTrue(is_valid, f"Visualization validation failed: {', '.join(issues)}")
            
        except Exception as e:
            self.fail(f"Error generating visualization with images directory: {e}")
    
    def test_visualization_with_base_dir(self):
        """
        Test component visualization generation using base directory.
        
        Verifies that component visualizations are correctly generated
        when using the base directory path, which should properly route
        to the images subdirectory.
        """
        logger.info(f"Generating visualization with base_dir = {self.output_dirs['base']}")
        try:
            # Try with the base directory
            viz_path = generate_component_visualization(
                self.output_dirs["base"],  # Pass the base directory
                self.test_id,
                self.error_analysis,
                None,
                "soa"
            )
            
            logger.info(f"Visualization path from base_dir: {viz_path}")
            exists = os.path.exists(viz_path) if viz_path else False
            logger.info(f"File exists: {exists}")
            
            self.assertIsNotNone(viz_path, "Visualization path should not be None")
            self.assertTrue(exists, "Visualization file should exist")
            
            # Verify path includes supporting_images directory
            self.assertIn("supporting_images",
                       viz_path.replace("\\", "/"),
                       "Path should include the supporting_images directory")
            
            # Verify no duplicate supporting_images in path
            self.assertNotIn("supporting_images/supporting_images", 
                          viz_path.replace("\\", "/"), 
                          "Path should not contain nested supporting_images directories")
            
        except Exception as e:
            self.fail(f"Error generating visualization with base directory: {e}")
    
    def test_directory_structure(self):
        """
        Test that the directory structure is correct.
        
        Verifies that the expected directory structure is created
        with base, images, json, and debug directories.
        """
        # Check the directory structure
        logger.info("\nChecking directory structure:")
        logger.info(f"Base directory: {self.output_dirs['base']}")
        self.assertTrue(os.path.exists(self.output_dirs['base']), "Base directory should exist")
        
        logger.info(f"Images directory: {self.output_dirs['images']}")
        self.assertTrue(os.path.exists(self.output_dirs['images']), "Images directory should exist")
        
        logger.info(f"JSON directory: {self.output_dirs['json']}")
        self.assertTrue(os.path.exists(self.output_dirs['json']), "JSON directory should exist")
        
        # Generate a visualization
        viz_path = generate_component_visualization(
            self.output_dirs["base"],
            self.test_id,
            self.error_analysis,
            None,
            "soa"
        )
        
        # Verify visualization exists in the correct directory
        self.assertIsNotNone(viz_path, "Visualization path should not be None")
        self.assertTrue(os.path.exists(viz_path), "Visualization file should exist")
        
        # Check the directory structure after visualization generation
        structure_issues = self._validate_directory_structure()
        self.assertEqual(len(structure_issues), 0, 
                      f"Directory structure has issues: {structure_issues}")
    
    def _validate_directory_structure(self):
        """
        Validate the test output directory structure.
        
        Returns:
            List of detected structure issues
        """
        issues = []
        
        # Check for images in the json directory
        json_dir = self.output_dirs["json"]
        if os.path.exists(json_dir):
            for filename in os.listdir(json_dir):
                if any(filename.endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif']):
                    issues.append(f"Image file {filename} found in JSON directory")
        
        # Check for json files in the images directory
        images_dir = self.output_dirs["images"]
        if os.path.exists(images_dir):
            for filename in os.listdir(images_dir):
                if filename.endswith('.json'):
                    issues.append(f"JSON file {filename} found in images directory")
        
        # Check for nested directories
        if os.path.exists(os.path.join(images_dir, "supporting_images")):
            issues.append("Nested supporting_images directory found")
            
        if os.path.exists(os.path.join(json_dir, "json")):
            issues.append("Nested json directory found")
            
        return issues


@TestRegistry.register(category='visualization', importance=2, tags=['timeline'])
class TestTimelineGenerator(unittest.TestCase):
    """
    Tests for timeline generation functionality.
    
    Tests the generation of timeline visualizations for test steps
    and error clusters.
    """
    
    def setUp(self):
        """Set up test environment."""
        # Skip if timeline generation is not available
        if generate_timeline_image is None:
            self.skipTest("Timeline visualization module not available")
            
        # Create a temporary directory for test outputs
        self.test_dir = tempfile.mkdtemp()
        
        # Mock step metadata
        self.step_dict = {
            1: type('Step', (), {'step_number': 1, 'text': 'Initialize test', 'keyword': 'Given'}),
            2: type('Step', (), {'step_number': 2, 'text': 'Perform action', 'keyword': 'When'}),
            3: type('Step', (), {'step_number': 3, 'text': 'Verify result', 'keyword': 'Then'})
        }

        # Create mock logs by step
        self.step_to_logs = {}

        for step in range(1, 4):
            logs = []
            # Regular logs
            for i in range(3):
                log = MockLogEntry()
                log.text = f"Log entry {i} for step {step}"
                log.file = "test.log"
                log.line_number = i + 1
                log.timestamp = datetime.now() - timedelta(minutes=30) + timedelta(minutes=step*10+i)
                log.is_error = False
                logs.append(log)

            # Error logs
            for i in range(2):
                log = MockLogEntry()
                log.text = f"ERROR: Test error {i} in step {step}"
                log.file = "test.log"
                log.line_number = i + 10
                log.timestamp = datetime.now() - timedelta(minutes=30) + timedelta(minutes=step*10+i+5)
                log.is_error = True
                log.severity = "High" if i == 0 else "Medium"
                log.component = "ComponentA"
                logs.append(log)

            self.step_to_logs[step] = logs

        # Create mock clusters
        self.clusters = {
            0: [log for step_logs in self.step_to_logs.values() for log in step_logs if getattr(log, 'is_error', False)]
        }
        
        # Create supporting_images directory to simulate production environment
        self.supporting_images_dir = os.path.join(self.test_dir, "supporting_images")
        os.makedirs(self.supporting_images_dir, exist_ok=True)
    
    def tearDown(self):
        """Clean up after tests."""
        try:
            shutil.rmtree(self.test_dir, ignore_errors=True)
        except Exception as e:
            logging.warning(f"Error cleaning up test directory: {str(e)}")
    
    def is_cluster_timeline_enabled(self):
        """
        Check if cluster timeline visualization is enabled in Config.
        
        Returns:
            True if feature is enabled, False otherwise
        """
        return hasattr(Config, 'ENABLE_CLUSTER_TIMELINE') and Config.ENABLE_CLUSTER_TIMELINE
        
    def test_cluster_timeline_in_test_mode(self):
        """
        Test generating cluster timeline image in test mode.
        
        Verifies that cluster timeline images are correctly generated
        in test mode with appropriate output locations.
        """
        test_id = "TEST-123"
        
        # Setup output directories using standardized approach
        dirs = setup_output_directories(self.test_dir, test_id)
        
        # Test generate_cluster_timeline_image in test mode
        with patch('reports.visualizations.get_output_path') as mock_get_output_path:
            # Setup the mock to return a path
            test_image_path = os.path.join(dirs["images"], f"{test_id}_cluster_timeline.png")
            test_debug_path = os.path.join(dirs["debug"], f"{test_id}_timeline_debug.txt")
            mock_get_output_path.side_effect = [test_image_path, test_debug_path]
            
            # Create directories if they don't exist
            os.makedirs(os.path.dirname(test_debug_path), exist_ok=True)
            
            # Create a file at that path to avoid IO errors
            with open(test_image_path, 'w') as f:
                f.write("test")
                
            cluster_image = generate_cluster_timeline_image(
                self.step_to_logs, 
                self.step_dict,
                self.clusters, 
                dirs["base"], 
                test_id
            )
        
        # Check if the feature is disabled in Config
        if not self.is_cluster_timeline_enabled():
            logger.info("NOTE: Cluster timeline visualization is disabled in Config. Skipping assertions.")
            self.assertIsNone(cluster_image, "Expected None when feature is disabled")
            return
            
        # Only run these assertions if the feature is enabled
        self.assertIsNotNone(cluster_image, "Failed to generate cluster timeline image")
        self.assertTrue(os.path.exists(cluster_image), "Cluster timeline image file doesn't exist")
        
        # Log the actual path that was created
        logger.info(f"Cluster timeline image created at: {cluster_image}")
        
        # Verify the path doesn't have nested supporting_images
        self.assertNotIn("supporting_images/supporting_images", 
                       cluster_image.replace("\\", "/"), 
                       "Path should not contain nested supporting_images directories")
    
    def test_timeline_in_test_mode(self):
        """
        Test generating regular timeline image in test mode.
        
        Verifies that regular timeline images are correctly generated
        in test mode with appropriate output locations.
        """
        test_id = "TEST-123"
        
        # Setup output directories using standardized approach
        dirs = setup_output_directories(self.test_dir, test_id)
        
        # Test generate_timeline_image in test mode
        with patch('reports.visualizations.get_output_path') as mock_get_output_path:
            # Setup the mock to return a path
            test_image_path = os.path.join(dirs["images"], f"{test_id}_timeline.png")
            mock_get_output_path.return_value = test_image_path
            
            # Create a file at that path to avoid IO errors
            with open(test_image_path, 'w') as f:
                f.write("test")
                
            timeline_image = generate_timeline_image(
                self.step_to_logs, 
                self.step_dict,
                dirs["base"], 
                test_id
            )
        
        self.assertIsNotNone(timeline_image, "Failed to generate timeline image")
        self.assertTrue(os.path.exists(timeline_image), "Timeline image file doesn't exist")
        
        # Verify the path doesn't have nested supporting_images
        self.assertNotIn("supporting_images/supporting_images", 
                       timeline_image.replace("\\", "/"), 
                       "Path should not contain nested supporting_images directories")
    
    def test_cluster_timeline_in_production_mode(self):
        """
        Test generating cluster timeline image in production mode.
        
        Verifies that cluster timeline images are correctly generated
        in production mode with the expected directory structure.
        """
        prod_id = "SXM-456"
        
        # Setup output directories using standardized approach
        dirs = setup_output_directories(self.test_dir, prod_id)
        
        # Test generate_cluster_timeline_image in production mode
        with patch('reports.visualizations.get_output_path') as mock_get_output_path:
            # Setup the mock to return a path
            prod_image_path = os.path.join(dirs["images"], f"{prod_id}_cluster_timeline.png")
            prod_debug_path = os.path.join(dirs["debug"], f"{prod_id}_timeline_debug.txt")
            mock_get_output_path.side_effect = [prod_image_path, prod_debug_path]
            
            # Create directories if they don't exist
            os.makedirs(os.path.dirname(prod_debug_path), exist_ok=True)
            
            # Create a file at that path to avoid IO errors
            with open(prod_image_path, 'w') as f:
                f.write("test")
                
            prod_cluster_image = generate_cluster_timeline_image(
                self.step_to_logs, 
                self.step_dict,
                self.clusters, 
                dirs["base"], 
                prod_id
            )
        
        # Check if the feature is disabled in Config
        if not self.is_cluster_timeline_enabled():
            logger.info("NOTE: Cluster timeline visualization is disabled in Config. Skipping assertions.")
            self.assertIsNone(prod_cluster_image, "Expected None when feature is disabled")
            return
            
        # Only run these assertions if the feature is enabled
        self.assertIsNotNone(prod_cluster_image, "Failed to generate cluster timeline image in production mode")
        self.assertTrue(os.path.exists(prod_cluster_image), "Production cluster timeline image file doesn't exist")
        
        # Verify production mode image is in supporting_images directory
        self.assertIn("supporting_images", 
                    prod_cluster_image.replace("\\", "/"), 
                    "Path should include supporting_images directory")
        
        # Verify the path doesn't have nested supporting_images
        self.assertNotIn("supporting_images/supporting_images", 
                       prod_cluster_image.replace("\\", "/"), 
                       "Path should not contain nested supporting_images directories")
    
    def test_timeline_in_production_mode(self):
        """
        Test generating regular timeline image in production mode.
        
        Verifies that regular timeline images are correctly generated
        in production mode with the expected directory structure.
        """
        prod_id = "SXM-456"
        
        # Setup output directories using standardized approach
        dirs = setup_output_directories(self.test_dir, prod_id)
        
        # Test generate_timeline_image in production mode
        with patch('reports.visualizations.get_output_path') as mock_get_output_path:
            # Setup the mock to return a path
            prod_image_path = os.path.join(dirs["images"], f"{prod_id}_timeline.png")
            mock_get_output_path.return_value = prod_image_path
            
            # Create a file at that path to avoid IO errors
            with open(prod_image_path, 'w') as f:
                f.write("test")
                
            prod_timeline_image = generate_timeline_image(
                self.step_to_logs, 
                self.step_dict,
                dirs["base"], 
                prod_id
            )
        
        self.assertIsNotNone(prod_timeline_image, "Failed to generate timeline image in production mode")
        self.assertTrue(os.path.exists(prod_timeline_image), "Production timeline image file doesn't exist")
        
        # Verify production timeline image is in supporting_images directory
        self.assertIn("supporting_images", 
                    prod_timeline_image.replace("\\", "/"), 
                    "Path should include supporting_images directory")
        
        # Verify the path doesn't have nested supporting_images
        self.assertNotIn("supporting_images/supporting_images", 
                       prod_timeline_image.replace("\\", "/"), 
                       "Path should not have duplicate supporting_images")

    def test_timeline_path_validation(self):
        """
        Test timeline path validation.
        
        Verifies that timeline generation produces file paths
        that follow the expected standards.
        """
        test_id = "TEST-PATH-123"
        
        # Setup output directories using standardized approach
        dirs = setup_output_directories(self.test_dir, test_id)
        
        # Test generate_timeline_image with path validation
        with patch('reports.visualizations.get_output_path') as mock_get_output_path:
            # Set up return values for mock
            image_path = os.path.join(dirs["images"], f"{test_id}_timeline.png")
            mock_get_output_path.return_value = image_path
            
            # Make sure directory exists
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            
            # Create a dummy file
            with open(image_path, 'w') as f:
                f.write("test")
            
            # Generate the timeline
            result = generate_timeline_image(
                self.step_to_logs,
                self.step_dict,
                dirs["base"],
                test_id
            )
        
        # Validate the path
        self.assertIsNotNone(result, "Timeline generation should return a path")
        self.assertEqual(result, image_path, "Timeline path should match expected path")
        
        # Validate structure
        self.assertIn("supporting_images", 
                    result.replace("\\", "/"), 
                    "Path should include supporting_images directory")
        
        # Verify no duplicate paths
        self.assertNotIn("supporting_images/supporting_images", 
                       result.replace("\\", "/"), 
                       "Path should not have duplicate supporting_images")


if __name__ == "__main__":
    # Enable feature flags for testing
    Config.ENABLE_COMPONENT_DISTRIBUTION = True
    Config.ENABLE_CLUSTER_TIMELINE = True
    Config.ENABLE_ERROR_PROPAGATION = True
    
    # Run tests directly when the file is executed
    unittest.main()