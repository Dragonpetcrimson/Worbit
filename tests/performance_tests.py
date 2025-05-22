"""
Performance tests for Orbit Analyzer.

This module measures the performance characteristics of key operations:
- Log parsing speed and scalability
- Error clustering performance with various dataset sizes
- Component analysis processing time
- Report generation performance
"""

import unittest
import time
import os
import sys
import random
import logging
from datetime import datetime, timedelta

# Add parent directory to path to find modules
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
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
    from test_utils import ConfigManager, timeit
except ImportError:
    # Simple timeit decorator as a placeholder
    def timeit(func):
        """Decorator to time function execution."""
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            result = func(*args, **kwargs)
            duration = (datetime.now() - start_time).total_seconds()
            logging.info(f"{func.__name__} completed in {duration:.2f} seconds")
            return result
        return wrapper
    
    # Minimal ConfigManager as a placeholder
    class ConfigManager:
        @classmethod
        def get(cls, key, default=None):
            """Get a configuration value."""
            return os.environ.get(key, default)

# Import modules to test with appropriate error handling
try:
    from log_analyzer import parse_logs
except ImportError:
    logging.warning("log_analyzer module not available")
    # Define a stub function
    def parse_logs(log_paths):
        logging.error("parse_logs called but module not available")
        return []

try:
    from error_clusterer import perform_error_clustering as cluster_errors
except ImportError:
    try:
        from error_clusterer import cluster_errors
    except ImportError:
        logging.warning("error_clusterer module not available")
        # Define a stub function
        def cluster_errors(errors, num_clusters=None):
            logging.error("cluster_errors called but module not available")
            return {}

try:
    from components.direct_component_analyzer import assign_components_and_relationships
except ImportError:
    logging.warning("direct_component_analyzer module not available")
    # Define a stub function
    def assign_components_and_relationships(errors):
        logging.error("assign_components_and_relationships called but module not available")
        return errors, [], "unknown"

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

@TestRegistry.register(category='performance', importance=2, slow=True, 
                     tags=['performance', 'benchmark'])
class PerformanceTests(unittest.TestCase):
    """
    Performance benchmark tests for various Orbit components.
    
    Measures the performance characteristics of key operations,
    including log parsing, error clustering, and component analysis.
    """
    
    @classmethod
    def setUpClass(cls):
        """
        Set up the test environment once for all tests.
        
        Creates test data of various sizes for benchmarking.
        """
        # Check if we should skip slow tests
        if os.environ.get("SKIP_SLOW_TESTS", "").lower() in ("true", "1", "yes"):
            raise unittest.SkipTest("Skipping slow performance tests as requested")
            
        # Create test directory
        cls.test_dir = os.path.join(os.path.dirname(__file__), "test_data", "performance")
        os.makedirs(cls.test_dir, exist_ok=True)
        
        # Generate log files of different sizes
        cls.large_log = os.path.join(cls.test_dir, "large_log.txt")
        cls.medium_log = os.path.join(cls.test_dir, "medium_log.txt")
        cls.small_log = os.path.join(cls.test_dir, "small_log.txt")
        
        # Generate test data
        cls.generate_large_log(cls.large_log, 2000)  # ~2000 entries
        cls.generate_large_log(cls.medium_log, 500)  # ~500 entries
        cls.generate_large_log(cls.small_log, 100)   # ~100 entries
        
        cls.test_id = "SXM-PERFTEST"
    
    @staticmethod
    def generate_large_log(filepath, num_entries):
        """
        Generate a large log file with random errors.
        
        Creates a test log file with a specified number of entries,
        including a mix of info, warning, and error messages from
        different components.
        
        Args:
            filepath: Path to create the log file
            num_entries: Number of log entries to generate
        """
        log_levels = ["INFO", "DEBUG", "WARNING", "ERROR", "FATAL"]
        components = ["soa", "android", "phoebe", "mimosa", "charles"]
        error_messages = [
            "Connection refused",
            "Timeout waiting for response",
            "Invalid state detected",
            "Resource not found",
            "Permission denied",
            "Unexpected response format"
        ]
        
        # Only generate if file doesn't exist or is empty
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            return
            
        logging.info(f"Generating test data file: {filepath} with {num_entries} entries")
        
        with open(filepath, "w") as f:
            for i in range(num_entries):
                # Generate timestamp
                timestamp = f"2025-03-26 {random.randint(0,23):02d}:{random.randint(0,59):02d}:{random.randint(0,59):02d}"
                
                # Determine log level - make ~20% be errors
                level = "ERROR" if random.random() < 0.2 else random.choice(log_levels)
                
                # Generate component
                component = random.choice(components)
                
                # Generate message
                message = random.choice(error_messages) if level in ["ERROR", "FATAL"] else f"Normal operation {i}"
                
                # Write log entry
                f.write(f"{timestamp} [{level}] - [{component}] - {message}\n")
    
    @classmethod
    def tearDownClass(cls):
        """
        Clean up after all tests in this class have run.
        
        Note: We keep the generated test files to avoid regenerating 
        them for future test runs.
        """
        pass  # Keep test files for future use
    
    def setUp(self):
        """Set up individual test."""
        # Check if slow tests should be skipped using ConfigManager
        try:
            if ConfigManager.get("SKIP_SLOW_TESTS", "").lower() in ("true", "1", "yes"):
                self.skipTest("Skipping slow performance tests as requested")
        except (AttributeError, ImportError):
            # Fall back to environment variable directly
            if os.environ.get("SKIP_SLOW_TESTS", "").lower() in ("true", "1", "yes"):
                self.skipTest("Skipping slow performance tests as requested")
    
    @timeit
    def test_log_parser_performance(self):
        """
        Test the performance of the log parser.
        
        Measures the time taken to parse a large log file and extract errors,
        validating both speed and correctness of the parser.
        """
        # Skip test if module isn't available
        if not hasattr(sys.modules.get('log_analyzer', {}), 'parse_logs'):
            self.skipTest("log_analyzer module not available")
            
        start_time = time.time()
        errors = parse_logs([self.large_log])
        duration = time.time() - start_time
        
        # Log results
        logging.info(f"\nLog Parser Performance:")
        logging.info(f"  File size: {os.path.getsize(self.large_log) / 1024:.2f} KB")
        logging.info(f"  Errors found: {len(errors)}")
        logging.info(f"  Processing time: {duration:.2f} seconds")
        
        # Calculate processing rate (with division by zero protection)
        if duration > 0:
            logging.info(f"  Processing rate: {len(errors) / duration:.2f} errors/second")
        else:
            logging.info(f"  Processing rate: >100,000 errors/second (too fast to measure)")
        
        # Assert performance meets expectations
        self.assertLess(duration, 5.0, f"Parsing took {duration:.2f}s (limit: 5s)")
        
        # Should have found approximately 400 errors (20% of 2000 entries)
        self.assertGreaterEqual(len(errors), 300, 
                             f"Found fewer errors than expected: {len(errors)} < 300")
    
    @timeit
    def test_clustering_performance(self):
        """
        Test the performance of error clustering.
        
        Measures the time taken to cluster errors extracted from logs,
        with validation of both speed and cluster quality.
        """
        # Skip test if modules aren't available
        if not (hasattr(sys.modules.get('log_analyzer', {}), 'parse_logs') and 
                (hasattr(sys.modules.get('error_clusterer', {}), 'cluster_errors') or 
                 hasattr(sys.modules.get('error_clusterer', {}), 'perform_error_clustering'))):
            self.skipTest("Required modules (log_analyzer or error_clusterer) not available")
            
        # First parse logs
        errors = parse_logs([self.large_log])
        if not errors:
            self.skipTest("No errors found in log file")
            
        # Then time the clustering
        start_time = time.time()
        clusters = cluster_errors(errors, num_clusters=5)
        duration = time.time() - start_time
        
        # Log results
        logging.info(f"\nClustering Performance:")
        logging.info(f"  Errors processed: {len(errors)}")
        logging.info(f"  Clusters created: {len(clusters)}")
        logging.info(f"  Processing time: {duration:.2f} seconds")
        
        # Calculate processing rate (with division by zero protection)
        if duration > 0:
            logging.info(f"  Processing rate: {len(errors) / duration:.2f} errors/second")
        else:
            logging.info(f"  Processing rate: >100,000 errors/second (too fast to measure)")
        
        # Assert performance meets expectations
        self.assertLess(duration, 3.0, f"Clustering took {duration:.2f}s (limit: 3s)")
        self.assertGreater(len(clusters), 0, "No clusters were created")
    
    @timeit
    def test_component_identification_performance(self):
        """
        Test the performance of component identification.
        
        Measures the time taken to identify components for errors,
        with validation of both speed and identification accuracy.
        """
        # Skip test if modules aren't available
        if not (hasattr(sys.modules.get('log_analyzer', {}), 'parse_logs') and 
                hasattr(sys.modules.get('components.direct_component_analyzer', {}), 
                        'assign_components_and_relationships')):
            self.skipTest("Required modules not available")
            
        # First parse logs
        errors = parse_logs([self.medium_log])
        if not errors:
            self.skipTest("No errors found in log file")
            
        # Time the component identification
        start_time = time.time()
        errors_with_components, component_summary, primary_component = assign_components_and_relationships(errors)
        duration = time.time() - start_time
        
        # Log results
        logging.info(f"\nComponent Identification Performance:")
        logging.info(f"  Errors processed: {len(errors)}")
        logging.info(f"  Primary component: {primary_component}")
        logging.info(f"  Components identified: {len(component_summary)}")
        logging.info(f"  Processing time: {duration:.2f} seconds")
        
        # Calculate processing rate (with division by zero protection)
        if duration > 0:
            logging.info(f"  Processing rate: {len(errors) / duration:.2f} errors/second")
        else:
            logging.info(f"  Processing rate: >100,000 errors/second (too fast to measure)")
        
        # Assert performance meets expectations
        self.assertLess(duration, 2.0, f"Component identification took {duration:.2f}s (limit: 2s)")

if __name__ == "__main__":
    # When run directly, use unittest's test runner
    unittest.main()
    
    # For integration with the new test runner system:
    # python run_all_tests.py --category performance