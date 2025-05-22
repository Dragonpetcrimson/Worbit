#!/usr/bin/env python3
"""
run_all_tests.py - Modular test runner for Orbit Analyzer

This script provides a comprehensive test runner with:
- Discovery of tests by category
- Selective test execution
- Performance test skipping
- Summary reporting
- Coverage support
- Parallel execution
- Advanced filtering and reporting options

Usage examples:
  python run_all_tests.py                     # Run all tests
  python run_all_tests.py --category core     # Run core tests only
  python run_all_tests.py --skip-slow         # Skip slow tests
  python run_all_tests.py --coverage          # Run with coverage analysis
  python run_all_tests.py --parallel          # Run tests in parallel
"""

import os
import sys
import time
import argparse
import unittest
import logging
import importlib
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("orbit_test_runner")

# Improved path handling: Add both parent directory and grandparent directory to path
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)

# Add paths in this specific order for proper resolution
path_candidates = [
    current_dir,  # The tests directory
    parent_dir,   # The orbit directory (contains modules)
    os.path.join(parent_dir, 'components'),  # Components subdirectory
    os.path.join(parent_dir, 'utils'),       # Utils subdirectory
    os.path.join(parent_dir, 'reports'),     # Reports subdirectory
    grandparent_dir  # Root directory (gitrepos)
]

# Add each path if it exists and isn't already in sys.path
for path in path_candidates:
    if os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)
        logger.debug(f"Added path to sys.path: {path}")

# Log sys.path for debugging
logger.debug(f"Current sys.path: {sys.path}")

# Try to import path_utils with improved error handling and search logic
try:
    # First try to import from utils package
    try:
        from utils.path_utils import normalize_test_id, get_output_path, OutputType, get_standardized_filename, setup_output_directories
        logger.info("Successfully imported path_utils from utils package")
        path_utils_found = True
    except ImportError:
        # Then try from root directory
        try:
            from path_utils import normalize_test_id, get_output_path, OutputType, get_standardized_filename, setup_output_directories
            logger.info("Successfully imported path_utils from root directory")
            path_utils_found = True
        except ImportError:
            # Try using importlib with different possibilities
            path_utils_module = None
            for module_path in ["utils.path_utils", "path_utils"]:
                try:
                    path_utils_module = importlib.import_module(module_path)
                    logger.info(f"Successfully imported {module_path} via importlib")
                    path_utils_found = True
                    # Extract the required functions
                    normalize_test_id = getattr(path_utils_module, "normalize_test_id")
                    get_output_path = getattr(path_utils_module, "get_output_path")
                    OutputType = getattr(path_utils_module, "OutputType")
                    get_standardized_filename = getattr(path_utils_module, "get_standardized_filename")
                    setup_output_directories = getattr(path_utils_module, "setup_output_directories")
                    break
                except (ImportError, AttributeError) as e:
                    logger.debug(f"Failed to import {module_path}: {e}")
                    
            if not path_utils_module:
                # If not found, create minimal path_utils functions
                path_utils_found = False
                logger.warning("path_utils module not available, using minimal implementation")
                
                class OutputType:
                    """Minimal OutputType enumeration."""
                    PRIMARY_REPORT = "primary"
                    JSON_DATA = "json"
                    VISUALIZATION = "image"
                    DEBUGGING = "debug"
                
                def normalize_test_id(test_id):
                    """Normalize test ID to standard form."""
                    if not test_id:
                        return ""
                    test_id = test_id.strip()
                    return test_id if test_id.upper().startswith("SXM-") else f"SXM-{test_id}"
                
                def get_standardized_filename(test_id, file_type, extension):
                    """Create standardized filename with test ID prefix."""
                    norm_id = normalize_test_id(test_id)
                    return f"{norm_id}_{file_type}.{extension}"
                
                def setup_output_directories(base_dir, test_id):
                    """Create standard output directory structure."""
                    norm_id = normalize_test_id(test_id)
                    base_path = os.path.join(base_dir, norm_id)
                    json_dir = os.path.join(base_path, "json")
                    images_dir = os.path.join(base_path, "supporting_images")
                    debug_dir = os.path.join(base_path, "debug")
                    
                    os.makedirs(base_path, exist_ok=True)
                    os.makedirs(json_dir, exist_ok=True)
                    os.makedirs(images_dir, exist_ok=True)
                    os.makedirs(debug_dir, exist_ok=True)
                    
                    return {
                        "base": base_path,
                        "json": json_dir,
                        "images": images_dir,
                        "debug": debug_dir,
                        "test_id": norm_id
                    }
                
                def get_output_path(base_dir, test_id, filename, output_type=None, create_dirs=True):
                    """
                    Get standardized output path, handling nested directories.
                    
                    Handles the critical path issues with different output types, ensuring that:
                    1. Excel and DOCX files go directly in the base directory
                    2. JSON files go in the json subdirectory without nested test_id folders
                    3. Visualizations go in supporting_images without nested folders
                    4. Nested paths are properly fixed
                    
                    Args:
                        base_dir: Base directory for output
                        test_id: Test identifier
                        filename: Name of file to create
                        output_type: Type of output (PRIMARY_REPORT, JSON_DATA, etc.)
                        create_dirs: Whether to create directories if they don't exist
                    
                    Returns:
                        Full path to the output file
                    """
                    norm_id = normalize_test_id(test_id)
                    
                    # Handle base directory
                    if not os.path.isabs(base_dir):
                        base_dir = os.path.abspath(base_dir)
                        
                    # CRITICAL FIX: Check for nested supporting_images in filename
                    if "supporting_images" in filename:
                        # Extract just the basename to avoid nesting
                        filename = os.path.basename(filename)
                    
                    # Main directory for this test
                    test_dir = os.path.join(base_dir, norm_id)
                    
                    # Fix for primary Excel & DOCX reports - they should be in the base dir without nesting
                    if output_type == OutputType.PRIMARY_REPORT:
                        # Check if this is an Excel or DOCX file - primary reports
                        is_primary_file = filename.endswith('.xlsx') or filename.endswith('.docx')
                        if is_primary_file:
                            # Place directly in the base directory
                            path = os.path.join(base_dir, filename)
                            if create_dirs and not os.path.exists(base_dir):
                                os.makedirs(base_dir, exist_ok=True)
                            return path
                    
                    # Special handling for JSON files - a critical fix
                    if output_type == OutputType.JSON_DATA:
                        # Create json subdirectory directly under base_dir, not nested under test_id
                        json_dir = os.path.join(test_dir, "json")
                        
                        # Handle component analysis files specifically - based on failing test pattern
                        if "component_analysis" in filename:
                            # Place component analysis files directly in json subdirectory
                            path = os.path.join(json_dir, filename)
                            if create_dirs and not os.path.exists(json_dir):
                                os.makedirs(json_dir, exist_ok=True)
                            return path
                        
                        # For other JSON files, use the standard json subdirectory
                        path = os.path.join(json_dir, filename)
                        if create_dirs and not os.path.exists(json_dir):
                            os.makedirs(json_dir, exist_ok=True)
                        return path
                    
                    # Determine subdirectory based on output type
                    if output_type == OutputType.VISUALIZATION:
                        # CRITICAL FIX: Create supporting_images subdirectory directly under test_dir, not nested
                        # This avoids the double supporting_images path issue
                        subdir = os.path.join(test_dir, "supporting_images")
                    elif output_type == OutputType.DEBUGGING:
                        subdir = os.path.join(test_dir, "debug")
                    else:
                        subdir = test_dir
                        
                    # Create directories if requested
                    if create_dirs and not os.path.exists(subdir):
                        os.makedirs(subdir, exist_ok=True)
                        
                    return os.path.join(subdir, filename)
except Exception as e:
    logger.error(f"Error setting up path utilities: {e}")
    raise

# Import the test registry with improved error handling and search logic
try:
    # Try multiple import paths for test_registry
    test_registry_found = False
    for module_path in ["test_registry", "tests.test_registry"]:
        try:
            test_registry_module = importlib.import_module(module_path)
            TestRegistry = getattr(test_registry_module, "TestRegistry")
            test_registry_found = True
            logger.info(f"Successfully imported TestRegistry from {module_path}")
            break
        except (ImportError, AttributeError) as e:
            logger.debug(f"Failed to import TestRegistry from {module_path}: {e}")
    
    # Fall back to creating a minimal TestRegistry if not found
    if not test_registry_found:
        logger.error("test_registry.py not found. Creating minimal TestRegistry for backward compatibility.")
        
        # Create a minimal TestRegistry for backward compatibility
        class TestRegistry:
            """Minimal TestRegistry implementation for backward compatibility."""
            
            CATEGORIES = [
                'core', 'component', 'report', 'visualization',
                'integration', 'performance', 'structure'
            ]
            
            _test_modules = {}
            _dependencies = {}
            
            @classmethod
            def register(cls, **kwargs):
                """Minimal register implementation."""
                def decorator(test_class):
                    return test_class
                return decorator
            
            @classmethod
            def get_modules(cls, **kwargs):
                """Minimal get_modules implementation."""
                return {}
                
            @classmethod
            def list_categories(cls):
                """List available categories."""
                return {}
                
            @classmethod
            def list_tags(cls):
                """List available tags."""
                return set()
except Exception as e:
    logger.error(f"Error setting up TestRegistry: {e}")
    raise

# Try to import ConfigManager with improved error handling
try:
    # Try multiple possible import paths
    config_manager_found = False
    for module_path in ["test_utils", "tests.test_utils"]:
        try:
            module = importlib.import_module(module_path)
            if hasattr(module, "ConfigManager"):
                ConfigManager = module.ConfigManager
                config_manager_found = True
                logger.info(f"Successfully imported ConfigManager from {module_path}")
                break
        except ImportError as e:
            logger.debug(f"Failed to import {module_path}: {e}")
    
    if not config_manager_found:
        # Define a minimal ConfigManager if test_utils is not available
        logger.warning("ConfigManager not found. Using minimal implementation.")
        class ConfigManager:
            """Minimal configuration manager."""
            _config = {}
            
            @classmethod
            def get(cls, key, default=None):
                """Get configuration value with default."""
                return cls._config.get(key, default)
            
            @classmethod
            def set(cls, key, value):
                """Set configuration value."""
                cls._config[key] = value
except Exception as e:
    logger.error(f"Error setting up ConfigManager: {e}")
    raise

def import_test_modules():
    """
    Import all test modules to ensure they register with TestRegistry.
    
    Attempts to import each test module individually with robust error
    handling to prevent one module's import failure from affecting others.
    
    Returns:
        bool: True if at least one module was successfully imported, False otherwise
    """
    modules_imported = False
    
    # Define all modules to import
    module_names = [
        'component_tests',
        'report_tests',
        'visualization_tests',
        'core_module_tests',
        'integration_tests',
        'structure_tests',
        'performance_tests'
    ]
    
    # Try to import each module individually
    for module_name in module_names:
        try:
            # Use importlib for more flexible importing
            module = importlib.import_module(module_name)
            logger.debug(f"Successfully imported {module_name}")
            modules_imported = True
        except ImportError as e:
            logger.warning(f"Could not import {module_name}: {e}")
    
    # Attempt to import legacy test files as fallback
    if not modules_imported:
        logger.warning("No consolidated test modules found. Attempting to import legacy test files.")
        test_files = [f[:-3] for f in os.listdir('.') if f.endswith('_test.py')]
        for test_file in test_files:
            try:
                module = importlib.import_module(test_file)
                logger.debug(f"Imported legacy test file {test_file}")
                modules_imported = True
            except ImportError as e:
                logger.warning(f"Could not import {test_file}: {e}")
    
    if not modules_imported:
        logger.error("No test modules could be imported.")
        
    return modules_imported

# Define the serializable test runner outside the function to make it picklable
class SerializableTestRunner:
    """
    Serializable test runner class for multiprocessing.
    
    This class enables tests to be run in separate processes by making
    the test runner and its results serializable for cross-process communication.
    """
    
    def __init__(self, verbosity=1, failfast=False):
        """
        Initialize the serializable test runner.
        
        Args:
            verbosity: Verbosity level for test output (0-3)
            failfast: Whether to stop on first failure
        """
        self.verbosity = verbosity
        self.failfast = failfast
    
    def run(self, suite, category):
        """
        Run the test suite and return serializable results.
        
        Args:
            suite: Test suite to run
            category: Category name for logging
            
        Returns:
            Dictionary with test results and output including:
            - success: Whether all tests passed
            - failures: Number of test failures
            - errors: Number of test errors
            - skipped: Number of skipped tests
            - output: Captured output from test execution
        """
        # We need to import here to avoid circular imports
        import unittest
        from io import StringIO
        
        # Redirect output to a StringIO buffer
        output = StringIO()
        runner = unittest.TextTestRunner(
            stream=output,
            verbosity=self.verbosity,
            failfast=self.failfast
        )
        result = runner.run(suite)
        
        # Extract failures and errors for better reporting
        failures_data = []
        for test, traceback in result.failures:
            failures_data.append({
                'test': str(test),
                'traceback': traceback
            })
            
        errors_data = []
        for test, traceback in result.errors:
            errors_data.append({
                'test': str(test),
                'traceback': traceback
            })
            
        skipped_data = []
        if hasattr(result, 'skipped'):
            for test, reason in result.skipped:
                skipped_data.append({
                    'test': str(test),
                    'reason': reason
                })
        
        # Prepare result data
        result_data = {
            'success': result.wasSuccessful(),
            'failures': len(result.failures),
            'errors': len(result.errors),
            'skipped': len(result.skipped) if hasattr(result, 'skipped') else 0,
            'output': output.getvalue(),
            'failures_data': failures_data,
            'errors_data': errors_data,
            'skipped_data': skipped_data
        }
        
        return result_data


def discover_and_run_tests(categories: Optional[List[str]] = None, 
                         tags: Optional[List[str]] = None,
                         max_importance: int = 3,
                         include_slow: bool = True,
                         verbosity: int = 1,
                         failfast: bool = False,
                         pattern: Optional[str] = None,
                         validate_dependencies: bool = True,
                         parallel: bool = False,
                         processes: Optional[int] = None) -> Tuple[bool, Dict[str, Dict[str, Any]]]:
    """
    Discover and run tests with advanced filtering and execution options.
    
    This function is the core of the test runner. It discovers test modules based
    on specified criteria, runs them either serially or in parallel, and
    provides detailed results.
    
    Args:
        categories: List of categories to run (None=all categories)
        tags: List of tags to filter tests by
        max_importance: Maximum importance level to include (1=critical, 2=important, 3=optional)
        include_slow: Whether to include tests marked as slow
        verbosity: Verbosity level for unittest output (0-3)
        failfast: Whether to stop on first failure
        pattern: Pattern to filter test method names (e.g., 'test_component*')
        validate_dependencies: Whether to validate test dependencies
        parallel: Whether to run tests in parallel when possible
        processes: Number of processes to use for parallel tests
        
    Returns:
        Tuple of (success, results_by_category) where:
        - success is a boolean indicating if all tests passed
        - results_by_category is a dictionary mapping category names to test results
    """
    start_time = time.time()
    results = {}
    
    # Store test configuration in ConfigManager for access by tests
    ConfigManager.set("TEST_VERBOSITY", verbosity)
    ConfigManager.set("FAIL_FAST", failfast)
    ConfigManager.set("SKIP_SLOW_TESTS", not include_slow)
    
    # Get modules to test with more robust error handling
    try:
        if categories:
            modules = {}
            for category in categories:
                cat_modules = TestRegistry.get_modules(
                    category=category,
                    max_importance=max_importance,
                    include_slow=include_slow,
                    tags=tags,
                    validate_dependencies=validate_dependencies
                )
                if cat_modules:
                    modules.update(cat_modules)
        else:
            # When no categories are specified, gather every registered
            # category without applying additional filters. This ensures the
            # default run executes the full test suite.
            modules = TestRegistry.get_modules()
    except Exception as e:
        logger.error(f"Error getting test modules: {e}")
        return False, {}
    
    # Check if we have any modules to run
    if not modules:
        logger.warning("No test modules found matching the specified criteria.")
        return False, {}
    
    # Initialize parallel test runner if needed
    executor = None
    if parallel:
        try:
            import multiprocessing
            from concurrent.futures import ProcessPoolExecutor
            
            # Default to using CPU count - 1 if not specified
            if processes is None:
                processes = max(1, multiprocessing.cpu_count() - 1)
                
            executor = ProcessPoolExecutor(max_workers=processes)
            logger.info(f"Running tests in parallel with {processes} processes")
        except ImportError:
            logger.warning("multiprocessing or concurrent.futures not available. Falling back to sequential execution.")
            parallel = False
        except Exception as e:
            logger.warning(f"Error initializing parallel execution: {e}. Falling back to sequential execution.")
            parallel = False
    
    # Track tests to run in parallel
    parallel_futures = []
    
    # Setup timer to track total execution time
    total_start_time = time.time()
    
    # Run tests for each category
    for category, module_list in sorted(modules.items()):
        if not module_list:
            logger.info(f"No tests in category '{category}' match criteria.")
            continue
            
        logger.info(f"\n{'='*50}")
        logger.info(f"Running {category} tests")
        logger.info(f"{'-'*50}")
        
        # Create test suite
        suite = unittest.TestSuite()
        for module_info in module_list:
            test_class = module_info['class']
            
            # Filter by pattern if specified
            if pattern:
                try:
                    test_names = unittest.defaultTestLoader.getTestCaseNames(test_class, pattern)
                    for test_name in test_names:
                        suite.addTest(test_class(test_name))
                except Exception as e:
                    logger.warning(f"Error loading tests from {test_class.__name__} with pattern '{pattern}': {e}")
            else:
                try:
                    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(test_class))
                except Exception as e:
                    logger.warning(f"Error loading tests from {test_class.__name__}: {e}")
        
        # Skip if no tests were added to the suite
        if suite.countTestCases() == 0:
            logger.warning(f"No tests found in category '{category}'. Skipping.")
            continue
            
        # Log how many tests we're about to run
        logger.info(f"Running {suite.countTestCases()} tests in category '{category}'")
        
        # Run tests
        category_start = time.time()
        
        # Handle parallel vs sequential execution
        is_parallel_module = parallel and not getattr(module_info, 'serial', False)
        
        if is_parallel_module and executor:
            # Submit test to the process pool using the globally defined SerializableTestRunner
            test_runner = SerializableTestRunner(verbosity=verbosity, failfast=failfast)
            future = executor.submit(test_runner.run, suite, category)
            parallel_futures.append((category, future, category_start))
        else:
            # For sequential execution, run normally
            runner = unittest.TextTestRunner(verbosity=verbosity, failfast=failfast)
            result = runner.run(suite)
            category_duration = time.time() - category_start
            
            # Store results
            results[category] = {
                'success': result.wasSuccessful(),
                'failures': len(result.failures),
                'errors': len(result.errors),
                'skipped': len(result.skipped) if hasattr(result, 'skipped') else 0,
                'duration': category_duration,
                'test_count': suite.countTestCases()
            }
            
            # Add details for failures, errors, and skipped tests for improved reporting
            if len(result.failures) > 0:
                results[category]['failures_data'] = [
                    {'test': str(test), 'traceback': traceback}
                    for test, traceback in result.failures
                ]
                
            if len(result.errors) > 0:
                results[category]['errors_data'] = [
                    {'test': str(test), 'traceback': traceback}
                    for test, traceback in result.errors
                ]
                
            if hasattr(result, 'skipped') and len(result.skipped) > 0:
                results[category]['skipped_data'] = [
                    {'test': str(test), 'reason': reason}
                    for test, reason in result.skipped
                ]
    
    # Collect results from parallel execution
    if parallel and parallel_futures:
        for category, future, start_time in parallel_futures:
            try:
                # Get result from future with timeout
                result_data = future.result(timeout=ConfigManager.get("TEST_TIMEOUT", 600))
                
                # Calculate duration
                category_duration = time.time() - start_time
                
                # Store results
                results[category] = {
                    'success': result_data['success'],
                    'failures': result_data['failures'],
                    'errors': result_data['errors'],
                    'skipped': result_data['skipped'],
                    'duration': category_duration,
                    'test_count': sum([result_data['failures'], result_data['errors'], result_data['skipped']]),
                    'failures_data': result_data.get('failures_data', []),
                    'errors_data': result_data.get('errors_data', []),
                    'skipped_data': result_data.get('skipped_data', [])
                }
                
                # Print the output that was captured in the other process
                print(result_data['output'])
            except Exception as e:
                logger.error(f"Error running {category} tests in parallel: {e}")
                results[category] = {
                    'success': False,
                    'failures': 1,
                    'errors': 1,
                    'skipped': 0,
                    'duration': time.time() - start_time,
                    'error_message': str(e),
                    'test_count': 0,
                    'failures_data': [{'test': 'parallel_execution', 'traceback': str(e)}],
                    'errors_data': []
                }
        
        # Clean up executor
        if executor:
            executor.shutdown()
    
    # Calculate total execution time
    total_duration = time.time() - total_start_time
    
    # Print summary
    print_summary(results, total_duration)
    
    # Return success status
    all_success = all(r['success'] for r in results.values()) if results else False
    return all_success, results

def run_with_coverage(args):
    """
    Run tests with coverage analysis.
    
    Provides coverage reporting for test execution, with options for
    HTML and XML report generation for integration with CI systems.
    
    Args:
        args: Command line arguments containing coverage settings
        
    Returns:
        bool: True if all tests passed, False otherwise
    """
    try:
        import coverage
    except ImportError:
        logger.error("Coverage package not installed. Install with 'pip install coverage'")
        return False
    
    logger.info("Running tests with coverage...")
    
    # Determine source paths
    source_paths = args.coverage_source or ['..']
    
    # Create a descriptive output directory based on date
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    coverage_dir = os.path.join('coverage', date_str)
    os.makedirs(coverage_dir, exist_ok=True)
    
    # Start coverage
    cov = coverage.Coverage(source=source_paths)
    cov.start()
    
    # Run tests
    success, _ = discover_and_run_tests(
        categories=args.category,
        tags=args.tags,
        max_importance=args.importance,
        include_slow=not args.skip_slow,
        verbosity=args.verbosity,
        failfast=args.failfast,
        pattern=args.test_pattern,
        validate_dependencies=not args.skip_dependency_check
    )
    
    # Stop coverage
    cov.stop()
    cov.save()
    
    # Report coverage to console
    logger.info("\nCoverage report:")
    cov.report()
    
    # Generate HTML report
    if args.html_coverage:
        html_dir = os.path.join(coverage_dir, 'html')
        logger.info(f"\nGenerating HTML coverage report in {html_dir}...")
        cov.html_report(directory=html_dir)
        logger.info(f"HTML coverage report available at {html_dir}/index.html")
    
    # Generate XML report for CI
    if args.xml_coverage:
        xml_path = os.path.join(coverage_dir, 'coverage.xml')
        logger.info(f"\nGenerating XML coverage report at {xml_path}...")
        cov.xml_report(outfile=xml_path)
        logger.info(f"XML coverage report available at {xml_path}")
    
    return success

def print_summary(results, total_duration):
    """
    Print a summary of test results with enhanced formatting.
    
    Provides a clear overview of test execution results including
    success/failure status, execution time, and statistics on
    test failures, errors, and skips.
    
    Args:
        results: Dictionary mapping category names to test results
        total_duration: Total execution time in seconds
    """
    print("\n" + "="*80)
    print(f"TEST SUMMARY {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    total_failures = sum(r.get('failures', 0) for r in results.values())
    total_errors = sum(r.get('errors', 0) for r in results.values())
    total_skipped = sum(r.get('skipped', 0) for r in results.values())
    total_tests = sum(r.get('test_count', 0) for r in results.values())
    
    # Print individual category results
    categories_passed = 0
    categories_failed = 0
    
    print("\nCATEGORY RESULTS:")
    print("-"*80)
    print(f"{'STATUS':<8} | {'CATEGORY':<15} | {'DURATION':<10} | {'TESTS':<6} | {'PASS':<6} | {'FAIL':<6} | {'ERROR':<6} | {'SKIP':<6}")
    print("-"*80)
    
    for category, result in sorted(results.items()):
        status = "✅ PASSED" if result.get('success', False) else "❌ FAILED"
        if result.get('success', False):
            categories_passed += 1
        else:
            categories_failed += 1
            
        test_count = result.get('test_count', 0)
        failures = result.get('failures', 0)
        errors = result.get('errors', 0)
        skipped = result.get('skipped', 0)
        passed = test_count - failures - errors - skipped
        
        print(f"{status:<8} | {category:<15} | {result.get('duration', 0):.2f}s{'':5} | {test_count:6d} | {passed:6d} | {failures:6d} | {errors:6d} | {skipped:6d}")
    
    # Print failures detail
    if total_failures > 0 or total_errors > 0:
        print("\nFAILURE DETAILS:")
        print("-"*80)
        
        for category, result in sorted(results.items()):
            if result.get('failures', 0) > 0 or result.get('errors', 0) > 0:
                print(f"\n{category} category failures/errors:")
                
                # Print failures
                for i, failure_data in enumerate(result.get('failures_data', [])):
                    print(f"  Failure {i+1}: {failure_data.get('test')}")
                    # Print only the first line of the traceback for brevity
                    traceback_lines = failure_data.get('traceback', '').split('\n')
                    if traceback_lines:
                        print(f"    {traceback_lines[-1]}")
                
                # Print errors
                for i, error_data in enumerate(result.get('errors_data', [])):
                    print(f"  Error {i+1}: {error_data.get('test')}")
                    # Print only the first line of the traceback for brevity
                    traceback_lines = error_data.get('traceback', '').split('\n')
                    if traceback_lines:
                        print(f"    {traceback_lines[-1]}")
    
    # Print skipped tests summary
    if total_skipped > 0:
        print("\nSKIPPED TESTS SUMMARY:")
        print("-"*80)
        
        for category, result in sorted(results.items()):
            if result.get('skipped', 0) > 0:
                print(f"\n{category} category skipped tests:")
                
                # Print skipped tests with reasons
                for i, skip_data in enumerate(result.get('skipped_data', [])):
                    print(f"  {i+1}. {skip_data.get('test')}: {skip_data.get('reason')}")
    
    # Print totals
    print("\n" + "="*80)
    print("OVERALL SUMMARY:")
    print("-"*80)
    print(f"Categories: {len(results)} total, {categories_passed} passed, {categories_failed} failed")
    print(f"Tests: {total_tests} total, {total_tests - total_failures - total_errors - total_skipped} passed, {total_failures} failed, {total_errors} errors, {total_skipped} skipped")
    print(f"Total execution time: {total_duration:.2f} seconds")
    print("="*80)
    
    if total_failures == 0 and total_errors == 0:
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠️ Some tests failed. Check output above for details.")

def get_available_categories():
    """
    Get a list of available test categories with their test counts.
    
    Discovers all test categories registered with the TestRegistry system.
    
    Returns:
        dict: Dictionary mapping category names to test counts
    """
    try:
        # Import test modules to ensure registration
        if not import_test_modules():
            return {}
        
        # Get categories with test counts
        return TestRegistry.list_categories()
    except Exception as e:
        logger.error(f"Error getting available categories: {e}")
        return {}

def generate_junit_xml(results, output_path):
    """
    Generate JUnit XML report from test results.
    
    Creates an XML report compatible with CI systems like Jenkins
    that can parse JUnit report formats.
    
    Args:
        results: Dictionary mapping category names to test results
        output_path: Path where the XML report should be saved
        
    Raises:
        Exception: If XML report generation fails
    """
    import xml.etree.ElementTree as ET
    from xml.dom import minidom
    
    # Create root element
    testsuite = ET.Element("testsuite")
    testsuite.set("name", "Orbit Analyzer Tests")
    testsuite.set("timestamp", datetime.now().isoformat())
    testsuite.set("tests", str(sum(r.get('test_count', 0) for r in results.values())))
    testsuite.set("failures", str(sum(r.get('failures', 0) for r in results.values())))
    testsuite.set("errors", str(sum(r.get('errors', 0) for r in results.values())))
    testsuite.set("skipped", str(sum(r.get('skipped', 0) for r in results.values())))
    
    # Process results by category
    for category, result in results.items():
        # Create testcase element for each category
        testcase = ET.SubElement(testsuite, "testcase")
        testcase.set("classname", category)
        testcase.set("name", category)
        testcase.set("time", str(result.get('duration', 0)))
        
        # Add failure elements if necessary
        if not result.get('success', False):
            # Add failures
            for failure_data in result.get('failures_data', []):
                failure = ET.SubElement(testcase, "failure")
                failure.set("type", "AssertionError")
                failure.set("message", f"Test failed: {failure_data.get('test', '')}")
                failure.text = failure_data.get('traceback', '')
            
            # Add errors
            for error_data in result.get('errors_data', []):
                error = ET.SubElement(testcase, "error")
                error.set("type", "Error")
                error.set("message", f"Test error: {error_data.get('test', '')}")
                error.text = error_data.get('traceback', '')
        
        # Add skipped elements
        for skip_data in result.get('skipped_data', []):
            skipped = ET.SubElement(testcase, "skipped")
            skipped.set("message", skip_data.get('reason', 'Test skipped'))
    
    # Generate XML string with proper formatting
    rough_string = ET.tostring(testsuite, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="  ")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(pretty_xml)

def generate_html_report(results, output_path):
    """
    Generate HTML report from test results.
    
    Creates a readable HTML report with test results that can be
    viewed in a browser. Includes detailed information about test
    execution, failures, and skipped tests.
    
    Args:
        results: Dictionary mapping category names to test results
        output_path: Path where the HTML report should be saved
        
    Raises:
        Exception: If HTML report generation fails
    """
    # Calculate summary data
    total_categories = len(results)
    passed_categories = sum(1 for r in results.values() if r.get('success', False))
    failed_categories = total_categories - passed_categories
    total_duration = sum(r.get('duration', 0) for r in results.values())
    overall_status = "PASSED" if passed_categories == total_categories else "FAILED"
    overall_status_class = "passed" if passed_categories == total_categories else "failed"
    
    total_tests = sum(r.get('test_count', 0) for r in results.values())
    total_passed = total_tests - sum(r.get('failures', 0) for r in results.values()) - sum(r.get('errors', 0) for r in results.values()) - sum(r.get('skipped', 0) for r in results.values())
    total_failures = sum(r.get('failures', 0) for r in results.values())
    total_errors = sum(r.get('errors', 0) for r in results.values())
    total_skipped = sum(r.get('skipped', 0) for r in results.values())
    
    # Generate category results HTML
    category_results = ""
    for category, result in sorted(results.items()):
        status = "PASSED" if result.get('success', False) else "FAILED"
        status_class = "passed" if result.get('success', False) else "failed"
        
        failures_html = ""
        if result.get('failures', 0) > 0:
            failures_html += "<div class='failures'><h4>Failures:</h4><ul>"
            for failure in result.get('failures_data', []):
                failures_html += f"<li><div class='test-name'>{failure.get('test', '')}</div>"
                failures_html += f"<pre class='error-trace'>{failure.get('traceback', '')}</pre></li>"
            failures_html += "</ul></div>"
            
        errors_html = ""
        if result.get('errors', 0) > 0:
            errors_html += "<div class='errors'><h4>Errors:</h4><ul>"
            for error in result.get('errors_data', []):
                errors_html += f"<li><div class='test-name'>{error.get('test', '')}</div>"
                errors_html += f"<pre class='error-trace'>{error.get('traceback', '')}</pre></li>"
            errors_html += "</ul></div>"
            
        skipped_html = ""
        if result.get('skipped', 0) > 0:
            skipped_html += "<div class='skipped'><h4>Skipped:</h4><ul>"
            for skip in result.get('skipped_data', []):
                skipped_html += f"<li><div class='test-name'>{skip.get('test', '')}</div>"
                skipped_html += f"<div class='skip-reason'>Reason: {skip.get('reason', 'Unknown')}</div></li>"
            skipped_html += "</ul></div>"
        
        # Calculate test pass rate percentage
        test_count = result.get('test_count', 0)
        failures = result.get('failures', 0)
        errors = result.get('errors', 0)
        skipped = result.get('skipped', 0)
        passed = test_count - failures - errors - skipped
        pass_rate = (passed / test_count * 100) if test_count > 0 else 0
        
        category_results += f"""
        <div class="category">
            <div class="category-header {status_class}">
                <h3>{category} - <span class="{status_class}">{status}</span></h3>
            </div>
            <div class="category-body">
                <div class="category-summary">
                    <p><strong>Duration:</strong> {result.get('duration', 0):.2f}s</p>
                    <p><strong>Tests:</strong> {test_count} total, {passed} passed ({pass_rate:.1f}%)</p>
                    <p><strong>Failures:</strong> {failures}</p>
                    <p><strong>Errors:</strong> {errors}</p>
                    <p><strong>Skipped:</strong> {skipped}</p>
                </div>
                {failures_html}
                {errors_html}
                {skipped_html}
            </div>
        </div>
        """

    # Discover step report files relative to the HTML output directory
    step_reports = []
    search_dir = os.path.dirname(os.path.abspath(output_path))
    for root_dir, _, files in os.walk(search_dir):
        for file in files:
            if file.endswith('_step_report.html'):
                step_reports.append(os.path.relpath(os.path.join(root_dir, file), search_dir))

    step_reports_html = ""
    if step_reports:
        step_reports_html += "<h2>Step Reports</h2><ul>"
        for path in sorted(step_reports):
            step_reports_html += f'<li><a href="{path}">{os.path.basename(path)}</a></li>'
        step_reports_html += "</ul>"
    
    # Enhanced HTML template with better formatting
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Orbit Analyzer Test Results - {datetime.now().strftime("%Y-%m-%d %H:%M")}</title>
    <style>
        body {{ font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif; margin: 0; padding: 20px; color: #333; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ border-bottom: 2px solid #eee; padding-bottom: 10px; color: #444; }}
        .summary {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        .category {{ border: 1px solid #ddd; margin-bottom: 20px; border-radius: 5px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        .category-header {{ padding: 10px 15px; }}
        .category-header.passed {{ background-color: #e8f5e9; }}
        .category-header.failed {{ background-color: #ffebee; }}
        .category-body {{ padding: 15px; }}
        .passed {{ color: #4CAF50; }}
        .failed {{ color: #F44336; }}
        .stats {{ display: flex; flex-wrap: wrap; justify-content: space-between; margin: 20px 0; gap: 10px; }}
        .stat-card {{ background-color: #f5f5f5; padding: 20px; border-radius: 5px; text-align: center; flex: 1; min-width: 200px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        .stat-value {{ font-size: 2em; font-weight: bold; margin: 10px 0; }}
        .failures, .errors, .skipped {{ margin-top: 15px; }}
        pre.error-trace {{ background: #f8f8f8; padding: 10px; border-left: 3px solid #e74c3c; overflow: auto; max-height: 200px; }}
        ul {{ list-style-type: none; padding-left: 0; }}
        li {{ margin-bottom: 10px; }}
        .test-name {{ font-weight: bold; margin-bottom: 5px; }}
        .skip-reason {{ color: #777; font-style: italic; }}
        .timestamp {{ color: #777; margin-top: 5px; font-size: 0.9em; }}
        @media (max-width: 768px) {{
            .stat-card {{ min-width: 140px; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Orbit Analyzer Test Results</h1>
        
        <div class="summary">
            <p><strong>Date:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            <p><strong>Total Categories:</strong> {total_categories}</p>
            <p><strong>Overall Status:</strong> <span class="{overall_status_class}">{overall_status}</span></p>
            <p><strong>Total Test Count:</strong> {total_tests}</p>
            <p><strong>Test Pass Rate:</strong> {total_passed}/{total_tests} ({(total_passed/total_tests*100):.1f}% passed)</p>
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <h3>Categories</h3>
                <div class="stat-value">{total_categories}</div>
                <div>{passed_categories} passed, {failed_categories} failed</div>
            </div>
            <div class="stat-card">
                <h3>Tests Passed</h3>
                <div class="stat-value passed">{total_passed}</div>
                <div>{(total_passed/total_tests*100):.1f}%</div>
            </div>
            <div class="stat-card">
                <h3>Tests Failed</h3>
                <div class="stat-value failed">{total_failures + total_errors}</div>
                <div>{total_failures} failures, {total_errors} errors</div>
            </div>
            <div class="stat-card">
                <h3>Tests Skipped</h3>
                <div class="stat-value">{total_skipped}</div>
                <div>{(total_skipped/total_tests*100):.1f}%</div>
            </div>
            <div class="stat-card">
                <h3>Duration</h3>
                <div class="stat-value">{total_duration:.2f}s</div>
            </div>
        </div>
        
        <h2>Category Results</h2>
        {category_results}

        {step_reports_html}

        <div class="timestamp">Report generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
    </div>
</body>
</html>
"""
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

def main():
    """
    Main entry point for the test runner.
    
    Parses command line arguments, sets up the testing environment,
    runs tests according to the specified criteria, and generates reports.
    
    Returns:
        int: 0 if all tests passed, 1 otherwise
    """
    # Define command line arguments
    parser = argparse.ArgumentParser(description="Run Orbit Analyzer tests")
    
    # Test selection arguments
    selection_group = parser.add_argument_group("Test Selection")
    selection_group.add_argument("--category", action="append", 
                      help="Test categories to run (can specify multiple)")
    selection_group.add_argument("--tag", action="append", dest="tags",
                      help="Only run tests with specified tags (can specify multiple)")
    selection_group.add_argument("--importance", type=int, default=3, 
                      help="Maximum importance level (1=critical, 2=important, 3=optional)")
    selection_group.add_argument("--test-pattern", type=str,
                      help="Only run tests matching this pattern (e.g., 'test_component*')")
    selection_group.add_argument("--skip-slow", action="store_true", 
                      help="Skip slow tests")
    
    # Test execution arguments
    execution_group = parser.add_argument_group("Test Execution")
    execution_group.add_argument("--verbosity", type=int, default=1, 
                      help="Test verbosity level (0-3)")
    execution_group.add_argument("--failfast", action="store_true", 
                      help="Stop on first failure")
    execution_group.add_argument("--skip-dependency-check", action="store_true",
                      help="Skip test dependency validation")
    execution_group.add_argument("--timeout", type=int, default=60,
                      help="Test timeout in seconds")
    execution_group.add_argument("--parallel", action="store_true",
                      help="Run tests in parallel (when possible)")
    execution_group.add_argument("--processes", type=int, default=None,
                      help="Number of processes to use for parallel testing")
    
    # Coverage arguments
    coverage_group = parser.add_argument_group("Coverage")
    coverage_group.add_argument("--coverage", action="store_true", 
                      help="Run tests with coverage")
    coverage_group.add_argument("--html-coverage", action="store_true", 
                      help="Generate HTML coverage report")
    coverage_group.add_argument("--xml-coverage", action="store_true",
                      help="Generate XML coverage report for CI integration")
    coverage_group.add_argument("--coverage-source", action="append",
                      help="Specify module to measure coverage for (can specify multiple)")
    
    # Reporting arguments
    reporting_group = parser.add_argument_group("Reporting")
    reporting_group.add_argument("--junit-xml", type=str,
                      help="Generate JUnit XML report at specified path")
    reporting_group.add_argument("--html-report", type=str,
                      help="Generate HTML test report at specified path")
    reporting_group.add_argument("--save-results", type=str,
                      help="Save test results to specified JSON file")
    
    # Information arguments
    info_group = parser.add_argument_group("Information")
    info_group.add_argument("--list-categories", action="store_true", 
                      help="List available test categories and exit")
    info_group.add_argument("--list-tags", action="store_true",
                      help="List available test tags and exit")
    info_group.add_argument("--show-dependencies", action="store_true",
                      help="Show test dependencies and exit")
    info_group.add_argument("--version", action="store_true",
                      help="Show version information and exit")
    
    # Debugging arguments
    debug_group = parser.add_argument_group("Debugging")
    debug_group.add_argument("--debug-paths", action="store_true",
                      help="Print Python module search paths for debugging import issues")
    debug_group.add_argument("--debug-imports", action="store_true",
                      help="Attempt to import key modules and report results")
    
    # Backwards compatibility
    compat_group = parser.add_argument_group("Backwards Compatibility")
    compat_group.add_argument("--legacy-mode", action="store_true",
                      help="Use legacy test runner for backwards compatibility")
    
    args = parser.parse_args()
    
    # Debug paths if requested
    if args.debug_paths:
        print("Python module search paths (sys.path):")
        for i, path in enumerate(sys.path):
            print(f"{i}: {path}")
            if os.path.exists(path):
                subpaths = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
                print(f"   Subdirectories: {', '.join(subpaths[:10])}{'...' if len(subpaths) > 10 else ''}")
        print("\nAdditional paths added by run_all_tests.py:")
        for path in path_candidates:
            print(f" - {path} {'(exists)' if os.path.exists(path) else '(not found)'}")
        return 0
    
    # Debug imports if requested
    if args.debug_imports:
        print("\nAttempting to import key modules:")
        key_modules = [
            "controller", "log_analyzer", "error_clusterer", "gpt_summarizer", 
            "components.component_analyzer", "direct_component_analyzer",
            "step_aware_analyzer", "batch_processor", "reports"
        ]
        for module_name in key_modules:
            try:
                module = importlib.import_module(module_name)
                print(f"✅ Successfully imported {module_name} from {module.__file__}")
            except ImportError as e:
                print(f"❌ Failed to import {module_name}: {e}")
        return 0
    
    # Store CLI arguments in ConfigManager
    try:
        ConfigManager.set("TEST_VERBOSITY", args.verbosity)
        ConfigManager.set("FAIL_FAST", args.failfast)
        ConfigManager.set("SKIP_SLOW_TESTS", args.skip_slow)
        ConfigManager.set("TEST_TIMEOUT", args.timeout)
    except Exception as e:
        logger.warning(f"Could not store config values: {e}")
    
    # Version information
    if args.version:
        import platform
        from importlib import metadata
        try:
            version = metadata.version("orbit_analyzer")
        except Exception:
            version = "development"
        python_version = sys.version.split()[0]
        platform_name = platform.platform()
        print(f"Orbit Analyzer Test Runner v{version}")
        print(f"Python: {python_version}")
        print(f"Platform: {platform_name}")
        return 0
    
    # Information display
    if args.list_categories:
        # Import test modules to ensure registration
        if import_test_modules():
            categories = TestRegistry.list_categories()
            if categories:
                print("\nAvailable test categories:")
                for category, count in sorted(categories.items()):
                    print(f"  - {category:<15} ({count} tests)")
            else:
                print("\nNo test categories available.")
        else:
            print("\nError: Failed to import test modules.")
        return 0
    
    if args.list_tags:
        # Import test modules to ensure registration
        if import_test_modules():
            tags = TestRegistry.list_tags()
            if tags:
                print("\nAvailable test tags:")
                for tag in sorted(tags):
                    print(f"  - {tag}")
            else:
                print("\nNo tags defined in tests.")
        else:
            print("\nError: Failed to import test modules.")
        return 0
    
    if args.show_dependencies:
        # Import test modules to ensure registration
        if import_test_modules():
            dependencies = getattr(TestRegistry, '_dependencies', {})
            if dependencies:
                print("\nTest dependencies:")
                for test, deps in sorted(dependencies.items()):
                    print(f"  - {test} depends on: {', '.join(deps)}")
            else:
                print("\nNo dependencies defined between tests.")
        else:
            print("\nError: Failed to import test modules.")
        return 0
    
    # Legacy mode
    if args.legacy_mode:
        print("Running tests in legacy mode...")
        # Import and run the original test runner
        sys.path.insert(0, os.path.dirname(__file__))
        try:
            from legacy_test_runner import run_all_legacy_tests
            logger.info("Using legacy test runner")
            return 0 if run_all_legacy_tests() else 1
        except ImportError:
            legacy_runner_path = os.path.join(os.path.dirname(__file__), 'legacy_test_runner.py')
            logger.error(f"Error: Legacy test runner not found at {legacy_runner_path}")
            logger.error("You may need to create this file for backward compatibility.")
            return 1
    
    # Import test modules
    if not import_test_modules():
        logger.warning("Could not import any test modules. Please check your test files.")
    
    # Run tests with or without coverage
    if args.coverage or args.html_coverage or args.xml_coverage:
        success = run_with_coverage(args)
    else:
        success, results = discover_and_run_tests(
            categories=args.category,
            tags=args.tags,
            max_importance=args.importance,
            include_slow=not args.skip_slow,
            verbosity=args.verbosity,
            failfast=args.failfast,
            pattern=args.test_pattern,
            validate_dependencies=not args.skip_dependency_check,
            parallel=args.parallel,
            processes=args.processes
        )
        
        # Generate reports if requested
        if args.save_results and results:
            try:
                # Ensure directory exists
                os.makedirs(os.path.dirname(os.path.abspath(args.save_results)), exist_ok=True)
                
                # Convert data to JSON-serializable format
                json_results = {}
                for category, result in results.items():
                    # Create clean version without complex objects
                    json_results[category] = {}
                    for key, value in result.items():
                        # Skip complex data for simple summary
                        if key not in ['failures_data', 'errors_data', 'skipped_data']:
                            json_results[category][key] = value
                
                with open(args.save_results, 'w') as f:
                    json.dump(json_results, f, indent=2, default=str)
                print(f"\nTest results saved to {args.save_results}")
            except Exception as e:
                print(f"\nError saving test results: {e}")
        
        # Generate JUnit XML report if requested
        if args.junit_xml:
            try:
                generate_junit_xml(results, args.junit_xml)
                print(f"\nJUnit XML report saved to {args.junit_xml}")
            except Exception as e:
                print(f"\nError generating JUnit XML report: {e}")
        
        # Generate HTML report if requested
        if args.html_report:
            try:
                generate_html_report(results, args.html_report)
                print(f"\nHTML report saved to {args.html_report}")
            except Exception as e:
                print(f"\nError generating HTML report: {e}")
    
    return 0 if success else 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nTest execution interrupted by user.")
        sys.exit(130)  # Standard Unix exit code for interrupt
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)