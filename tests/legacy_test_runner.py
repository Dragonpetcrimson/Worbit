#!/usr/bin/env python3
"""
legacy_test_runner.py - Backward compatibility for running tests with the old system

This script provides a way to run tests using the original approach during the transition
to the new TestRegistry-based system. It discovers and runs test modules with the old
naming convention and function-based tests.

Usage:
    python legacy_test_runner.py                        # Run all tests
    python legacy_test_runner.py --include-performance  # Include performance tests
"""

import os
import sys
import unittest
import logging
import importlib
import time
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("legacy_test_runner")

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

def discover_legacy_test_modules() -> List[str]:
    """
    Discover all existing test modules in the current directory with *_test.py pattern.
    
    This simulates the old test discovery mechanism by finding all Python
    files with the _test.py suffix in the current directory.
    
    Returns:
        List of test module names
    """
    test_modules = []
    
    # Get the directory this script is in
    script_dir = os.path.dirname(os.path.realpath(__file__))
    
    # Find all Python files matching the pattern
    for filename in os.listdir(script_dir):
        if filename.endswith('_test.py') and filename != 'parameterized_tests.py':
            module_name = filename[:-3]  # Remove .py extension
            test_modules.append(module_name)
    
    # Log discovered modules
    if test_modules:
        logger.info(f"Discovered {len(test_modules)} test modules")
        logger.debug(f"Module list: {', '.join(test_modules)}")
    else:
        logger.warning("No test modules discovered. Check directory location.")
    
    return sorted(test_modules)

def check_module_importable(module_name: str) -> bool:
    """
    Check if a module can be imported without errors.
    
    Args:
        module_name: Name of the module to check
        
    Returns:
        True if the module can be imported, False otherwise
    """
    try:
        # Try to import the module
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False
    except Exception:
        return False

def run_single_test_module(module_name: str) -> Tuple[bool, Dict[str, Any]]:
    """
    Run tests from a single module.
    
    Attempts to import the module and run all test functions and test
    classes contained within it. Handles errors gracefully and reports results.
    
    Args:
        module_name: Name of the test module to run
        
    Returns:
        Tuple of (success_status, result_stats)
    """
    logger.info(f"Running tests from {module_name}...")
    
    result_stats = {
        'passed': False,
        'total': 0,
        'failures': 0,
        'errors': 0,
        'skipped': 0,
        'duration': 0
    }
    
    try:
        # Try to import the module
        module = importlib.import_module(module_name)
        
        # Prepare a test suite
        suite = unittest.TestSuite()
        
        # Look for test_* functions in the module
        function_count = 0
        for name in dir(module):
            if name.startswith('test_') and callable(getattr(module, name)):
                # Create a test case for standalone function
                test_func = getattr(module, name)
                
                # If __test__ exists and is False, skip it
                if hasattr(test_func, '__test__') and not test_func.__test__:
                    logger.debug(f"Skipping disabled test function: {name}")
                    result_stats['skipped'] += 1
                    continue
                
                # Create a TestCase class for this function
                class FunctionTestCase(unittest.TestCase):
                    def test_function(self):
                        test_func()
                
                FunctionTestCase.__name__ = f"{name}_TestCase"
                suite.addTest(FunctionTestCase("test_function"))
                function_count += 1
        
        # Log standalone functions found
        if function_count > 0:
            logger.debug(f"Found {function_count} standalone test functions in {module_name}")
        
        # Also find all TestCase classes
        class_count = 0
        for name in dir(module):
            obj = getattr(module, name)
            if isinstance(obj, type) and issubclass(obj, unittest.TestCase) and obj != unittest.TestCase:
                test_case = obj
                tests = unittest.defaultTestLoader.loadTestsFromTestCase(test_case)
                suite.addTests(tests)
                class_count += 1
        
        # Log TestCase classes found
        if class_count > 0:
            logger.debug(f"Found {class_count} TestCase classes in {module_name}")
        
        # Check if any tests were found
        if function_count == 0 and class_count == 0:
            logger.warning(f"No tests found in {module_name}")
            result_stats['skipped'] += 1
            return False, result_stats
        
        # Run the tests
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        # Update result statistics
        result_stats['total'] = result.testsRun
        result_stats['failures'] = len(result.failures)
        result_stats['errors'] = len(result.errors)
        result_stats['skipped'] = len(getattr(result, 'skipped', []))
        result_stats['passed'] = result.wasSuccessful()
        
        return result.wasSuccessful(), result_stats
        
    except ImportError as e:
        logger.error(f"Failed to import module {module_name}: {e}")
        result_stats['errors'] += 1
        return False, result_stats
    except Exception as e:
        logger.error(f"Error running tests from {module_name}: {e}")
        import traceback
        traceback.print_exc()
        result_stats['errors'] += 1
        return False, result_stats

def run_all_legacy_tests(exclude_performance: bool = True) -> bool:
    """
    Run all legacy tests.
    
    Discovers and runs all test modules with the legacy naming convention.
    Provides summary statistics and handles errors gracefully.
    
    Args:
        exclude_performance: Whether to exclude performance tests (default: True)
        
    Returns:
        True if all tests passed, False otherwise
    """
    start_time = time.time()
    
    # Discover all test modules
    test_modules = discover_legacy_test_modules()
    
    # Filter out performance tests if requested
    if exclude_performance:
        original_count = len(test_modules)
        test_modules = [m for m in test_modules if 'performance' not in m]
        if original_count != len(test_modules):
            logger.info(f"Excluded {original_count - len(test_modules)} performance test modules")
    
    logger.info(f"Preparing to run {len(test_modules)} test modules: {', '.join(test_modules)}")
    
    # Track results
    results = {}
    total_tests = 0
    total_failures = 0
    total_errors = 0
    total_skipped = 0
    
    # Run tests from each module
    for module_name in test_modules:
        module_start_time = time.time()
        
        # Check if module is importable before attempting to run tests
        if not check_module_importable(module_name):
            logger.warning(f"Module {module_name} cannot be imported, skipping")
            results[module_name] = {
                'passed': False,
                'total': 0,
                'failures': 0,
                'errors': 1,  # Count as an error
                'skipped': 0,
                'duration': 0,
                'message': "Module could not be imported"
            }
            total_errors += 1
            continue
        
        # Run the tests in this module
        passed, stats = run_single_test_module(module_name)
        module_duration = time.time() - module_start_time
        
        # Update stats with duration
        stats['duration'] = module_duration
        results[module_name] = stats
        
        # Update totals
        total_tests += stats.get('total', 0)
        total_failures += stats.get('failures', 0)
        total_errors += stats.get('errors', 0)
        total_skipped += stats.get('skipped', 0)
    
    # Print summary
    total_duration = time.time() - start_time
    print("\n" + "="*70)
    print("LEGACY TEST SUMMARY")
    print("=" * 70)
    
    passed_count = sum(1 for r in results.values() if r.get('passed', False))
    
    for module_name, result in sorted(results.items()):
        status = "✅ PASSED" if result.get('passed', False) else "❌ FAILED"
        details = f"Tests: {result.get('total', 0)}, "
        details += f"Failures: {result.get('failures', 0)}, "
        details += f"Errors: {result.get('errors', 0)}, "
        details += f"Skipped: {result.get('skipped', 0)}"
        
        print(f"{status} | {module_name:<30} | {result.get('duration', 0):.2f}s | {details}")
    
    print("-" * 70)
    print(f"Total modules: {len(results)}")
    print(f"Modules passed: {passed_count}/{len(results)}")
    print(f"Total tests: {total_tests}")
    print(f"Total failures: {total_failures}")
    print(f"Total errors: {total_errors}")
    print(f"Total skipped: {total_skipped}")
    print(f"Total duration: {total_duration:.2f}s")
    
    if passed_count == len(results):
        print("\n🎉 All test modules passed!")
    else:
        print("\n⚠️ Some test modules failed. Check output above for details.")
    
    # Return overall success
    return passed_count == len(results)

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Run legacy tests")
    parser.add_argument("--include-performance", action="store_true", 
                      help="Include performance tests")
    parser.add_argument("--module", type=str, 
                      help="Run only the specified test module")
    parser.add_argument("--list", action="store_true",
                      help="List available test modules without running them")
    args = parser.parse_args()
    
    # Just list modules if requested
    if args.list:
        modules = discover_legacy_test_modules()
        print("\nAvailable test modules:")
        for module in modules:
            print(f"  - {module}")
        sys.exit(0)
    
    # Run a single module if specified
    if args.module:
        if check_module_importable(args.module):
            passed, _ = run_single_test_module(args.module)
            sys.exit(0 if passed else 1)
        else:
            logger.error(f"Module {args.module} cannot be imported")
            sys.exit(1)
    
    # Run all tests
    success = run_all_legacy_tests(exclude_performance=not args.include_performance)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)