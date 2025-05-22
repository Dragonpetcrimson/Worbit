"""
TestRegistry - Test registration and discovery system for Orbit Analyzer.

This module provides a mechanism for registering test classes with metadata such as
category, importance, tags, and dependencies, allowing for flexible test selection
and execution patterns.

Categories:
- core: Tests for core processing modules like log_analyzer, error_clusterer
- component: Tests for component-related functionality and identification
- report: Tests for report generation in various formats
- visualization: Tests for visualization generation
- integration: Tests that integrate multiple components together
- performance: Tests that measure performance characteristics
- structure: Tests for project structure, paths, and organization
"""

import sys
import platform as plt
from typing import Dict, List, Set, Any, Optional, Tuple


class TestRegistry:
    """
    Registry system for test modules enabling advanced categorization, 
    tagging, dependency tracking, and filtering.
    """
    
    # Standard categories
    CATEGORIES = [
        'core',        # Core module tests (log_analyzer, log_segmenter, etc.)
        'component',   # Component-related tests
        'report',      # Report generation tests
        'visualization', # Visualization tests
        'integration', # Integration tests
        'performance', # Performance tests
        'structure'    # Path and directory structure tests
    ]
    
    _test_modules = {}
    _dependencies = {}  # Track dependencies between tests
    _cache = {}  # Cache for get_modules results
    
    @classmethod
    def register(cls, category='core', importance=1, slow=False, tags=None, depends_on=None, 
                platforms=None, python_version=None):
        """
        Decorator to register a test class with the registry.
        
        Args:
            category: Test category for grouping (must be in CATEGORIES)
            importance: Importance level (1=critical, 2=important, 3=optional)
            slow: Whether this test is slow and should be skipped in quick test mode
            tags: Additional tags for fine-grained filtering (list of strings)
            depends_on: Test classes this test depends on (list of class names)
            platforms: Platforms where this test is valid (list of 'windows', 'linux', 'macos')
            python_version: Minimum Python version required (tuple like (3, 7))
            
        Returns:
            Decorator function
        """
        def decorator(test_class):
            # Validate category
            if category not in cls.CATEGORIES:
                raise ValueError(f"Invalid category '{category}'. Must be one of {cls.CATEGORIES}")
            
            # Initialize tags if None
            test_tags = tags or []
            
            # Store test info
            cls._test_modules.setdefault(category, []).append({
                'class': test_class,
                'name': test_class.__name__,
                'importance': importance,
                'slow': slow,
                'tags': test_tags,
                'platforms': platforms or ['windows', 'linux', 'macos'],
                'python_version': python_version or (3, 7)
            })
            
            # Store dependencies
            if depends_on:
                cls._dependencies[test_class.__name__] = depends_on
            
            # Clear cache when new tests are registered
            cls._cache.clear()
            
            return test_class
        return decorator
    
    @classmethod
    def get_modules(cls, category=None, max_importance=3, include_slow=True,
                   tags=None, platform=None, validate_dependencies=True):
        """
        Get registered test modules, with advanced filtering options.
        
        Args:
            category: Specific category to return (None=all categories)
            max_importance: Maximum importance level to include (lower=more important)
            include_slow: Whether to include tests marked as slow
            tags: Only include tests with these tags (list of strings)
            platform: Current platform ('windows', 'linux', 'macos')
            validate_dependencies: Check and include dependencies
            
        Returns:
            Dictionary of test modules by category
        """
        # Check cache first
        cache_key = (
            str(category), 
            max_importance, 
            include_slow, 
            str(tags), 
            str(platform), 
            validate_dependencies
        )
        if cache_key in cls._cache:
            return cls._cache[cache_key]
            
        # Determine current platform if not specified
        if platform is None:
            system = plt.system().lower()
            if 'windows' in system:
                platform = 'windows'
            elif 'darwin' in system:
                platform = 'macos'
            else:
                platform = 'linux'
        
        # Create python version tuple
        python_version = tuple(map(int, sys.version.split('.')[0:2]))
        
        # Filter modules by category
        if category:
            # Handle single category or list of categories
            if isinstance(category, list):
                categories = category
            else:
                categories = [category]
        else:
            categories = cls.CATEGORIES
        
        # Apply filters
        modules = {}
        for cat in categories:
            filtered_modules = []
            
            for module_info in cls._test_modules.get(cat, []):
                # Apply standard filters
                if module_info['importance'] > max_importance:
                    continue
                if not include_slow and module_info['slow']:
                    continue
                if platform not in module_info['platforms']:
                    continue
                if module_info['python_version'] > python_version:
                    continue
                
                # Apply tag filter
                if tags:
                    if not any(tag in module_info['tags'] for tag in tags):
                        continue
                
                filtered_modules.append(module_info)
            
            if filtered_modules:
                modules[cat] = filtered_modules
        
        # Add dependencies if needed
        if validate_dependencies and cls._dependencies:
            modules = cls._add_dependencies(modules)
        
        # Cache results
        cls._cache[cache_key] = modules
            
        return modules
    
    @classmethod
    def _add_dependencies(cls, modules):
        """
        Add dependencies to the filtered modules.
        
        Args:
            modules: Dictionary of filtered modules by category
            
        Returns:
            Updated modules dictionary with dependencies included
        """
        # Create a flat list of included test names
        included_tests = set()
        for cat_modules in modules.values():
            for module_info in cat_modules:
                included_tests.add(module_info['name'])
        
        # Check for missing dependencies
        missing_deps = {}
        for cat_modules in modules.values():
            for module_info in cat_modules:
                test_name = module_info['name']
                if test_name in cls._dependencies:
                    for dep in cls._dependencies[test_name]:
                        if dep not in included_tests:
                            missing_deps.setdefault(test_name, []).append(dep)
        
        # If there are missing dependencies, add them to the modules
        if missing_deps:
            # Log the missing dependencies for better diagnostics
            for test_name, deps in missing_deps.items():
                dep_list = ", ".join(deps)
                print(f"Adding missing dependencies for {test_name}: {dep_list}")
                
            # Find each missing dependency
            for test_name, deps in missing_deps.items():
                for dep in deps:
                    # Find the dependency in the full registry
                    for cat, cat_modules in cls._test_modules.items():
                        for module_info in cat_modules:
                            if module_info['name'] == dep:
                                # Add the dependency to the modules
                                if cat not in modules:
                                    modules[cat] = []
                                if module_info not in modules[cat]:
                                    modules[cat].append(module_info)
                                included_tests.add(dep)
                                
                                # Check if this dependency has its own dependencies
                                if dep in cls._dependencies:
                                    for sub_dep in cls._dependencies[dep]:
                                        if sub_dep not in included_tests:
                                            missing_deps.setdefault(dep, []).append(sub_dep)
        
        return modules
                
    @classmethod
    def list_categories(cls):
        """
        List all available categories with count of tests in each.
        
        Returns:
            Dictionary mapping category names to test counts
        """
        return {cat: len(modules) for cat, modules in cls._test_modules.items()}
        
    @classmethod
    def list_tags(cls):
        """
        List all available tags across all tests.
        
        Returns:
            Set of all tags used in the test registry
        """
        tags = set()
        for cat_modules in cls._test_modules.values():
            for module_info in cat_modules:
                tags.update(module_info['tags'])
        return tags
        
    @classmethod
    def validate_dependencies(cls, test_names=None):
        """
        Validate that all dependencies exist and are properly registered.
        
        Args:
            test_names: Optional list of test names to validate (None=all tests)
            
        Returns:
            Tuple of (valid, missing_dependencies) where missing_dependencies is a
            dictionary mapping test names to lists of missing dependencies
        """
        # Get all registered test names
        all_test_names = set()
        for cat_modules in cls._test_modules.values():
            for module_info in cat_modules:
                all_test_names.add(module_info['name'])
                
        # Determine which tests to validate
        if test_names is None:
            tests_to_validate = set(cls._dependencies.keys())
        else:
            tests_to_validate = set(test_names).intersection(cls._dependencies.keys())
            
        # Check for missing dependencies
        missing_dependencies = {}
        for test_name in tests_to_validate:
            deps = cls._dependencies.get(test_name, [])
            missing = [dep for dep in deps if dep not in all_test_names]
            if missing:
                missing_dependencies[test_name] = missing
                
        # Return validation results
        return (len(missing_dependencies) == 0, missing_dependencies)
        
    @classmethod
    def clear_cache(cls):
        """Clear the module cache, forcing re-computation on next get_modules call."""
        cls._cache.clear()
        
    @classmethod
    def get_test_by_name(cls, test_name):
        """
        Get a specific test class by name.
        
        Args:
            test_name: Name of the test class to retrieve
            
        Returns:
            Test class if found, None otherwise
        """
        for cat_modules in cls._test_modules.values():
            for module_info in cat_modules:
                if module_info['name'] == test_name:
                    return module_info['class']
        return None