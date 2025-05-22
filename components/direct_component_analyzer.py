"""
Direct Component Analyzer - Identifies components from filenames and basic patterns

This module provides direct component identification based on filenames and basic patterns.
It is designed to be efficient and reliable for quick component assignment.
"""

import os
import logging
import re
import copy
from typing import Dict, List, Tuple, Optional, Any, Set, Union
from collections import defaultdict

from components.component_model import get_component_registry, create_component_info
from components.component_utils import (
    identify_component_from_file, 
    enrich_with_component_info,
    determine_primary_component
)

# Component information with descriptions (static reference data)
COMPONENT_INFO = {
    "soa": {
        "name": "SOA",
        "description": "SiriusXM application built on Android",
        "related_to": ["phoebe", "mimosa", "charles"],
        "parent": "android",  # Parent relationship
        "key_indicators": ["siriusxm", "channel", "playback", "audio", "stream"]
    },
    "android": {
        "name": "Android",
        "description": "Android system and platform errors",
        "related_to": ["soa"],
        "children": ["soa"],  # Child relationship
        "key_indicators": ["activity", "intent", "fragment", "service", "broadcast"]
    },
    "mimosa": {
        "name": "Mimosa",
        "description": "Provides fake testing data (Satellite/IP channel)",
        "related_to": ["soa", "lapetus"],
        "key_indicators": ["data", "signal", "simulate"]
    },
    "charles": {
        "name": "Charles",
        "description": "Proxy for live data",
        "related_to": ["soa", "phoebe"],
        "key_indicators": ["http", "connection", "timeout"]
    },
    "phoebe": {
        "name": "Phoebe",
        "description": "Proxy to run data to SOA",
        "related_to": ["soa", "arecibo", "lapetus", "charles"],
        "key_indicators": ["proxy", "transmission", "data"]
    },
    "arecibo": {
        "name": "Arecibo",
        "description": "Monitors traffic from Phoebe",
        "related_to": ["phoebe"],
        "key_indicators": ["monitoring", "traffic"]
    },
    "translator": {
        "name": "Translator",
        "description": "Translates commands between test framework and SOA",
        "related_to": ["soa", "smite"],
        "key_indicators": ["translator", "command", "autosmite"]
    },
    "telesto": {
        "name": "Telesto",
        "description": "Coordinates components",
        "related_to": ["mimosa", "phoebe", "lapetus", "arecibo"],
        "key_indicators": ["coordination", "component"]
    },
    "lapetus": {
        "name": "Lapetus",
        "description": "API to add channel and categories",
        "related_to": ["phoebe", "telesto", "mimosa"],
        "key_indicators": ["api", "channel", "configuration"]
    },
    "ip_traffic": {
        "name": "IP Traffic",
        "description": "Network traffic and HTTP communication",
        "related_to": ["charles", "soa", "phoebe"],
        "key_indicators": ["http", "get", "post", "status"]
    }
}

# Create a component identifier cache to avoid redundant identification
# This significantly improves performance for repeated analysis of similar filenames
class ComponentCache:
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
    
    def get(self, text: str) -> Optional[str]:
        """Get component from cache if it exists"""
        if not text:
            return None
        # Use a more efficient hash for cache key to avoid storing large strings
        cache_key = hash(text.lower())
        return self.cache.get(cache_key)
    
    def set(self, text: str, component: str) -> None:
        """Add component to cache, manage cache size"""
        if not text:
            return
        
        # Use a more efficient hash for cache key
        cache_key = hash(text.lower())
        
        # Manage cache size - use simple LRU mechanism if needed
        if len(self.cache) >= self.max_size:
            # Simple approach - clear half the cache when full
            # In a production system, we would use a proper LRU implementation
            items = list(self.cache.items())
            self.cache = dict(items[len(items)//2:])
        
        self.cache[cache_key] = component

# Global component cache instance
component_cache = ComponentCache()

def identify_component_from_filename(filename: str) -> Tuple[str, str]:
    """
    Identify component based on filename pattern.
    
    Args:
        filename: Filename to analyze
        
    Returns:
        Tuple of (component, source)
    """
    return identify_component_from_file(filename)

def trace_component_changes(error_before, error_after, operation_name="unknown"):
    """
    Trace component changes for debugging purposes.
    
    Args:
        error_before: Error dictionary before operation
        error_after: Error dictionary after operation
        operation_name: Name of the operation for logging
        
    Returns:
        True if changes were detected, False otherwise
    """
    # Skip if either error is None
    if error_before is None or error_after is None:
        return False
        
    # Get component fields from base module if available 
    try:
        from reports.base import COMPONENT_FIELDS
    except ImportError:
        # Fall back to a basic set if import fails
        COMPONENT_FIELDS = {
            'component', 'component_source', 'source_component', 'root_cause_component',
            'primary_issue_component', 'affected_components', 'expected_component'
        }
        
    # Check for changes in component fields
    changes_detected = False
    for field in COMPONENT_FIELDS:
        if field in error_before and field in error_after:
            if error_before[field] != error_after[field]:
                logging.warning(
                    f"Component field '{field}' changed during {operation_name}: "
                    f"'{error_before[field]}' -> '{error_after[field]}'"
                )
                changes_detected = True
                
    return changes_detected

class ComponentAnalyzer:
    """
    Optimized Component Analyzer that performs component analysis in a single pass
    and maintains a cache of results to prevent redundant processing.
    """
    def __init__(self):
        """Initialize the analyzer with component registry."""
        self.registry = get_component_registry()
        # Class-level cache to store already processed error IDs
        self._cache = {}
        self._component_counts = defaultdict(int)
        self._primary_issue_component = 'unknown'
        self._component_summary = []
        self._processed_errors = 0
    
    def reset(self) -> None:
        """Reset internal state for a new analysis"""
        self._cache.clear()
        self._component_counts = defaultdict(int)
        self._primary_issue_component = 'unknown'
        self._component_summary = []
        self._processed_errors = 0
    
    def _get_error_id(self, error: Dict) -> str:
        """
        Generate a unique identifier for an error to use as cache key
        For thread safety, the key needs to be immutable
        """
        # Use a combination of file and first 100 chars of text if available
        file_str = str(error.get('file', ''))
        text_str = str(error.get('text', ''))[:100] if 'text' in error else ''
        return f"{file_str}:{hash(text_str)}"
    
    def assign_component_to_error(self, error: Dict) -> None:
        """
        Assign component to a single error based on analysis rules.
        Uses caching to avoid redundant processing.
        
        Args:
            error: Error dictionary to be updated with component info
        """
        if not error:
            return
            
        # Skip if already has a valid component
        if 'component' in error and error['component'] != 'unknown':
            self._component_counts[error['component']] += 1
            self._processed_errors += 1
            return
        
        # Keep original for tracing
        error_before = None
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            error_before = copy.deepcopy(error)
        
        # Check cache for this error
        error_id = self._get_error_id(error)
        if error_id in self._cache:
            cached_result = self._cache[error_id]
            error['component'] = cached_result['component']
            error['component_source'] = cached_result['component_source']
            self._component_counts[error['component']] += 1
            self._processed_errors += 1
            
            # Trace any component changes if in debug mode
            if error_before is not None:
                trace_component_changes(error_before, error, "component_assignment_from_cache")
            return
        
        # Get filename (convert to string in case it's not already)
        filename = str(error.get('file', '')).lower()
        
        # Identify component based on filename
        component, source = identify_component_from_filename(filename)
        
        # Update error with component info
        error['component'] = component
        error['component_source'] = source
        
        # Cache the result
        self._cache[error_id] = {
            'component': component,
            'component_source': source
        }
        
        # Update component counts
        self._component_counts[component] += 1
        self._processed_errors += 1
        
        # Enrich with full component information
        enrich_with_component_info(error, component, source)
        
        # Trace any component changes if in debug mode
        if error_before is not None:
            trace_component_changes(error_before, error, "component_assignment")
    
    def identify_primary_component(self) -> str:
        """
        Identify the primary component with issues based on component counts.
        Applies special logic for SOA vs Android determination.
        
        Returns:
            Primary issue component string
        """
        if not self._component_counts:
            return 'unknown'
        
        # Filter out 'unknown' for primary component selection
        filtered_counts = {k: v for k, v in self._component_counts.items() 
                         if k != 'unknown'}
        
        # Apply special SOA vs Android logic
        if 'soa' in filtered_counts and 'android' in filtered_counts:
            # If SOA errors are at least 50% of Android errors, prefer SOA as primary
            if filtered_counts['soa'] >= filtered_counts['android'] * 0.5:
                # Boost SOA weight for primary component selection
                filtered_counts['soa'] = filtered_counts['soa'] * 1.5
        
        # Select primary component - prefer non-unknown components
        if filtered_counts:
            primary_issue_component = max(filtered_counts.items(), key=lambda x: x[1])[0]
        else:
            # If all are unknown, just use the max
            primary_issue_component = max(self._component_counts.items(), key=lambda x: x[1])[0]
        
        self._primary_issue_component = primary_issue_component
        return primary_issue_component
    
    def generate_component_summary(self) -> List[Dict]:
        """
        Generate summary of components for error report.
        Includes component metadata from registry.
        
        Returns:
            List of component summary dictionaries
        """
        component_summary = []
        
        # Add entries for each component with errors (sorted by error count)
        for comp_id, count in sorted(self._component_counts.items(), 
                                    key=lambda x: x[1], reverse=True):
            if comp_id != 'unknown' and count > 0:
                # Use registry to get component info
                component = self.registry.get_component(comp_id)
                
                # Create summary entry
                summary_entry = {
                    "id": comp_id,
                    "name": component.name,
                    "description": component.description,
                    "error_count": count,
                    "percentage": (count / self._processed_errors * 100) if self._processed_errors > 0 else 0
                }
                
                # Add related components directly from registry
                if hasattr(component, 'related_to'):
                    summary_entry['related_to'] = component.related_to
                else:
                    # Fall back to static component info if needed
                    summary_entry['related_to'] = COMPONENT_INFO.get(comp_id, {}).get('related_to', [])
                
                component_summary.append(summary_entry)
        
        self._component_summary = component_summary
        return component_summary
    
    def add_root_cause_info(self, errors: List[Dict]) -> None:
        """
        Add root cause information to the first error
        
        Args:
            errors: List of error dictionaries
        """
        if not errors or self._primary_issue_component == 'unknown':
            return
        
        # Keep original for tracing
        error_before = None
        if logging.getLogger().isEnabledFor(logging.DEBUG) and errors:
            error_before = copy.deepcopy(errors[0])
        
        # Add root cause info to first error without overriding existing values
        first_error = errors[0]
        if 'root_cause_component' not in first_error:
            first_error['root_cause_component'] = self._primary_issue_component

        # Get component from registry for description
        component = self.registry.get_component(self._primary_issue_component)
        if 'root_cause_description' not in first_error:
            first_error['root_cause_description'] = component.description
        
        # Use related_to from registry if available
        related_components = []
        if hasattr(component, 'related_to'):
            related_components = component.related_to
        else:
            # Fall back to static component info if needed
            related_components = COMPONENT_INFO.get(self._primary_issue_component, {}).get('related_to', [])
        
        # Filter to only affected components (those found in errors)
        affected_components = [comp for comp in related_components if comp in self._component_counts]
        errors[0]['affected_components'] = ', '.join(affected_components)
        
        # Get parent/child relationships
        parent = None
        if hasattr(component, 'parent'):
            parent = component.parent
        else:
            parent = COMPONENT_INFO.get(self._primary_issue_component, {}).get('parent', None)
        
        children = []
        if hasattr(component, 'children'):
            children = component.children
        else:
            children = COMPONENT_INFO.get(self._primary_issue_component, {}).get('children', [])
        
        if parent:
            errors[0]['parent_component'] = parent
        if children:
            errors[0]['child_components'] = ', '.join(children)
            
        # Trace any component changes if in debug mode
        if error_before is not None:
            trace_component_changes(error_before, errors[0], "root_cause_info_addition")

def assign_components_and_relationships(errors: List[Dict]) -> Tuple[List[Dict], List[Dict], str]:
    """
    Optimized main function to assign components to errors and identify relationships.
    Uses caching to avoid redundant processing and eliminates deep copying.
    
    Args:
        errors: List of error dictionaries
        
    Returns:
        Tuple of (updated errors, component summary, primary issue component)
    """
    # Create an analyzer instance
    analyzer = ComponentAnalyzer()
    
    # Make a copy of errors to avoid modifying the original
    processed_errors = copy.deepcopy(errors)
    
    # Assign components to each error
    for error in processed_errors:
        analyzer.assign_component_to_error(error)
    
    # Identify the primary component responsible for issues
    primary_issue_component = analyzer.identify_primary_component()
    
    # Generate component summary for reporting
    component_summary = analyzer.generate_component_summary()
    
    # Add root cause information to the first error if available
    if processed_errors:
        analyzer.add_root_cause_info(processed_errors)
    
    logging.info(f"Component analysis complete. Primary issue component: {primary_issue_component}")
    logging.info(f"Component distribution: {analyzer._component_counts}")
    
    return processed_errors, component_summary, primary_issue_component