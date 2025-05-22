"""
Component Utilities Module

This module provides standardized utility functions for component operations
throughout the Orbit Analyzer system. It ensures consistent component information
handling across the entire application lifecycle.
"""

import logging
import copy
from typing import Dict, Optional, Tuple, List, Any, Set

# Define component-related fields globally to ensure consistency
COMPONENT_FIELDS = {
    'component', 'component_source', 'source_component', 'root_cause_component',
    'primary_issue_component', 'affected_components', 'expected_component',
    'component_scores', 'component_distribution', 'parent_component', 
    'child_components', 'related_components'
}

def extract_component_fields(data: Dict) -> Dict:
    """
    Extract all component-related fields from a dictionary.
    
    Args:
        data: Dictionary containing component information
        
    Returns:
        Dictionary with only component-related fields
    """
    if not data or not isinstance(data, dict):
        return {}
        
    return {
        field: data[field] for field in COMPONENT_FIELDS 
        if field in data and data[field] is not None
    }

def apply_component_fields(data: Dict, component_fields: Dict) -> Dict:
    """
    Apply component fields to a dictionary.
    
    Args:
        data: Dictionary to apply component fields to
        component_fields: Component fields to apply
        
    Returns:
        Dictionary with component fields applied
    """
    # Properly handle the case where data is None or not a dictionary
    if data is None:
        data = {}
    elif not isinstance(data, dict):
        return data
        
    # Handle the case where component_fields is None or empty
    if not component_fields:
        return data
        
    # Create a copy to avoid modifying the original
    result = copy.deepcopy(data)
    
    # Apply component fields
    for field, value in component_fields.items():
        result[field] = value
        
    return result

def preserve_component_fields(source: Dict, target: Dict) -> Dict:
    """
    Preserve component fields from source to target.
    
    Args:
        source: Source dictionary with component fields
        target: Target dictionary to preserve fields to
        
    Returns:
        Target dictionary with component fields preserved
    """
    if not source or not target:
        return target
        
    component_fields = extract_component_fields(source)
    return apply_component_fields(target, component_fields)

def enrich_with_component_info(data: Dict, component_id: Optional[str] = None, source: str = 'default') -> Dict:
    """
    Enrich data with component information.
    
    Args:
        data: Dictionary to enrich
        component_id: Component ID to use (if None, uses 'component' from data)
        source: Source of component identification
        
    Returns:
        Enriched dictionary with component information
    """
    if not data or not isinstance(data, dict):
        return data
        
    # Determine component ID
    if not component_id:
        component_id = data.get('component', 'unknown')
        
    # Get component info
    from components.component_model import get_component_registry
    registry = get_component_registry()
    component = registry.get_component(component_id)
    
    # Create a copy to avoid modifying the original
    result = copy.deepcopy(data)
    
    # Set component fields
    result['component'] = component.id
    result['component_source'] = source or data.get('component_source', 'default')
    result['component_name'] = component.name
    result['component_description'] = component.description
    result['component_type'] = component.type
    
    # Set relationship fields if available
    if component._parent:
        result['parent_component'] = component._parent
    if component._children:
        result['child_components'] = component._children
    if component._related_to:
        result['related_components'] = component._related_to
        
    return result

def identify_component_from_file(filename: str) -> Tuple[str, str]:
    """
    Identify component based on filename.
    
    Args:
        filename: Filename to analyze
        
    Returns:
        Tuple of (component_id, source)
    """
    registry = get_component_registry()
    component_id = registry.identify_component_from_filename(filename)
    return component_id, 'filename'

def determine_primary_component(errors: List[Dict]) -> str:
    """
    Determine primary component from a list of errors.
    
    Args:
        errors: List of error objects
        
    Returns:
        Component ID of the likely primary component
    """
    if not errors:
        return 'unknown'
        
    # Count components
    component_counts = {}
    for error in errors:
        component = error.get('component', 'unknown')
        if component not in component_counts:
            component_counts[component] = 0
        component_counts[component] += 1
    
    # Identify primary component
    from components.component_model import get_component_registry
    registry = get_component_registry()
    primary_component_id = registry.identify_primary_component(component_counts)
    return primary_component_id

def validate_component_data(data: Dict, primary_issue_component: Optional[str] = None) -> Dict:
    """
    Validate component data and ensure required fields.
    
    Args:
        data: Dictionary to validate
        primary_issue_component: Primary issue component
        
    Returns:
        Validated dictionary with required fields
    """
    if not data or not isinstance(data, dict):
        return data
        
    # Create a copy to avoid modifying the original
    result = copy.deepcopy(data)
    
    # Ensure required fields
    if 'component' not in result or not result['component']:
        result['component'] = 'unknown'
        
    if 'component_source' not in result or not result['component_source']:
        result['component_source'] = 'default'
        
    # Add primary_issue_component if provided and not present
    if primary_issue_component and 'primary_issue_component' not in result:
        result['primary_issue_component'] = primary_issue_component
        
    return result

def verify_component_preservation(original: Dict, processed: Dict) -> bool:
    """
    Verify that component information is preserved.
    
    Args:
        original: Original dictionary with component fields
        processed: Processed dictionary to verify
        
    Returns:
        True if component information is preserved, False otherwise
    """
    if not original or not processed:
        return True
        
    if isinstance(original, dict) and isinstance(processed, dict):
        # Check component fields
        for field in COMPONENT_FIELDS:
            if field in original and original[field] is not None:
                if field not in processed or processed[field] != original[field]:
                    logging.warning(f"Component field '{field}' not preserved: '{original[field]}' -> '{processed.get(field, 'missing')}'")
                    return False
                    
    return True

def normalize_component_fields(data: Dict) -> Dict:
    """
    Normalize component fields to standard format.
    
    Args:
        data: Dictionary to normalize
        
    Returns:
        Normalized dictionary
    """
    if not data or not isinstance(data, dict):
        return data
        
    # Create a copy to avoid modifying the original
    result = copy.deepcopy(data)
    
    # Normalize component ID
    if 'component' in result and result['component']:
        result['component'] = result['component'].lower()
        
    # Normalize primary_issue_component
    if 'primary_issue_component' in result and result['primary_issue_component']:
        result['primary_issue_component'] = result['primary_issue_component'].lower()
        
    # Normalize root_cause_component
    if 'root_cause_component' in result and result['root_cause_component']:
        result['root_cause_component'] = result['root_cause_component'].lower()
        
    return result

def count_components(errors: List[Dict]) -> Dict[str, int]:
    """
    Count component occurrences in a list of errors.
    
    Args:
        errors: List of error dictionaries
        
    Returns:
        Dictionary mapping component IDs to counts
    """
    component_counts = {}
    
    for error in errors:
        if not isinstance(error, dict):
            continue
            
        component = error.get('component', 'unknown')
        
        if component not in component_counts:
            component_counts[component] = 0
            
        component_counts[component] += 1
        
    return component_counts

def count_component_sources(errors: List[Dict]) -> Dict[str, int]:
    """
    Count component sources in a list of errors.
    
    Args:
        errors: List of error dictionaries
        
    Returns:
        Dictionary mapping component sources to counts
    """
    source_counts = {}
    
    for error in errors:
        if not isinstance(error, dict):
            continue
            
        source = error.get('component_source', 'default')
        
        if source not in source_counts:
            source_counts[source] = 0
            
        source_counts[source] += 1
        
    return source_counts

def propagate_primary_component(errors: List[Dict], primary_issue_component: str) -> List[Dict]:
    """
    Propagate primary_issue_component to all errors.
    
    Args:
        errors: List of error dictionaries
        primary_issue_component: Primary issue component
        
    Returns:
        List of error dictionaries with primary_issue_component
    """
    if not errors or not primary_issue_component:
        return errors
        
    # Create a deep copy to avoid modifying the original
    result = copy.deepcopy(errors)
    
    # Set primary_issue_component on all errors
    for error in result:
        if isinstance(error, dict):
            error['primary_issue_component'] = primary_issue_component
            
    return result

def propagate_primary_component_to_clusters(clusters: Dict[int, List[Dict]], primary_issue_component: str) -> Dict[int, List[Dict]]:
    """
    Propagate primary_issue_component to all errors in clusters.
    
    Args:
        clusters: Dictionary mapping cluster IDs to lists of errors
        primary_issue_component: Primary issue component
        
    Returns:
        Clusters with primary_issue_component propagated
    """
    if not clusters or not primary_issue_component:
        return clusters
        
    # Create a deep copy to avoid modifying the original
    result = copy.deepcopy(clusters)
    
    # Set primary_issue_component on all errors in all clusters
    for cluster_id, errors in result.items():
        for error in errors:
            if isinstance(error, dict):
                error['primary_issue_component'] = primary_issue_component
                
    return result

def get_component_error_distribution(errors: List[Dict]) -> Dict[str, int]:
    """
    Get distribution of errors by component.
    
    Args:
        errors: List of error dictionaries
        
    Returns:
        Dictionary mapping component IDs to error counts
    """
    return count_components(errors)

def ensure_component_consistency(data: Dict, reference_component: Optional[str] = None) -> Dict:
    """
    Ensure component fields are consistent with each other.
    
    Args:
        data: Dictionary to ensure consistency
        reference_component: Reference component to use if needed
        
    Returns:
        Dictionary with consistent component fields
    """
    if not data or not isinstance(data, dict):
        return data
        
    # Create a copy to avoid modifying the original
    result = copy.deepcopy(data)
    
    # If component is missing but source_component is present, use source_component
    if ('component' not in result or not result['component'] or result['component'] == 'unknown') and 'source_component' in result and result['source_component']:
        result['component'] = result['source_component']
        result['component_source'] = 'inferred_from_source'
        
    # If reference_component is provided and component is unknown, use reference
    if reference_component and ('component' not in result or not result['component'] or result['component'] == 'unknown'):
        result['component'] = reference_component
        result['component_source'] = 'inferred_from_reference'
        
    return result

def merge_component_info(source: Dict, target: Dict, override: bool = False) -> Dict:
    """
    Merge component information from source to target.
    
    Args:
        source: Source dictionary with component information
        target: Target dictionary to merge into
        override: Whether to override existing fields
        
    Returns:
        Target dictionary with merged component information
    """
    if not source or not target:
        return target
        
    # Create a copy to avoid modifying the original
    result = copy.deepcopy(target)
    
    # Extract component fields from source
    component_fields = extract_component_fields(source)
    
    # Apply fields based on override setting
    for field, value in component_fields.items():
        # Skip if field exists and override is False
        if field in result and not override:
            continue
            
        result[field] = value
        
    return result

# Import the component registry function at module level to avoid circular imports
from components.component_model import get_component_registry