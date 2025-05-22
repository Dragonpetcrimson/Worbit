"""
reports/data_preprocessor.py - Data normalization and validation

Enhanced with component information preservation mechanisms to ensure consistent 
component identification across the analysis pipeline.
"""

import logging
import copy
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
from datetime import datetime

from reports.base import COMPONENT_FIELDS, ensure_datetime


def extract_component_fields(data: Dict) -> Dict:
    """
    Extract all component-related fields from a dictionary.
    
    Args:
        data: Dictionary to extract component fields from
        
    Returns:
        Dictionary containing only component-related fields
    """
    if not data or not isinstance(data, dict):
        return {}
    
    return {
        field: data[field] for field in COMPONENT_FIELDS 
        if field in data and data[field] is not None
    }


def apply_component_fields(target: Dict, component_fields: Dict) -> Dict:
    """
    Apply component fields to a dictionary without overriding existing values.
    
    Args:
        target: Target dictionary
        component_fields: Component fields to apply
        
    Returns:
        Target dictionary with component fields applied
    """
    if not target or not isinstance(target, dict) or not component_fields:
        return target
    
    # Create a copy to avoid modifying the original
    result = copy.deepcopy(target)
    
    # Apply component fields, but don't override existing values except unknown
    for field, value in component_fields.items():
        if field not in result or result[field] is None or result[field] == 'unknown':
            result[field] = value
    
    return result


def preserve_component_fields(source: Dict, target: Dict) -> Dict:
    """
    Preserve component fields from source to target dictionary.
    
    Args:
        source: Source dictionary
        target: Target dictionary
        
    Returns:
        Target dictionary with component fields preserved
    """
    if not source or not target:
        return target
    
    component_fields = extract_component_fields(source)
    return apply_component_fields(target, component_fields)


def count_components(data: List[Dict]) -> Dict[str, int]:
    """
    Count component occurrences in a list of error dictionaries.
    
    Args:
        data: List of error dictionaries
        
    Returns:
        Dictionary mapping components to their counts
    """
    component_counts = {}
    for item in data:
        if isinstance(item, dict) and 'component' in item:
            comp = item.get('component', 'unknown')
            component_counts[comp] = component_counts.get(comp, 0) + 1
    return component_counts


def count_component_sources(data: List[Dict]) -> Dict[str, int]:
    """
    Count component_source occurrences in a list of error dictionaries.
    
    Args:
        data: List of error dictionaries
        
    Returns:
        Dictionary mapping component_sources to their counts
    """
    source_counts = {}
    for item in data:
        if isinstance(item, dict) and 'component_source' in item:
            source = item.get('component_source', 'unknown')
            source_counts[source] = source_counts.get(source, 0) + 1
    return source_counts


def extract_primary_component(errors: List[Dict], fallback: str = "unknown") -> str:
    """
    Extract the primary component from error data.
    
    Args:
        errors: List of error dictionaries
        fallback: Fallback value if no primary component can be determined
        
    Returns:
        Primary component identifier
    """
    # Count components first
    component_counts = count_components(errors)
    
    # Filter out 'unknown' for primary component selection
    filtered_counts = {k: v for k, v in component_counts.items() if k != 'unknown'}
    
    # Apply special SOA vs Android logic
    if 'soa' in filtered_counts and 'android' in filtered_counts:
        # If SOA errors are at least 50% of Android errors, prefer SOA as primary
        if filtered_counts['soa'] >= filtered_counts['android'] * 0.5:
            # Boost SOA weight for primary component selection
            filtered_counts['soa'] = filtered_counts['soa'] * 1.5
    
    # Select primary component - prefer non-unknown components
    if filtered_counts:
        return max(filtered_counts.items(), key=lambda x: x[1])[0]
    elif component_counts:
        return max(component_counts.items(), key=lambda x: x[1])[0]
    
    return fallback


def normalize_timestamps_in_dict(data: Any, 
                               preserve_component_info: bool = True, 
                               primary_issue_component: Optional[str] = None) -> Any:
    """
    Recursively normalize timestamp values in a dictionary while preserving component information.
    
    Args:
        data: Dictionary, list, or scalar value to normalize
        preserve_component_info: Flag to control component preservation
        primary_issue_component: Primary component for reference only
        
    Returns:
        Normalized data structure with preserved component information
    """
    # Base case: if data is None, return None
    if data is None:
        return None
    
    # Handle dictionary case with careful component preservation
    if isinstance(data, dict):
        # Extract component fields before modification
        component_fields = extract_component_fields(data) if preserve_component_info else {}
        
        # Create a copy to avoid modifying the original
        result = {}
        
        # Process all fields
        for key, value in data.items():
            # Normalize timestamps
            if key == 'timestamp' and isinstance(value, (str, datetime)):
                result[key] = ensure_datetime(value)
            else:
                # Recursively process nested structures
                result[key] = normalize_timestamps_in_dict(
                    value, 
                    preserve_component_info, 
                    primary_issue_component
                )
                
        # Reapply component fields
        if preserve_component_info and component_fields:
            result = apply_component_fields(result, component_fields)
            
        # Add primary_issue_component if provided and not present
        if primary_issue_component and preserve_component_info and 'primary_issue_component' not in result:
            result['primary_issue_component'] = primary_issue_component
            
        return result
        
    # Handle list case
    elif isinstance(data, list):
        return [
            normalize_timestamps_in_dict(item, preserve_component_info, primary_issue_component)
            for item in data
        ]
        
    # Return primitive types unchanged
    return data


def validate_component_fields(data: Any, 
                            primary_issue_component: Optional[str] = None, 
                            validate_depth: bool = True) -> Any:
    """
    Validate component fields in data structures without overriding existing values.
    
    Args:
        data: Dictionary, list, or scalar value to validate
        primary_issue_component: Primary component for reference only
        validate_depth: Whether to recursively validate nested structures
        
    Returns:
        Validated data with consistent component fields
    """
    # Base case: if data is None or not a dict/list, return as-is
    if data is None or not isinstance(data, (dict, list)):
        return data
    
    # Handle dictionary case
    if isinstance(data, dict):
        # Create a copy to avoid modifying the original
        result = copy.deepcopy(data)
        
        # RULE 1: Only infer source_component from component if source is missing
        if 'component' in result and result['component'] != 'unknown':
            if 'source_component' not in result or result['source_component'] == 'unknown':
                result['source_component'] = result['component']
                if 'component_source' not in result:
                    result['component_source'] = 'component_validation'
        
        # RULE 2: Only infer component from source_component if component is missing
        elif ('component' not in result or result['component'] == 'unknown') and 'source_component' in result and result['source_component'] != 'unknown':
            result['component'] = result['source_component']
            if 'component_source' not in result:
                result['component_source'] = 'derived_from_source'
        
        # RULE 3: Add primary_issue_component as reference if missing, but NEVER replace existing values
        if primary_issue_component and 'primary_issue_component' not in result:
            result['primary_issue_component'] = primary_issue_component
        
        # RULE 4: Apply file-based component inference ONLY if no component is set
        if ('component' not in result or result['component'] == 'unknown') and 'file' in result:
            file_name = result.get('file', '').lower()
            
            # Detect translator component
            if 'translator' in file_name:
                result['component'] = 'translator'
                result['component_source'] = 'filename'
            # Detect mimosa component
            elif 'mimosa' in file_name:
                result['component'] = 'mimosa'
                result['component_source'] = 'filename'
            # Detect phoebe component
            elif 'phoebe' in file_name:
                result['component'] = 'phoebe'
                result['component_source'] = 'filename'
            # Detect charles component
            elif 'charles' in file_name:
                result['component'] = 'charles'
                result['component_source'] = 'filename'
            # Content-based inference for app_debug.log ONLY if component not already set
            elif 'app_debug.log' in file_name:
                content = str(result.get('text', '')).lower()
                if any(marker in content for marker in ['bluetoothmanagerservice', 'activitymanager', 'packagemanager']):
                    result['component'] = 'android'
                    result['component_source'] = 'content_analysis'
                elif any(marker in content for marker in ['siriusxm', 'sxm', 'channel', 'playback']):
                    result['component'] = 'soa'
                    result['component_source'] = 'content_analysis'
                # Translator detection in app_debug
                elif any(marker in content for marker in ['translator', 'smite', 'command', 'response']):
                    result['component'] = 'translator'
                    result['component_source'] = 'content_analysis'
        
        # Recursively validate nested structures if needed
        if validate_depth:
            for k, v in result.items():
                if isinstance(v, (dict, list)) and k not in COMPONENT_FIELDS:
                    result[k] = validate_component_fields(v, primary_issue_component, validate_depth)
        
        return result
    
    # Handle list case
    elif isinstance(data, list):
        return [validate_component_fields(item, primary_issue_component, validate_depth) 
               for item in data]


def normalize_component_fields(data: Dict) -> Dict:
    """
    Normalize component fields to standard format.
    
    Args:
        data: Dictionary to normalize
        
    Returns:
        Dictionary with normalized component fields
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


def verify_component_preservation(original: Dict, processed: Dict) -> bool:
    """
    Verify that component information was preserved during processing.
    
    Args:
        original: Original dictionary
        processed: Processed dictionary
        
    Returns:
        True if component information was preserved, False otherwise
    """
    if not original or not processed:
        return True
    
    if isinstance(original, dict) and isinstance(processed, dict):
        # Check component fields
        for field in COMPONENT_FIELDS:
            if field in original and original[field] is not None:
                if field not in processed or processed[field] != original[field]:
                    logging.warning(f"Component field {field} not preserved: {original[field]} -> {processed.get(field, 'missing')}")
                    return False
    
    return True


def ensure_error_flags(errors: List[Dict]) -> List[Dict]:
    """
    Ensure all errors have required flags and attributes for visualization.
    
    Args:
        errors: List of error dictionaries
        
    Returns:
        List of errors with required flags
    """
    for err in errors:
        if isinstance(err, dict):
            # Ensure is_error flag
            err['is_error'] = True
            
            # Ensure severity has a valid value 
            if 'severity' not in err or not err['severity']:
                err['severity'] = 'Medium'
                
            # Ensure timestamp if missing (using current time as fallback)
            if 'timestamp' not in err or not err['timestamp']:
                err['timestamp'] = datetime.now()
    
    return errors


def preprocess_errors(errors: List[Dict], 
                    primary_issue_component: str, 
                    component_diagnostic: Optional[Dict] = None) -> Tuple[List[Dict], str]:
    """
    Preprocess errors to ensure component information is consistent without overriding.
    
    Args:
        errors: List of error dictionaries
        primary_issue_component: Primary issue component
        component_diagnostic: Optional diagnostic data
        
    Returns:
        Tuple of (processed errors, primary_issue_component)
    """
    # First, create deep copies to avoid modifying originals
    errors_copy = copy.deepcopy(errors)
    
    # Store original component information
    original_component_info = {}
    for i, error in enumerate(errors_copy):
        if isinstance(error, dict):
            original_component_info[i] = extract_component_fields(error)
    
    # Log initial component distribution
    initial_counts = count_components(errors_copy[:20])
    logging.info(f"Initial component distribution: {initial_counts}")
    
    # Use component_diagnostic data if available to enhance ONLY unknown components
    if component_diagnostic:
        logging.info("Using component diagnostic data to identify unknown components")
        
        # Extract file-to-component mapping
        file_component_map = {}
        if "files_by_component" in component_diagnostic:
            for component, files in component_diagnostic["files_by_component"].items():
                for file in files:
                    file_component_map[file] = component
        
        # Apply file-component mapping ONLY to entries without component information
        for i, err in enumerate(errors_copy):
            if isinstance(err, dict) and 'file' in err:
                # Only apply to entries without component information
                if 'component' not in err or err['component'] == 'unknown':
                    file_name = err['file']
                    if file_name in file_component_map:
                        errors_copy[i]['component'] = file_component_map[file_name]
                        errors_copy[i]['component_source'] = 'diagnostic'
    
    # Detect translator logs
    for i, err in enumerate(errors_copy):
        if isinstance(err, dict):
            # Check if it's a translator log but hasn't been identified yet
            if ('component' not in err or err['component'] == 'unknown'):
                # Check file name for translator indicators
                if 'file' in err and any(marker in err['file'].lower() for marker in ['translator', 'smite']):
                    errors_copy[i]['component'] = 'translator'
                    errors_copy[i]['component_source'] = 'filename'
                # Check log content for translator indicators
                elif 'text' in err:
                    text = str(err['text']).lower()
                    if any(marker in text for marker in ['"type":"command"', '"type":"response"', 'autosmitetranslator']):
                        errors_copy[i]['component'] = 'translator'
                        errors_copy[i]['component_source'] = 'content_analysis'
    
    # Check if we need component mapping for remaining unknown components
    unknown_count = sum(1 for err in errors_copy if isinstance(err, dict) 
                      and ('component' not in err or err['component'] == 'unknown'))
    
    if unknown_count > 0:
        # Use direct component mapping only for entries without components
        try:
            logging.info(f"Found {unknown_count} entries without component information, applying targeted mapping")
            # Create a list with just the unknown component errors
            unknown_errors = [err for err in errors_copy if isinstance(err, dict) 
                           and ('component' not in err or err['component'] == 'unknown')]
            
            # Apply component relationships only to unknown errors
            from components.direct_component_analyzer import assign_components_and_relationships
            mapped_unknown_errors, _, _ = assign_components_and_relationships(unknown_errors)
            
            # Merge the mapped unknown errors back into the main list
            mapped_index = 0
            for i, err in enumerate(errors_copy):
                if isinstance(err, dict) and ('component' not in err or err['component'] == 'unknown'):
                    if mapped_index < len(mapped_unknown_errors):
                        # Only copy the component fields, not the entire error
                        for field in COMPONENT_FIELDS:
                            if field in mapped_unknown_errors[mapped_index]:
                                errors_copy[i][field] = mapped_unknown_errors[mapped_index][field]
                        mapped_index += 1
        except Exception as e:
            logging.error(f"Error applying targeted component mapping: {str(e)}")
    
    # Only calculate if primary_issue_component is truly "unknown"
    if primary_issue_component == "unknown":
        logging.info("Primary component not provided, will derive it from error distribution")
        derived_primary = extract_primary_component(errors_copy)
        if derived_primary != "unknown":
            primary_issue_component = derived_primary
            logging.info(f"Derived primary component from error distribution: {primary_issue_component}")
    else:
        # Log that we're using the provided primary component
        logging.info(f"Using provided primary component: {primary_issue_component}")
    
    # Validate errors with gentle consistency checks - no overriding
    validated_errors = []
    for i, err in enumerate(errors_copy):
        if isinstance(err, dict):
            # Use validate_component_fields with primary_issue_component as reference only
            validated_err = validate_component_fields(err, primary_issue_component)
            
            # Ensure original component fields are preserved
            if i in original_component_info:
                for field, value in original_component_info[i].items():
                    # Only apply original fields that weren't properly preserved
                    # and aren't "unknown"
                    if (field not in validated_err or validated_err[field] != value) and value != "unknown":
                        validated_err[field] = value
            
            # Normalize component fields (lowercase, etc.)
            validated_err = normalize_component_fields(validated_err)
            
            validated_errors.append(validated_err)
        else:
            validated_errors.append(err)
    
    # Apply ensure_error_flags to add required flags for visualization
    processed_errors = ensure_error_flags(validated_errors)
    
    # Final component distribution check
    final_counts = count_components(processed_errors[:20])
    source_counts = count_component_sources(processed_errors[:20])
    logging.info(f"Component distribution after preprocessing: {final_counts}")
    logging.info(f"Component source distribution after preprocessing: {source_counts}")
    
    # Verify component preservation
    preservation_success = True
    for i, error in enumerate(processed_errors[:5]):
        if i in original_component_info and isinstance(error, dict):
            if not verify_component_preservation(original_component_info[i], error):
                preservation_success = False
                break
    
    if not preservation_success:
        logging.warning("Component information was not fully preserved during preprocessing")
    else:
        logging.info("Component information was successfully preserved during preprocessing")
    
    return processed_errors, primary_issue_component


def preprocess_clusters(clusters: Dict[int, List[Dict]], 
                       primary_issue_component: str,
                       component_diagnostic: Optional[Dict] = None) -> Dict[int, List[Dict]]:
    """
    Preprocess clusters with consistent component handling.
    
    Args:
        clusters: Dictionary mapping cluster IDs to lists of errors
        primary_issue_component: Primary issue component
        component_diagnostic: Optional diagnostic data
        
    Returns:
        Processed clusters
    """
    if not clusters:
        return {}
    
    # Create deep copy to avoid modifying original
    clusters_copy = copy.deepcopy(clusters)
    
    # Process each cluster
    for cluster_id, errors in clusters_copy.items():
        # Process each error in the cluster
        for i, error in enumerate(errors):
            # Validate and normalize component fields
            error = validate_component_fields(error, primary_issue_component)
            
            # Normalize component fields (lowercase, etc.)
            error = normalize_component_fields(error)
            
            # Ensure primary_issue_component is set
            if primary_issue_component and 'primary_issue_component' not in error:
                error['primary_issue_component'] = primary_issue_component
            
            # Update error in the list
            errors[i] = error
    
    return clusters_copy


def normalize_data(errors: List[Dict], 
                  clusters: Dict[int, List[Dict]], 
                  primary_issue_component: str) -> Tuple[List[Dict], Dict[int, List[Dict]]]:
    """
    Normalize and validate data for reports.
    
    Args:
        errors: List of error dictionaries
        clusters: Dictionary mapping cluster IDs to lists of errors
        primary_issue_component: Primary issue component
        
    Returns:
        Tuple of (normalized errors, normalized clusters)
    """
    # Store original component information
    original_error_components = {}
    for i, error in enumerate(errors):
        if isinstance(error, dict):
            original_error_components[i] = extract_component_fields(error)
    
    # Store original cluster component information
    original_cluster_components = {}
    for cluster_id, cluster_errors in clusters.items():
        original_cluster_components[cluster_id] = {}
        for i, error in enumerate(cluster_errors):
            if isinstance(error, dict):
                original_cluster_components[cluster_id][i] = extract_component_fields(error)
    
    # Normalize timestamps and preserve component information in errors
    normalized_errors = [
        normalize_timestamps_in_dict(error, True, primary_issue_component)
        for error in errors
    ]
    
    # Verify component preservation in errors
    for i, error in enumerate(normalized_errors):
        if i in original_error_components and isinstance(error, dict):
            if not verify_component_preservation(original_error_components[i], error):
                # Apply original component fields if not preserved
                for field, value in original_error_components[i].items():
                    if field not in error or error[field] != value:
                        error[field] = value
    
    # Normalize timestamps and preserve component information in clusters
    normalized_clusters = {}
    for cluster_id, cluster_errors in clusters.items():
        normalized_clusters[cluster_id] = [
            normalize_timestamps_in_dict(error, True, primary_issue_component)
            for error in cluster_errors
        ]
        
        # Verify component preservation in cluster errors
        if cluster_id in original_cluster_components:
            for i, error in enumerate(normalized_clusters[cluster_id]):
                if i in original_cluster_components[cluster_id] and isinstance(error, dict):
                    if not verify_component_preservation(original_cluster_components[cluster_id][i], error):
                        # Apply original component fields if not preserved
                        for field, value in original_cluster_components[cluster_id][i].items():
                            if field not in error or error[field] != value:
                                error[field] = value
    
    return normalized_errors, normalized_clusters