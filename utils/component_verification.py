"""
Component verification utilities for debugging component information preservation.

This module provides utility functions to verify and diagnose issues with
component information preservation throughout the processing pipeline.
It helps identify where component information might be lost or changed.
"""

import logging
import json
import os
from typing import Dict, List, Any, Tuple, Optional, Set, Union

# Try to import COMPONENT_FIELDS from reports.base
try:
    from reports.base import COMPONENT_FIELDS
except ImportError:
    # Fallback definition if import fails
    COMPONENT_FIELDS = {
        'component', 'component_source', 'source_component', 'root_cause_component',
        'primary_issue_component', 'affected_components', 'expected_component',
        'component_scores', 'component_distribution', 'parent_component', 'child_components',
        'related_components'
    }


def verify_component_preservation(source: Any, target: Any, path: str = "") -> bool:
    """
    Verify component information is preserved between two objects.
    
    Args:
        source: Source object (dict, list, or primitive)
        target: Target object (dict, list, or primitive)
        path: Current path for nested objects (for logging)
        
    Returns:
        Boolean indicating whether component information is preserved
    """
    # Handle None values
    if source is None and target is None:
        return True
    if source is None or target is None:
        if path:
            logging.warning(f"Mismatch at {path}: One object is None")
        return False
        
    # Handle non-dict cases
    if not isinstance(source, dict) or not isinstance(target, dict):
        if isinstance(source, list) and isinstance(target, list):
            # Compare lists - check length first
            if len(source) != len(target):
                if path:
                    logging.warning(f"List length mismatch at {path}: {len(source)} != {len(target)}")
                return False
                
            # Compare list items
            for i in range(min(len(source), len(target))):
                item_path = f"{path}[{i}]" if path else f"[{i}]"
                if not verify_component_preservation(source[i], target[i], item_path):
                    return False
            return True
        
        # For non-dict and non-list objects, return True
        # Since they can't contain component fields
        return True
        
    # Check component fields
    preservation_status = True
    for field in COMPONENT_FIELDS:
        if field in source and field in target:
            # Skip unknown values - these are defaults and might be overridden
            if source[field] == "unknown" and target[field] != "unknown":
                continue
                
            if source[field] != target[field]:
                field_path = f"{path}.{field}" if path else field
                logging.warning(f"Component field '{field_path}' not preserved: '{source[field]}' -> '{target[field]}'")
                preservation_status = False
        elif field in source and source[field] not in (None, "", "unknown") and field not in target:
            field_path = f"{path}.{field}" if path else field
            logging.warning(f"Component field '{field_path}' was lost: '{source[field]}' -> missing")
            preservation_status = False
                
    # Check nested structures
    for key in source:
        if key in target:
            next_path = f"{path}.{key}" if path else key
            
            if isinstance(source[key], dict) and isinstance(target[key], dict):
                if not verify_component_preservation(source[key], target[key], next_path):
                    preservation_status = False
            elif isinstance(source[key], list) and isinstance(target[key], list):
                for i in range(min(len(source[key]), len(target[key]))):
                    item_path = f"{next_path}[{i}]"
                    src_item = source[key][i] if i < len(source[key]) else None
                    tgt_item = target[key][i] if i < len(target[key]) else None
                    
                    if isinstance(src_item, dict) and isinstance(tgt_item, dict):
                        if not verify_component_preservation(src_item, tgt_item, item_path):
                            preservation_status = False
                            
    return preservation_status


def verify_component_preservation_in_file(source_path: str, target_path: str) -> bool:
    """
    Verify component information is preserved between two JSON files.
    
    Args:
        source_path: Path to source JSON file
        target_path: Path to target JSON file
        
    Returns:
        Boolean indicating whether component information is preserved
    """
    try:
        # Check if files exist
        if not os.path.exists(source_path):
            logging.error(f"Source file does not exist: {source_path}")
            return False
        if not os.path.exists(target_path):
            logging.error(f"Target file does not exist: {target_path}")
            return False
            
        # Load JSON files
        with open(source_path, 'r', encoding='utf-8') as sf:
            source_data = json.load(sf)
        with open(target_path, 'r', encoding='utf-8') as tf:
            target_data = json.load(tf)
            
        # Verify component preservation
        return verify_component_preservation(source_data, target_data)
    except Exception as e:
        logging.error(f"Error verifying component preservation in files: {str(e)}")
        return False


def verify_component_fields_in_list(source_list: List[Dict], target_list: List[Dict]) -> bool:
    """
    Checks component fields in lists of objects.
    
    Args:
        source_list: Source list of dictionaries
        target_list: Target list of dictionaries
        
    Returns:
        Boolean success status
    """
    if not source_list or not target_list:
        return True  # Empty lists are considered preserved
        
    if len(source_list) != len(target_list):
        logging.warning(f"List length mismatch: {len(source_list)} != {len(target_list)}")
        
    # Verify items up to the minimum length of both lists
    min_length = min(len(source_list), len(target_list))
    preservation_status = True
    
    for i in range(min_length):
        source_item = source_list[i]
        target_item = target_list[i]
        
        if not verify_component_preservation(source_item, target_item, f"list[{i}]"):
            preservation_status = False
            
    return preservation_status


def verify_component_fields_in_clusters(source_clusters: Dict[int, List[Dict]], 
                                       target_clusters: Dict[int, List[Dict]]) -> bool:
    """
    Validates component information in error clusters.
    
    Args:
        source_clusters: Source clusters dictionary
        target_clusters: Target clusters dictionary
        
    Returns:
        Boolean success status
    """
    if not source_clusters or not target_clusters:
        return True  # Empty clusters are considered preserved
        
    # Check cluster keys match
    if set(source_clusters.keys()) != set(target_clusters.keys()):
        logging.warning(f"Cluster keys mismatch: {set(source_clusters.keys())} != {set(target_clusters.keys())}")
        # Continue checking matching clusters
        
    preservation_status = True
    
    # Check each cluster that exists in both
    common_clusters = set(source_clusters.keys()) & set(target_clusters.keys())
    for cluster_id in common_clusters:
        source_errors = source_clusters[cluster_id]
        target_errors = target_clusters[cluster_id]
        
        if not verify_component_fields_in_list(source_errors, target_errors):
            logging.warning(f"Component fields not preserved in cluster {cluster_id}")
            preservation_status = False
            
    return preservation_status


def count_component_fields(data: Any) -> Dict[str, int]:
    """
    Counts occurrences of component fields in data.
    
    Args:
        data: Data to analyze (dict, list, or other)
        
    Returns:
        Dictionary of field counts
    """
    counts = {field: 0 for field in COMPONENT_FIELDS}
    
    def _count_in_object(obj: Any):
        if isinstance(obj, dict):
            # Count component fields in this dict
            for field in COMPONENT_FIELDS:
                if field in obj and obj[field] not in (None, "", "unknown"):
                    counts[field] += 1
                    
            # Recursively count in nested structures
            for key, value in obj.items():
                _count_in_object(value)
        elif isinstance(obj, list):
            # Recursively count in list items
            for item in obj:
                _count_in_object(item)
    
    # Start the recursive counting
    _count_in_object(data)
    
    return counts


def generate_component_diagnostic(file_path: str) -> Dict[str, Any]:
    """
    Creates diagnostic information about component fields in a JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Dictionary with diagnostic data
    """
    result = {
        "success": False,
        "file_path": file_path,
        "component_counts": {},
        "component_source_counts": {},
        "primary_issue_component": "unknown",
        "components_by_source": {},
        "error_count": 0
    }
    
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            logging.error(f"File does not exist: {file_path}")
            result["error"] = "File does not exist"
            return result
            
        # Load JSON file
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Count component fields
        result["component_counts"] = count_component_fields(data)
        
        # Extract primary_issue_component
        if isinstance(data, dict) and "primary_issue_component" in data:
            result["primary_issue_component"] = data["primary_issue_component"]
            
        # Analyze errors if present
        if isinstance(data, dict) and "errors" in data and isinstance(data["errors"], list):
            result["error_count"] = len(data["errors"])
            
            # Count components and sources
            component_counts = {}
            source_counts = {}
            components_by_source = {}
            
            for error in data["errors"]:
                if isinstance(error, dict):
                    # Count component
                    component = error.get("component", "unknown")
                    component_counts[component] = component_counts.get(component, 0) + 1
                    
                    # Count component_source
                    source = error.get("component_source", "unknown")
                    source_counts[source] = source_counts.get(source, 0) + 1
                    
                    # Group components by source
                    if source not in components_by_source:
                        components_by_source[source] = {}
                    components_by_source[source][component] = components_by_source[source].get(component, 0) + 1
            
            result["component_counts_in_errors"] = component_counts
            result["component_source_counts"] = source_counts
            result["components_by_source"] = components_by_source
            
        result["success"] = True
            
    except Exception as e:
        logging.error(f"Error generating component diagnostic: {str(e)}")
        result["error"] = str(e)
        
    return result


def trace_component_changes(obj: Dict, before_op: str, after_op: str) -> Dict[str, Any]:
    """
    Create a snapshot of component fields before/after an operation for comparison.
    
    Args:
        obj: Object to trace changes on
        before_op: Description of operation (before)
        after_op: Description of operation (after)
        
    Returns:
        Dictionary with component field values
    """
    if not isinstance(obj, dict):
        return {}
        
    trace = {
        "timestamp": __import__('datetime').datetime.now().isoformat(),
        "operation": f"{before_op} -> {after_op}",
        "fields": {}
    }
    
    # Extract component fields
    for field in COMPONENT_FIELDS:
        if field in obj:
            trace["fields"][field] = obj[field]
            
    return trace


def audit_component_changes(before_snapshot: Dict, after_snapshot: Dict) -> Dict[str, Any]:
    """
    Compare component field snapshots to detect changes.
    
    Args:
        before_snapshot: Snapshot before operation
        after_snapshot: Snapshot after operation
        
    Returns:
        Dictionary with detected changes
    """
    if not before_snapshot or not after_snapshot:
        return {}
        
    changes = {
        "detected": False,
        "changes": {}
    }
    
    # Compare fields
    for field in COMPONENT_FIELDS:
        before_value = before_snapshot.get("fields", {}).get(field)
        after_value = after_snapshot.get("fields", {}).get(field)
        
        if before_value is not None and after_value is not None and before_value != after_value:
            # Skip unknown -> specific value transitions (this is expected enhancement)
            if before_value == "unknown" and after_value != "unknown":
                continue
                
            changes["detected"] = True
            changes["changes"][field] = {
                "before": before_value,
                "after": after_value
            }
            
    # Add operation information
    if changes["detected"]:
        changes["operation"] = before_snapshot.get("operation")
        
    return changes


def component_info_summary(data: Any) -> Dict[str, Any]:
    """
    Generate a summary of component information in data.
    
    Args:
        data: Data to analyze
        
    Returns:
        Summary dictionary
    """
    summary = {
        "component_fields_present": {},
        "primary_component": "unknown",
        "component_distribution": {},
        "component_sources": set(),
        "field_consistency": {},
        "problematic_fields": []
    }
    
    # Check if primary_issue_component is directly available
    if isinstance(data, dict) and "primary_issue_component" in data:
        summary["primary_component"] = data["primary_issue_component"]
    
    # Track component field presence and values
    for field in COMPONENT_FIELDS:
        summary["field_consistency"][field] = {"consistent": True, "values": set()}
    
    def _analyze_object(obj: Any):
        """Recursively analyze objects for component information."""
        if isinstance(obj, dict):
            # Check component fields
            for field in COMPONENT_FIELDS:
                if field in obj and obj[field] not in (None, ""):
                    # Track field presence
                    summary["component_fields_present"][field] = summary["component_fields_present"].get(field, 0) + 1
                    
                    # Track consistency
                    value = obj[field]
                    if isinstance(value, (list, dict, set)):
                        # Skip complex values for consistency check
                        pass
                    else:
                        summary["field_consistency"][field]["values"].add(value)
                    
            # Track component distribution
            if "component" in obj and obj["component"] not in (None, ""):
                component = obj["component"]
                summary["component_distribution"][component] = summary["component_distribution"].get(component, 0) + 1
                
            # Track component sources
            if "component_source" in obj and obj["component_source"] not in (None, ""):
                summary["component_sources"].add(obj["component_source"])
                
            # Recursively analyze nested objects
            for value in obj.values():
                _analyze_object(value)
                
        elif isinstance(obj, list):
            # Recursively analyze list items
            for item in obj:
                _analyze_object(item)
    
    # Start recursive analysis
    _analyze_object(data)
    
    # Check consistency
    for field, info in summary["field_consistency"].items():
        # If more than 1 value (excluding unknown), mark as inconsistent
        values = set(v for v in info["values"] if v != "unknown")
        if len(values) > 1:
            info["consistent"] = False
            summary["problematic_fields"].append(field)
    
    return summary


def export_diagnostic_report(data: Any, output_path: str, primary_issue_component: str = "unknown") -> str:
    """
    Export a detailed diagnostic report for component information.
    
    Args:
        data: Data to analyze
        output_path: Path to save the report
        primary_issue_component: Primary issue component for context
        
    Returns:
        Path to the exported report
    """
    # Create diagnostic data
    diagnostic = {
        "timestamp": __import__('datetime').datetime.now().isoformat(),
        "primary_issue_component": primary_issue_component,
        "summary": component_info_summary(data),
        "component_field_counts": count_component_fields(data)
    }
    
    # Add sample data for specific key structures
    samples = {}
    
    def _collect_samples(obj: Any, path: str = ""):
        """Collect samples of objects with component fields."""
        if isinstance(obj, dict):
            # Check if this dict has component fields
            has_component_fields = any(field in obj for field in COMPONENT_FIELDS)
            if has_component_fields:
                # Add to samples if not too many already
                if len(samples) < 20:
                    samples[path or "root"] = {
                        field: obj.get(field) 
                        for field in COMPONENT_FIELDS 
                        if field in obj
                    }
            
            # Recursively collect from nested structures
            for key, value in obj.items():
                next_path = f"{path}.{key}" if path else key
                _collect_samples(value, next_path)
                
        elif isinstance(obj, list):
            # Skip lists longer than 20 items
            if len(obj) <= 20:
                for i, item in enumerate(obj):
                    item_path = f"{path}[{i}]" if path else f"[{i}]"
                    _collect_samples(item, item_path)
    
    # Collect samples
    _collect_samples(data)
    diagnostic["samples"] = samples
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Use custom encoder for datetime objects
    class DateTimeEncoder(json.JSONEncoder):
        def default(self, obj):
            if hasattr(obj, 'isoformat'):
                return obj.isoformat()
            return super().default(obj)
    
    # Save to file
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(diagnostic, f, cls=DateTimeEncoder, indent=2)
        return output_path
    except Exception as e:
        logging.error(f"Error exporting diagnostic report: {str(e)}")
        return ""
