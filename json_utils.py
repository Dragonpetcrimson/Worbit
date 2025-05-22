"""
JSON utilities with component preservation support.

This module provides utilities for serializing and deserializing JSON data
while preserving component information. It handles proper usage of 
ComponentAwareEncoder to ensure component fields are maintained throughout
the serialization process.
"""

import json
import os
import logging
from typing import Any, Dict, Union, Optional, IO, TextIO


def serialize_with_component_awareness(
    data: Any, 
    file_obj: TextIO, 
    primary_issue_component: Optional[str] = None, 
    indent: int = 2
) -> None:
    """
    Serialize data with component awareness.
    
    This helper function correctly handles ComponentAwareEncoder instantiation.
    
    Args:
        data: Data to serialize
        file_obj: File object to write to
        primary_issue_component: Primary issue component for reference
        indent: JSON indentation level
        
    Returns:
        None (data is written to file_obj)
    
    Example:
        with open('output.json', 'w') as f:
            serialize_with_component_awareness(data, f, primary_issue_component="soa")
    """
    from reports.base import ComponentAwareEncoder
    
    # Use a lambda to create a class factory that returns our encoder instance
    encoder_factory = lambda *args, **kwargs: ComponentAwareEncoder(
        *args, primary_issue_component=primary_issue_component, **kwargs)
    
    # Dump JSON with our encoder factory
    return json.dump(data, file_obj, cls=encoder_factory, indent=indent)


def parse_with_component_awareness(file_obj_or_string: Union[TextIO, str]) -> Any:
    """
    Parse JSON data with component awareness.
    
    Args:
        file_obj_or_string: File object or JSON string to parse
        
    Returns:
        Parsed JSON data
    """
    # Handle file objects vs strings
    if hasattr(file_obj_or_string, 'read'):
        # It's a file-like object
        return json.load(file_obj_or_string)
    else:
        # It's a string
        return json.loads(file_obj_or_string)


def serialize_to_json_file(
    data: Any, 
    file_path: str, 
    primary_issue_component: Optional[str] = None, 
    indent: int = 2
) -> str:
    """
    Serialize data to a JSON file with component awareness.
    
    Args:
        data: Data to serialize
        file_path: Path to the output file
        primary_issue_component: Primary issue component for reference
        indent: JSON indentation level
        
    Returns:
        Path to the output file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Write with component awareness
    with open(file_path, 'w', encoding='utf-8') as f:
        serialize_with_component_awareness(
            data, 
            f, 
            primary_issue_component=primary_issue_component,
            indent=indent
        )
    
    # Verify component information
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
            
        # Check for component fields if data is a dictionary
        if isinstance(data, dict) and isinstance(loaded_data, dict):
            if 'component' in data and data['component'] != loaded_data.get('component'):
                logging.warning(
                    f"Component field 'component' not preserved: '{data['component']}' -> "
                    f"'{loaded_data.get('component', 'missing')}'"
                )
            if 'primary_issue_component' in data and data['primary_issue_component'] != loaded_data.get('primary_issue_component'):
                logging.warning(
                    f"Component field 'primary_issue_component' not preserved: '{data['primary_issue_component']}' -> "
                    f"'{loaded_data.get('primary_issue_component', 'missing')}'"
                )
    except Exception as e:
        logging.warning(f"Could not verify component preservation: {str(e)}")
    
    return file_path


def verify_component_preservation(
    source_data: Any, 
    loaded_data: Any,
    component_fields: Optional[set] = None
) -> bool:
    """
    Verify component information is preserved after serialization.
    
    Args:
        source_data: Original data before serialization
        loaded_data: Data after serialization and deserialization
        component_fields: Set of component fields to check (default: None)
        
    Returns:
        True if component information is preserved, False otherwise
    """
    if component_fields is None:
        # Import here to avoid circular imports
        try:
            from reports.base import COMPONENT_FIELDS
            component_fields = COMPONENT_FIELDS
        except ImportError:
            # Fallback if cannot import
            component_fields = {
                'component', 'component_source', 'source_component', 'root_cause_component',
                'primary_issue_component', 'affected_components', 'expected_component',
                'component_scores', 'component_distribution', 'parent_component', 
                'child_components', 'related_components'
            }
    
    # Helper function for recursive checking
    def check_components(src, tgt, path=""):
        if isinstance(src, dict) and isinstance(tgt, dict):
            # Check component fields
            for field in component_fields:
                if field in src and src[field] not in (None, '', 'unknown'):
                    if field not in tgt:
                        logging.warning(f"{path}.{field} missing in target")
                        return False
                    if src[field] != tgt[field]:
                        logging.warning(
                            f"{path}.{field} changed: '{src[field]}' -> '{tgt[field]}'"
                        )
                        return False
            
            # Check all keys recursively
            for key in src:
                if key in tgt:
                    if not check_components(src[key], tgt[key], f"{path}.{key}"):
                        return False
        
        elif isinstance(src, list) and isinstance(tgt, list):
            # Check lists item by item if lengths match
            if len(src) == len(tgt):
                for i, (src_item, tgt_item) in enumerate(zip(src, tgt)):
                    if not check_components(src_item, tgt_item, f"{path}[{i}]"):
                        return False
        
        return True
    
    # Start the recursive check
    return check_components(source_data, loaded_data)


def load_json_file(file_path: str) -> Any:
    """
    Load JSON data from a file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Parsed JSON data
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def dump_with_component_preservation(
    data: Any, 
    primary_issue_component: Optional[str] = None, 
    indent: int = 2
) -> str:
    """
    Dump data to a JSON string with component awareness.
    
    Args:
        data: Data to serialize
        primary_issue_component: Primary issue component for reference
        indent: JSON indentation level
        
    Returns:
        JSON string
    """
    from reports.base import ComponentAwareEncoder
    
    # Use a lambda to create a class factory that returns our encoder instance
    encoder_factory = lambda *args, **kwargs: ComponentAwareEncoder(
        *args, primary_issue_component=primary_issue_component, **kwargs)
    
    # Dump JSON with our encoder factory
    return json.dumps(data, cls=encoder_factory, indent=indent)
