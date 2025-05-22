"""
reports/base.py - Common utilities and base classes for report generation
"""

import os
import re
import json
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, date, time
import copy

# Define component-related fields globally to ensure consistency
COMPONENT_FIELDS = {
    'component', 'component_source', 'source_component', 'root_cause_component',
    'primary_issue_component', 'affected_components', 'expected_component',
    'component_scores', 'component_distribution', 'parent_component', 'child_components',
    'related_components'  # Added to track component relationships
}

class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles datetime objects."""
    def default(self, obj):
        if isinstance(obj, (datetime, date, time)):
            return obj.isoformat()
        return super().default(obj)


class ComponentAwareEncoder(DateTimeEncoder):
    """
    Enhanced JSON encoder that carefully preserves component information during serialization.
    This encoder ensures that component fields retain their original values without overriding
    and properly handles nested structures to prevent component information loss.
    
    Usage:
        # INCORRECT - Do not instantiate before passing to json.dump()
        encoder = ComponentAwareEncoder(primary_issue_component="soa")
        json.dump(data, file, cls=encoder)  # This will fail
        
        # CORRECT - Option 1: Use the helper function
        from json_utils import serialize_with_component_awareness
        serialize_with_component_awareness(data, file, primary_issue_component="soa")
        
        # CORRECT - Option 2: Create a lambda factory
        json.dump(data, file, 
                cls=lambda *a, **kw: ComponentAwareEncoder(primary_issue_component="soa"),
                indent=2)
    """
    
    def __init__(self, *args, primary_issue_component=None, **kwargs):
        """
        Initialize encoder with optional primary_issue_component reference.
        
        Args:
            primary_issue_component: Primary component for reference only
            *args, **kwargs: Standard encoder parameters
        """
        super().__init__(*args, **kwargs)
        self.primary_issue_component = primary_issue_component
        self.component_fields = COMPONENT_FIELDS
        # Added tracking for component field transformations
        self.field_transformations = {}
    
    def default(self, obj):
        """
        Enhanced encoding that preserves component information without modification.
        
        Args:
            obj: Object to encode
            
        Returns:
            JSON-serializable representation with preserved component information
        """
        # First handle datetime objects with parent class
        if isinstance(obj, (datetime, date, time)):
            return super().default(obj)
        
        # For objects with a to_dict method, convert to dict and process
        if hasattr(obj, 'to_dict') and callable(getattr(obj, 'to_dict')):
            return self.default(obj.to_dict())
        
        # Special handling for dictionaries with component fields
        if isinstance(obj, dict):
            # Store original component information for validation
            original_component_info = self._extract_component_info(obj)
            
            # Create a deep copy to avoid modifying the original
            result = copy.deepcopy(obj)
            
            # Process the dictionary with component preservation
            self._preserve_component_fields(result)
            
            # Verify component preservation and track transformations
            if original_component_info:
                processed_component_info = self._extract_component_info(result)
                if not self.validate_component_preservation(original_component_info, processed_component_info):
                    logging.warning(f"Component information not fully preserved during serialization")
                    # Track transformations for debugging
                    for field in self.component_fields:
                        if field in original_component_info and field in processed_component_info:
                            if original_component_info[field] != processed_component_info[field]:
                                self.field_transformations[field] = {
                                    'original': original_component_info[field],
                                    'transformed': processed_component_info[field]
                                }
                                # Restore original value for critical fields
                                if field == 'component' and original_component_info[field] not in (None, '', 'unknown'):
                                    result[field] = original_component_info[field]
            
            return result
            
        # For lists, recursively process each item to ensure component preservation
        elif isinstance(obj, list):
            return [self.default(item) for item in obj]
            
        # Default behavior for other types
        return super().default(obj)
        
    def _preserve_component_fields(self, data_dict):
        """
        Carefully preserve component fields in a dictionary and all its nested structures.
        
        Args:
            data_dict: Dictionary containing component data to preserve
            
        Returns:
            Processed dictionary or None if data_dict was None
        """
        # Early return for None
        if data_dict is None:
            return None
            
        # Validate input
        if not isinstance(data_dict, dict):
            return data_dict
        
        # First, collect component information from this level
        component_info = self._extract_component_info(data_dict)
        
        # Store original component value for verification
        original_component = data_dict.get('component')
        
        # Apply primary_issue_component if specified and missing
        if self.primary_issue_component and 'primary_issue_component' not in component_info:
            component_info['primary_issue_component'] = self.primary_issue_component
        
        # Ensure consistency among component fields at this level
        component_info = self._ensure_component_consistency_internal(component_info)
        
        # Validate component values
        component_info = self._validate_component_values_internal(component_info)
        
        # Apply the validated component info back to the data dictionary
        self._apply_component_info(data_dict, component_info)
        
        # Verify component value wasn't changed
        if original_component is not None and original_component != 'unknown':
            current_component = data_dict.get('component')
            if current_component != original_component:
                logging.warning(
                    f"Component field 'component' not preserved: "
                    f"'{original_component}' -> '{current_component}'"
                )
                # Restore original value
                data_dict['component'] = original_component
        
        # Now process all nested dictionaries and lists
        self._process_nested_structures(data_dict, component_info)
        
        return data_dict
    
    def _extract_component_info(self, data_dict):
        """
        Extract all component-related information from a dictionary.
        
        Args:
            data_dict: Dictionary to extract from
            
        Returns:
            Dictionary containing only component-related fields
        """
        if data_dict is None:
            return {}
            
        result = {}
        for field in self.component_fields:
            if field in data_dict:
                result[field] = copy.deepcopy(data_dict[field])
        return result
    
    def _apply_component_info(self, data_dict, component_info):
        """
        Apply component information to a dictionary.
        
        Args:
            data_dict: Dictionary to apply component info to
            component_info: Component information to apply
            
        Returns:
            Updated dictionary or None if data_dict was None
        """
        # Early return for None inputs
        if data_dict is None:
            return None
            
        # Handle empty component_info
        if not component_info:
            return data_dict
            
        # Apply component fields
        for field, value in component_info.items():
            if value is not None:  # Only apply non-None values
                data_dict[field] = value
                
        return data_dict
    
    def _process_nested_structures(self, data_dict, parent_component_info):
        """
        Process nested dictionaries and lists, propagating component information.
        
        Args:
            data_dict: Dictionary containing nested structures
            parent_component_info: Component info from parent to propagate
            
        Returns:
            Processed dictionary or None if data_dict was None
        """
        # Handle None values
        if data_dict is None:
            return None
            
        for key, value in list(data_dict.items()):  # Use list() to allow modification during iteration
            if isinstance(value, dict):
                # Process nested dictionary
                self._process_nested_dict(value, parent_component_info)
            elif isinstance(value, list):
                # Process nested list
                data_dict[key] = self._process_nested_list(value, parent_component_info)
                
        return data_dict
    
    def _process_nested_dict(self, nested_dict, parent_component_info):
        """
        Process a nested dictionary, preserving and propagating component information.
        
        Args:
            nested_dict: Nested dictionary to process
            parent_component_info: Component info from parent to propagate
            
        Returns:
            Processed dictionary or None if nested_dict was None
        """
        # Early return for None inputs
        if nested_dict is None:
            return None
            
        # Store original component values for verification
        original_component = nested_dict.get('component')
        
        # Extract existing component info from this nested dictionary
        local_component_info = self._extract_component_info(nested_dict)
        
        # If this dict has no component info but parent does, consider propagation
        if not local_component_info and parent_component_info:
            # Only propagate component info if this dict has none
            # Do not override existing values!
            for field, value in parent_component_info.items():
                if field not in nested_dict and value is not None:
                    nested_dict[field] = value
        
        # Process nested structures within this dictionary
        self._process_nested_structures(nested_dict, local_component_info or parent_component_info)
        
        # Verify component value wasn't changed
        if original_component is not None and original_component != 'unknown':
            current_component = nested_dict.get('component')
            if current_component != original_component:
                logging.warning(
                    f"Nested component changed: "
                    f"'{original_component}' -> '{current_component}'"
                )
                # Restore original value
                nested_dict['component'] = original_component
        
        return nested_dict
    
    def _process_nested_list(self, nested_list, parent_component_info):
        """
        Process a nested list, handling any dictionaries within it.
        
        Args:
            nested_list: List to process
            parent_component_info: Component info from parent to propagate
            
        Returns:
            Processed list with preserved component information
        """
        # Handle None values
        if nested_list is None:
            return None
            
        result = []
        for item in nested_list:
            if isinstance(item, dict):
                # Create a copy to avoid modifying the original
                item_copy = copy.deepcopy(item)
                
                # Process the dictionary with parent component info
                self._process_nested_dict(item_copy, parent_component_info)
                result.append(item_copy)
            elif isinstance(item, list):
                # Recursively process nested lists
                result.append(self._process_nested_list(item, parent_component_info))
            else:
                # For non-dict, non-list items, just append directly
                result.append(item)
        return result
    
    def _ensure_component_consistency_internal(self, component_info):
        """
        Ensure consistency between related component fields.
        
        Args:
            component_info: Dictionary containing component data
            
        Returns:
            Updated component info with consistent fields
        """
        # Create a copy to avoid modifying the original
        result = component_info.copy()
        
        # Infer source_component from component if missing
        if 'component' in result and result['component'] not in (None, 'unknown', ''):
            if 'source_component' not in result:
                result['source_component'] = result['component']
                
        # Ensure component_source and source_component are consistent
        if 'source_component' in result and 'component_source' not in result:
            result['component_source'] = 'default'  # Default source if missing
        elif 'component_source' in result and 'source_component' not in result:
            result['source_component'] = result['component'] if 'component' in result else 'unknown'
            
        # Handle affected_components relationship with component
        if 'component' in result and 'affected_components' in result:
            # If affected_components is a list and component is not in it, add it
            if isinstance(result['affected_components'], list) and result['component'] not in result['affected_components']:
                if result['component'] not in (None, 'unknown', ''):
                    result['affected_components'].append(result['component'])
        
        # Ensure primary_issue_component is set if we have it
        if self.primary_issue_component and 'primary_issue_component' not in result:
            result['primary_issue_component'] = self.primary_issue_component
        
        # Ensure root_cause_component is consistent with primary_issue_component if not already set
        if 'primary_issue_component' in result and 'root_cause_component' not in result:
            result['root_cause_component'] = result['primary_issue_component']
        
        return result
    
    def _validate_component_values_internal(self, component_info):
        """
        Validate component field values to ensure they are properly formatted.
        
        Args:
            component_info: Dictionary containing component data
            
        Returns:
            Updated component info with validated values
        """
        # Create a copy to avoid modifying the original
        result = component_info.copy()
        
        # Ensure component values are lowercase for consistency
        for field in ['component', 'primary_issue_component', 'root_cause_component', 'source_component']:
            if field in result and isinstance(result[field], str):
                result[field] = result[field].lower()
        
        # Ensure component values are not None or empty strings
        for field in self.component_fields:
            if field in result and result[field] in (None, ''):
                result[field] = 'unknown'
                
        # Handle component_scores validation
        if 'component_scores' in result and isinstance(result['component_scores'], dict):
            # Create copy of component_scores to avoid modifying during iteration
            scores = copy.deepcopy(result['component_scores'])
            
            # Ensure all scores are numeric
            for component, score in scores.items():
                if not isinstance(score, (int, float)):
                    try:
                        scores[component] = float(score)
                    except (ValueError, TypeError):
                        scores[component] = 0.0
            
            result['component_scores'] = scores
                        
        # Handle component_distribution validation
        if 'component_distribution' in result and isinstance(result['component_distribution'], dict):
            # Create copy of component_distribution to avoid modifying during iteration
            distribution = copy.deepcopy(result['component_distribution'])
            
            # Ensure all distribution values are valid
            for component, value in distribution.items():
                if not isinstance(value, (int, float)):
                    try:
                        distribution[component] = float(value)
                    except (ValueError, TypeError):
                        distribution[component] = 0.0
            
            result['component_distribution'] = distribution
            
        # Ensure related_components is always a list
        if 'related_components' in result:
            if not isinstance(result['related_components'], list):
                if result['related_components'] is None:
                    result['related_components'] = []
                elif isinstance(result['related_components'], str):
                    result['related_components'] = [result['related_components']]
                else:
                    try:
                        result['related_components'] = list(result['related_components'])
                    except:
                        result['related_components'] = []
        
        return result

    def validate_component_preservation(self, original, processed):
        """
        Validate that component information is preserved.
        
        Args:
            original: Original dictionary with component information
            processed: Processed dictionary to validate
            
        Returns:
            True if component information is preserved, False otherwise
        """
        if not original or not processed:
            return True  # Nothing to validate
            
        preserved = True
        for field in self.component_fields:
            if field in original and original[field] not in (None, '', 'unknown'):
                if field not in processed:
                    logging.warning(f"Component field {field} missing in processed data")
                    preserved = False
                elif processed[field] != original[field]:
                    logging.warning(f"Component field {field} changed from '{original[field]}' to '{processed[field]}'")
                    preserved = False
                    
        return preserved


def apply_component_fields(data_dict, source_dict):
    """
    Apply component fields from source_dict to data_dict.
    
    Args:
        data_dict: Target dictionary to update (or None)
        source_dict: Source dictionary with component fields
        
    Returns:
        Updated data_dict or None if data_dict is None
    """
    # Early return for None inputs
    if data_dict is None:
        return None
        
    # Handle None source_dict
    if source_dict is None:
        return data_dict
        
    # Copy component fields from source to target
    for field in COMPONENT_FIELDS:
        if field in source_dict and source_dict[field] is not None:
            data_dict[field] = source_dict[field]
            
    return data_dict


def ensure_datetime(timestamp_value):
    """
    Ensure a timestamp is a datetime object.
    
    Args:
        timestamp_value: A timestamp which could be a string or datetime object
        
    Returns:
        datetime object or None if conversion fails
    """
    if timestamp_value is None or timestamp_value == "No timestamp":
        return None
        
    # If it's already a datetime object, return it
    if isinstance(timestamp_value, datetime):
        return timestamp_value
        
    # If it's a string, try to convert it
    if isinstance(timestamp_value, str):
        # Try standard ISO format first
        try:
            return datetime.fromisoformat(timestamp_value)
        except ValueError:
            pass
            
        # Try common formats
        formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%a %b %d %H:%M:%S %Y",
            "%d/%m/%Y %H:%M:%S",
            "%m/%d/%Y %H:%M:%S"
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(timestamp_value, fmt)
            except ValueError:
                continue
                
        # Try to extract timestamp using regex
        # Look for patterns like HH:MM:SS or HH:MM:SS.microseconds
        match = re.search(r'(\d{2}:\d{2}:\d{2}(?:\.\d+)?)', timestamp_value)
        if match:
            time_str = match.group(1)
            try:
                # Assume today's date with the extracted time
                today = datetime.now().strftime("%Y-%m-%d")
                return datetime.strptime(f"{today} {time_str}", "%Y-%m-%d %H:%M:%S.%f" if "." in time_str else "%Y-%m-%d %H:%M:%S")
            except ValueError:
                pass
    
    # For any other type or if all conversions fail
    logging.warning(f"Could not convert timestamp: {timestamp_value} ({type(timestamp_value)})")
    return None


def sanitize_text(value):
    """Sanitize a string value for safe output in reports."""
    if isinstance(value, str):
        # Replace control characters
        sanitized = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', value)
        
        # More aggressive sanitization for specific problematic strings
        if "cannot be used in worksheets" in sanitized:
            sanitized = sanitized.replace("cannot be used in worksheets", "[filtered text]")
        
        # Remove any other potentially problematic characters
        sanitized = ''.join(c for c in sanitized if c.isprintable() or c in ['\n', '\r', '\t'])
        
        return sanitized
    return value


class ReportConfig:
    """Configuration settings for report generation."""
    
    def __init__(self, 
                output_dir: str, 
                test_id: str, 
                primary_issue_component: str = "unknown",
                enable_excel: bool = True,
                enable_markdown: bool = True,
                enable_json: bool = True,
                enable_docx: bool = True,
                enable_component_report: bool = True,
                enable_step_report: bool = True,
                enable_component_html: bool = True):
        """
        Initialize report configuration.
        
        Args:
            output_dir: Directory to write reports to
            test_id: Test ID for reports
            primary_issue_component: Primary component for issue
            enable_excel: Whether to generate Excel reports
            enable_markdown: Whether to generate Markdown reports
            enable_json: Whether to generate JSON reports
            enable_docx: Whether to generate DOCX reports
            enable_component_report: Whether to generate component visualizations
            enable_step_report: Whether to generate the step-aware HTML report
            enable_component_html: Whether to generate the component analysis HTML report
        """
        self.output_dir = output_dir
        self.test_id = test_id
        # Normalize primary_issue_component to lowercase for consistency
        self.primary_issue_component = primary_issue_component.lower() if primary_issue_component else "unknown"
        self.enable_excel = enable_excel
        self.enable_markdown = enable_markdown
        self.enable_json = enable_json
        self.enable_docx = enable_docx
        self.enable_component_report = enable_component_report
        self.enable_step_report = enable_step_report
        self.enable_component_html = enable_component_html
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)


class ReportData:
    """Container for data used in report generation."""
    
    def __init__(self,
                errors: List[Dict],
                summary: str,
                clusters: Dict[int, List[Dict]],
                ocr_data: List[Dict] = None,
                background_text: str = "",
                scenario_text: str = "",
                ymir_flag: bool = False,
                component_analysis: Dict[str, Any] = None,
                component_diagnostic: Dict[str, Any] = None):
        """
        Initialize report data.
        
        Args:
            errors: List of error dictionaries
            summary: AI-generated summary
            clusters: Dictionary mapping cluster IDs to lists of errors
            ocr_data: List of OCR data dictionaries
            background_text: Background section from feature file
            scenario_text: Scenario section from feature file
            ymir_flag: Whether this is a Ymir test
            component_analysis: Results from component relationship analysis
            component_diagnostic: Additional diagnostic information for components
        """
        self.errors = errors
        self.summary = summary
        self.clusters = clusters
        self.ocr_data = ocr_data or []
        self.background_text = background_text
        self.scenario_text = scenario_text
        self.ymir_flag = ymir_flag
        self.component_analysis = component_analysis
        self.component_diagnostic = component_diagnostic


class ReportGenerator:
    """Base class for report generators."""
    
    def __init__(self, config: ReportConfig):
        """
        Initialize a report generator.
        
        Args:
            config: Report configuration
        """
        self.config = config
    
    def generate(self, data: ReportData) -> str:
        """
        Generate a report.
        
        Args:
            data: Report data
            
        Returns:
            Path to the generated report
        """
        raise NotImplementedError("Subclasses must implement generate()")