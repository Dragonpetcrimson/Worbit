"""
reports/json_generator.py - JSON report generation
"""

import os
import json
import logging
import copy
from typing import Dict, List, Any, Optional

from reports.base import ComponentAwareEncoder, ReportGenerator, ReportConfig, ReportData, COMPONENT_FIELDS
from utils.path_utils import (
    get_output_path,
    OutputType,
    normalize_test_id,
    get_standardized_filename,
    sanitize_base_directory
)

class JsonReportGenerator(ReportGenerator):
    """Generator for JSON reports."""
    
    def generate(self, data: ReportData) -> str:
        """
        Generate a JSON report.
        
        Args:
            data: Report data
            
        Returns:
            Path to the generated report
        """
        # Create JSON data structure
        json_data = {
            "test_id": self.config.test_id,
            "summary": data.summary,
            "errors": data.errors,
            "ocr": data.ocr_data,
            "clusters": {str(k): v for k, v in data.clusters.items()},
            "background": data.background_text,
            "scenario": data.scenario_text,
            "ymir": data.ymir_flag,
            "primary_issue_component": self.config.primary_issue_component,
            "component_analysis": data.component_analysis
        }
        
        # Calculate component distribution for JSON output
        component_distribution = {}
        for err in data.errors:
            if isinstance(err, dict) and 'component' in err:
                comp = err.get('component')
                if comp != 'unknown':
                    component_distribution[comp] = component_distribution.get(comp, 0) + 1
        
        json_data["component_distribution"] = component_distribution
        
        # Use utilities for filename and path
        filename = get_standardized_filename(self.config.test_id, "log_analysis", "json")
        return self.write_json_report(json_data, filename)
    
    def write_json_report(self, data: Dict, filename: str) -> str:
        """
        Write JSON report with component-preserving encoding.
        
        Args:
            data: Data to serialize
            filename: Output filename
            
        Returns:
            Path to the written file
        """
        # Sanitize the output directory to prevent nested directories
        output_dir = sanitize_base_directory(self.config.output_dir, "json")
        
        # Use path utilities to get proper path
        output_path = get_output_path(
            output_dir,
            self.config.test_id,
            filename,
            OutputType.JSON_DATA
        )
        
        try:
            # Create a copy of the data to avoid modifying the original
            json_data = copy.deepcopy(data)
            
            # Only set the top-level primary_issue_component if not present
            if "primary_issue_component" not in json_data or not json_data["primary_issue_component"]:
                json_data["primary_issue_component"] = self.config.primary_issue_component
            
            # Process any image data to ensure they're in the supporting_images directory
            self._process_image_references(json_data)
            
            # Check component field counts before serialization
            component_field_counts_before = self._count_component_fields(json_data)
            logging.debug(f"Component field counts before serialization: {component_field_counts_before}")
            
            # Write JSON with component-aware encoder
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, 
                        cls=lambda *args, **kwargs: ComponentAwareEncoder(
                            *args, primary_issue_component=self.config.primary_issue_component, **kwargs), 
                        indent=2)
            
            # Verification - load data back and check component fields
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    loaded_data = json.load(f)
                    
                # Check component fields are preserved
                component_field_counts_after = self._count_component_fields(loaded_data)
                
                # Log counts for verification
                logging.info(f"Component field counts before serialization: {component_field_counts_before}")
                logging.info(f"Component field counts after serialization: {component_field_counts_after}")
                
                # Check for any field count differences
                if not all(component_field_counts_before.get(field, 0) == component_field_counts_after.get(field, 0) 
                       for field in COMPONENT_FIELDS):
                    logging.warning("Component field counts changed during serialization")
                    
                # Extract and log component counts in errors from the serialized data
                if "errors" in loaded_data and isinstance(loaded_data["errors"], list):
                    component_counts = {}
                    for err in loaded_data["errors"][:20]:
                        if isinstance(err, dict) and "component" in err:
                            comp = err["component"]
                            component_counts[comp] = component_counts.get(comp, 0) + 1
                    
                    if component_counts:
                        logging.info(f"Component distribution in serialized JSON: {component_counts}")
            except Exception as e:
                logging.warning(f"Could not verify serialized JSON: {str(e)}")
            
            logging.info(f"Successfully wrote JSON report to {output_path}")
            return output_path
        except Exception as e:
            logging.error(f"Error writing JSON report: {str(e)}")
            return ""
    
    def _process_image_references(self, data: Any) -> None:
        """
        Process data to ensure any image references point to the supporting_images directory.
        
        Args:
            data: Data structure to process
        """
        if isinstance(data, dict):
            # Process image fields in dictionaries
            for key, value in list(data.items()):
                # Check if this is an image field
                if isinstance(key, str) and ("image" in key.lower() or "path" in key.lower()) and isinstance(value, str):
                    # If it contains an image file extension, ensure it points to supporting_images
                    if any(ext in value.lower() for ext in ['.png', '.jpg', '.jpeg', '.gif', '.svg']):
                        # Extract the filename
                        base_filename = os.path.basename(value)
                        # Update the path to point to supporting_images
                        data[key] = os.path.join("supporting_images", base_filename)
                # Recursively process nested structures
                elif isinstance(value, (dict, list)):
                    self._process_image_references(value)
        
        elif isinstance(data, list):
            # Process lists recursively
            for item in data:
                if isinstance(item, (dict, list)):
                    self._process_image_references(item)
    
    def _count_component_fields(self, data):
        """Count component fields in data structure."""
        counts = {field: 0 for field in COMPONENT_FIELDS}
        
        def count_in_dict(d):
            if not isinstance(d, dict):
                return
                
            for field in COMPONENT_FIELDS:
                if field in d and d[field] is not None:
                    counts[field] += 1
                    
            for value in d.values():
                if isinstance(value, dict):
                    count_in_dict(value)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            count_in_dict(item)
                            
        count_in_dict(data)
        return counts