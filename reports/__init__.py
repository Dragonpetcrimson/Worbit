"""
reports/__init__.py - Package initialization and main entry point for report generation
"""

import sys
import os
import logging
import copy
from datetime import datetime
from typing import Dict, List, Optional, Any

# Import required classes and functions
from reports.base import ReportConfig, ReportData
from reports.report_manager import ReportManager
from reports.component_analyzer import generate_component_report

# Import visualization functions directly
from reports.visualizations import generate_cluster_timeline_image, generate_timeline_image

# Import component utilities for consistent handling
from utils.path_utils import normalize_test_id, sanitize_base_directory
from utils.path_validator import fix_directory_structure

# Create dummy modules for backward compatibility
class DummyModule:
    def __getattr__(self, name):
        # If the missing attribute is one of our timeline functions,
        # return the real implementation from visualizations.py
        if name == 'generate_timeline_image':
            from reports.visualizations import generate_timeline_image
            return generate_timeline_image
        elif name == 'generate_cluster_timeline_image':
            from reports.visualizations import generate_cluster_timeline_image
            return generate_cluster_timeline_image
        return None

# Register dummy modules to prevent import errors
sys.modules['timeline_image_generator'] = DummyModule()
sys.modules['cluster_timeline_generator'] = DummyModule()
sys.modules['reports.timeline_image_generator'] = DummyModule()
sys.modules['reports.cluster_timeline_generator'] = DummyModule()

# Log successful initialization
logging.info(f"Reports module initialized successfully at {datetime.now().isoformat()}")

def write_reports(
    output_dir: str,
    test_id: str,
    summary: str,
    errors: list,
    ocr_data: list,
    clusters: dict,
    ymir_flag: bool = False,
    background_text: str = "",
    scenario_text: str = "",
    component_analysis: dict = None,
    primary_issue_component: str = "unknown",
    component_report_path: str = None,
    component_diagnostic: dict = None,
    enable_step_report: Optional[bool] = None,
    enable_component_html: Optional[bool] = None
) -> dict:
    """
    Write comprehensive reports with enhanced component information preservation.
    
    Args:
        output_dir: Directory to write reports to
        test_id: Test ID for the report
        summary: AI-generated summary
        errors: List of error dictionaries
        ocr_data: List of OCR data dictionaries
        clusters: Dictionary mapping cluster IDs to lists of errors
        ymir_flag: Whether this is a Ymir test
        background_text: Background section from feature file
        scenario_text: Scenario section from feature file
        component_analysis: Results from component relationship analysis
        primary_issue_component: Primary component identified as causing issues
        component_report_path: Path to component report file
        component_diagnostic: Additional diagnostic information for components
        
    Returns:
        Dictionary with report information
    """
    # Normalize test_id
    test_id = normalize_test_id(test_id)
    
    # Normalize primary_issue_component
    if primary_issue_component:
        primary_issue_component = primary_issue_component.lower()
    
    # Log component information for debugging
    component_distribution = {}
    for error in errors[:20]:  # Sample for logging
        component = error.get('component', 'unknown')
        if component not in component_distribution:
            component_distribution[component] = 0
        component_distribution[component] += 1
    
    logging.info(f"Component distribution (sample): {component_distribution}")
    logging.info(f"Primary issue component: {primary_issue_component}")
    
    # Determine optional feature flags
    if enable_step_report is None:
        env_flag = os.getenv("ENABLE_STEP_REPORT")
        enable_step_report = str(env_flag).lower() in ("true", "1", "yes") if env_flag is not None else True

    if enable_component_html is None:
        env_flag = os.getenv("ENABLE_COMPONENT_HTML")
        enable_component_html = str(env_flag).lower() in ("true", "1", "yes") if env_flag is not None else True

    # Create config
    config = ReportConfig(
        output_dir=output_dir,
        test_id=test_id,
        primary_issue_component=primary_issue_component,
        enable_excel=True,
        enable_markdown=True,
        enable_json=True,
        enable_docx=True,
        enable_component_report=True,
        enable_step_report=enable_step_report,
        enable_component_html=enable_component_html
    )
    
    # Create data container
    data = ReportData(
        errors=errors,
        summary=summary,
        clusters=clusters,
        ocr_data=ocr_data,
        background_text=background_text,
        scenario_text=scenario_text,
        ymir_flag=ymir_flag,
        component_analysis=component_analysis,
        component_diagnostic=component_diagnostic
    )
    
    # Create report manager
    manager = ReportManager(config)
    
    # Generate reports
    results = manager.generate_reports(data)
    
    # Verify and fix directory structure
    issues = fix_directory_structure(output_dir, test_id)
    
    # Log issues and fixes
    if issues.get("fixed_files"):
        logging.info(f"Fixed {len(issues.get('fixed_files', []))} files with directory structure issues")
    
    return results