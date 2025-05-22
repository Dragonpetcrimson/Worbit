#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Timeline Visualization Fix Script

This script diagnoses and fixes issues with the timeline visualization in the Orbit Analyzer.
It checks required modules, configuration, directories, and performs a fix for timeline visualization
generation and HTML report integration.
"""

import os
import sys
import re
import json
import logging
import datetime
import traceback
import glob
import importlib
import inspect
from typing import Dict, List, Any, Optional, Tuple
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Global variables
TEST_ID = "SXM-2302295"
LOG_DIR = "./logs"
OUTPUT_DIR = "./output"
FEATURE_FILE_PATH = None

def log_system_info():
    """Log information about the system environment."""
    logging.info("Starting timeline visualization diagnostic script")
    logging.info(f"Python version: {sys.version}")
    logging.info(f"Current working directory: {os.getcwd()}")
    logging.info(f"System encoding: {sys.getdefaultencoding()}")
    logging.info(f"File system encoding: {sys.getfilesystemencoding()}")

def find_feature_file(test_id: str) -> Optional[str]:
    """
    Find feature file for the given test ID.
    
    Args:
        test_id: The test ID to search for
        
    Returns:
        Path to the feature file if found, None otherwise
    """
    logging.info(f"Searching for feature files for test {test_id}...")
    
    # Try in test-specific log directory
    dir_path = os.path.join(LOG_DIR, test_id)
    logging.debug(f"Searching in {dir_path}")
    feature_files = glob.glob(os.path.join(dir_path, "*.feature"))
    feature_files += glob.glob(os.path.join(dir_path, f"*{test_id}*.feature"))
    
    # Try in general log directory
    if not feature_files:
        logging.debug(f"Searching in {LOG_DIR}")
        feature_files = glob.glob(os.path.join(LOG_DIR, "*.feature"))
        feature_files += glob.glob(os.path.join(LOG_DIR, f"*{test_id}*.feature"))
    
    # Try in current directory
    if not feature_files:
        logging.debug(f"Searching in ./")
        feature_files = glob.glob(f"*{test_id}*.feature")
    
    if feature_files:
        logging.info(f"Found feature files with pattern '*{test_id}*.feature' in ./:")
        for file in feature_files:
            logging.info(f"  - {file}")
        logging.info(f"Using feature file: {feature_files[0]}")
        return feature_files[0]
    else:
        logging.warning(f"No feature files found for test ID {test_id}")
        return None

def check_required_modules():
    """Check if all required modules are available."""
    logging.info("=== CHECKING REQUIRED MODULES ===")
    
    required_modules = [
        ("matplotlib", "Visualization generation"),
        ("networkx", "Graph operations"),
        ("pandas", "Data manipulation"),
        ("PIL", "Image verification")
    ]
    
    for module_name, purpose in required_modules:
        try:
            module = importlib.import_module(module_name)
            if module_name == "matplotlib":
                logging.debug(f"matplotlib data path: {module.get_data_path()}")
                logging.debug(f"CONFIGDIR={module.get_configdir()}")
                logging.debug(f"interactive is {module.is_interactive()}")
                logging.debug(f"platform is {module.get_backend()}")
            logging.info(f"[OK] {module_name} - Available ({purpose})")
        except ImportError:
            logging.error(f"[MISSING] {module_name} - Not available ({purpose})")
            return False
    
    return True

def check_configuration():
    """Check configuration settings."""
    logging.info("=== CHECKING CONFIGURATION ===")
    
    try:
        from config import Config
        Config.setup_logging()
        
        logging.info("Config module imported successfully")
        logging.info(f"Config.ENABLE_STEP_REPORT_IMAGES = {Config.ENABLE_STEP_REPORT_IMAGES}")
        logging.info(f"Config.ENABLE_VISUALIZATION_PLACEHOLDERS = {Config.ENABLE_VISUALIZATION_PLACEHOLDERS}")
        logging.info(f"Config.LOG_BASE_DIR = {Config.LOG_BASE_DIR}")
        logging.info(f"Config.OUTPUT_BASE_DIR = {Config.OUTPUT_BASE_DIR}")
        logging.info(f"Config._logging_initialized = {Config._logging_initialized}")
        
        # Update globals based on Config
        global LOG_DIR, OUTPUT_DIR
        LOG_DIR = Config.LOG_BASE_DIR
        OUTPUT_DIR = Config.OUTPUT_BASE_DIR
        
        return True
    except Exception as e:
        logging.error(f"Error in configuration: {str(e)}")
        traceback.print_exc()
        return False

def check_directories_and_files(test_id: str):
    """Check if necessary directories and files exist."""
    logging.info("=== CHECKING DIRECTORIES AND FILES ===")
    
    # Check log directory
    log_dir = os.path.join(LOG_DIR, test_id)
    if os.path.exists(log_dir):
        logging.info(f"Log directory exists: {log_dir}")
        
        # Check log files
        log_files = [f for f in os.listdir(log_dir) if f.endswith('.log')]
        logging.info(f"Found {len(log_files)} log files in {log_dir}")
        for log_file in log_files:
            logging.info(f"  - {log_file}")
    else:
        logging.warning(f"Log directory does not exist: {log_dir}")
    
    # Check output directory
    output_dir = os.path.join(OUTPUT_DIR, test_id)
    if os.path.exists(output_dir):
        logging.info(f"Output directory exists: {output_dir}")
    else:
        logging.warning(f"Output directory does not exist: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Created output directory: {output_dir}")
    
    # Check supporting_images directory
    supporting_images_dir = os.path.join(output_dir, "supporting_images")
    if os.path.exists(supporting_images_dir):
        logging.info(f"Supporting images directory exists: {supporting_images_dir}")
    else:
        os.makedirs(supporting_images_dir, exist_ok=True)
        logging.info(f"Created supporting_images directory: {supporting_images_dir}")
    
    # Check for HTML report
    html_report_path = os.path.join(output_dir, f"{test_id}_step_report.html")
    if os.path.exists(html_report_path):
        logging.info(f"HTML report exists: {html_report_path}")
    else:
        logging.warning(f"HTML report does not exist: {html_report_path}")
    
    return True

def check_visualization_module():
    """Check visualization module functionality."""
    logging.info("=== CHECKING VISUALIZATION MODULE ===")
    
    try:
        from reports import visualizations
        
        # Log timestamp to verify module import
        now = datetime.datetime.now().isoformat()
        logging.info(f"Reports module initialized successfully at {now}")
        
        # Check for relevant functions
        logging.info("Successfully imported visualization functions")
        
        # Inspect generate_timeline_image function
        timeline_sig = str(inspect.signature(visualizations.generate_timeline_image))
        logging.info(f"generate_timeline_image signature: {timeline_sig}")
        
        # Inspect generate_cluster_timeline_image function
        cluster_timeline_sig = str(inspect.signature(visualizations.generate_cluster_timeline_image))
        logging.info(f"generate_cluster_timeline_image signature: {cluster_timeline_sig}")
        
        # Inspect generate_visualization_placeholder function
        placeholder_sig = str(inspect.signature(visualizations.generate_visualization_placeholder))
        logging.info(f"generate_visualization_placeholder signature: {placeholder_sig}")
        
        # Get function source code
        timeline_source = inspect.getsource(visualizations.generate_timeline_image)
        logging.info(f"generate_timeline_image source code length: {len(timeline_source)}")
        
        return True
    except Exception as e:
        logging.error(f"Error checking visualization module: {str(e)}")
        traceback.print_exc()
        return False

def check_step_aware_analyzer():
    """Check step-aware analyzer module."""
    logging.info("=== CHECKING STEP-AWARE ANALYZER MODULE ===")
    
    try:
        import step_aware_analyzer
        
        logging.info("Successfully imported step-aware analyzer functions")
        
        # Inspect generate_step_report function
        step_report_sig = str(inspect.signature(step_aware_analyzer.generate_step_report))
        logging.info(f"generate_step_report signature: {step_report_sig}")
        
        # Inspect run_step_aware_analysis function
        step_analysis_sig = str(inspect.signature(step_aware_analyzer.run_step_aware_analysis))
        logging.info(f"run_step_aware_analysis signature: {step_analysis_sig}")
        
        # Check if generate_step_report calls generate_timeline_image
        source = inspect.getsource(step_aware_analyzer.generate_step_report)
        if "generate_timeline_image" in source:
            logging.info("generate_step_report calls generate_timeline_image")
        else:
            logging.warning("generate_step_report does not call generate_timeline_image")
        
        # Check if generate_step_report calls generate_cluster_timeline_image
        if "generate_cluster_timeline_image" in source:
            logging.info("generate_step_report calls generate_cluster_timeline_image")
        else:
            logging.warning("generate_step_report does not call generate_cluster_timeline_image")
        
        return True
    except Exception as e:
        logging.error(f"Error checking step-aware analyzer: {str(e)}")
        traceback.print_exc()
        return False

def check_gherkin_log_correlator():
    """Check gherkin log correlator module."""
    logging.info("=== CHECKING GHERKIN LOG CORRELATOR ===")
    
    try:
        import gherkin_log_correlator
        
        logging.info("Successfully imported Gherkin log correlator functions")
        
        # Inspect correlate_logs_with_steps function
        correlate_sig = str(inspect.signature(gherkin_log_correlator.correlate_logs_with_steps))
        logging.info(f"correlate_logs_with_steps signature: {correlate_sig}")
        
        return True
    except Exception as e:
        logging.error(f"Error checking Gherkin log correlator: {str(e)}")
        traceback.print_exc()
        return False

def check_path_utilities():
    """Check path utilities module."""
    logging.info("=== CHECKING PATH UTILITIES ===")
    
    try:
        from utils import path_utils
        
        logging.info("Successfully imported path utilities")
        
        # Inspect get_output_path function
        output_path_sig = str(inspect.signature(path_utils.get_output_path))
        logging.info(f"get_output_path signature: {output_path_sig}")
        
        # Inspect get_standardized_filename function
        filename_sig = str(inspect.signature(path_utils.get_standardized_filename))
        logging.info(f"get_standardized_filename signature: {filename_sig}")
        
        # Check OutputType enum
        logging.info(f"OutputType enum values: {list(path_utils.OutputType)}")
        
        return True
    except Exception as e:
        logging.error(f"Error checking path utilities: {str(e)}")
        traceback.print_exc()
        return False

def extract_step_data_from_html(test_id: str) -> Tuple[Dict[int, Dict], Dict[int, List]]:
    """
    Extract step data from HTML report for the given test ID.
    
    Args:
        test_id: The test ID to search for
        
    Returns:
        Tuple of (step_dict, step_to_logs)
    """
    logging.info("Extracting step data from HTML report...")
    
    html_report_path = os.path.join(OUTPUT_DIR, test_id, f"{test_id}_step_report.html")
    if not os.path.exists(html_report_path):
        logging.warning(f"HTML report not found: {html_report_path}")
        return {}, {}
    
    logging.info(f"Reading HTML report from: {html_report_path}")
    
    with open(html_report_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Extract step data from HTML using regex
    # Updated pattern to match the actual HTML structure
    step_pattern = r'<div class="step">\s*<div class="step-header">\s*<h3>Step (\d+): (.*?)</h3>'
    step_matches = re.findall(step_pattern, html_content)
    
    step_dict = {}
    for step_num, step_text in step_matches:
        step_num = int(step_num)
        step_dict[step_num] = {
            'step_number': step_num,
            'text': step_text,
            'step_name': step_text  # Adding this for compatibility with visualization functions
        }
    
    logging.info(f"Extracted data for {len(step_dict)} steps from HTML report")
    
    # Create mock step_to_logs
    # In a real scenario, this would be extracted from the HTML or re-generated
    step_to_logs = {step_num: [] for step_num in step_dict}
    
    return step_dict, step_to_logs

def create_fixed_timeline(test_id: str, step_dict: Dict[int, Dict], step_to_logs: Dict[int, List]):
    """
    Create a fixed timeline visualization.
    
    Args:
        test_id: The test ID
        step_dict: Dictionary of step data
        step_to_logs: Dictionary mapping step numbers to logs
    
    Returns:
        Path to the generated visualization, or None if failed
    """
    logging.info("=== CREATING TIMELINE VISUALIZATION ===")
    
    try:
        # Import required modules - include os import to avoid UnboundLocalError
        import os  # Import os here to fix the UnboundLocalError
        from reports.visualizations import generate_timeline_image
        from utils.path_utils import OutputType, get_output_path, get_standardized_filename
        
        # Define output directory
        test_output_dir = os.path.join(OUTPUT_DIR, test_id)
        logging.info(f"Test output directory: {test_output_dir}")
        
        # Create supporting_images directory
        supporting_images_dir = os.path.join(test_output_dir, "supporting_images")
        os.makedirs(supporting_images_dir, exist_ok=True)
        logging.info(f"Supporting images directory: {supporting_images_dir}")
        
        # Define expected output path for timeline visualization
        expected_filename = get_standardized_filename(test_id, "timeline", "png")
        expected_path = os.path.join(supporting_images_dir, expected_filename)
        logging.info(f"Expected timeline visualization path: {expected_path}")
        
        # Generate timeline visualization
        logging.info("Calling generate_timeline_image...")
        timeline_path = generate_timeline_image(
            step_to_logs=step_to_logs,
            step_dict=step_dict,
            output_dir=test_output_dir,
            test_id=test_id
        )
        
        if timeline_path and os.path.exists(timeline_path):
            logging.info(f"Timeline visualization generated successfully: {timeline_path}")
            logging.info(f"File size: {os.path.getsize(timeline_path)} bytes")
            return timeline_path
        else:
            if timeline_path:
                logging.error(f"Timeline path returned but file does not exist: {timeline_path}")
            else:
                logging.error("Timeline generation function returned None or empty path")
            return None
    except Exception as e:
        logging.error(f"Error creating timeline visualization: {str(e)}")
        traceback.print_exc()
        return None

def update_html_report(test_id: str, timeline_path: str):
    """
    Update HTML report to include timeline visualization.
    
    Args:
        test_id: The test ID
        timeline_path: Path to timeline visualization
    
    Returns:
        True if successful, False otherwise
    """
    logging.info("=== UPDATING HTML REPORT ===")
    
    try:
        html_report_path = os.path.join(OUTPUT_DIR, test_id, f"{test_id}_step_report.html")
        logging.info(f"HTML report path: {html_report_path}")
        
        if not os.path.exists(html_report_path):
            logging.warning(f"HTML report not found: {html_report_path}")
            return False
        
        # Create backup of original HTML
        backup_path = f"{html_report_path}.bak"
        import shutil
        shutil.copy2(html_report_path, backup_path)
        logging.info(f"Created backup of HTML report: {backup_path}")
        
        # Read HTML file
        with open(html_report_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Check if timeline section exists with updated pattern to match the entire section
        timeline_section_pattern = r'<div id="timeline-section"[^>]*>.*?</div>'
        match = re.search(timeline_section_pattern, html_content, re.DOTALL)
        
        if match:
            logging.info(f"Found timeline section")
            
            # Get relative path to image
            rel_timeline_path = os.path.relpath(timeline_path, os.path.dirname(html_report_path))
            rel_timeline_path = rel_timeline_path.replace('\\', '/')  # Ensure forward slashes
            logging.info(f"Relative path to timeline image: {rel_timeline_path}")
            
            # Replace entire timeline section with new content including the image
            updated_timeline_section = f'<div id="timeline-section">\n    <h2>Test Execution Timeline</h2>\n    <img src="{rel_timeline_path}" alt="Test Execution Timeline" class="timeline-image" />\n</div>'
            updated_html = re.sub(timeline_section_pattern, updated_timeline_section, html_content, flags=re.DOTALL)
            
            # Write updated HTML
            with open(html_report_path, 'w', encoding='utf-8') as f:
                f.write(updated_html)
            
            logging.info(f"HTML report updated with timeline visualization")
            return True
        else:
            logging.warning("Timeline section not found in HTML report")
            
            # Try to find the section by just looking for the heading
            heading_pattern = r'<h2>Test Execution Timeline</h2>'
            match = re.search(heading_pattern, html_content)
            
            if match:
                logging.info("Found Test Execution Timeline heading, but not in expected pattern")
                # Check surrounding context
                start_idx = max(0, match.start() - 100)
                end_idx = min(len(html_content), match.end() + 100)
                context = html_content[start_idx:end_idx]
                logging.info(f"Context: {context}")
                
                # Try to inject the timeline image after the heading
                rel_timeline_path = os.path.relpath(timeline_path, os.path.dirname(html_report_path))
                rel_timeline_path = rel_timeline_path.replace('\\', '/')  # Ensure forward slashes
                img_tag = f'<img src="{rel_timeline_path}" alt="Test Execution Timeline" class="timeline-image" />'
                
                updated_html = html_content.replace(match.group(0), match.group(0) + '\n    ' + img_tag)
                
                # Write updated HTML
                with open(html_report_path, 'w', encoding='utf-8') as f:
                    f.write(updated_html)
                
                logging.info(f"HTML report updated with timeline visualization by inserting after heading")
                return True
            
            # If all else fails, try to append the timeline section at the end of the <body>
            body_end_pattern = r'</body>'
            match = re.search(body_end_pattern, html_content)
            
            if match:
                logging.info("Injecting timeline section before </body>")
                
                rel_timeline_path = os.path.relpath(timeline_path, os.path.dirname(html_report_path))
                rel_timeline_path = rel_timeline_path.replace('\\', '/')  # Ensure forward slashes
                
                timeline_section = f'\n<div id="timeline-section">\n    <h2>Test Execution Timeline</h2>\n    <img src="{rel_timeline_path}" alt="Test Execution Timeline" class="timeline-image" />\n</div>\n'
                
                updated_html = html_content.replace(match.group(0), timeline_section + match.group(0))
                
                # Write updated HTML
                with open(html_report_path, 'w', encoding='utf-8') as f:
                    f.write(updated_html)
                
                logging.info(f"HTML report updated with timeline visualization by adding before </body>")
                return True
                
            return False
    except Exception as e:
        logging.error(f"Error updating HTML report: {str(e)}")
        traceback.print_exc()
        return False

def run_diagnostic():
    """Run diagnostic and fix for timeline visualization."""
    log_system_info()
    
    # Get test ID from command line
    parser = argparse.ArgumentParser(description='Timeline visualization fix for Orbit Analyzer')
    parser.add_argument('--test-id', type=str, default="SXM-2302295", 
                        help='Test ID to fix (default: SXM-2302295)')
    args = parser.parse_args()
    
    global TEST_ID
    TEST_ID = args.test_id
    
    logging.info(f"=== STARTING TIMELINE VISUALIZATION FIX FOR {TEST_ID} ===")
    logging.info(f"Log directory: {LOG_DIR}")
    logging.info(f"Output directory: {OUTPUT_DIR}")
    
    # Find feature file
    global FEATURE_FILE_PATH
    FEATURE_FILE_PATH = find_feature_file(TEST_ID)
    
    # Run checks
    checks = {
        "required_modules": check_required_modules(),
        "configuration": check_configuration(),
        "directories_and_files": check_directories_and_files(TEST_ID),
        "visualization_module": check_visualization_module(),
        "step_aware_analyzer": check_step_aware_analyzer(),
        "gherkin_log_correlator": check_gherkin_log_correlator(),
        "path_utilities": check_path_utilities()
    }
    
    # Extract step data
    step_dict, step_to_logs = extract_step_data_from_html(TEST_ID)
    
    # Create fixed timeline visualization
    timeline_path = create_fixed_timeline(TEST_ID, step_dict, step_to_logs)
    checks["timeline_generation"] = timeline_path is not None
    
    # Update HTML report
    html_updated = False
    if timeline_path:
        html_updated = update_html_report(TEST_ID, timeline_path)
    checks["html_report_update"] = html_updated
    
    # Print diagnostic summary
    logging.info("\n=== DIAGNOSTIC SUMMARY ===")
    for check, result in checks.items():
        status = "OK" if result else "FAILED"
        logging.info(f"{check.replace('_', ' ').title()}: {status}")
    
    # Print fix implementation status
    logging.info("\n=== FIX IMPLEMENTATION STATUS ===")
    if not checks["timeline_generation"]:
        logging.error("FAILED: Could not create timeline visualization.")
    elif not checks["html_report_update"]:
        logging.error("FAILED: Could not update HTML report.")
    else:
        logging.info("SUCCESS: Timeline visualization created and HTML report updated.")
    
    if not checks["timeline_generation"] or not checks["html_report_update"]:
        logging.info("Check the logs for detailed error information.")
    
    # Print final output locations
    if timeline_path:
        logging.info(f"\nTimeline visualization: {timeline_path}")
    
    html_report_path = os.path.join(OUTPUT_DIR, TEST_ID, f"{TEST_ID}_step_report.html")
    if os.path.exists(html_report_path):
        logging.info(f"HTML report: {html_report_path}")
    
    logging.info("\nFix script completed. Check logs for details.")

if __name__ == "__main__":
    run_diagnostic()