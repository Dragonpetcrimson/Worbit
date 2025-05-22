"""
utils/path_utils.py - Centralized path handling utilities for Orbit

This module provides standardized path handling functions to ensure consistent
file organization across the application. It has been enhanced to prevent
nested directory creation and provide cleanup capabilities.
"""

import os
import logging
import traceback
import shutil
from enum import Enum
from typing import Dict, Optional, List

# Try to import Config; use defaults if not available
try:
    from config import Config
except ImportError:
    # Mock Config for standalone use
    class Config:
        LOG_BASE_DIR = "./logs"
        OUTPUT_BASE_DIR = "./output"

class OutputType(Enum):
    """Enumeration of output file types with their destinations"""
    PRIMARY_REPORT = "primary"  # Goes in root directory (Excel, DOCX, HTML)
    JSON_DATA = "json"          # Goes in json/ subdirectory
    VISUALIZATION = "image"     # Goes in supporting_images/ subdirectory
    DEBUGGING = "debug"         # Goes in debug/ subdirectory (optional)

def normalize_test_id(test_id: str) -> str:
    """
    Normalize test ID to standard SXM-#### format
    
    Args:
        test_id: Input test ID
    
    Returns:
        Normalized test ID with SXM- prefix
    """
    if not test_id:
        return ""
    test_id = test_id.strip()
    return test_id if test_id.upper().startswith("SXM-") else f"SXM-{test_id}"

def sanitize_base_directory(base_dir: str, expected_subdir: str = None) -> str:
    """
    Sanitize the base directory to prevent nested subdirectories.
    
    Args:
        base_dir: Base directory path to sanitize
        expected_subdir: Optional subdirectory to check for
        
    Returns:
        Sanitized base directory path
    """
    if not base_dir or not isinstance(base_dir, str):
        return base_dir
        
    # Normalize path separators
    norm_path = os.path.normpath(base_dir)
    parts = norm_path.replace('\\', '/').split('/')
    
    # Check if last part is a known subdirectory
    known_subdirs = ["json", "supporting_images", "debug"]
    
    if expected_subdir and parts and parts[-1] == expected_subdir:
        # Directory already contains the expected subdirectory
        logging.info(f"Sanitizing path: '{base_dir}' contains '{expected_subdir}' subdirectory")
        # Use parent directory
        return os.path.dirname(norm_path)
        
    if parts and parts[-1] in known_subdirs:
        # Using a known subdirectory directly
        logging.info(f"Sanitizing path: '{base_dir}' is a known subdirectory")
        # Use parent directory
        return os.path.dirname(norm_path)
        
    return base_dir

def get_output_path(
    base_dir: str, 
    test_id: str, 
    filename: str, 
    output_type: OutputType = OutputType.PRIMARY_REPORT,
    create_dirs: bool = True
) -> str:
    """
    Get standardized output path based on file type.
    
    This function now includes validation to detect and prevent directory nesting.
    
    Args:
        base_dir: Base output directory
        test_id: Test ID (will be normalized)
        filename: Filename to use
        output_type: Type of output determining subdirectory
        create_dirs: Whether to create directories if they don't exist
        
    Returns:
        Full path for the output file
    """
    test_id = normalize_test_id(test_id)
    
    # Sanitize the base directory to prevent nested directories
    expected_subdir = None
    if output_type == OutputType.JSON_DATA:
        expected_subdir = "json"
    elif output_type == OutputType.VISUALIZATION:
        expected_subdir = "supporting_images"
    elif output_type == OutputType.DEBUGGING:
        expected_subdir = "debug"
        
    base_dir = sanitize_base_directory(base_dir, expected_subdir)
    
    # Determine the output directory based on type
    if output_type == OutputType.JSON_DATA:
        output_dir = os.path.join(base_dir, "json")
    elif output_type == OutputType.VISUALIZATION:
        output_dir = os.path.join(base_dir, "supporting_images")
    elif output_type == OutputType.DEBUGGING:
        output_dir = os.path.join(base_dir, "debug")
    else:  # PRIMARY_REPORT - root directory
        output_dir = base_dir
        
    # Create directories if needed
    if create_dirs:
        try:
            os.makedirs(output_dir, exist_ok=True)
            logging.debug(f"Ensured directory exists: {output_dir}")
        except Exception as e:
            logging.error(f"Failed to create directory {output_dir}: {str(e)}")
            traceback.print_exc()
        
    # Return full path
    full_path = os.path.join(output_dir, filename)
    logging.debug(f"Generated output path: {full_path}")
    return full_path

def setup_output_directories(base_dir: str, test_id: str) -> Dict[str, str]:
    """
    Create standard output directory structure
    
    Args:
        base_dir: Base output directory
        test_id: Test ID
        
    Returns:
        Dictionary of paths for each directory
    """
    # Normalize test ID
    test_id = normalize_test_id(test_id)
    
    # Sanitize base directory in case it already contains subdirectories
    base_dir = sanitize_base_directory(base_dir)
    
    # Define directory paths
    json_dir = os.path.join(base_dir, "json")
    images_dir = os.path.join(base_dir, "supporting_images")
    debug_dir = os.path.join(base_dir, "debug")
    
    # Create directories
    created_dirs = []
    try:
        os.makedirs(base_dir, exist_ok=True)
        created_dirs.append(base_dir)
        
        os.makedirs(json_dir, exist_ok=True)
        created_dirs.append(json_dir)
        
        os.makedirs(images_dir, exist_ok=True)
        created_dirs.append(images_dir)
        
        os.makedirs(debug_dir, exist_ok=True)
        created_dirs.append(debug_dir)
        
        logging.info(f"Created/verified output directories: {', '.join(created_dirs)}")
    except Exception as e:
        logging.error(f"Error creating output directories: {str(e)}")
        traceback.print_exc()
    
    # Return paths
    return {
        "base": base_dir,
        "json": json_dir,
        "images": images_dir,
        "debug": debug_dir,
        "test_id": test_id
    }

def get_standardized_filename(test_id: str, file_type: str, extension: str) -> str:
    """
    Create standardized filename with test ID prefix
    
    Args:
        test_id: Test ID (will be normalized)
        file_type: Type identifier (e.g., log_analysis, component_report)
        extension: File extension (without dot)
        
    Returns:
        Standardized filename
    """
    test_id = normalize_test_id(test_id)
    filename = f"{test_id}_{file_type}.{extension}"
    logging.debug(f"Generated standardized filename: {filename}")
    return filename

def get_visualization_path(output_dir: str, test_id: str, viz_type: str, extension: str = "png") -> str:
    """
    Get standardized path for a visualization file.
    
    This is a convenience function specifically for visualizations like timelines.
    
    Args:
        output_dir: Base output directory
        test_id: Test ID
        viz_type: Type of visualization (e.g., timeline, cluster_timeline)
        extension: File extension (default: png)
        
    Returns:
        Full path to the visualization file
    """
    filename = get_standardized_filename(test_id, viz_type, extension)
    return get_output_path(output_dir, test_id, filename, OutputType.VISUALIZATION)

def cleanup_nested_directories(base_dir: str) -> Dict[str, int]:
    """
    Clean up nested directories that should not exist.
    
    This is useful when reports are regenerated to prevent accumulation
    of nested directories from previous runs.
    
    Args:
        base_dir: Base output directory
        
    Returns:
        Dictionary with counts of fixes made
    """
    results = {
        "json_dirs_fixed": 0,
        "images_dirs_fixed": 0,
        "debug_dirs_fixed": 0,
        "dirs_removed": 0,
    }
    
    if not os.path.exists(base_dir):
        return results
    
    # Check for nested json directory
    nested_json = os.path.join(base_dir, "json", "json")
    if os.path.exists(nested_json):
        for filename in os.listdir(nested_json):
            source_path = os.path.join(nested_json, filename)
            if os.path.isfile(source_path):
                target_path = os.path.join(base_dir, "json", filename)
                try:
                    if not os.path.exists(target_path):
                        shutil.move(source_path, target_path)
                    else:
                        os.remove(source_path)
                    results["json_dirs_fixed"] += 1
                except Exception as e:
                    logging.error(f"Error fixing nested json file: {str(e)}")
        try:
            shutil.rmtree(nested_json)
            results["dirs_removed"] += 1
        except Exception as e:
            logging.error(f"Error removing nested json directory: {str(e)}")
    
    # Check for nested supporting_images directory
    nested_images = os.path.join(base_dir, "supporting_images", "supporting_images")
    if os.path.exists(nested_images):
        for filename in os.listdir(nested_images):
            source_path = os.path.join(nested_images, filename)
            if os.path.isfile(source_path):
                target_path = os.path.join(base_dir, "supporting_images", filename)
                try:
                    if not os.path.exists(target_path):
                        shutil.move(source_path, target_path)
                    else:
                        os.remove(source_path)
                    results["images_dirs_fixed"] += 1
                except Exception as e:
                    logging.error(f"Error fixing nested image file: {str(e)}")
        try:
            shutil.rmtree(nested_images)
            results["dirs_removed"] += 1
        except Exception as e:
            logging.error(f"Error removing nested supporting_images directory: {str(e)}")
    
    # Check for nested debug directory
    nested_debug = os.path.join(base_dir, "debug", "debug")
    if os.path.exists(nested_debug):
        for filename in os.listdir(nested_debug):
            source_path = os.path.join(nested_debug, filename)
            if os.path.isfile(source_path):
                target_path = os.path.join(base_dir, "debug", filename)
                try:
                    if not os.path.exists(target_path):
                        shutil.move(source_path, target_path)
                    else:
                        os.remove(source_path)
                    results["debug_dirs_fixed"] += 1
                except Exception as e:
                    logging.error(f"Error fixing nested debug file: {str(e)}")
        try:
            shutil.rmtree(nested_debug)
            results["dirs_removed"] += 1
        except Exception as e:
            logging.error(f"Error removing nested debug directory: {str(e)}")
    
    # Log results
    total_fixed = results["json_dirs_fixed"] + results["images_dirs_fixed"] + results["debug_dirs_fixed"]
    if total_fixed > 0 or results["dirs_removed"] > 0:
        logging.info(
            f"Moved {total_fixed} files and removed {results['dirs_removed']} nested directories"
        )
        
    return results

def get_path_reference(path: str, base_dir: str, reference_type: str = "html") -> Optional[str]:
    """
    Convert a full path to a standardized reference format.
    
    Args:
        path: Full path to the file
        base_dir: Base directory for reference calculation
        reference_type: Type of reference to generate (html, json, relative)
        
    Returns:
        A standardized reference string
    """
    if not path or not os.path.exists(path):
        logging.warning(f"Cannot create path reference for non-existent path: {path}")
        return None
        
    # Normalize paths
    path = os.path.normpath(path)
    base_dir = os.path.normpath(base_dir)
    
    # Get the base filename
    filename = os.path.basename(path)
    
    # Determine output type from path
    if "/json/" in path.replace("\\", "/") or "\\json\\" in path.replace("/", "\\"):
        output_type = "json"
    elif "/supporting_images/" in path.replace("\\", "/") or "\\supporting_images\\" in path.replace("/", "\\"):
        output_type = "supporting_images"
    elif "/debug/" in path.replace("\\", "/") or "\\debug\\" in path.replace("/", "\\"):
        output_type = "debug"
    else:
        output_type = None
    
    # Create appropriate reference
    if reference_type == "html":
        if output_type == "supporting_images":
            ref_path = f"supporting_images/{filename}"
        elif output_type == "json":
            ref_path = f"json/{filename}"
        elif output_type == "debug":
            ref_path = f"debug/{filename}"
        else:
            ref_path = filename
        logging.debug(f"Created HTML reference path: {ref_path} from {path}")
        return ref_path
    elif reference_type == "json":
        # Full path for JSON serialization
        return path
    elif reference_type == "relative":
        # Path relative to base_dir
        try:
            rel_path = os.path.relpath(path, base_dir)
            logging.debug(f"Created relative path: {rel_path} from {path} (base: {base_dir})")
            return rel_path
        except ValueError:
            # If paths are on different drives, return full path
            logging.warning(f"Could not create relative path from {path} to {base_dir}")
            return path
    
    # Default to full path
    return path

def verify_visualization_directory(base_dir: str) -> bool:
    """
    Verify that the supporting_images directory exists and is accessible.
    
    This is specifically for ensuring visualization images can be saved.
    
    Args:
        base_dir: Base directory for the output structure
        
    Returns:
        True if directory exists and is accessible, False otherwise
    """
    # Sanitize the base directory
    base_dir = sanitize_base_directory(base_dir)
    
    # Get the visualization directory path
    viz_dir = os.path.join(base_dir, "supporting_images")
    
    # Check if directory exists
    if not os.path.exists(viz_dir):
        try:
            os.makedirs(viz_dir, exist_ok=True)
            logging.info(f"Created visualization directory: {viz_dir}")
        except Exception as e:
            logging.error(f"Failed to create visualization directory {viz_dir}: {str(e)}")
            return False
    
    # Check if directory is writable
    try:
        # Try to create a test file
        test_file = os.path.join(viz_dir, ".test_file")
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        logging.debug(f"Verified visualization directory is writable: {viz_dir}")
        return True
    except Exception as e:
        logging.error(f"Visualization directory exists but is not writable: {viz_dir}: {str(e)}")
        return False

def ensure_test_dirs(test_id: str) -> Dict[str, str]:
    """
    Create or ensure both logs/{test_id} and output/{test_id} directory structures exist.
    
    This function recreates the standard directory structure used by the log analyzer:
    - logs/{test_id}/ - For log files
    - output/{test_id}/ - For output files
      - output/{test_id}/json/ - For JSON data
      - output/{test_id}/supporting_images/ - For visualizations
      - output/{test_id}/debug/ - For debug information
    
    Args:
        test_id: Test ID to use for directory names
        
    Returns:
        Dictionary with paths to created directories
    """
    # Normalize test ID
    test_id = normalize_test_id(test_id)
    
    result_dirs = {}
    
    # Create logs directory
    logs_base_dir = Config.LOG_BASE_DIR
    test_log_dir = os.path.join(logs_base_dir, test_id)
    
    try:
        os.makedirs(test_log_dir, exist_ok=True)
        result_dirs["logs"] = test_log_dir
        logging.info(f"Created/verified logs directory: {test_log_dir}")
    except Exception as e:
        logging.error(f"Error creating logs directory {test_log_dir}: {str(e)}")
        traceback.print_exc()
    
    # Create output directories
    output_base_dir = Config.OUTPUT_BASE_DIR
    test_output_dir = os.path.join(output_base_dir, test_id)
    
    # Use existing function to set up output directories
    output_dirs = setup_output_directories(test_output_dir, test_id)
    
    # Combine results
    result_dirs.update(output_dirs)
    
    return result_dirs