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
        os.makedirs(output_dir, exist_ok=True)
        
    # Return full path
    return os.path.join(output_dir, filename)

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
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)
    
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
    return f"{test_id}_{file_type}.{extension}"

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
        "debug_dirs_fixed": 0
    }
    
    if not os.path.exists(base_dir):
        return results
    
    # Check for nested json directory
    nested_json = os.path.join(base_dir, "json", "json")
    if os.path.exists(nested_json):
        # Copy files to the parent directory
        for filename in os.listdir(nested_json):
            source_path = os.path.join(nested_json, filename)
            if os.path.isfile(source_path):
                target_path = os.path.join(base_dir, "json", filename)
                try:
                    if not os.path.exists(target_path):
                        shutil.copy2(source_path, target_path)
                        results["json_dirs_fixed"] += 1
                except Exception as e:
                    logging.error(f"Error fixing nested json file: {str(e)}")
    
    # Check for nested supporting_images directory
    nested_images = os.path.join(base_dir, "supporting_images", "supporting_images")
    if os.path.exists(nested_images):
        # Copy files to the parent directory
        for filename in os.listdir(nested_images):
            source_path = os.path.join(nested_images, filename)
            if os.path.isfile(source_path):
                target_path = os.path.join(base_dir, "supporting_images", filename)
                try:
                    if not os.path.exists(target_path):
                        shutil.copy2(source_path, target_path)
                        results["images_dirs_fixed"] += 1
                except Exception as e:
                    logging.error(f"Error fixing nested image file: {str(e)}")
    
    # Check for nested debug directory
    nested_debug = os.path.join(base_dir, "debug", "debug")
    if os.path.exists(nested_debug):
        # Copy files to the parent directory
        for filename in os.listdir(nested_debug):
            source_path = os.path.join(nested_debug, filename)
            if os.path.isfile(source_path):
                target_path = os.path.join(base_dir, "debug", filename)
                try:
                    if not os.path.exists(target_path):
                        shutil.copy2(source_path, target_path)
                        results["debug_dirs_fixed"] += 1
                except Exception as e:
                    logging.error(f"Error fixing nested debug file: {str(e)}")
    
    # Log results
    total_fixed = sum(results.values())
    if total_fixed > 0:
        logging.info(f"Fixed {total_fixed} files in nested directories")
        
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
            return f"supporting_images/{filename}"
        elif output_type == "json":
            return f"json/{filename}"
        elif output_type == "debug":
            return f"debug/{filename}"
        else:
            return filename
    elif reference_type == "json":
        # Full path for JSON serialization
        return path
    elif reference_type == "relative":
        # Path relative to base_dir
        try:
            return os.path.relpath(path, base_dir)
        except ValueError:
            # If paths are on different drives, return full path
            return path
    
    # Default to full path
    return path