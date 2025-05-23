"""
utils/path_validator.py - Validates and fixes correct file placement
"""

import os
import logging
import traceback
import glob
import shutil
import re
from typing import Dict, List, Set

def validate_file_structure(base_dir: str, test_id: str) -> Dict[str, List[str]]:
    """
    Validate that files are in their proper locations.
    
    Args:
        base_dir: Base output directory
        test_id: Test ID
        
    Returns:
        Dictionary of misplaced files by category
    """
    issues = {
        "json_dir_images": [],
        "images_dir_json": [],
        "nested_directories": [],
        "expected_but_missing": []
    }
    
    # Check that directories exist
    json_dir = os.path.join(base_dir, "json")
    images_dir = os.path.join(base_dir, "supporting_images")
    debug_dir = os.path.join(base_dir, "debug")
    
    # Check for nested directories
    if os.path.exists(os.path.join(json_dir, "json")):
        issues["nested_directories"].append("json/json")
    
    if os.path.exists(os.path.join(images_dir, "supporting_images")):
        issues["nested_directories"].append("supporting_images/supporting_images")
    
    # Check for misplaced files
    if os.path.exists(json_dir):
        for file in os.listdir(json_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                issues["json_dir_images"].append(os.path.join("json", file))
    
    if os.path.exists(images_dir):
        for file in os.listdir(images_dir):
            if file.lower().endswith('.json'):
                issues["images_dir_json"].append(os.path.join("supporting_images", file))
    
    # Look for expected files
    expected_files = [
        (os.path.join(base_dir, f"{test_id}_bug_report.docx"), "Primary report"),
        (os.path.join(base_dir, f"{test_id}_log_analysis.xlsx"), "Primary report"),
        (os.path.join(json_dir, f"{test_id}_log_analysis.json"), "JSON data"),
        (os.path.join(images_dir, f"{test_id}_component_errors.png"), "Component Visualization")
    ]
    
    # Calculate standard paths
    component_error_path = os.path.join(images_dir, f"{test_id}_component_errors.png")

    # Report missing component_errors.png
    if not os.path.exists(component_error_path):
        issues["expected_but_missing"].append(component_error_path)
    
    # Check the other expected files
    for file_path, file_type in expected_files:
        if file_path != component_error_path and not os.path.exists(file_path):
            issues["expected_but_missing"].append(f"{file_path} ({file_type})")
    
    return issues

def check_html_references(html_file: str) -> Dict[str, List[str]]:
    """
    Check HTML file for correct references to supporting files.
    
    Args:
        html_file: Path to HTML file
        
    Returns:
        Dictionary of issues
    """
    issues = {
        "missing_supporting_images_prefix": [],
        "references_to_nonexistent_files": []
    }
    
    if not os.path.exists(html_file):
        return {"file_missing": [html_file]}
    
    try:
        with open(html_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check for image references that don't use supporting_images prefix
        img_refs = []
        img_start = 0
        while True:
            img_start = content.find('<img src="', img_start)
            if img_start == -1:
                break
            
            img_end = content.find('"', img_start + 10)
            if img_end == -1:
                break
                
            img_src = content[img_start + 10:img_end]
            img_refs.append(img_src)
            img_start = img_end
        
        # Check each image reference
        base_dir = os.path.dirname(html_file)
        for img_src in img_refs:
            if not img_src.startswith("supporting_images/") and not img_src.startswith("data:"):
                issues["missing_supporting_images_prefix"].append(img_src)

            # Report missing referenced files
            if not img_src.startswith("data:"):
                file_path = os.path.join(base_dir, img_src)
                if not os.path.exists(file_path):
                    issues["references_to_nonexistent_files"].append(img_src)
        
        return issues
            
    except Exception as e:
        logging.error(f"Error checking HTML references: {str(e)}")
        return {"error": [str(e)]}

def print_validation_results(base_dir: str, test_id: str):
    """
    Run validation and print results in a user-friendly format.
    
    Args:
        base_dir: Base output directory
        test_id: Test ID
    """
    print(f"\n=== Validation for {test_id} in {base_dir} ===")
    structure_issues = validate_file_structure(base_dir, test_id)
    
    print("\nDirectory Structure Issues:")
    total_structure_issues = sum(len(issues) for issues in structure_issues.values())
    if total_structure_issues == 0:
        print("  ✅ No directory structure issues found")
    else:
        for issue_type, issues in structure_issues.items():
            if issues:
                print(f"  ❌ {issue_type}: {len(issues)} issues")
                for issue in issues[:3]:  # Show at most 3 issues
                    print(f"    - {issue}")
                if len(issues) > 3:
                    print(f"    - ... and {len(issues) - 3} more")
    
    # Print overall result
    if total_structure_issues == 0:
        print("\n✅ VALIDATION PASSED: All files are in correct locations")
    else:
        print(f"\n❌ VALIDATION FAILED: Found {total_structure_issues} structure issues")

def fix_directory_structure(base_dir: str, test_id: str) -> Dict[str, List[str]]:
    """
    Find and fix directory structure issues.
    
    Args:
        base_dir: Base output directory
        test_id: Test ID
        
    Returns:
        Dictionary of fixes made by category
    """
    issues = {
        "json_dir_images": [],
        "images_dir_json": [],
        "nested_directories": [],
        "misplaced_visualizations": [],
        "fixed_files": [],
        "expected_but_missing": []
    }
    
    # First, clean up nested directories
    from utils.path_utils import cleanup_nested_directories
    cleanup_results = cleanup_nested_directories(base_dir)
    
    # Check if any visualizations are in the json directory
    json_dir = os.path.join(base_dir, "json")
    if os.path.exists(json_dir):
        for file in os.listdir(json_dir):
            file_path = os.path.join(json_dir, file)
            if os.path.isfile(file_path) and any(file.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif']):
                issues["json_dir_images"].append(file_path)
                
                # Move file to supporting_images directory
                target_path = os.path.join(base_dir, "supporting_images", file)
                if not os.path.exists(target_path):
                    try:
                        os.makedirs(os.path.dirname(target_path), exist_ok=True)
                        shutil.copy2(file_path, target_path)
                        issues["fixed_files"].append(f"Moved image: {file_path} -> {target_path}")
                        logging.info(f"Moved visualization from json directory: {file_path} -> {target_path}")
                    except Exception as e:
                        logging.error(f"Error fixing misplaced image file: {str(e)}")
    
    # Check if any json files are in the supporting_images directory
    images_dir = os.path.join(base_dir, "supporting_images")
    if os.path.exists(images_dir):
        for file in os.listdir(images_dir):
            file_path = os.path.join(images_dir, file)
            if os.path.isfile(file_path) and file.lower().endswith('.json'):
                issues["images_dir_json"].append(file_path)
                
                # Move file to json directory
                target_path = os.path.join(base_dir, "json", file)
                if not os.path.exists(target_path):
                    try:
                        os.makedirs(os.path.dirname(target_path), exist_ok=True)
                        shutil.copy2(file_path, target_path)
                        
                        # Verify component preservation for JSON files
                        try:
                            from utils.component_verification import verify_component_preservation
                            if not verify_component_preservation(file_path, target_path):
                                logging.error(f"Component information was lost when copying {file_path} to {target_path}")
                        except ImportError:
                            # Component verification not available yet, log warning
                            logging.warning("Component verification module not available, skipping verification")
                        
                        issues["fixed_files"].append(f"Moved JSON: {file_path} -> {target_path}")
                        logging.info(f"Moved JSON from supporting_images directory: {file_path} -> {target_path}")
                    except Exception as e:
                        logging.error(f"Error fixing misplaced JSON file: {str(e)}")
    
    # Check for visualization files in the wrong places (e.g., json/supporting_images/)
    json_images_dir = os.path.join(base_dir, "json", "supporting_images")
    if os.path.exists(json_images_dir):
        issues["nested_directories"].append(json_images_dir)
        
        # Copy visualization files to the correct location
        for file in os.listdir(json_images_dir):
            file_path = os.path.join(json_images_dir, file)
            if os.path.isfile(file_path):
                target_path = os.path.join(base_dir, "supporting_images", file)
                if not os.path.exists(target_path):
                    try:
                        os.makedirs(os.path.dirname(target_path), exist_ok=True)
                        shutil.copy2(file_path, target_path)
                        issues["fixed_files"].append(f"Moved visualization: {file_path} -> {target_path}")
                        issues["misplaced_visualizations"].append(file_path)
                        logging.info(f"Moved visualization from nested directory: {file_path} -> {target_path}")
                    except Exception as e:
                        logging.error(f"Error fixing nested visualization file: {str(e)}")
    
    # Report presence of component visualization file
    component_error_path = os.path.join(images_dir, f"{test_id}_component_errors.png")

    # Report missing component_errors.png
    if not os.path.exists(component_error_path):
        issues["expected_but_missing"].append(component_error_path)
    
    # Fix all HTML files to ensure correct references
    html_files = glob.glob(os.path.join(base_dir, f"{test_id}*.html"))
    for html_file in html_files:
        html_fixes = fix_html_references(html_file, base_dir)
        if html_fixes:
            issues["fixed_files"].extend(html_fixes)

    total_moved = (
        len(issues.get("fixed_files", []))
        + cleanup_results.get("json_dirs_fixed", 0)
        + cleanup_results.get("images_dirs_fixed", 0)
        + cleanup_results.get("debug_dirs_fixed", 0)
    )
    if total_moved > 0 or cleanup_results.get("dirs_removed", 0) > 0:
        logging.info(
            f"Directory cleanup moved {total_moved} files and removed {cleanup_results.get('dirs_removed', 0)} directories"
        )

    return issues

def fix_html_references(html_path: str, base_dir: str) -> List[str]:
    """
    Check HTML references to supporting files without modifying them.

    Args:
        html_path: Path to HTML file
        base_dir: Base directory for the test

    Returns:
        List of issues found
    """
    issues: List[str] = []

    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            content = f.read()

        results = check_html_references(html_path)
        for values in results.values():
            issues.extend(values)

        if 'style="display:none"' in content:
            issues.append('display_none_style_present')

        if '<div style="display:none">' in content:
            issues.append('hidden_container_present')

    except Exception as e:
        logging.error(f"Error checking HTML references in {html_path}: {str(e)}")

    return issues
