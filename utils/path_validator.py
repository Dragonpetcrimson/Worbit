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
        (os.path.join(base_dir, f"{test_id}_component_report.html"), "Primary report"),
        (os.path.join(base_dir, f"{test_id}_log_analysis.xlsx"), "Primary report"),
        (os.path.join(json_dir, f"{test_id}_log_analysis.json"), "JSON data"),
        (os.path.join(images_dir, f"{test_id}_cluster_timeline.png"), "Visualization"),
        (os.path.join(images_dir, f"{test_id}_component_errors.png"), "Component Visualization")
    ]
    
    # Calculate standard paths
    normalized_test_id = test_id
    component_error_path = os.path.join(images_dir, f"{normalized_test_id}_component_errors.png")
    component_dist_path = os.path.join(images_dir, f"{normalized_test_id}_component_distribution.png")
    
    # Check existing files
    has_error_file = os.path.exists(component_error_path)
    has_dist_file = os.path.exists(component_dist_path)
    
    # Handle standardization needs
    if not has_error_file and has_dist_file:
        # Copy from distribution to error (the standard we want to enforce)
        try:
            shutil.copy2(component_dist_path, component_error_path)
            logging.info(f"Created {component_error_path} from {component_dist_path} for standardization")
            has_error_file = True  # We've created it now
        except Exception as e:
            logging.error(f"Failed to copy {component_dist_path} to {component_error_path}: {str(e)}")
    
    # During transition, we want both files to exist for backward compatibility
    # Copy from error to distribution if needed
    if has_error_file and not has_dist_file:
        try:
            shutil.copy2(component_error_path, component_dist_path)
            logging.info(f"Created {component_dist_path} from {component_error_path} for backward compatibility")
        except Exception as e:
            logging.error(f"Failed to copy {component_error_path} to {component_dist_path}: {str(e)}")
    
    # Handle expected file checking for component_errors.png only
    # component_distribution.png is maintained only for compatibility
    if not has_error_file and not has_dist_file:
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
            
            # Check if referenced file exists
            if not img_src.startswith("data:"):
                file_path = os.path.join(base_dir, img_src)
                
                # Special handling for component visualization alternatives
                filename = os.path.basename(img_src)
                test_id = os.path.basename(html_file).split('_')[0]
                
                # The component visualization reference checking logic
                if filename == f"{test_id}_component_distribution.png":
                    # If this references component_distribution.png, check if component_errors.png exists instead
                    component_error_path = os.path.join(base_dir, "supporting_images", f"{test_id}_component_errors.png")
                    if os.path.exists(component_error_path):
                        # We found the standardized file, don't report as nonexistent
                        continue
                    
                # Standard existence check
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
    
    # Check HTML references
    html_file = os.path.join(base_dir, f"{test_id}_component_report.html")
    
    # Initialize total_html_issues
    total_html_issues = 0
    
    if os.path.exists(html_file):
        html_issues = check_html_references(html_file)
        
        print("\nHTML Reference Issues:")
        total_html_issues = sum(len(issues) for issues in html_issues.values())
        if total_html_issues == 0:
            print("  ✅ No HTML reference issues found")
        else:
            for issue_type, issues in html_issues.items():
                if issues:
                    print(f"  ❌ {issue_type}: {len(issues)} issues")
                    for issue in issues[:3]:
                        print(f"    - {issue}")
                    if len(issues) > 3:
                        print(f"    - ... and {len(issues) - 3} more")
    else:
        print(f"\n⚠️ HTML file not found: {html_file}")
    
    # Print overall result
    if total_structure_issues == 0 and total_html_issues == 0:
        print("\n✅ VALIDATION PASSED: All files are in correct locations and properly referenced")
    else:
        print(f"\n❌ VALIDATION FAILED: Found {total_structure_issues} structure issues and {total_html_issues} HTML issues")

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
        "fixed_files": []
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
    
    # CORRECTED APPROACH: When checking for expected files, standardize on component_errors.png
    # Create component_errors.png from component_distribution.png if needed
    component_error_path = os.path.join(images_dir, f"{test_id}_component_errors.png")
    component_dist_path = os.path.join(images_dir, f"{test_id}_component_distribution.png")
    
    # Check existing files
    has_error_file = os.path.exists(component_error_path)
    has_dist_file = os.path.exists(component_dist_path)
    
    # Handle standardization needs
    if not has_error_file and has_dist_file:
        # Copy from distribution to error (the standard we want to enforce)
        try:
            shutil.copy2(component_dist_path, component_error_path)
            logging.info(f"Created {component_error_path} from {component_dist_path} for standardization")
            issues["fixed_files"].append(f"Copied {component_dist_path} to {component_error_path}")
            has_error_file = True  # We've created it now
        except Exception as e:
            logging.error(f"Failed to copy {component_dist_path} to {component_error_path}: {str(e)}")
    
    # During transition, we want both files to exist for backward compatibility
    # Copy from error to distribution if needed
    if has_error_file and not has_dist_file:
        try:
            shutil.copy2(component_error_path, component_dist_path)
            logging.info(f"Created {component_dist_path} from {component_error_path} for backward compatibility")
            issues["fixed_files"].append(f"Copied {component_error_path} to {component_dist_path}")
        except Exception as e:
            logging.error(f"Failed to copy {component_error_path} to {component_dist_path}: {str(e)}")
    
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
    Fix HTML references to supporting files.
    
    Args:
        html_path: Path to HTML file
        base_dir: Base directory for the test
        
    Returns:
        List of fixes made
    """
    fixes = []
    
    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        original_content = content
        
        # Fix references without supporting_images prefix
        for match in re.finditer(r'<img\s+src="([^"]+\.(png|jpg|jpeg|gif))"', content):
            img_ref = match.group(1)
            if not img_ref.startswith("supporting_images/"):
                filename = os.path.basename(img_ref)
                new_ref = f"supporting_images/{filename}"
                content = content.replace(f'src="{img_ref}"', f'src="{new_ref}"')
                fixes.append(f"Fixed reference: {img_ref} -> {new_ref}")
                logging.info(f"Fixed HTML reference: {img_ref} -> {new_ref}")
        
        # CORRECTED: Update HTML to reference component_errors.png
        # Check for references to component_distribution.png
        test_id = os.path.basename(html_path).split('_')[0]
        if "component_distribution.png" in content:
            # Update HTML to reference component_errors.png
            new_content = content.replace(
                f"supporting_images/{test_id}_component_distribution.png", 
                f"supporting_images/{test_id}_component_errors.png"
            )
            new_content = new_content.replace(
                f"{test_id}_component_distribution.png", 
                f"supporting_images/{test_id}_component_errors.png"
            )
            
            if new_content != content:
                content = new_content
                fixes.append("Updated component_distribution.png references to component_errors.png")
                logging.info(f"Updated component_distribution.png references to component_errors.png in {html_path}")
        
        # Fix style="display:none" on images
        if 'style="display:none"' in content:
            content_before = content
            content = re.sub(r'<img([^>]+)style="display:none"([^>]*)>', r'<img\1\2>', content)
            if content != content_before:
                fixes.append("Removed display:none style from images")
                logging.info(f"Removed display:none style from images in {html_path}")
            
        # Fix hidden div containers
        if '<div style="display:none">' in content:
            content_before = content
            content = re.sub(r'<div style="display:none">', r'<div class="visualization">', content)
            if content != content_before:
                fixes.append("Fixed hidden visualization containers")
                logging.info(f"Fixed hidden visualization containers in {html_path}")
        
        # Write changes back to file if needed
        if content != original_content:
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
    except Exception as e:
        logging.error(f"Error fixing HTML references in {html_path}: {str(e)}")
    
    return fixes