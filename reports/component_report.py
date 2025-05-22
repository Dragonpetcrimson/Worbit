# reports/component_report.py
import os
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
import json
from datetime import datetime
import traceback
from jinja2 import Environment, FileSystemLoader, select_autoescape

# Import path utilities
from utils.path_utils import (
    get_output_path,
    OutputType,
    normalize_test_id,
    get_standardized_filename,
    sanitize_base_directory
)

# Try to import visualization utilities, with fallbacks if not available
try:
    from utils.visualization_utils import (
        configure_matplotlib_backend,
        verify_visualization_data,
        save_figure_with_cleanup,
        verify_image_file
    )
except ImportError:
    # Fallback implementations if visualization_utils is not available
    def configure_matplotlib_backend():
        """Configure matplotlib to work in any environment."""
        import matplotlib
        matplotlib.use('Agg', force=True)
        return matplotlib.pyplot

    def verify_visualization_data(data, data_type):
        """Simple fallback for data verification."""
        if not data:
            return False, "No data available"
        return True, ""
    
    def save_figure_with_cleanup(fig, image_path, dpi=100):
        """Save figure with guaranteed cleanup."""
        import matplotlib.pyplot as plt
        try:
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            fig.savefig(image_path, dpi=dpi, bbox_inches='tight')
            return image_path
        finally:
            plt.close(fig)
    
def verify_image_file(path):
    """Simple fallback for image verification."""
    return os.path.exists(path) and os.path.getsize(path) > 0

# Jinja2 environment for HTML templates
TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")
env = Environment(
    loader=FileSystemLoader(TEMPLATE_DIR),
    autoescape=select_autoescape(["html", "xml"]),
)

def generate_component_report(
    output_dir: str,
    test_id: str,
    analysis_results: Dict[str, Any],
    primary_issue_component: Optional[str] = None
) -> str:
    """
    Generate an HTML report for component relationship analysis.
    
    Args:
        output_dir: Directory to save the report
        test_id: Test ID for the title
        analysis_results: Results from ComponentIntegration.analyze_logs
        primary_issue_component: Optional component identified as root cause
        
    Returns:
        Path to the generated HTML report
    """
    # Verify directory structure before generating report
    from utils.path_validator import fix_directory_structure, check_html_references
    fix_directory_structure(output_dir, test_id)
    
    # Sanitize output directory to prevent nesting
    output_dir = sanitize_base_directory(output_dir)
    logging.info(f"Component report using output directory: {output_dir}")
    
    # Use path utilities for report path
    report_filename = get_standardized_filename(test_id, "component_report", "html")
    report_path = get_output_path(
        output_dir,
        test_id,
        report_filename,
        OutputType.PRIMARY_REPORT
    )
    
    # Extract component analysis and clusters data from analysis_results if available
    error_analysis = None
    clusters_data = None
    
    # Load component analysis data if not directly provided
    analysis_path = analysis_results.get("analysis_files", {}).get("component_analysis")
    if analysis_path and os.path.exists(analysis_path):
        try:
            with open(analysis_path, 'r', encoding='utf-8') as f:
                error_analysis = json.load(f)
                logging.info(f"Loaded component analysis data from {analysis_path}")
        except Exception as e:
            logging.error(f"Error loading component analysis: {str(e)}")
            error_analysis = {}
    
    # Load clusters data if not directly provided
    clusters_path = analysis_results.get("analysis_files", {}).get("enhanced_clusters")
    if clusters_path and os.path.exists(clusters_path):
        try:
            with open(clusters_path, 'r', encoding='utf-8') as f:
                clusters_data = json.load(f)
                logging.info(f"Loaded clusters data from {clusters_path}")
        except Exception as e:
            logging.error(f"Error loading clusters data: {str(e)}")
            clusters_data = {}
    
    # Extract a root cause component from analysis_results if not provided directly
    if primary_issue_component is None:
        # First try from analysis_results directly
        primary_issue_component = analysis_results.get("primary_issue_component")
        
        # If not found, try from error_analysis
        if not primary_issue_component and error_analysis:
            primary_issue_component = error_analysis.get("root_cause_component")
    
    logging.info(f"Using primary_issue_component: {primary_issue_component}")
    
    # Generate the component visualization
    visualization_path = generate_component_visualization(
        output_dir=output_dir, 
        test_id=test_id, 
        error_analysis=error_analysis,
        clusters_data=clusters_data,
        primary_issue_component=primary_issue_component
    )
    
    # Standardize visualization references in HTML
    viz_files = {}
    for key, file_path in analysis_results.get("analysis_files", {}).items():
        if file_path and os.path.exists(file_path):
            # Extract just the base filename for HTML reference
            base_filename = os.path.basename(file_path)
            
            # Always ensure paths start with supporting_images/ for HTML references
            if any(ext in base_filename.lower() for ext in ['.png', '.jpg', '.jpeg', '.gif', '.svg']):
                # Set standard path reference for HTML
                viz_files[key] = f"supporting_images/{base_filename}"
                
                # Ensure the file exists in supporting_images directory
                expected_path = os.path.join(output_dir, "supporting_images", base_filename)
                if not os.path.exists(expected_path):
                    # Log the issue but don't try to fix it here
                    logging.warning(f"Expected visualization file not found: {expected_path}")
            else:
                # For non-image files, keep as is
                viz_files[key] = file_path
    
    # Add the newly generated visualization to the references
    if visualization_path and os.path.exists(visualization_path):
        base_filename = os.path.basename(visualization_path)
        # Store under both keys for backward compatibility
        viz_files["component_errors"] = f"supporting_images/{base_filename}"
        viz_files["component_distribution"] = f"supporting_images/{base_filename}"
        logging.info(f"Added component visualization to HTML references: supporting_images/{base_filename}")
    
    # Build context for template
    context = {
        "title": f"Component Analysis for {test_id}",
        "test_id": test_id,
        "timestamp": analysis_results.get("timestamp", datetime.now().isoformat()),
        "metrics": analysis_results.get("metrics", {}),
        "component_viz": viz_files.get("component_errors") or viz_files.get("component_distribution"),
        "relationships_viz": viz_files.get("component_diagram") or viz_files.get("component_relationships"),
    }

    template = env.get_template("component_report.html.j2")
    html = template.render(**context)
    
    
    # Write HTML to file
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    # Validate HTML references
    try:
        html_issues = check_html_references(report_path)
        total_issues = sum(len(v) for v in html_issues.values())
        if total_issues > 0:
            logging.warning(f"HTML reference issues found in {report_path}:")
            for issue_type, issues in html_issues.items():
                for issue in issues:
                    logging.warning(f"  {issue_type}: {issue}")
    except Exception as e:
        logging.error(f"Failed to validate HTML references: {e}")

    # Log what we've done
    logging.info(f"Generated component report at {report_path}")

    return report_path


def generate_component_visualization(
    output_dir: str, 
    test_id: str, 
    error_analysis: Optional[Dict[str, Any]] = None,
    clusters_data: Optional[Dict[str, Any]] = None,
    primary_issue_component: Optional[str] = None
) -> Optional[str]:
    """
    Generate a horizontal bar chart showing component distribution by cluster.
    
    Args:
        output_dir: Directory to save the visualization
        test_id: Test ID for the filename
        error_analysis: Component error analysis data
        clusters_data: Enhanced clustering data
        primary_issue_component: The component identified as root cause
        
    Returns:
        Path to the generated PNG file or None if visualization couldn't be created
    """
    try:
        # Check feature flag with getattr and defensive imports
        try:
            from config import Config
            if not getattr(Config, 'ENABLE_COMPONENT_DISTRIBUTION', True):
                logging.info(f"Component error distribution visualization is disabled by feature flag")
                return None
        except ImportError:
            # If config can't be imported, assume the feature is enabled
            pass
        
        # Configure matplotlib
        plt = configure_matplotlib_backend()
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Sanitize output directory
        output_dir = sanitize_base_directory(output_dir, "supporting_images")
        
        # Use component_errors as the primary filename for consistency
        image_path = get_output_path(
            output_dir,
            test_id,
            get_standardized_filename(test_id, "component_errors", "png"),
            OutputType.VISUALIZATION
        )
        
        # Verify we have valid data for visualization
        if not error_analysis or not error_analysis.get("component_error_counts"):
            logging.warning("No component error data available for visualization")
            return None
        
        # Extract component data
        component_counts = error_analysis.get("component_error_counts", {})
        
        # Filter out "unknown" component for better visualization
        if "unknown" in component_counts and len(component_counts) > 1:
            component_counts = {k: v for k, v in component_counts.items() if k != "unknown"}
        
        # Sort components by error count (descending)
        sorted_components = sorted(component_counts.items(), key=lambda x: x[1], reverse=True)
        
        # If no components after filtering, return None (no visualization)
        if not sorted_components:
            logging.warning("No identified components with errors for visualization")
            return None
        
        # Extract component names and error counts
        components, counts = zip(*sorted_components)
        
        # Define component colors
        component_colors = {
            "soa": "#3498db",       # Blue
            "mimosa": "#2ecc71",    # Green
            "charles": "#9b59b6",   # Purple
            "phoebe": "#d35400",    # Dark Orange
            "translator": "#16a085", # Green-blue
            "android": "#7f8c8d",   # Gray
            "ip_traffic": "#2980b9", # Light blue
            "telesto": "#f39c12",   # Orange
            "arecibo": "#1abc9c",   # Turquoise
            "lapetus": "#34495e",   # Navy
        }
        default_color = "#95a5a6"  # Light gray
        root_cause_color = "#d81b60"  # Magenta/red
        
        # Determine colors based on component
        colors = []
        for comp in components:
            if comp == primary_issue_component:
                colors.append(root_cause_color)
            else:
                colors.append(component_colors.get(comp, default_color))
        
        # Create horizontal bar chart
        fig = plt.figure(figsize=(10, 6))
        y_pos = range(len(components))
        
        # Convert component names to uppercase for display
        display_components = [comp.upper() for comp in components]
        
        # Create the horizontal bar chart
        bars = plt.barh(y_pos, counts, align='center', color=colors)
        plt.yticks(y_pos, display_components)
        plt.xlabel('Error Count')
        plt.title('Component Error Distribution')
        
        # Add error count values at the end of each bar
        for i, v in enumerate(counts):
            plt.text(v + 0.1, i, str(v), va='center')
        
        # Add a legend for primary component if it exists
        if primary_issue_component and primary_issue_component in components:
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=root_cause_color, label=f'Root Cause: {primary_issue_component.upper()}')
            ]
            plt.legend(handles=legend_elements, loc='lower right')
        
        # Save the visualization with proper cleanup
        plt.tight_layout()
        image_path = save_figure_with_cleanup(fig, image_path)
        
        # Verify the image was created successfully
        if verify_image_file(image_path):
            logging.info(f"Generated component error distribution visualization: {image_path}")
            
            # Also save with the alternative name for backward compatibility
            alt_image_path = get_output_path(
                output_dir,
                test_id,
                get_standardized_filename(test_id, "component_distribution", "png"),
                OutputType.VISUALIZATION
            )
            
            # Copy the file instead of regenerating the figure
            import shutil
            shutil.copy2(image_path, alt_image_path)
            logging.info(f"Created backward compatible visualization copy: {alt_image_path}")
            
            return image_path
        else:
            logging.error(f"Failed to create valid visualization file at {image_path}")
            return None
        
    except Exception as e:
        logging.error(f"Error generating component visualization: {str(e)}")
        traceback.print_exc()
        return None