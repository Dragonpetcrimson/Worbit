# reports/component_report.py
import os
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
import json
from datetime import datetime
import traceback

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
    from utils.path_validator import fix_directory_structure
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
    
    # Build HTML
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Component Analysis for {test_id}</title>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; color: #333; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            h1, h2, h3 {{ color: #444; }}
            h1 {{ border-bottom: 2px solid #eee; padding-bottom: 10px; }}
            .summary {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            .visualization {{ margin: 20px 0; text-align: center; }}
            .visualization img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }}
            .card {{ border: 1px solid #ddd; margin-bottom: 20px; border-radius: 5px; overflow: hidden; }}
            .card-header {{ padding: 10px 15px; background-color: #f5f5f5; }}
            .card-body {{ padding: 15px; }}
            table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
            th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
            .error-message {{ background-color: #fff0f0; border-left: 3px solid #ff5555; padding: 10px; margin: 10px 0; }}
            .component-badge {{ 
                display: inline-block;
                padding: 3px 8px;
                background-color: #eee;
                border-radius: 3px;
                margin-right: 5px;
                font-size: 0.9em;
            }}
            .component-badge.root-cause {{
                background-color: #d81b60;
                color: white;
            }}
            .component-badge.affected {{
                background-color: #3949ab;
                color: white;
            }}
            .severity-high {{ color: #d13948; }}
            .severity-medium {{ color: #e48743; }}
            .severity-low {{ color: #4287f5; }}
            .error-path {{ 
                border-left: 3px solid #e74c3c; 
                padding-left: 15px; 
                margin: 10px 0; 
            }}
            .error-path-item {{
                padding: 5px;
                margin: 5px 0;
                background-color: #f9f9f9;
                border-radius: 3px;
            }}
            .error-path-arrow {{
                text-align: center;
                color: #888;
                margin: 5px 0;
            }}
            .visualization-notice {{
                background-color: #f8f9fa;
                border-left: 3px solid #6c757d;
                padding: 15px;
                margin: 15px 0;
                border-radius: 5px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Component Analysis for {test_id}</h1>
            
            <div class="summary">
                <p><strong>Analysis Date:</strong> {analysis_results.get("timestamp", datetime.now().isoformat())}</p>
                <p><strong>Components Tagged:</strong> {analysis_results.get("metrics", {}).get("component_tagged_logs", 0)} log entries</p>
                <p><strong>Errors Tagged:</strong> {analysis_results.get("metrics", {}).get("component_tagged_errors", 0)} errors</p>
                <p><strong>Clusters Identified:</strong> {analysis_results.get("metrics", {}).get("clusters", 0)} clusters</p>
                <p><strong>Root Cause Errors:</strong> {analysis_results.get("metrics", {}).get("root_cause_errors", 0)} errors</p>
                <p><em>This report shows the relationships between components and how errors propagate through the system.</em></p>
            </div>
    """
    
    # Add component error distribution visualization - check both component_errors and component_distribution keys
    component_viz_path = viz_files.get("component_errors") or viz_files.get("component_distribution")
    if component_viz_path:
        expected_file_path = os.path.join(output_dir, component_viz_path)
        
        if os.path.exists(expected_file_path):
            # File exists, show the visualization
            html += f"""
                <div class="visualization">
                    <h2>Component Error Distribution</h2>
                    <img src="{component_viz_path}" alt="Component Error Distribution">
                </div>
            """
        else:
            # File doesn't exist, show placeholder message
            html += f"""
                <div class="visualization-notice">
                    <h2>Component Error Distribution</h2>
                    <p>Visual component diagrams are temporarily unavailable.</p>
                    <p>Please refer to the component analysis text below for relationship details.</p>
                </div>
            """
    
    # Add component relationship diagram
    if viz_files.get("component_diagram") or viz_files.get("component_relationships"):
        # Try both keys for backward compatibility
        image_path = viz_files.get("component_diagram") or viz_files.get("component_relationships")
        if image_path:
            expected_file_path = os.path.join(output_dir, image_path)
            
            if os.path.exists(expected_file_path):
                # File exists, show the visualization
                html += f"""
                    <div class="visualization">
                        <h2>Component Relationship Diagram</h2>
                        <img src="{image_path}" alt="Component Relationships">
                    </div>
                """
            else:
                # File doesn't exist, show placeholder message
                html += f"""
                    <div class="visualization-notice">
                        <h2>Component Relationship Diagram</h2>
                        <p>Component relationship diagrams are temporarily unavailable.</p>
                        <p>Please refer to the text description below for component relationships.</p>
                    </div>
                """
    
    # Add error propagation diagram
    if viz_files.get("error_propagation"):
        image_path = viz_files.get("error_propagation")
        expected_file_path = os.path.join(output_dir, image_path)
        
        if os.path.exists(expected_file_path):
            # File exists, show the visualization
            html += f"""
                <div class="visualization">
                    <h2>Error Propagation Analysis</h2>
                    <img src="{image_path}" alt="Error Propagation">
                </div>
            """
        else:
            # File doesn't exist, show placeholder message
            html += f"""
                <div class="visualization-notice">
                    <h2>Error Propagation Analysis</h2>
                    <p>Error propagation diagrams are temporarily unavailable.</p>
                    <p>Please refer to the text description below for error propagation details.</p>
                </div>
            """
    
    # Add component error heatmap (if different from component_distribution)
    if viz_files.get("error_heatmap") and viz_files.get("error_heatmap") != viz_files.get("component_distribution"):
        image_path = viz_files.get("error_heatmap")
        expected_file_path = os.path.join(output_dir, image_path)
        
        if os.path.exists(expected_file_path):
            # File exists, show the visualization
            html += f"""
                <div class="visualization">
                    <h2>Component Error Heatmap</h2>
                    <img src="{image_path}" alt="Component Error Heatmap">
                </div>
            """
        else:
            # File doesn't exist, show placeholder message
            html += f"""
                <div class="visualization-notice">
                    <h2>Component Error Heatmap</h2>
                    <p>Component error heatmaps are temporarily unavailable.</p>
                    <p>Please refer to the text description below for error distribution details.</p>
                </div>
            """
    
    # Add component analysis section
    if error_analysis:
        html += f"""
            <h2>Component Analysis Results</h2>
            
            <div class="card">
                <div class="card-header">
                    <h3>Root Cause Component</h3>
                </div>
                <div class="card-body">
        """
        
        # Add root cause component info - Using the primary_issue_component parameter consistently
        root_cause = primary_issue_component if primary_issue_component else error_analysis.get("root_cause_component", "unknown")
        if root_cause and root_cause.lower() != "unknown":
            # Add more detailed component description
            error_count = error_analysis.get("component_error_counts", {}).get(root_cause, 0)
            html += f"""
                    <p><span class="component-badge root-cause">{root_cause.upper()}</span> has been identified as the likely ROOT CAUSE component.</p>
                    <p>This component had {error_count} errors during the test execution.</p>
                    
                    <p><strong>Description:</strong> {
                        error_analysis.get("component_info", {}).get(root_cause, {}).get("description", 
                        "This component experienced critical failures during testing.")}
                    </p>
                    
                    <p><strong>Role in the System:</strong> {
                        error_analysis.get("component_info", {}).get(root_cause, {}).get("role", 
                        "This component handles important functionality within the system.")}
                    </p>
            """
        else:
            html += """
                    <p>No clear root cause component could be identified.</p>
            """
        
        html += """
                </div>
            </div>
            
            <div class="card">
                <div class="card-header">
                    <h3>Affected Components</h3>
                </div>
                <div class="card-body">
        """
        
        # Add affected components table
        components = error_analysis.get("components_with_issues", [])
        if components:
            html += """
                    <table>
                        <tr>
                            <th>Component</th>
                            <th>Error Count</th>
                            <th>Severity Distribution</th>
                            <th>Relationship to Root Cause</th>
                        </tr>
            """
            
            for component in components:
                if component != "unknown":
                    error_count = error_analysis.get("component_error_counts", {}).get(component, 0)
                    severity_data = error_analysis.get("severity_by_component", {}).get(component, {})
                    
                    # Add relationship information
                    relationship = "Unknown"
                    if component == root_cause:
                        relationship = "Root Cause Component"
                    elif root_cause and root_cause != "unknown":
                        if component in error_analysis.get("component_info", {}).get(root_cause, {}).get("related_to", []):
                            relationship = "Directly Related"
                        else:
                            # Check for paths between components
                            for path in error_analysis.get("propagation_paths", []):
                                if component in path and root_cause in path:
                                    relationship = "Indirectly Related (Error Chain)"
                                    break
                    
                    html += f"""
                        <tr>
                            <td>{"⚠️ " if component == root_cause else ""}{component.upper()}</td>
                            <td>{error_count}</td>
                            <td>
                                <span class="severity-high">{severity_data.get("High", 0)} High</span>,
                                <span class="severity-medium">{severity_data.get("Medium", 0)} Medium</span>,
                                <span class="severity-low">{severity_data.get("Low", 0)} Low</span>
                            </td>
                            <td>{relationship}</td>
                        </tr>
                    """
            
            html += """
                    </table>
            """
        else:
            html += """
                    <p>No affected components identified.</p>
            """
        
        html += """
                </div>
            </div>
        """
        
        # Add error propagation paths
        if error_analysis.get("propagation_paths"):
            html += """
            <div class="card">
                <div class="card-header">
                    <h3>Error Propagation Paths</h3>
                </div>
                <div class="card-body">
            """
            
            for i, path in enumerate(error_analysis.get("propagation_paths", [])[:5]):
                html += f"""
                    <div class="error-path">
                        <p><strong>Path {i+1}</strong></p>
                """
                
                for j, component in enumerate(path):
                    is_root = component == root_cause
                    html += f"""
                        <div class="error-path-item">
                            <span class="component-badge {'root-cause' if is_root else 'affected'}">{component.upper()}</span>
                            {" (Root Cause)" if is_root else ""}
                        </div>
                    """
                    
                    if j < len(path) - 1:
                        html += """
                        <div class="error-path-arrow">↓</div>
                        """
                
                html += """
                    </div>
                """
            
            html += """
                </div>
            </div>
            """
    
    # Add enhanced clustering section
    if clusters_data:
        html += """
            <h2>Enhanced Error Clustering</h2>
            
            <div class="card">
                <div class="card-header">
                    <h3>Root Cause Errors</h3>
                </div>
                <div class="card-body">
        """
        
        # Add root cause errors
        root_cause_errors = clusters_data.get("root_cause_errors", [])
        if root_cause_errors:
            html += """
                <p>The following errors were identified as potential root causes:</p>
                <table>
                    <tr>
                        <th>Component</th>
                        <th>Severity</th>
                        <th>Error</th>
                    </tr>
            """
            
            for error in root_cause_errors[:5]:  # Show top 5
                component = error.get("component", "unknown")
                severity = error.get("severity", "Low")
                text = error.get("text", "").replace("<", "&lt;").replace(">", "&gt;")
                
                html += f"""
                    <tr>
                        <td><span class="component-badge">{component.upper()}</span></td>
                        <td class="severity-{severity.lower()}">{severity}</td>
                        <td>{text[:150]}{"..." if len(text) > 150 else ""}</td>
                    </tr>
                """
            
            html += """
                </table>
            """
        else:
            html += """
                <p>No clear root cause errors identified.</p>
            """
        
        html += """
                </div>
            </div>
        """
        
        # Add causal chains section
        causal_chains = clusters_data.get("causality_paths", [])
        if causal_chains:
            html += """
            <div class="card">
                <div class="card-header">
                    <h3>Error Causal Chains</h3>
                </div>
                <div class="card-body">
                <p>The following error sequences represent potential cause-effect relationships:</p>
            """
            
            for i, chain in enumerate(causal_chains[:3]):  # Show top 3
                html += f"""
                    <div class="error-path">
                        <p><strong>Causal Chain {i+1}</strong></p>
                """
                
                for j, error in enumerate(chain):
                    component = error.get("component", "unknown")
                    severity = error.get("severity", "Low")
                    text = error.get("text", "").replace("<", "&lt;").replace(">", "&gt;")
                    
                    html += f"""
                        <div class="error-path-item">
                            <span class="component-badge">{component.upper()}</span>
                            <span class="severity-{severity.lower()}">({severity})</span>
                            <div>{text[:100]}{"..." if len(text) > 100 else ""}</div>
                        </div>
                    """
                    
                    if j < len(chain) - 1:
                        html += """
                        <div class="error-path-arrow">↓</div>
                        """
                
                html += """
                    </div>
                """
            
            html += """
                </div>
            </div>
            """
    
    # Close HTML
    html += """
        </div>
    </body>
    </html>
    """
    
    # Write HTML to file
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
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