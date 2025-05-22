"""
Visualization generators for the log analysis system.
This module creates timeline visualizations and component relationship charts.
Enhanced with thread safety, memory management, and path standardization.
"""

import os
import sys
import logging
import traceback
import threading
import time
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

# Import base classes
from reports.base import ReportGenerator, ReportData

# Try to import visualization utilities, with fallbacks if not available
try:
    from utils.visualization_utils import (
        verify_visualization_data,
        handle_empty_data,
        save_figure_with_cleanup,
        configure_matplotlib_backend,
        generate_placeholder,
        get_visualization_path,
        verify_image_file
    )
    HAS_VISUALIZATION_UTILS = True
except ImportError:
    HAS_VISUALIZATION_UTILS = False
    logging.warning("Visualization utilities not available, using internal fallbacks")

# Import path utilities with fallbacks
try:
    from utils.path_utils import (
        get_output_path,
        OutputType,
        normalize_test_id,
        get_standardized_filename,
        sanitize_base_directory
    )
    HAS_PATH_UTILS = True
except ImportError:
    HAS_PATH_UTILS = False
    logging.warning("Path utilities not available, using internal fallbacks")

# Try to import matplotlib and related libraries with robust error handling
try:
    import matplotlib
    matplotlib.use('Agg', force=True)  # Force non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.lines import Line2D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logging.error("Matplotlib not available - visualizations will not be generated")

# Try to import PIL for image verification with fallback
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    logging.warning("PIL not available - image verification will be limited")

# Try to import networkx with fallbacks
try:
    import networkx as nx
    HAS_NETWORKX = True
    
    # Try to use pydot/graphviz for better layouts
    try:
        import pydot
        from networkx.drawing.nx_pydot import graphviz_layout
        HAS_GRAPHVIZ = True
    except ImportError:
        HAS_GRAPHVIZ = False
        logging.info("Using fallback layout system (pydot/graphviz not available)")
except ImportError:
    HAS_NETWORKX = False
    HAS_GRAPHVIZ = False
    logging.error("NetworkX not available - component visualizations will not be generated")

# Try to import numpy for array operations
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    logging.warning("NumPy not available - some visualization features may be limited")

# Import Config for feature flags
try:
    from config import Config
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False
    logging.warning("Config not available - using default feature flags")

# Thread-local storage for visualization state
_visualization_local = threading.local()

# Feature flag cache to avoid repeated lookups
_feature_flags = {}

# Visualization locks to prevent concurrent generation of the same visualization
_visualization_locks = {
    "timeline": threading.Lock(),
    "cluster_timeline": threading.Lock(),
    "component": threading.Lock(),
    "component_errors": threading.Lock(),
    "error_propagation": threading.Lock(),
    "placeholder": threading.Lock()
}

def _is_feature_enabled(feature_name: str, default: bool = False) -> bool:
    """
    Check if a feature is enabled with thread-safe fallback.
    
    Args:
        feature_name: Name of the feature flag in Config
        default: Default value if flag doesn't exist
        
    Returns:
        Boolean indicating if feature is enabled
    """
    # Use thread-local cache if available
    if not hasattr(_visualization_local, 'feature_cache'):
        _visualization_local.feature_cache = {}
    
    # Check cache first
    if feature_name in _visualization_local.feature_cache:
        return _visualization_local.feature_cache[feature_name]
    
    # Get from config
    try:
        if HAS_CONFIG:
            result = getattr(Config, feature_name, default)
        else:
            result = default
    except Exception:
        # If config can't be accessed, use default
        result = default
    
    # Cache for future use
    _visualization_local.feature_cache[feature_name] = result
    
    return result

def _is_placeholder_enabled() -> bool:
    """
    Check if visualization placeholders are enabled.
    
    Returns:
        Boolean indicating if placeholders should be generated
    """
    return _is_feature_enabled('ENABLE_VISUALIZATION_PLACEHOLDERS', False)

def _configure_matplotlib_backend_internal():
    """
    Internal fallback implementation to configure matplotlib backend.
    Only used if utils.visualization_utils is not available.
    """
    # Import matplotlib and explicitly use a non-GUI backend
    import matplotlib
    
    # Force Agg backend to avoid tkinter thread issues completely
    matplotlib.use('Agg', force=True)
    
    import matplotlib.pyplot as plt
    
    # Configure global settings
    plt.rcParams['figure.max_open_warning'] = 50  # Prevent warnings for many figures
    plt.rcParams['font.size'] = 10  # Readable default font size
    plt.rcParams['figure.dpi'] = 100  # Default DPI
    
    return plt

def _save_figure_with_cleanup_internal(fig, image_path, dpi=100):
    """
    Internal fallback implementation to save figure with cleanup.
    Only used if utils.visualization_utils is not available.
    
    Args:
        fig: Matplotlib figure
        image_path: Path to save the image
        dpi: Resolution in dots per inch
        
    Returns:
        Path to the saved image
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        
        # Save figure with specified DPI
        fig.savefig(image_path, bbox_inches='tight', dpi=dpi)
        return image_path
    finally:
        # Always close figure to free memory, even if save fails
        plt.close(fig)

def _verify_image_internal(image_path, description="visualization"):
    """
    Internal fallback implementation to verify image validity.
    Only used if utils.visualization_utils is not available.
    
    Args:
        image_path: Path to the image file
        description: Description for logging
        
    Returns:
        True if valid, False otherwise
    """
    if not os.path.exists(image_path):
        logging.warning(f"Generated {description} file does not exist: {image_path}")
        return False
        
    # File size check - very basic validation
    if os.path.getsize(image_path) < 100:  # Less than 100 bytes is suspicious
        logging.warning(f"Generated {description} file is too small: {image_path}")
        return False
    
    # Try PIL if available
    if HAS_PIL:
        try:
            img = Image.open(image_path)
            img.verify()
            return True
        except Exception as e:
            logging.warning(f"Generated {description} is invalid: {str(e)}")
            return False
    
    # If we get here without PIL, assume valid
    return True

def _get_visualization_path_internal(output_dir, test_id, visualization_type, extension="png"):
    """
    Internal fallback implementation to get standardized visualization path.
    Only used if utils.path_utils is not available.
    
    Args:
        output_dir: Output directory
        test_id: Test ID
        visualization_type: Type of visualization (component, errors, etc.)
        extension: File extension (default: png)
        
    Returns:
        Standardized path for the visualization
    """
    # Basic sanitization
    if test_id and not test_id.startswith("SXM-"):
        test_id = f"SXM-{test_id}"
    
    # Check for nested directories
    if "supporting_images" in output_dir:
        # Already has supporting_images, don't nest further
        viz_dir = output_dir
    else:
        # Add supporting_images subdirectory
        viz_dir = os.path.join(output_dir, "supporting_images")
    
    # Create directory
    os.makedirs(viz_dir, exist_ok=True)
    
    # Create filename and path
    filename = f"{test_id}_{visualization_type}.{extension}"
    return os.path.join(viz_dir, filename)

def _generate_placeholder_internal(output_dir, test_id, message="Visualization not available"):
    """
    Internal fallback implementation to generate placeholder visualization.
    Only used if utils.visualization_utils is not available.
    
    Args:
        output_dir: Output directory
        test_id: Test ID
        message: Message for placeholder
        
    Returns:
        Path to placeholder file or None
    """
    # Check if placeholders are enabled
    if not _is_placeholder_enabled():
        return None
    
    # Configure matplotlib
    if HAS_MATPLOTLIB:
        _configure_matplotlib_backend_internal()
        
        # Get path for placeholder
        placeholder_path = _get_visualization_path_internal(
            output_dir, test_id, "visualization_placeholder", "png")
        
        try:
            fig = plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, message, ha='center', va='center', fontsize=14, wrap=True)
            plt.axis('off')
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(placeholder_path), exist_ok=True)
            
            # Save with cleanup
            return _save_figure_with_cleanup_internal(fig, placeholder_path)
        except Exception as e:
            logging.error(f"Error generating placeholder: {str(e)}")
            plt.close('all')  # Ensure cleanup
            return None
    else:
        return None

def configure_matplotlib():
    """
    Configure matplotlib for non-interactive use.
    Falls back to internal implementation if visualization_utils not available.
    """
    if HAS_VISUALIZATION_UTILS:
        return configure_matplotlib_backend()
    else:
        return _configure_matplotlib_backend_internal()

def save_figure(fig, image_path, dpi=100):
    """
    Save a figure with proper cleanup to prevent memory leaks.
    Falls back to internal implementation if visualization_utils not available.
    
    Args:
        fig: Matplotlib figure
        image_path: Path to save figure
        dpi: Resolution in dots per inch
        
    Returns:
        Path to saved image
    """
    if HAS_VISUALIZATION_UTILS:
        return save_figure_with_cleanup(fig, image_path, dpi)
    else:
        return _save_figure_with_cleanup_internal(fig, image_path, dpi)

def verify_image(image_path, description="visualization"):
    """
    Verify that an image file is valid and not corrupted.
    Falls back to internal implementation if visualization_utils not available.
    
    Args:
        image_path: Path to image file
        description: Description for logging
        
    Returns:
        True if valid, False otherwise
    """
    if HAS_VISUALIZATION_UTILS:
        return verify_image_file(image_path, description)
    else:
        return _verify_image_internal(image_path, description)

def get_viz_path(output_dir, test_id, visualization_type, extension="png"):
    """
    Get standardized path for visualization file.
    Falls back to internal implementation if path_utils not available.
    
    Args:
        output_dir: Output directory
        test_id: Test ID
        visualization_type: Type of visualization
        extension: File extension (default: png)
        
    Returns:
        Standardized path for visualization
    """
    if HAS_VISUALIZATION_UTILS:
        return get_visualization_path(output_dir, test_id, visualization_type, extension)
    elif HAS_PATH_UTILS:
        # Sanitize output directory to prevent nested directories
        output_dir = sanitize_base_directory(output_dir, "supporting_images")
        
        # Use path utilities consistently
        return get_output_path(
            output_dir,
            test_id,
            get_standardized_filename(test_id, visualization_type, extension),
            OutputType.VISUALIZATION
        )
    else:
        return _get_visualization_path_internal(output_dir, test_id, visualization_type, extension)

def generate_placeholder(output_dir, test_id, message="Visualization not available"):
    """
    Generate a placeholder image with an error message when visualization fails.
    Falls back to internal implementation if visualization_utils not available.
    
    Args:
        output_dir: Output directory
        test_id: Test ID
        message: Message for placeholder
        
    Returns:
        Path to placeholder file or None
    """
    # Check if placeholders are enabled
    if not _is_placeholder_enabled():
        return None
        
    if HAS_VISUALIZATION_UTILS:
        return generate_placeholder(output_dir, test_id, message)
    else:
        return _generate_placeholder_internal(output_dir, test_id, message)

def validate_timeline_in_report(output_dir: str, test_id: str) -> bool:
    """
    Validate that the timeline image is properly included in the HTML report.
    Logs a warning if the timeline section is missing or if the image is not properly linked.

    Args:
        output_dir: Output directory
        test_id: Test ID

    Returns:
        True if validation passes, False otherwise
    """
    # Skip validation if diagnostic checks are disabled
    if not _is_feature_enabled('ENABLE_DIAGNOSTIC_CHECKS', False):
        return True
        
    try:
        import re
        
        # Normalize test ID to ensure consistent format
        if HAS_PATH_UTILS:
            test_id = normalize_test_id(test_id)
        elif not test_id.startswith("SXM-"):
            test_id = f"SXM-{test_id}"
            
        # Define path to HTML report
        html_report_path = os.path.join(output_dir, f"{test_id}_step_report.html")
        
        # Check if HTML report exists
        if not os.path.exists(html_report_path):
            logging.warning(f"Cannot validate timeline in report: HTML report not found at {html_report_path}")
            return False
            
        logging.info(f"Validating timeline in HTML report: {html_report_path}")
        
        # Read HTML content
        with open(html_report_path, 'r', encoding='utf-8', errors='ignore') as f:
            html_content = f.read()
            
        # Check if timeline section exists
        timeline_section_pattern = r'<div id="timeline-section"[^>]*>.*?</div>'
        match = re.search(timeline_section_pattern, html_content, re.DOTALL)
        
        if not match:
            # Try alternative pattern - just the heading
            heading_pattern = r'<h2>Test Execution Timeline</h2>'
            if re.search(heading_pattern, html_content):
                logging.warning("Timeline heading found but not in proper section format in HTML report")
            else:
                logging.warning("Timeline section not found in HTML report")
            return False
            
        # Check if timeline image is included
        timeline_filename = f"{test_id}_timeline.png"
        expected_image_pattern = f'<img[^>]*src="[^"]*{timeline_filename}"[^>]*>'
        
        if not re.search(expected_image_pattern, html_content, re.DOTALL):
            logging.warning(f"Timeline image ({timeline_filename}) reference not found in HTML report")
            return False
            
        # If we get here, validation passed
        logging.info("SUCCESS: Timeline validation passed - image properly referenced in HTML report")
        return True
            
    except Exception as e:
        logging.error(f"Error validating timeline in report: {str(e)}")
        traceback.print_exc()
        return False

def generate_cluster_timeline_image(step_to_logs, step_dict, clusters, output_dir, test_id):
    """
    Generate a cluster timeline image with enhanced thread safety and data validation.
    Shows errors colored by cluster rather than severity.
    
    Args:
        step_to_logs: Dictionary mapping step numbers to log entries
        step_dict: Dictionary mapping step numbers to step objects
        clusters: Dictionary mapping cluster IDs to lists of errors
        output_dir: Directory to write the image
        test_id: Test ID for the filename
        
    Returns:
        Path to the generated image or None if generation fails or is disabled
    """
    # Check feature flag with thread-safe method
    if not _is_feature_enabled('ENABLE_CLUSTER_TIMELINE', False):
        logging.info(f"Cluster timeline visualization is disabled by feature flag")
        return None
    
    # Acquire lock to prevent concurrent generation of the same visualization
    with _visualization_locks["cluster_timeline"]:
        # Validate input data
        if not step_to_logs or not step_dict or not clusters:
            logging.warning("Missing required data for cluster timeline visualization")
            return handle_empty_data(output_dir, test_id, "cluster_timeline", 
                                   "Insufficient data for cluster timeline visualization")
    
        # Configure matplotlib
        configure_matplotlib()
        
        # Get path for visualization
        image_path = get_viz_path(output_dir, test_id, "cluster_timeline")
        
        # Get path for debug log
        if HAS_PATH_UTILS:
            debug_log_path = get_output_path(
                output_dir, 
                test_id, 
                get_standardized_filename(test_id, "cluster_timeline_debug", "txt"),
                OutputType.DEBUGGING
            )
        else:
            # Fallback for debug log path
            debug_dir = os.path.join(output_dir, "debug")
            os.makedirs(debug_dir, exist_ok=True)
            debug_log_path = os.path.join(debug_dir, f"{test_id}_cluster_timeline_debug.txt")
        
        # Log the exact path being used for debugging
        logging.info(f"Saving cluster timeline to: {image_path}")
        
        try:
            # Create figure
            fig = plt.figure(figsize=(12, 8))
            
            # Use a temp debug log for immediate information
            temp_debug = []
            temp_debug.append(f"Generating cluster timeline for {test_id}")
            temp_debug.append(f"Step count: {len(step_dict)}")
            temp_debug.append(f"Cluster count: {len(clusters)}")
            
            # Build error to cluster mapping with multiple identification methods
            error_to_cluster = {}
            
            # Method 1: Direct object identity mapping
            for cluster_id, cluster_errors in clusters.items():
                for error in cluster_errors:
                    error_id = id(error)  # Use object id as identifier
                    error_to_cluster[error_id] = cluster_id
                    
                    # Method 2: File/line based mapping
                    if isinstance(error, dict):
                        # For dictionary-style errors
                        file_name = error.get('file', 'unknown')
                        line_num = error.get('line_num', 0)
                        error_key = f"{file_name}:{line_num}"
                        error_to_cluster[error_key] = cluster_id
                        
                        # Method 3: Text-based mapping
                        text = error.get('text', '')
                        if text:
                            text_key = text[:50].strip()  # Use start of text as key
                            if text_key:
                                error_to_cluster[text_key] = cluster_id
                    elif hasattr(error, 'file') and hasattr(error, 'line_num'):
                        # For object-style errors
                        error_key = f"{error.file}:{error.line_num}"
                        error_to_cluster[error_key] = cluster_id
                        if hasattr(error, 'text') and error.text:
                            text_key = error.text[:50].strip()
                            if text_key:
                                error_to_cluster[text_key] = cluster_id
            
            # Debug logging
            temp_debug.append(f"Created mapping with {len(error_to_cluster)} entries")
            
            # Determine the start and end times
            all_timestamps = []
            for step, logs in step_to_logs.items():
                for log in logs:
                    if hasattr(log, 'timestamp') and log.timestamp is not None:
                        # Handle multiple timestamp formats
                        ts = log.timestamp
                        if isinstance(ts, str):
                            try:
                                ts = datetime.fromisoformat(ts)
                            except ValueError:
                                try:
                                    # Try common formats
                                    formats = [
                                        "%Y-%m-%d %H:%M:%S",
                                        "%Y-%m-%d %H:%M:%S.%f",
                                        "%Y-%m-%dT%H:%M:%S",
                                        "%Y-%m-%dT%H:%M:%S.%f",
                                        "%d/%m/%Y %H:%M:%S",
                                        "%m/%d/%Y %H:%M:%S"
                                    ]
                                    for fmt in formats:
                                        try:
                                            ts = datetime.strptime(ts, fmt)
                                            break
                                        except ValueError:
                                            continue
                                except Exception:
                                    continue
                        if isinstance(ts, datetime):
                            all_timestamps.append(ts)
            
            # Check if we have valid timestamps
            if not all_timestamps:
                temp_debug.append("No valid timestamps found")
                
                # Create linear timeline without timestamps
                plt.title(f"Cluster Timeline (no timestamps available)")
                plt.xlabel("Step")
                plt.ylabel("Events")
                
                steps = sorted(step_dict.keys())
                for step_num in steps:
                    step = step_dict[step_num]
                    plt.axvline(x=step_num, color='gray', linestyle='--', alpha=0.5)
                    step_text = getattr(step, 'text', f"Step {step_num}")
                    if isinstance(step_text, str) and len(step_text) > 30:
                        step_text = step_text[:27] + "..."
                    plt.text(step_num, 0.5, step_text, 
                             rotation=90, verticalalignment='center')
                
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                # Save with cleanup
                image_path = save_figure(fig, image_path)
                
                # Write debug log
                with open(debug_log_path, 'w') as f:
                    f.write('\n'.join(temp_debug))
                
                # Verify image
                verify_image(image_path, "cluster timeline")
                
                # Validate if timeline is properly included in the report if diagnostics are enabled
                validate_timeline_in_report(output_dir, test_id)
                
                return image_path
            
            # Calculate time range with buffer
            start_time = min(all_timestamps) - timedelta(seconds=30)
            end_time = max(all_timestamps) + timedelta(seconds=30)
            
            # Log time range
            temp_debug.append(f"Time range: {start_time} to {end_time}")
            
            # Setup plot
            plt.title(f"Error Cluster Timeline")
            plt.xlabel("Time")
            plt.ylabel("Steps")
            
            # Configure date format based on time range
            time_range = end_time - start_time
            if time_range.total_seconds() < 3600:  # Less than 1 hour
                date_format = '%H:%M:%S'
            elif time_range.total_seconds() < 86400:  # Less than 1 day
                date_format = '%H:%M'
            else:
                date_format = '%m-%d %H:%M'
                
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(date_format))
            plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
            
            # Color cycle for clusters
            cluster_colors = [
                '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
            ]
            
            # Track which clusters we've seen for the legend
            clusters_seen = set()
            
            # Draw step boundaries
            for step_num, step in step_dict.items():
                step_logs = step_to_logs.get(step_num, [])
                if not step_logs:
                    continue
                    
                # Calculate step timespan
                step_timestamps = [
                    log.timestamp for log in step_logs 
                    if hasattr(log, 'timestamp') and log.timestamp
                    and isinstance(log.timestamp, datetime)
                ]
                if not step_timestamps:
                    continue
                    
                step_start = min(step_timestamps)
                
                # Plot step boundaries
                plt.axvline(x=step_start, color='gray', linestyle='--', alpha=0.5)
                
                # Step label with vertical text
                step_text = getattr(step, 'text', f"Step {step_num}")
                if isinstance(step_text, str) and len(step_text) > 30:
                    step_text = step_text[:27] + "..."
                plt.text(
                    step_start, 0.95, f"Step {step_num}",
                    rotation=90, 
                    verticalalignment='top', 
                    transform=plt.gca().get_xaxis_transform()
                )
                
                # Group errors by cluster
                cluster_points = {}
                
                # Track errors for this step in debug log
                step_error_count = 0
                step_cluster_counts = {}
                
                for log in step_logs:
                    if hasattr(log, 'is_error') and log.is_error and hasattr(log, 'timestamp') and log.timestamp:
                        if not isinstance(log.timestamp, datetime):
                            continue
                            
                        step_error_count += 1
                        
                        # Try multiple methods to find cluster ID
                        cluster_id = -1  # Default to unclustered
                        
                        # Method 1: Object identity
                        error_id = id(log)
                        if error_id in error_to_cluster:
                            cluster_id = error_to_cluster[error_id]
                        
                        # Method 2: File/line key
                        if cluster_id == -1 and hasattr(log, 'file') and hasattr(log, 'line_num'):
                            error_key = f"{log.file}:{log.line_num}"
                            if error_key in error_to_cluster:
                                cluster_id = error_to_cluster[error_key]
                        
                        # Method 3: Text snippet key
                        if cluster_id == -1 and hasattr(log, 'text') and log.text:
                            text_key = log.text[:50].strip()
                            if text_key and text_key in error_to_cluster:
                                cluster_id = error_to_cluster[text_key]
                        
                        # Initialize cluster in points dict if not present
                        if cluster_id not in cluster_points:
                            cluster_points[cluster_id] = []
                        
                        # Add this error to the cluster's points
                        cluster_points[cluster_id].append(log.timestamp)
                        
                        # Track clusters seen for this step
                        if cluster_id not in step_cluster_counts:
                            step_cluster_counts[cluster_id] = 0
                        step_cluster_counts[cluster_id] += 1
                
                # Plot errors by cluster
                for cluster_id, points in cluster_points.items():
                    if not points:
                        continue
                        
                    if cluster_id >= 0:  # Skip unclustered (-1)
                        color = cluster_colors[cluster_id % len(cluster_colors)]
                        label = f'Cluster {cluster_id}' if cluster_id not in clusters_seen else ""
                        plt.plot(points, [step_num] * len(points), 'o', color=color, label=label)
                        clusters_seen.add(cluster_id)
                    else:
                        # Plot unclustered errors in gray
                        label = 'Unclustered' if -1 not in clusters_seen else ""
                        plt.plot(points, [step_num] * len(points), 'o', color='gray', label=label)
                        clusters_seen.add(-1)
                
                # Debug logging for this step
                temp_debug.append(f"Step {step_num}: {step_error_count} errors across {len(cluster_points)} clusters")
                for cid, count in sorted(step_cluster_counts.items()):
                    if cid >= 0:
                        temp_debug.append(f"  Cluster {cid}: {count} errors")
                    else:
                        temp_debug.append(f"  Unclustered: {count} errors")
            
            # Add legend if we plotted any errors
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))  # Remove duplicate labels
            if by_label:
                plt.legend(by_label.values(), by_label.keys(), loc='upper right')
            
            # Set the x-axis limits with 5% padding
            time_range = end_time - start_time
            padding = time_range * 0.05
            plt.xlim(start_time - padding, end_time + padding)
            
            # Rotate x-axis labels for better readability
            plt.gcf().autofmt_xdate()
            
            # Add grid
            plt.grid(True, alpha=0.3)
            
            # Save the image with proper cleanup
            plt.tight_layout()
            image_path = save_figure(fig, image_path)
            
            # Write debug log
            with open(debug_log_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(temp_debug))
            
            # Verify the image was created successfully
            verify_image(image_path, "cluster timeline")
            
            # Validate if timeline is properly included in the report if diagnostics are enabled
            validate_timeline_in_report(output_dir, test_id)
            
            return image_path
            
        except Exception as e:
            logging.error(f"Error generating cluster timeline: {str(e)}")
            traceback.print_exc()
            
            # Write exception to debug log
            try:
                temp_debug.append(f"ERROR: {str(e)}")
                temp_debug.append(traceback.format_exc())
                with open(debug_log_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(temp_debug))
            except Exception:
                pass
            
            # Return placeholder if enabled
            return generate_placeholder(
                output_dir, test_id, f"Error generating cluster timeline: {str(e)}"
            )

def generate_timeline_image(step_to_logs, step_dict, output_dir, test_id):
    """
    Generate a PNG image of the timeline visualization with enhanced error handling.
    
    Args:
        step_to_logs: Dictionary mapping step numbers to log entries
        step_dict: Dictionary mapping step numbers to step objects with step_name, start_time, end_time
        output_dir: Directory to write the image
        test_id: Test ID for the filename
        
    Returns:
        Path to the generated image or None if generation fails
    """
    # Acquire lock to prevent concurrent generation of the same visualization
    with _visualization_locks["timeline"]:
        # Validate input data
        if not step_to_logs or not step_dict:
            logging.warning("Missing required data for timeline visualization")
            return handle_empty_data(output_dir, test_id, "timeline", 
                                   "Insufficient data for timeline visualization")
    
        # Configure matplotlib
        configure_matplotlib()
        
        # Get path for visualization
        image_path = get_viz_path(output_dir, test_id, "timeline")
        
        # Get path for debug log
        if HAS_PATH_UTILS:
            debug_log_path = get_output_path(
                output_dir, 
                test_id, 
                get_standardized_filename(test_id, "timeline_debug", "txt"),
                OutputType.DEBUGGING
            )
        else:
            # Fallback for debug log path
            debug_dir = os.path.join(output_dir, "debug")
            os.makedirs(debug_dir, exist_ok=True)
            debug_log_path = os.path.join(debug_dir, f"{test_id}_timeline_debug.txt")
        
        # Log the path being used
        logging.info(f"Saving timeline visualization to: {image_path}")
        
        # Create temp debug log for immediate information
        temp_debug = []
        temp_debug.append(f"Generating timeline for {test_id}")
        temp_debug.append(f"Step count: {len(step_dict)}")
        
        try:
            # Create figure
            fig = plt.figure(figsize=(12, 8))
            
            # Check if step_dict contains proper step metadata with start_time and end_time
            has_proper_step_metadata = any(
                isinstance(step, dict) and 'start_time' in step and 'end_time' in step
                for step in step_dict.values()
            )
            
            # Log the metadata status
            temp_debug.append(f"Has proper step metadata: {has_proper_step_metadata}")
            
            # Determine the start and end times - prioritize step metadata if available
            all_timestamps = []
            
            if has_proper_step_metadata:
                # Use the start_time and end_time from step_dict
                for step_num, step_info in step_dict.items():
                    if isinstance(step_info, dict):
                        if 'start_time' in step_info and isinstance(step_info['start_time'], datetime):
                            all_timestamps.append(step_info['start_time'])
                        if 'end_time' in step_info and isinstance(step_info['end_time'], datetime):
                            all_timestamps.append(step_info['end_time'])
                            
                # Log the timestamps extracted from step metadata
                temp_debug.append(f"Using {len(all_timestamps)} timestamps from step metadata")
            
            # If no timestamps from metadata, or too few, extract from logs as fallback
            if len(all_timestamps) < 2:
                temp_debug.append("Insufficient timestamps from metadata, extracting from logs")
                all_timestamps = []
                for step, logs in step_to_logs.items():
                    for log in logs:
                        if hasattr(log, 'timestamp') and log.timestamp is not None:
                            # Handle multiple timestamp formats
                            ts = log.timestamp
                            if isinstance(ts, str):
                                try:
                                    ts = datetime.fromisoformat(ts)
                                except ValueError:
                                    try:
                                        # Try common formats
                                        formats = [
                                            "%Y-%m-%d %H:%M:%S",
                                            "%Y-%m-%d %H:%M:%S.%f",
                                            "%Y-%m-%dT%H:%M:%S",
                                            "%Y-%m-%dT%H:%M:%S.%f",
                                            "%d/%m/%Y %H:%M:%S",
                                            "%m/%d/%Y %H:%M:%S"
                                        ]
                                        for fmt in formats:
                                            try:
                                                ts = datetime.strptime(ts, fmt)
                                                break
                                            except ValueError:
                                                continue
                                    except Exception:
                                        continue
                            if isinstance(ts, datetime):
                                all_timestamps.append(ts)
                                
                # Log the timestamps extracted from logs
                temp_debug.append(f"Extracted {len(all_timestamps)} timestamps from logs")
            
            # Check if we have valid timestamps
            if not all_timestamps:
                temp_debug.append("No valid timestamps found")
                
                # Create linear timeline without timestamps
                plt.title(f"Test Timeline (no timestamps available)")
                plt.xlabel("Step")
                plt.ylabel("Events")
                
                steps = sorted(step_dict.keys())
                for step_num in steps:
                    step = step_dict[step_num]
                    plt.axvline(x=step_num, color='gray', linestyle='--', alpha=0.5)
                    
                    # Extract step text based on the structure of step_dict
                    if isinstance(step, dict) and 'step_name' in step:
                        step_text = step['step_name']
                    else:
                        step_text = getattr(step, 'text', f"Step {step_num}")
                        
                    if isinstance(step_text, str) and len(step_text) > 30:
                        step_text = step_text[:27] + "..."
                    plt.text(step_num, 0.5, step_text, 
                             rotation=90, verticalalignment='center')
                
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                # Save with cleanup
                image_path = save_figure(fig, image_path)
                
                # Write debug log
                with open(debug_log_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(temp_debug))
                
                # Verify image
                verify_image(image_path, "timeline")
                
                # Validate if timeline is properly included in the report if diagnostics are enabled
                validate_timeline_in_report(output_dir, test_id)
                
                return image_path
            
            # Calculate time range with buffer
            start_time = min(all_timestamps) - timedelta(seconds=30)
            end_time = max(all_timestamps) + timedelta(seconds=30)
            
            # Log time range
            temp_debug.append(f"Time range: {start_time} to {end_time}")
            
            # Setup plot
            plt.title(f"Test Timeline with Steps")
            plt.xlabel("Time")
            plt.ylabel("Steps")
            
            # Configure date format based on time range
            time_range = end_time - start_time
            if time_range.total_seconds() < 3600:  # Less than 1 hour
                date_format = '%H:%M:%S'
            elif time_range.total_seconds() < 86400:  # Less than 1 day
                date_format = '%H:%M'
            else:
                date_format = '%m-%d %H:%M'
                
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(date_format))
            plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
            
            # Track if we need to include legend items
            has_high = False
            has_medium = False
            has_low = False
            
            # Draw step boundaries
            for step_num, step in step_dict.items():
                step_logs = step_to_logs.get(step_num, [])
                
                # Determine step_start time - prioritize step metadata
                step_start = None
                
                # Try to get start_time from step metadata
                if isinstance(step, dict) and 'start_time' in step and isinstance(step['start_time'], datetime):
                    step_start = step['start_time']
                    temp_debug.append(f"Using metadata start time for step {step_num}: {step_start}")
                # Fall back to calculating from logs
                elif step_logs:
                    step_timestamps = [
                        log.timestamp for log in step_logs 
                        if hasattr(log, 'timestamp') and log.timestamp
                        and isinstance(log.timestamp, datetime)
                    ]
                    if step_timestamps:
                        step_start = min(step_timestamps)
                        temp_debug.append(f"Using calculated start time for step {step_num}: {step_start}")
                
                if step_start is None:
                    temp_debug.append(f"No valid start time for step {step_num}, skipping")
                    continue
                    
                # Plot step boundaries
                plt.axvline(x=step_start, color='gray', linestyle='--', alpha=0.5)
                
                # Step label with vertical text
                # Extract step text based on the structure of step_dict
                if isinstance(step, dict) and 'step_name' in step:
                    step_text = step['step_name']
                else:
                    step_text = getattr(step, 'text', f"Step {step_num}")
                    
                if isinstance(step_text, str) and len(step_text) > 30:
                    step_text = step_text[:27] + "..."
                plt.text(
                    step_start, 0.95, f"Step {step_num}",
                    rotation=90, 
                    verticalalignment='top', 
                    transform=plt.gca().get_xaxis_transform()
                )
                
                # Collect error points by severity
                error_points_high = []
                error_points_med = []
                error_points_low = []
                
                for log in step_logs:
                    if hasattr(log, 'is_error') and log.is_error and hasattr(log, 'timestamp') and log.timestamp:
                        if not isinstance(log.timestamp, datetime):
                            continue
                            
                        # Get severity with fallback to Medium
                        severity = getattr(log, 'severity', 'Medium')
                        if isinstance(severity, str):
                            severity = severity.lower()
                        else:
                            severity = 'medium'  # Default
                        
                        if severity == 'high':
                            error_points_high.append(log.timestamp)
                            has_high = True
                        elif severity == 'low':
                            error_points_low.append(log.timestamp)
                            has_low = True
                        else:
                            error_points_med.append(log.timestamp)
                            has_medium = True
                
                # Plot errors with different colors and markers
                if error_points_high:
                    plt.plot(
                        error_points_high, 
                        [step_num] * len(error_points_high), 
                        'ro',  # Red circles
                        label='High Severity' if step_num == sorted(step_dict.keys())[0] else "",
                        markersize=8
                    )
                
                if error_points_med:
                    plt.plot(
                        error_points_med, 
                        [step_num] * len(error_points_med), 
                        'yo',  # Yellow circles
                        label='Medium Severity' if step_num == sorted(step_dict.keys())[0] else "",
                        markersize=6
                    )
                
                if error_points_low:
                    plt.plot(
                        error_points_low, 
                        [step_num] * len(error_points_low), 
                        'bo',  # Blue circles
                        label='Low Severity' if step_num == sorted(step_dict.keys())[0] else "",
                        markersize=5
                    )
                
                # Debug logging for this step
                temp_debug.append(
                    f"Step {step_num}: {len(step_logs)} logs, "
                    f"{len(error_points_high)} high errors, "
                    f"{len(error_points_med)} medium errors, "
                    f"{len(error_points_low)} low errors"
                )
            
            # Add legend if we plotted any errors
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))  # Remove duplicate labels
            if by_label:
                plt.legend(by_label.values(), by_label.keys(), loc='upper right')
            
            # Set the x-axis limits with 5% padding
            time_range = end_time - start_time
            padding = time_range * 0.05
            plt.xlim(start_time - padding, end_time + padding)
            
            # Rotate x-axis labels for better readability
            plt.gcf().autofmt_xdate()
            
            # Add grid
            plt.grid(True, alpha=0.3)
            
            # Save the image with proper cleanup
            plt.tight_layout()
            image_path = save_figure(fig, image_path)
            
            # Write debug log
            with open(debug_log_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(temp_debug))
            
            # Verify the image was created successfully
            verify_image(image_path, "timeline")
            
            # Validate if timeline is properly included in the report if diagnostics are enabled
            validate_timeline_in_report(output_dir, test_id)
            
            return image_path
            
        except Exception as e:
            logging.error(f"Error generating timeline: {str(e)}")
            traceback.print_exc()
            
            # Write exception to debug log
            try:
                temp_debug.append(f"ERROR: {str(e)}")
                temp_debug.append(traceback.format_exc())
                with open(debug_log_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(temp_debug))
            except Exception:
                pass
            
            # Return placeholder if enabled
            return generate_placeholder(
                output_dir, test_id, f"Error generating timeline: {str(e)}"
            )

class VisualizationGenerator(ReportGenerator):
    """
    Generator for report visualizations with enhanced thread safety and error handling.
    
    This class orchestrates the generation of all visualizations used in reports, including:
    - Component relationship diagrams
    - Component error distribution charts
    - Error propagation visualizations
    - Timeline visualizations
    - Cluster timeline visualizations
    
    It implements a robust approach to visualization generation with thread safety,
    timeout protection, memory management, and comprehensive error handling.
    """
    
    def generate(self, data: ReportData) -> Dict[str, str]:
        """
        Generate visualizations for the report with enhanced error handling.
        
        Args:
            data: Report data containing errors, clusters, and component information
            
        Returns:
            Dictionary mapping visualization types to file paths
        """
        visualization_paths = {}
        
        try:
            # Initialize thread-local storage for this thread
            if not hasattr(_visualization_local, 'feature_cache'):
                _visualization_local.feature_cache = {}
            
            # Ensure matplotlib is properly configured
            configure_matplotlib()
            
            # Ensure we're using the base output directory
            if HAS_PATH_UTILS:
                base_output_dir = sanitize_base_directory(self.config.output_dir)
            else:
                # Simple check for nested directories
                if "supporting_images" in self.config.output_dir or "json" in self.config.output_dir:
                    base_output_dir = os.path.dirname(self.config.output_dir)
                else:
                    base_output_dir = self.config.output_dir
                    
            logging.info(f"Visualization generator using base directory: {base_output_dir}")
            
            # Get and validate component analysis data
            component_analysis = data.component_analysis or {}
            
            # Check if we need to enhance component data
            if not component_analysis or "component_summary" not in component_analysis or not component_analysis["component_summary"]:
                logging.warning("Component analysis data missing or incomplete, attempting to create from errors")
                try:
                    # Try to import and use build_component_analysis
                    from reports.component_analyzer import build_component_analysis
                    component_analysis = build_component_analysis(
                        data.errors, 
                        self.config.primary_issue_component, 
                        component_analysis
                    )
                    logging.info("Successfully built component analysis from errors")
                except ImportError:
                    logging.warning("Could not import component_analyzer, visualization may be limited")
                except Exception as e:
                    logging.error(f"Error building component analysis: {str(e)}")
            
            # Extract component data for visualization
            component_summary = component_analysis.get("component_summary", [])
            relationships = component_analysis.get("relationships", [])
            primary_component = self.config.primary_issue_component
            
            # Generate visualizations with thread safety and timeout protection
            
            # 1. Component relationship visualization
            self._generate_with_timeout(
                visualization_paths,
                "component",
                'ENABLE_COMPONENT_REPORT_IMAGES',
                True,  # Default to enabled
                30,    # 30 second timeout
                "generate_component_visualization",
                base_output_dir,
                self.config.test_id,
                component_summary,
                relationships,
                primary_component
            )
            
            # 2. Component error distribution visualization
            self._generate_with_timeout(
                visualization_paths,
                "component_errors",
                'ENABLE_COMPONENT_DISTRIBUTION',
                True,  # Default to enabled
                30,    # 30 second timeout
                "generate_component_error_distribution",
                base_output_dir,
                self.config.test_id,
                component_summary,
                data.clusters,
                primary_component
            )
            
            # Maintain backward compatibility for component distribution/errors
            if "component_errors" in visualization_paths:
                visualization_paths["component_distribution"] = visualization_paths["component_errors"]
            
            # 3. Error propagation visualization
            self._generate_with_timeout(
                visualization_paths,
                "error_propagation",
                'ENABLE_ERROR_PROPAGATION',
                False,  # Default to disabled
                30,     # 30 second timeout
                "generate_error_propagation_diagram",
                base_output_dir,
                self.config.test_id,
                component_analysis.get("error_graph")
            )
            
            # 4. Timeline visualizations if we have step data
            if hasattr(data, "step_to_logs") and data.step_to_logs and hasattr(data, "step_dict") and data.step_dict:
                # Ensure logs have required attributes for visualization
                for step_num, logs in data.step_to_logs.items():
                    for log in logs:
                        if not hasattr(log, 'is_error'):
                            log.is_error = False
                        if not hasattr(log, 'severity'):
                            log.severity = 'Medium'
                
                # Generate standard timeline visualization - always enabled
                self._generate_with_timeout(
                    visualization_paths,
                    "timeline",
                    None,  # No feature flag, always enabled
                    30,    # 30 second timeout
                    "generate_timeline_image",
                    base_output_dir,
                    self.config.test_id,
                    data.step_to_logs,
                    data.step_dict
                )
                
                # Generate cluster timeline if we have clusters and it's enabled
                if data.clusters:
                    self._generate_with_timeout(
                        visualization_paths,
                        "cluster_timeline",
                        'ENABLE_CLUSTER_TIMELINE',
                        False,  # Default to disabled
                        30,     # 30 second timeout
                        "generate_cluster_timeline_image",
                        base_output_dir,
                        self.config.test_id,
                        data.step_to_logs,
                        data.step_dict,
                        data.clusters
                    )
            
            # Return visualization paths
            return visualization_paths
            
        except Exception as e:
            logging.error(f"Error in visualization generation: {str(e)}")
            traceback.print_exc()
            visualization_paths["error"] = str(e)
            return visualization_paths
            
        finally:
            # Always clean up resources
            if HAS_MATPLOTLIB:
                plt.close('all')
                
            # Clean up thread-local storage
            if hasattr(_visualization_local, 'feature_cache'):
                delattr(_visualization_local, 'feature_cache')
    
    def _generate_with_timeout(self, paths_dict, key, feature_flag, default_value, timeout_seconds, func_name, *args, **kwargs):
        """
        Generate a visualization with timeout protection and comprehensive error handling.
        
        Args:
            paths_dict: Dictionary to store the visualization path
            key: Key to use in the paths dictionary
            feature_flag: Name of the feature flag to check
            default_value: Default value if feature flag doesn't exist
            timeout_seconds: Timeout in seconds
            func_name: Name of the generation function to call
            *args, **kwargs: Arguments for the generator function
        """
        # Check if feature is enabled
        if feature_flag is not None and not _is_feature_enabled(feature_flag, default_value):
            logging.info(f"{key} visualization is disabled by feature flag {feature_flag}")
            return
        
        # Determine which function to call
        if func_name == "generate_component_visualization":
            from components.component_visualizer import generate_component_visualization as generator_func
        elif func_name == "generate_component_error_distribution":
            from components.component_visualizer import generate_component_error_distribution as generator_func
        elif func_name == "generate_error_propagation_diagram":
            from components.component_visualizer import generate_error_propagation_diagram as generator_func
        elif func_name == "generate_timeline_image":
            generator_func = generate_timeline_image
        elif func_name == "generate_cluster_timeline_image":
            generator_func = generate_cluster_timeline_image
        else:
            logging.error(f"Unknown visualization function: {func_name}")
            return
        
        # Set up thread safety for visualization
        result = [None]
        error = [None]
        
        def generate_with_timeout():
            try:
                # Call the generator function with the provided arguments
                result[0] = generator_func(*args, **kwargs)
            except Exception as e:
                error[0] = e
        
        # Create and start the thread
        thread = threading.Thread(target=generate_with_timeout)
        thread.daemon = True
        thread.start()
        
        # Wait with timeout
        thread.join(timeout=timeout_seconds)
        
        if thread.is_alive():
            # Timeout occurred
            logging.warning(f"Timeout generating {key} visualization after {timeout_seconds} seconds")
            # Thread continues to run but we don't wait for it
            
            # Generate a placeholder if enabled
            placeholder_path = generate_placeholder(
                args[0], args[1], f"Timeout while generating {key} visualization"
            )
            if placeholder_path:
                paths_dict[key] = placeholder_path
            return
        
        if error[0]:
            # Exception occurred
            logging.error(f"Error in {key} visualization: {str(error[0])}")
            traceback.print_exc()
            
            # Generate a placeholder if enabled
            placeholder_path = generate_placeholder(
                args[0], args[1], f"Error generating {key} visualization: {str(error[0])}"
            )
            if placeholder_path:
                paths_dict[key] = placeholder_path
            return
        
        # If generation was successful
        path = result[0]
        if path and os.path.exists(path):
            paths_dict[key] = path
            
            # Verify image is valid
            if not verify_image(path, key):
                # If verification fails, create a placeholder
                placeholder_path = generate_placeholder(
                    args[0], args[1], f"Generated {key} visualization is invalid"
                )
                if placeholder_path:
                    paths_dict[key] = placeholder_path
        else:
            logging.warning(f"No valid path returned for {key} visualization")
            
generate_visualization_placeholder = generate_placeholder