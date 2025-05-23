"""
Visualization utilities for the Orbit Analyzer system.
Provides centralized functions for visualization management, path handling,
and error handling with proper memory management and thread safety.
"""

import os
import sys
import logging
import warnings
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

# Try to import path utilities with fallbacks
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
    logging.warning("Path utilities not available in visualization_utils, using internal fallbacks")

# Try to import matplotlib and related libraries with robust error handling
try:
    import matplotlib
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logging.error("Matplotlib not available - visualization utilities will be limited")

# Try to import PIL for image verification with fallback
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    logging.warning("PIL not available - image verification will be limited")

# Try to import Config for feature flags
try:
    from config import Config
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False
    logging.warning("Config not available in visualization_utils - using default feature flags")

def _is_feature_enabled(feature_name: str, default: bool = False) -> bool:
    """
    Check if a feature is enabled with fallback.
    
    Args:
        feature_name: Name of the feature flag in Config
        default: Default value if flag doesn't exist
        
    Returns:
        Boolean indicating if feature is enabled
    """
    # Get from config
    try:
        if HAS_CONFIG:
            return getattr(Config, feature_name, default)
        else:
            return default
    except Exception:
        # If config can't be accessed, use default
        return default

def _is_placeholder_enabled() -> bool:
    """
    Check if visualization placeholders are enabled.
    
    Returns:
        Boolean indicating if placeholders should be generated
    """
    return _is_feature_enabled('ENABLE_VISUALIZATION_PLACEHOLDERS', False)

def verify_visualization_data(data: Any, data_type: str) -> Tuple[bool, str]:
    """
    Verifies data is valid for visualization generation.
    
    Args:
        data: Data to verify
        data_type: Type of data/visualization
        
    Returns:
        Tuple of (is_valid, message)
    """
    if data is None:
        return False, f"No data provided for {data_type} visualization"
    
    # Data type-specific validation
    if data_type == "component_summary":
        # Component summary should be a list of dictionaries
        if not isinstance(data, list):
            return False, f"Component summary should be a list, got {type(data)}"
        if len(data) == 0:
            return False, f"Component summary is empty"
        if not all(isinstance(item, dict) for item in data):
            return False, f"Component summary items should be dictionaries"
        return True, ""
        
    elif data_type == "relationships":
        # Relationships should be a list of dictionaries
        if not isinstance(data, list):
            return False, f"Relationships should be a list, got {type(data)}"
        # Empty relationships are allowed (no connections)
        return True, ""
        
    elif data_type == "clusters":
        # Clusters should be a dictionary mapping cluster IDs to lists of errors
        if not isinstance(data, dict):
            return False, f"Clusters should be a dictionary, got {type(data)}"
        if len(data) == 0:
            return False, f"Clusters dictionary is empty"
        return True, ""
        
    elif data_type == "error_graph":
        # Error graph can be a NetworkX DiGraph or a dictionary with nodes/edges
        if data is None:
            return False, f"Error graph is None"
        # We'll do more specific validation in the visualization function
        return True, ""
        
    elif data_type == "step_to_logs":
        # Step to logs should be a dictionary mapping step numbers to log entries
        if not isinstance(data, dict):
            return False, f"Step to logs should be a dictionary, got {type(data)}"
        if len(data) == 0:
            return False, f"Step to logs dictionary is empty"
        return True, ""
        
    elif data_type == "step_dict":
        # Step dict should be a dictionary mapping step numbers to step objects
        if not isinstance(data, dict):
            return False, f"Step dict should be a dictionary, got {type(data)}"
        if len(data) == 0:
            return False, f"Step dict dictionary is empty"
        return True, ""
    
    # Default validation for other data types
    return True, ""

def handle_empty_data(output_dir: str, test_id: str, 
                    data_type: str, message: Optional[str] = None) -> Optional[str]:
    """
    Handles empty data based on configuration.
    Returns a placeholder path if placeholders are enabled, otherwise None.
    
    Args:
        output_dir: Output directory
        test_id: Test ID
        data_type: Type of data/visualization
        message: Optional specific message for the placeholder
        
    Returns:
        Path to placeholder image or None
    """
    if not message:
        message = f"No data available for {data_type} visualization"
    
    # Check if placeholders are enabled
    if _is_placeholder_enabled():
        logging.warning(f"{message} - generating placeholder")
        return generate_placeholder(output_dir, test_id, message)
    else:
        logging.warning(f"{message} - no placeholder generated (disabled by configuration)")
        return None

def configure_matplotlib_backend():
    """
    Configure matplotlib to work in any environment.
    Forces Agg backend for thread safety and headless operation.
    
    Returns:
        Matplotlib pyplot module or None if matplotlib not available
    """
    if not HAS_MATPLOTLIB:
        logging.error("Matplotlib not available - cannot configure backend")
        return None
        
    # Force Agg backend to avoid tkinter thread issues completely
    matplotlib.use('Agg', force=True)
    
    # Configure global settings
    plt.rcParams['figure.max_open_warning'] = 50  # Prevent warnings for many figures
    plt.rcParams['font.size'] = 10  # Readable default font size
    plt.rcParams['figure.dpi'] = 100  # Default DPI
    
    return plt

def save_figure_with_cleanup(fig, image_path: str, dpi: int = 100) -> Optional[str]:
    """
    Saves figure with guaranteed cleanup, even in error paths.
    
    Args:
        fig: Matplotlib figure
        image_path: Path to save the image
        dpi: Resolution in dots per inch
        
    Returns:
        Path to the saved image or None if saving failed
    """
    if not HAS_MATPLOTLIB:
        logging.error("Matplotlib not available - cannot save figure")
        return None
        
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        
        # Save figure with specified DPI
        fig.savefig(image_path, bbox_inches='tight', dpi=dpi)
        
        # Verify file exists and has content
        if os.path.exists(image_path) and os.path.getsize(image_path) > 100:
            return image_path
        else:
            logging.warning(f"Generated image file is invalid or too small: {image_path}")
            return None
    except Exception as e:
        logging.error(f"Error saving figure: {str(e)}")
        return None
    finally:
        # Always close figure to free memory, even if save fails
        plt.close(fig)

def get_visualization_path(output_dir: str, test_id: str, 
                         visualization_type: str, extension: str = "png") -> str:
    """
    Get standardized path for a visualization file.
    
    Args:
        output_dir: Output directory
        test_id: Test ID
        visualization_type: Type of visualization (component, errors, etc.)
        extension: File extension (default: png)
        
    Returns:
        Standardized path for the visualization
    """
    if HAS_PATH_UTILS:
        # Sanitize output directory to prevent nested directories
        output_dir = sanitize_base_directory(output_dir)

        # Use path utilities consistently
        return get_output_path(
            output_dir,
            test_id,
            get_standardized_filename(test_id, visualization_type, extension),
            OutputType.PRIMARY_REPORT
        )
    else:
        # Basic sanitization
        if test_id and not test_id.startswith("SXM-"):
            test_id = f"SXM-{test_id}"
        
        viz_dir = output_dir
        os.makedirs(viz_dir, exist_ok=True)
        
        # Create filename and path
        filename = f"{test_id}_{visualization_type}.{extension}"
        return os.path.join(viz_dir, filename)

def verify_image_file(image_path: str, description: str = "visualization") -> bool:
    """
    Verify that an image file is valid and not corrupted.
    
    Args:
        image_path: Path to image file
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
            with Image.open(image_path) as img:
                img.verify()
                # Get image dimensions for additional validation
                if img.width < 10 or img.height < 10:
                    logging.warning(f"Generated {description} has suspicious dimensions: {img.width}x{img.height}")
                    return False
            return True
        except Exception as e:
            logging.warning(f"Generated {description} is invalid: {str(e)}")
            return False
    
    # If we get here without PIL, assume valid based on file size
    return True

def generate_placeholder(output_dir: str, test_id: str, 
                       message: str = "Visualization not available") -> Optional[str]:
    """
    Generate a placeholder image with an error message when visualization fails.
    
    Args:
        output_dir: Output directory
        test_id: Test ID
        message: Message to display in the placeholder
        
    Returns:
        Path to placeholder file or None
    """
    # Check if placeholders are enabled
    if not _is_placeholder_enabled():
        return None
    
    # Configure matplotlib
    if not HAS_MATPLOTLIB:
        logging.error("Matplotlib not available - cannot generate placeholder")
        return None
        
    configure_matplotlib_backend()
    
    # Get path for placeholder
    placeholder_path = get_visualization_path(
        output_dir, test_id, "visualization_placeholder", "png")
    
    try:
        fig = plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, message, ha='center', va='center', fontsize=14, wrap=True)
        plt.axis('off')
        
        # Save with cleanup
        return save_figure_with_cleanup(fig, placeholder_path)
    except Exception as e:
        logging.error(f"Error generating placeholder: {str(e)}")
        if HAS_MATPLOTLIB:
            plt.close('all')  # Ensure cleanup
        return None

def get_preferred_format(default_format: str = "png") -> str:
    """
    Get the preferred image format based on available backends.
    
    Args:
        default_format: Default format to use if no preference
        
    Returns:
        String representing the preferred format ('png', 'svg', etc.)
    """
    if not HAS_MATPLOTLIB:
        return default_format
        
    # SVG is better for web viewing but requires proper backend support
    try:
        from matplotlib.backends.backend_svg import FigureCanvasSVG
        return "svg"  # SVG is available
    except ImportError:
        pass
    
    return default_format  # Fall back to default (usually PNG)

def calculate_figure_size(graph_size: int) -> Tuple[float, float]:
    """
    Calculate appropriate figure size based on graph complexity.
    
    Args:
        graph_size: Number of nodes in the graph
        
    Returns:
        Tuple of (width, height) in inches
    """
    # Base size
    width = 10
    height = 8
    
    # Adjust based on graph size
    if graph_size <= 5:
        width = 8
        height = 6
    elif graph_size <= 10:
        width = 10
        height = 8
    elif graph_size <= 20:
        width = 12
        height = 10
    else:
        # For very large graphs
        width = 14
        height = 12
        
    return (width, height)

def setup_graph_visualization(graph_size: int) -> Tuple[Any, int, int]:
    """
    Set up a figure for graph visualization with appropriate size.
    
    Args:
        graph_size: Number of nodes in the graph
        
    Returns:
        Tuple of (figure, width, height) in pixels
    """
    if not HAS_MATPLOTLIB:
        logging.error("Matplotlib not available - cannot set up graph visualization")
        return None, 0, 0
        
    # Configure matplotlib
    configure_matplotlib_backend()
    
    # Calculate figure size in inches
    width_in, height_in = calculate_figure_size(graph_size)
    
    # Create figure
    fig = plt.figure(figsize=(width_in, height_in), dpi=100)
    
    # Calculate dimensions in pixels
    width_px = int(width_in * 100)
    height_px = int(height_in * 100)
    
    return fig, width_px, height_px