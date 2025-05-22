"""
Component visualization module for the Orbit Analyzer.
Generates component relationship diagrams and error distribution visualizations
with enhanced thread safety, layout algorithms, and error handling.

This module has been updated to remove the PyGraphviz dependency while
maintaining high-quality visualization capabilities through a multi-layered
layout approach that works across all environments.
"""

import os
import sys
import logging
import traceback
import warnings
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from collections import defaultdict

# Try to import visualization utilities with fallbacks
try:
    from utils.visualization_utils import (
        verify_visualization_data,
        handle_empty_data,
        save_figure_with_cleanup,
        configure_matplotlib_backend,
        generate_placeholder,
        get_visualization_path,
        verify_image_file,
        calculate_figure_size,
        setup_graph_visualization
    )
    HAS_VISUALIZATION_UTILS = True
except ImportError:
    HAS_VISUALIZATION_UTILS = False
    logging.warning("Visualization utilities not available, using internal fallbacks")

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
    logging.warning("Path utilities not available, using internal fallbacks")

# Try to import matplotlib and related libraries with robust error handling
try:
    import matplotlib
    matplotlib.use('Agg', force=True)  # Force non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
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

# Visualization locks to prevent concurrent generation
_visualization_locks = {
    "component": threading.Lock(),
    "component_errors": threading.Lock(),
    "error_propagation": threading.Lock()
}

# Internal helpers for missing utilities
def _configure_matplotlib_backend_internal():
    """
    Internal fallback implementation to configure matplotlib backend.
    Only used if utils.visualization_utils is not available.
    """
    if not HAS_MATPLOTLIB:
        return None
        
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
    if not HAS_MATPLOTLIB:
        return None
        
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
            with Image.open(image_path) as img:
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
    if HAS_CONFIG and hasattr(Config, 'ENABLE_VISUALIZATION_PLACEHOLDERS'):
        placeholders_enabled = Config.ENABLE_VISUALIZATION_PLACEHOLDERS
    else:
        placeholders_enabled = False
        
    if not placeholders_enabled:
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
            
            # Save with cleanup
            return _save_figure_with_cleanup_internal(fig, placeholder_path)
        except Exception as e:
            logging.error(f"Error generating placeholder: {str(e)}")
            plt.close('all')  # Ensure cleanup
            return None
    else:
        return None

def _is_feature_enabled(feature_name, default=False):
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

def _is_placeholder_enabled():
    """
    Check if visualization placeholders are enabled.
    
    Returns:
        Boolean indicating if placeholders should be generated
    """
    return _is_feature_enabled('ENABLE_VISUALIZATION_PLACEHOLDERS', False)

class VisualizationStateManager:
    """
    Thread-safe state manager for visualization generation.
    
    This class provides centralized state management for visualization
    generation, including feature flag checking, locking, and cleanup.
    """
    
    def __init__(self):
        """Initialize the visualization state manager."""
        # Thread-local storage for visualization state
        self.local = threading.local()
        
        # Feature flag cache
        self.feature_flags = {}
        
        # Visualization locks
        self.locks = {
            "component": threading.Lock(),
            "component_errors": threading.Lock(),
            "error_propagation": threading.Lock()
        }
    
    def initialize_thread(self):
        """Initialize thread-local storage for the current thread."""
        if not hasattr(self.local, 'initialized'):
            self.local.initialized = True
            self.local.feature_cache = {}
            self.local.visualizations = set()
    
    def is_feature_enabled(self, feature_name, default=False):
        """
        Check if a feature is enabled with thread-safe fallback.
        
        Args:
            feature_name: Name of the feature flag in Config
            default: Default value if flag doesn't exist
            
        Returns:
            Boolean indicating if feature is enabled
        """
        self.initialize_thread()
        
        # Check thread-local cache first
        if feature_name in self.local.feature_cache:
            return self.local.feature_cache[feature_name]
        
        # Check global cache
        if feature_name in self.feature_flags:
            result = self.feature_flags[feature_name]
        else:
            # Get from config
            try:
                if HAS_CONFIG:
                    result = getattr(Config, feature_name, default)
                else:
                    result = default
            except Exception:
                # If config can't be accessed, use default
                result = default
            
            # Cache in global cache
            self.feature_flags[feature_name] = result
        
        # Cache in thread-local storage
        self.local.feature_cache[feature_name] = result
        
        return result
    
    def is_placeholder_enabled(self):
        """
        Check if visualization placeholders are enabled.
        
        Returns:
            Boolean indicating if placeholders should be generated
        """
        return self.is_feature_enabled('ENABLE_VISUALIZATION_PLACEHOLDERS', False)
    
    def acquire_lock(self, visualization_type):
        """
        Acquire lock for a specific visualization type.
        
        Args:
            visualization_type: Type of visualization
            
        Returns:
            True if lock acquired, False otherwise
        """
        if visualization_type not in self.locks:
            logging.warning(f"Unknown visualization type for lock: {visualization_type}")
            return False
            
        self.locks[visualization_type].acquire()
        return True
    
    def release_lock(self, visualization_type):
        """
        Release lock for a specific visualization type.
        
        Args:
            visualization_type: Type of visualization
        """
        if visualization_type not in self.locks:
            logging.warning(f"Unknown visualization type for lock: {visualization_type}")
            return
            
        try:
            self.locks[visualization_type].release()
        except RuntimeError:
            # Lock may not be held, which is fine
            pass
    
    def cleanup(self):
        """Clean up resources for the current thread."""
        if hasattr(self.local, 'initialized'):
            if hasattr(self.local, 'feature_cache'):
                delattr(self.local, 'feature_cache')
            if hasattr(self.local, 'visualizations'):
                delattr(self.local, 'visualizations')
            delattr(self.local, 'initialized')

# Singleton instance
viz_state_manager = VisualizationStateManager()

class ComponentVisualizer:
    """
    Visualizer for component relationships and error propagation.
    Generates visualizations without relying on PyGraphviz for improved compatibility.
    """
    
    def __init__(self, component_schema_path: str = None, component_graph: Any = None):
        """
        Initialize the visualizer with component schema or graph.
        
        Args:
            component_schema_path: Path to the component schema JSON file
            component_graph: Existing component graph (optional)
        """
        # Configure matplotlib backend
        self._configure_matplotlib_backend()
        
        # Load schema if path provided
        self.component_schema = self._load_schema(component_schema_path) if component_schema_path else {}
        
        # Use provided graph or build from schema
        if component_graph is not None:
            self.component_graph = component_graph
        else:
            self.component_graph = self._build_component_graph()
            
        # Primary issue component (will be set later if needed)
        self.primary_issue_component = None
        
        # Define component colors for consistency
        self.component_colors = {
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
        
        # Default color for unknown components
        self.default_color = "#95a5a6"  # Light gray
        
        # Special color for root cause component
        self.root_cause_color = "#d81b60"  # Magenta/red
    
    def _configure_matplotlib_backend(self):
        """Configure matplotlib to work in any environment."""
        if HAS_VISUALIZATION_UTILS:
            return configure_matplotlib_backend()
        else:
            return _configure_matplotlib_backend_internal()
    
    def _load_schema(self, schema_path: str) -> Dict:
        """
        Load the component schema from a JSON file.
        
        Args:
            schema_path: Path to schema file
            
        Returns:
            Component schema dictionary or empty schema if not found
        """
        if not schema_path or not os.path.exists(schema_path):
            logging.warning(f"Component schema file not found: {schema_path}")
            return {}
            
        try:
            import json
            with open(schema_path, 'r', encoding='utf-8') as f:
                schema = json.load(f)
            logging.info(f"Loaded component schema from {schema_path}")
            return schema
        except Exception as e:
            logging.error(f"Error loading component schema: {str(e)}")
            return {}
    
    def _build_component_graph(self) -> Any:
        """
        Build a directed graph of component relationships from schema.
        
        Returns:
            NetworkX DiGraph of component relationships or None if NetworkX not available
        """
        if not HAS_NETWORKX:
            logging.error("NetworkX not available - cannot build component graph")
            return None
            
        G = nx.DiGraph()
        
        # Add components as nodes
        for component in self.component_schema.get("components", []):
            component_id = component.get("id")
            if component_id:
                G.add_node(component_id, **component)
        
        # Add data flows as edges
        for flow in self.component_schema.get("dataFlows", []):
            source = flow.get("source")
            target = flow.get("target")
            if source and target:
                G.add_edge(source, target, **flow)
        
        # Add parent-child relationships if not already present
        for component in self.component_schema.get("components", []):
            component_id = component.get("id")
            parent = component.get("parent")
            if component_id and parent and not G.has_edge(parent, component_id):
                G.add_edge(parent, component_id, relationship="parent-child")
        
        return G
    
    def _get_graph_layout(self, G):
        """
        Get a layout for the graph using available algorithms with robust fallbacks.
        Completely removes dependency on PyGraphviz while maintaining visualization quality.
        
        Args:
            G: NetworkX graph
            
        Returns:
            Dictionary of node positions
        """
        if not HAS_NETWORKX:
            logging.error("NetworkX not available - cannot compute graph layout")
            return {}
            
        # Check graph size to optimize layout approach
        node_count = G.number_of_nodes()
        if node_count == 0:
            return {}
        
        # For very small graphs, spring layout is sufficient and fast
        if node_count <= 3:
            return nx.spring_layout(G, seed=42)
        
        # First attempt: Try pydot (part of NetworkX)
        if HAS_GRAPHVIZ:
            try:
                # Silence warning messages from Pydot
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    return graphviz_layout(G, prog='dot')
            except Exception as e:
                logging.debug(f"Pydot layout failed ({str(e)}), trying next option")
        
        # Second attempt: Try spectral layout (good for component relationships)
        try:
            # Spectral layout works well for tree-like structures
            return nx.spectral_layout(G)
        except Exception as e:
            logging.debug(f"Spectral layout failed ({str(e)}), trying next option")
        
        # Third attempt: Try shell layout (good for visualizing hierarchies)
        try:
            # Group nodes by type or relationships
            groups = []
            seen = set()
            
            # Create groups based on node types or importance
            type_groups = defaultdict(list)
            for node in G.nodes():
                node_type = G.nodes[node].get("type", "unknown")
                type_groups[node_type].append(node)
                
            # Add groups in order of importance
            for group_type in ["application", "proxy", "platform", "unknown"]:
                if group_type in type_groups and type_groups[group_type]:
                    groups.append(type_groups[group_type])
                    seen.update(type_groups[group_type])
            
            # Add any remaining nodes
            remaining = [node for node in G.nodes() if node not in seen]
            if remaining:
                groups.append(remaining)
                
            # Only use shell layout if we have valid groups
            if groups:
                return nx.shell_layout(G, groups)
        except Exception as e:
            logging.debug(f"Shell layout failed ({str(e)}), falling back to spring layout")
        
        # Final fallback: Enhanced spring layout with optimized parameters
        return nx.spring_layout(
            G, 
            k=0.3 + (0.1 / max(node_count, 1)),  # Dynamic spacing based on node count
            iterations=100,                      # More iterations for better layout
            seed=42                              # Consistent layout between runs
        )
    
    def _save_figure_with_cleanup(self, fig, image_path, dpi=100):
        """
        Save figure and ensure proper cleanup to prevent memory leaks.
        
        Args:
            fig: Matplotlib figure
            image_path: Path to save the image
            dpi: Resolution in dots per inch
            
        Returns:
            Path to the saved image
        """
        if HAS_VISUALIZATION_UTILS:
            return save_figure_with_cleanup(fig, image_path, dpi)
        else:
            return _save_figure_with_cleanup_internal(fig, image_path, dpi)
    
    def _verify_image(self, image_path, description="visualization"):
        """
        Verify that an image is valid and not corrupted.
        
        Args:
            image_path: Path to the image file
            description: Description for logging
            
        Returns:
            True if valid, False otherwise
        """
        if HAS_VISUALIZATION_UTILS:
            return verify_image_file(image_path, description)
        else:
            return _verify_image_internal(image_path, description)
    
    def _get_component_name(self, component_id: str) -> str:
        """
        Get a friendly name for a component.
        
        Args:
            component_id: Component identifier
            
        Returns:
            Display name for the component from schema or the ID itself
        """
        # Try to get name from schema
        for component in self.component_schema.get("components", []):
            if component.get("id") == component_id:
                return component.get("name", component_id.upper())
        
        # Return uppercase ID as fallback
        return component_id.upper()
    
    def _get_color_for_component(self, component_id: str) -> str:
        """
        Get color for a specific component.
        
        Args:
            component_id: Component identifier
            
        Returns:
            Hex color code for the component
        """
        # Special highlight for primary issue component
        if component_id == self.primary_issue_component:
            return self.root_cause_color
            
        # Get from color dictionary, or use default
        return self.component_colors.get(component_id, self.default_color)
    
    def _calculate_figure_size(self, G) -> Tuple[float, float]:
        """
        Calculate appropriate figure size based on graph complexity.
        
        Args:
            G: NetworkX graph
            
        Returns:
            Tuple of (width, height) in inches
        """
        # Base size
        width = 10
        height = 8
        
        if not HAS_NETWORKX or G is None:
            return (width, height)
            
        # Get node count
        node_count = G.number_of_nodes()
        
        # Adjust based on graph size
        if node_count <= 5:
            width = 8
            height = 6
        elif node_count <= 10:
            width = 10
            height = 8
        elif node_count <= 20:
            width = 12
            height = 10
        else:
            # For very large graphs
            width = 14
            height = 12
            
        return (width, height)
    
    def set_primary_issue_component(self, component_id):
        """
        Mark a component as the primary issue component.
        
        Args:
            component_id: The component identified as root cause
        """
        self.primary_issue_component = component_id
    
    def generate_component_relationship_diagram(self, output_dir: str, test_id: str = None, width=800, height=600) -> str:
        """
        Generate a component relationship diagram using advanced layout techniques.
        
        Args:
            output_dir: Directory to save the diagram
            test_id: Test ID for the filename (optional)
            width: Width of the diagram in pixels (unused, maintained for compatibility)
            height: Height of the diagram in pixels (unused, maintained for compatibility)
            
        Returns:
            Path to the generated image
        """
        # Check if component relationships visualization is enabled
        if not _is_feature_enabled('ENABLE_COMPONENT_RELATIONSHIPS', True):
            logging.info(f"Component relationships visualization is disabled by feature flag")
            return None
        
        # Acquire lock to prevent concurrent generation
        with _visualization_locks["component"]:
            # Configure matplotlib backend
            self._configure_matplotlib_backend()
            
            # Sanitize output directory and use path utilities
            if HAS_PATH_UTILS:
                output_dir = sanitize_base_directory(output_dir, "supporting_images")
                image_path = get_output_path(
                    output_dir, 
                    test_id or "default", 
                    get_standardized_filename(test_id or "default", "component_relationships", "png"),
                    OutputType.VISUALIZATION
                )
            else:
                image_path = _get_visualization_path_internal(
                    output_dir, test_id or "default", "component_relationships", "png")
            
            try:
                # Check if we have components to visualize
                if not HAS_NETWORKX or not self.component_graph or self.component_graph.number_of_nodes() == 0:
                    logging.warning("No components to visualize, creating a placeholder")
                    
                    # If placeholders are disabled, return None
                    if not _is_placeholder_enabled():
                        return None
                    
                    # Create a placeholder visualization
                    fig = plt.figure(figsize=(8, 6))
                    plt.text(0.5, 0.5, "No component relationships to visualize\nCheck component identification and log analysis",
                            ha='center', va='center', fontsize=14)
                    plt.axis('off')
                    return self._save_figure_with_cleanup(fig, image_path)
                
                # Calculate appropriate figure size
                fig_size = self._calculate_figure_size(self.component_graph)
                fig = plt.figure(figsize=fig_size, dpi=100)
                
                # Get layout with optimized algorithm selection
                pos = self._get_graph_layout(self.component_graph)
                
                # Draw nodes with enhanced appearance
                node_colors = []
                node_sizes = []
                for node in self.component_graph.nodes():
                    # Get color based on component
                    color = self._get_color_for_component(node)
                    node_colors.append(color)
                    
                    # Size based on node importance (degree)
                    size = 2500 + (self.component_graph.degree(node) * 200)
                    node_sizes.append(size)
                
                # Draw edges with improved styling
                nx.draw_networkx_edges(
                    self.component_graph, pos, 
                    arrowsize=15, 
                    width=1.5,
                    edge_color="#555555",
                    alpha=0.7,
                    connectionstyle='arc3,rad=0.1'  # Curved edges for better visibility
                )
                
                # Draw nodes with larger size to prevent label overlap
                nx.draw_networkx_nodes(
                    self.component_graph, pos,
                    node_size=node_sizes,
                    node_color=node_colors,
                    alpha=0.9,
                    edgecolors='black',  # Add border for better contrast
                    linewidths=1
                )
                
                # Draw node labels with increased offset from the node center
                label_dict = {node: self._get_component_name(node) for node in self.component_graph.nodes()}
                nx.draw_networkx_labels(
                    self.component_graph, pos,
                    labels=label_dict,
                    font_size=11,
                    font_weight='bold',
                    font_color='black',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=4)
                )
                
                # Add edge labels with better formatting
                edge_labels = {
                    (u, v): d.get("dataType", "").replace("_", " ")
                    for u, v, d in self.component_graph.edges(data=True)
                    if "dataType" in d and d["dataType"]  # Only add labels if dataType exists and is not empty
                }
                
                # Only add edge labels if there are any
                if edge_labels:
                    nx.draw_networkx_edge_labels(
                        self.component_graph, pos,
                        edge_labels=edge_labels,
                        font_size=9,
                        font_color='darkblue',
                        alpha=0.7
                    )
                
                # Add legend for primary issue component if set
                if self.primary_issue_component:
                    # Create legend with primary component highlighted
                    root_cause_patch = mpatches.Patch(
                        color=self.root_cause_color, 
                        label=f'Root Cause: {self._get_component_name(self.primary_issue_component)}'
                    )
                    plt.legend(handles=[root_cause_patch], loc='upper right')
                
                # Add title with test ID
                if test_id:
                    plt.title(f"Component Relationships for {test_id}")
                else:
                    plt.title("Component Relationships")
                
                # Remove axis for cleaner look
                plt.axis('off')
                
                # Save with proper cleanup
                return self._save_figure_with_cleanup(fig, image_path)
                
            except Exception as e:
                logging.error(f"Error generating component diagram: {str(e)}")
                traceback.print_exc()
                
                # Create error visualization placeholder if enabled
                if _is_placeholder_enabled():
                    fig = plt.figure(figsize=(8, 6))
                    plt.text(0.5, 0.5, f"Error generating component diagram:\n{str(e)}", 
                            ha='center', va='center', fontsize=12)
                    plt.axis('off')
                    return self._save_figure_with_cleanup(fig, image_path)
                return None
            finally:
                # Ensure cleanup even on error
                if HAS_MATPLOTLIB:
                    plt.close('all')

    def generate_component_error_distribution(self, output_dir: str, test_id: str, 
                                         component_summary: List[Dict] = None, 
                                         clusters: Dict = None,
                                         primary_component: str = None) -> str:
        """
        Generate a visualization of error distribution across components.
        
        Args:
            output_dir: Directory to save the visualization
            test_id: Test ID
            component_summary: Summary of components involved
            clusters: Dictionary of error clusters (optional)
            primary_component: Primary issue component
            
        Returns:
            Path to the generated visualization file
        """
        # Check if component distribution visualization is enabled
        if not _is_feature_enabled('ENABLE_COMPONENT_DISTRIBUTION', True):
            logging.info(f"Component distribution visualization is disabled by feature flag")
            return None
        
        # Acquire lock to prevent concurrent generation
        with _visualization_locks["component_errors"]:
            # Configure matplotlib backend
            self._configure_matplotlib_backend()
            
            # Store primary component if provided
            if primary_component:
                self.primary_issue_component = primary_component
            
            # Sanitize output directory and use path utilities
            if HAS_PATH_UTILS:
                output_dir = sanitize_base_directory(output_dir, "supporting_images")
                # Both visualization filenames for backward compatibility
                distribution_path = get_output_path(
                    output_dir, 
                    test_id, 
                    get_standardized_filename(test_id, "component_distribution", "png"),
                    OutputType.VISUALIZATION
                )
                
                errors_path = get_output_path(
                    output_dir, 
                    test_id, 
                    get_standardized_filename(test_id, "component_errors", "png"),
                    OutputType.VISUALIZATION
                )
            else:
                distribution_path = _get_visualization_path_internal(
                    output_dir, test_id, "component_distribution", "png")
                errors_path = _get_visualization_path_internal(
                    output_dir, test_id, "component_errors", "png")
            
            try:
                # If component_summary is missing or empty, create a placeholder
                if not component_summary or len(component_summary) == 0:
                    logging.warning("No component summary data available")
                    
                    # Check if we can extract from clusters
                    component_counts = {}
                    if clusters:
                        for cluster_id, errors in clusters.items():
                            for error in errors:
                                if isinstance(error, dict) and 'component' in error:
                                    comp = error['component']
                                    if comp not in component_counts:
                                        component_counts[comp] = 0
                                    component_counts[comp] += 1
                    
                    # If we extracted some component data, use it
                    if component_counts:
                        logging.info(f"Extracted component data from clusters: {component_counts}")
                        component_summary = [
                            {"id": comp, "error_count": count}
                            for comp, count in component_counts.items()
                        ]
                    else:
                        # If still no data, create placeholder or return None
                        if not _is_placeholder_enabled():
                            return None
                        
                        # Create a placeholder visualization with instructions
                        fig = plt.figure(figsize=(8, 6))
                        plt.text(0.5, 0.5, "No component error data available\nCheck component identification in log analysis",
                                ha='center', va='center', fontsize=14)
                        plt.axis('off')
                        
                        # Save to both paths for compatibility
                        self._save_figure_with_cleanup(fig, distribution_path)
                        return self._save_figure_with_cleanup(fig, errors_path)
                
                # Extract component data with error counts
                components = []
                error_counts = []
                colors = []
                
                # Process each component
                for component in component_summary:
                    # Get component ID
                    component_id = component.get('id', 'unknown')
                    
                    # Get error count
                    error_count = component.get('error_count', 0)
                    
                    # Skip components with no errors or unknown components if we have others
                    if error_count <= 0 or (component_id == 'unknown' and len(component_summary) > 1):
                        continue
                    
                    components.append(component_id.upper())
                    error_counts.append(error_count)
                    
                    # Determine color based on whether this is the primary component
                    if component_id == self.primary_issue_component:
                        colors.append(self.root_cause_color)
                    else:
                        colors.append(self._get_color_for_component(component_id))
                
                # If no components with errors, create placeholder or return None
                if not components:
                    logging.warning("No components with errors found")
                    
                    if not _is_placeholder_enabled():
                        return None
                    
                    # Create a placeholder visualization with instructions
                    fig = plt.figure(figsize=(8, 6))
                    plt.text(0.5, 0.5, "No components with errors found\nCheck component identification in log analysis",
                            ha='center', va='center', fontsize=14)
                    plt.axis('off')
                    
                    # Save to both paths for compatibility
                    self._save_figure_with_cleanup(fig, distribution_path)
                    return self._save_figure_with_cleanup(fig, errors_path)
                
                # Sort by error count (descending)
                sorted_indices = sorted(range(len(error_counts)), key=lambda k: error_counts[k], reverse=True)
                components = [components[i] for i in sorted_indices]
                error_counts = [error_counts[i] for i in sorted_indices]
                colors = [colors[i] for i in sorted_indices]
                
                # Create a horizontal bar chart
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Create bars
                bars = ax.barh(components, error_counts, color=colors)
                
                # Add value labels
                for i, bar in enumerate(bars):
                    width = bar.get_width()
                    ax.text(
                        width + 0.5, 
                        bar.get_y() + bar.get_height()/2,
                        str(error_counts[i]),
                        ha='left',
                        va='center',
                        fontweight='bold'
                    )
                
                # Add labels and title
                ax.set_xlabel('Error Count')
                ax.set_ylabel('Component')
                ax.set_title(f'Error Distribution by Component for {test_id}')
                
                # Add grid lines
                ax.grid(axis='x', linestyle='--', alpha=0.7)
                
                # Adjust layout for better fit
                plt.tight_layout()
                
                # Save to both paths for backward compatibility
                self._save_figure_with_cleanup(fig, distribution_path)
                path = self._save_figure_with_cleanup(fig, errors_path)
                
                # Verify images
                self._verify_image(distribution_path, "component distribution")
                self._verify_image(errors_path, "component errors")
                
                return path
                
            except Exception as e:
                logging.error(f"Error generating component error distribution: {str(e)}")
                traceback.print_exc()
                
                # Create error visualization placeholder if enabled
                if _is_placeholder_enabled():
                    fig = plt.figure(figsize=(8, 6))
                    plt.text(0.5, 0.5, f"Error generating component error distribution:\n{str(e)}", 
                            ha='center', va='center', fontsize=12)
                    plt.axis('off')
                    
                    # Save to both paths for compatibility
                    self._save_figure_with_cleanup(fig, distribution_path)
                    return self._save_figure_with_cleanup(fig, errors_path)
                return None
            finally:
                # Ensure cleanup even on error
                if HAS_MATPLOTLIB:
                    plt.close('all')

    def generate_error_propagation_diagram(self, output_dir: str, test_id: str, error_graph=None) -> str:
        """
        Generate a visualization of error propagation paths.
        
        Args:
            output_dir: Directory to save the visualization
            test_id: Test ID for the filename
            error_graph: Error graph (NetworkX DiGraph or dictionary)
            
        Returns:
            Path to the generated visualization file
        """
        # Check if error propagation visualization is enabled
        if not _is_feature_enabled('ENABLE_ERROR_PROPAGATION', False):
            logging.info(f"Error propagation visualization is disabled by feature flag")
            return None
        
        # Acquire lock to prevent concurrent generation
        with _visualization_locks["error_propagation"]:
            # Configure matplotlib backend
            self._configure_matplotlib_backend()
            
            # Sanitize output directory and use path utilities
            if HAS_PATH_UTILS:
                output_dir = sanitize_base_directory(output_dir, "supporting_images")
                image_path = get_output_path(
                    output_dir, 
                    test_id, 
                    get_standardized_filename(test_id, "error_propagation", "png"),
                    OutputType.VISUALIZATION
                )
            else:
                image_path = _get_visualization_path_internal(
                    output_dir, test_id, "error_propagation", "png")
            
            try:
                # Check if NetworkX is available
                if not HAS_NETWORKX:
                    logging.error("NetworkX not available - cannot generate error propagation diagram")
                    
                    if not _is_placeholder_enabled():
                        return None
                    
                    # Create a placeholder visualization
                    fig = plt.figure(figsize=(8, 6))
                    plt.text(0.5, 0.5, "Error propagation visualization not available\nNetworkX library is required",
                            ha='center', va='center', fontsize=14)
                    plt.axis('off')
                    return self._save_figure_with_cleanup(fig, image_path)
                
                # Handle different formats for error_graph
                if isinstance(error_graph, dict):
                    # Convert dictionary to NetworkX graph
                    G = nx.DiGraph()
                    
                    # Add nodes
                    for node_data in error_graph.get("nodes", []):
                        if "id" in node_data:
                            node_id = node_data["id"]
                            G.add_node(node_id, **{k: v for k, v in node_data.items() if k != "id"})
                    
                    # Add edges
                    for edge_data in error_graph.get("edges", []):
                        if "source" in edge_data and "target" in edge_data:
                            source = edge_data["source"]
                            target = edge_data["target"]
                            G.add_edge(source, target, **{k: v for k, v in edge_data.items() if k not in ["source", "target"]})
                elif isinstance(error_graph, nx.DiGraph):
                    # Use the provided NetworkX graph directly
                    G = error_graph
                else:
                    # No usable graph
                    logging.warning("No error propagation data available")
                    
                    if not _is_placeholder_enabled():
                        return None
                    
                    # Create a placeholder visualization
                    fig = plt.figure(figsize=(8, 6))
                    plt.text(0.5, 0.5, "No error propagation data available\nCheck error analysis and component relationships", 
                            ha='center', va='center', fontsize=14)
                    plt.axis('off')
                    return self._save_figure_with_cleanup(fig, image_path)
                
                # Check if graph is empty
                if not G or G.number_of_nodes() == 0:
                    logging.warning("Error graph has no nodes")
                    
                    if not _is_placeholder_enabled():
                        return None
                    
                    # Create a placeholder visualization
                    fig = plt.figure(figsize=(8, 6))
                    plt.text(0.5, 0.5, "No error propagation data available\nCheck error analysis and component relationships", 
                            ha='center', va='center', fontsize=14)
                    plt.axis('off')
                    return self._save_figure_with_cleanup(fig, image_path)
                
                # Create figure with appropriate size
                fig_size = self._calculate_figure_size(G)
                fig = plt.figure(figsize=fig_size, dpi=100)
                
                # Get layout for the graph
                pos = self._get_graph_layout(G)
                
                # Determine node colors based on attributes
                node_colors = []
                for node in G.nodes():
                    node_data = G.nodes[node]
                    if node_data.get('is_root_cause', False):
                        node_colors.append('red')
                    elif node_data.get('is_symptom', False):
                        node_colors.append('orange')
                    else:
                        node_colors.append('lightblue')
                
                # Draw edges with weights determining thickness
                edge_widths = [G[u][v].get('weight', 1.0) * 2 for u, v in G.edges()]
                nx.draw_networkx_edges(
                    G, pos, 
                    edge_color='gray', 
                    alpha=0.7, 
                    arrows=True,
                    arrowsize=15,
                    width=edge_widths,
                    connectionstyle='arc3,rad=0.1'  # Curved edges
                )
                
                # Draw nodes with size reflecting importance
                nx.draw_networkx_nodes(
                    G, pos,
                    node_color=node_colors,
                    node_size=800,
                    alpha=0.9,
                    edgecolors='black',
                    linewidths=1
                )
                
                # Create labels with truncated error text
                labels = {}
                for node in G.nodes():
                    # Get error text or component as label
                    error_text = G.nodes[node].get('text', str(node))
                    component = G.nodes[node].get('component', 'unknown')
                    
                    # Create a short label with component prefix
                    if len(error_text) > 30:
                        label = f"{component.upper()}: {error_text[:27]}..."
                    else:
                        label = f"{component.upper()}: {error_text}"
                    
                    labels[node] = label
                
                # Draw labels
                nx.draw_networkx_labels(
                    G, pos,
                    labels=labels,
                    font_size=9,
                    font_weight='bold',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2)
                )
                
                # Add title
                plt.title(f"Error Propagation Analysis - {test_id}", fontsize=14)
                plt.axis('off')
                plt.tight_layout()
                
                # Add legend
                red_patch = mpatches.Patch(color='red', label='Root Cause')
                orange_patch = mpatches.Patch(color='orange', label='Symptom')
                lightblue_patch = mpatches.Patch(color='lightblue', label='Intermediate')
                plt.legend(handles=[red_patch, orange_patch, lightblue_patch], loc='upper right')
                
                # Save figure with cleanup
                return self._save_figure_with_cleanup(fig, image_path)
                
            except Exception as e:
                logging.error(f"Error generating error propagation diagram: {str(e)}")
                traceback.print_exc()
                
                # Create error visualization placeholder if enabled
                if _is_placeholder_enabled():
                    fig = plt.figure(figsize=(8, 6))
                    plt.text(0.5, 0.5, f"Error generating error propagation diagram:\n{str(e)}", 
                            ha='center', va='center', fontsize=12)
                    plt.axis('off')
                    return self._save_figure_with_cleanup(fig, image_path)
                return None
            finally:
                # Ensure cleanup even on error
                if HAS_MATPLOTLIB:
                    plt.close('all')

# Function for backward compatibility with reports/visualizations.py
def generate_component_visualization(output_dir, test_id, component_summary, relationships=None, primary_component=None):
    """
    Generate a component relationship diagram.
    This function is provided for compatibility with reports/visualizations.py.
    
    Args:
        output_dir: Directory to save the diagram
        test_id: Test ID for the filename
        component_summary: Component summary data
        relationships: Component relationships (optional)
        primary_component: Primary issue component (optional)
        
    Returns:
        Path to the generated visualization or None if generation fails
    """
    # Create visualizer
    visualizer = ComponentVisualizer()
    
    # Build component graph from component_summary and relationships
    if HAS_NETWORKX:
        G = nx.DiGraph()
        
        # Add components as nodes
        for component in component_summary:
            component_id = component.get('id')
            if component_id:
                G.add_node(component_id, **component)
        
        # Add relationships as edges
        if relationships:
            for rel in relationships:
                source = rel.get('source')
                target = rel.get('target')
                if source and target:
                    G.add_edge(source, target, **rel)
        
        # Set graph and primary component
        visualizer.component_graph = G
    
    # Set primary issue component if provided
    if primary_component:
        visualizer.primary_issue_component = primary_component
    
    # Generate visualization
    return visualizer.generate_component_relationship_diagram(output_dir, test_id)

# Function for backward compatibility with reports/visualizations.py
def generate_component_error_distribution(output_dir, test_id, component_summary=None, clusters=None, primary_component=None):
    """
    Generate a component error distribution visualization.
    This function is provided for compatibility with reports/visualizations.py.
    
    Args:
        output_dir: Directory to save the visualization
        test_id: Test ID for the filename
        component_summary: Component summary data (optional)
        clusters: Error clusters (optional)
        primary_component: Primary issue component (optional)
        
    Returns:
        Path to the generated visualization or None if generation fails
    """
    # Create visualizer
    visualizer = ComponentVisualizer()
    
    # Set primary issue component if provided
    if primary_component:
        visualizer.primary_issue_component = primary_component
    
    # Generate visualization
    return visualizer.generate_component_error_distribution(output_dir, test_id, component_summary, clusters, primary_component)

# Function for backward compatibility with reports/visualizations.py
def generate_error_propagation_diagram(output_dir, test_id, error_graph=None):
    """
    Generate an error propagation diagram.
    This function is provided for compatibility with reports/visualizations.py.
    
    Args:
        output_dir: Directory to save the visualization
        test_id: Test ID for the filename
        error_graph: Error graph data (optional)
        
    Returns:
        Path to the generated visualization or None if generation fails
    """
    # Create visualizer
    visualizer = ComponentVisualizer()
    
    # Generate visualization
    return visualizer.generate_error_propagation_diagram(output_dir, test_id, error_graph)