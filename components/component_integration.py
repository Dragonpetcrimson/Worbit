"""
components/component_integration.py - Component integration for analysis and visualization

This module integrates component analysis, relationship visualization, and error clustering
to provide comprehensive component-aware analysis.
"""

import os
import logging
import traceback
import json
import copy
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple, Union
import threading

# Import core modules
try:
    from components.component_analyzer import ComponentAnalyzer
    from components.component_visualizer import ComponentVisualizer, _is_feature_enabled
    from components.context_aware_clusterer import ContextAwareClusterer
except ImportError as e:
    logging.warning(f"Could not import component modules: {str(e)}")
    # Define placeholder classes if imports fail
    class ComponentAnalyzer:
        def __init__(self, *args, **kwargs): pass
        def enrich_log_entries_with_components(self, log_entries): return log_entries
        def analyze_component_failures(self, errors): return {}
    
    class ComponentVisualizer:
        def __init__(self, *args, **kwargs): pass
        def generate_component_relationship_diagram(self, *args, **kwargs): return ""
        def generate_error_propagation_diagram(self, *args, **kwargs): return ""
        def generate_component_error_distribution(self, *args, **kwargs): return ""
    
    class ContextAwareClusterer:
        def __init__(self, *args, **kwargs): pass
        def cluster_errors(self, errors, num_clusters=None): return {}
        def get_root_cause_errors(self, clusters): return []
        def get_causality_paths(self): return []
        def export_error_graph(self, output_path, test_id): return ""

# Import path utilities
try:
    from utils.path_utils import (
        get_output_path,
        OutputType,
        normalize_test_id,
        get_standardized_filename,
        sanitize_base_directory
    )
except ImportError:
    # Define fallbacks if path utilities unavailable
    def get_output_path(base_dir, test_id, filename, output_type=None):
        return os.path.join(base_dir, filename)
    
    def normalize_test_id(test_id):
        return test_id
    
    def get_standardized_filename(test_id, file_type, extension):
        return f"{test_id}_{file_type}.{extension}"
    
    def sanitize_base_directory(base_dir, expected_subdir=None):
        return base_dir
    
    class OutputType:
        JSON_DATA = "json"
        VISUALIZATION = "visualization"

# Thread-local storage for visualization state
_visualization_local = threading.local()

class ComponentIntegration:
    """
    Integration layer for component relationship analysis, visualizations,
    and enhanced error clustering.
    """
    
    def __init__(self, component_schema_path: str):
        """
        Initialize with component schema.
        
        Args:
            component_schema_path: Path to component schema JSON
        """
        # Initialize thread-local storage
        if not hasattr(_visualization_local, 'feature_cache'):
            _visualization_local.feature_cache = {}
            
        # Initialize component modules
        self.analyzer = ComponentAnalyzer(component_schema_path)
        self.visualizer = ComponentVisualizer(component_schema_path)
        self.clusterer = ContextAwareClusterer(component_schema_path)
        
        # Store schema path
        self.schema_path = component_schema_path
        
        # Track any feature caching
        self.feature_cache = {}
    
    def _is_feature_enabled(self, feature_name, default=False):
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
        from config import Config
        result = getattr(Config, feature_name, default)
        
        # Cache for future use
        _visualization_local.feature_cache[feature_name] = result
        
        return result
    
    def analyze_logs(self, log_entries: List[Any], errors: List[Any], output_dir: str, test_id: str) -> Dict[str, Any]:
        """
        Perform comprehensive component-aware analysis with simplified component identification.
        
        Args:
            log_entries: List of log entries from all files
            errors: List of detected errors
            output_dir: Directory for output files
            test_id: Test ID for file naming
            
        Returns:
            Analysis results and paths to generated files
        """
        # Initialize thread-local storage for this thread
        if not hasattr(_visualization_local, 'feature_cache'):
            _visualization_local.feature_cache = {}
        
        results = {
            "test_id": test_id,
            "timestamp": datetime.now().isoformat(),
            "analysis_files": {},
            "metrics": {}
        }
        
        try:
            # Step 1: Enrich logs with component information
            log_entries = self.analyzer.enrich_log_entries_with_components(log_entries)
            errors = self.analyzer.enrich_log_entries_with_components(errors)
            
            # Set the JSON directory
            json_dir = os.path.join(output_dir, "json")
            os.makedirs(json_dir, exist_ok=True)
            
            # Step 2: Generate baseline component relationship diagram
            if self._is_feature_enabled("ENABLE_COMPONENT_RELATIONSHIPS", True):
                logging.info("Generating component relationship diagram")
                try:
                    relationship_path = self.visualizer.generate_component_relationship_diagram(
                        output_dir, test_id
                    )
                    if relationship_path:
                        results["analysis_files"]["component_diagram"] = relationship_path
                except Exception as e:
                    logging.error(f"Error generating component diagram: {str(e)}")
            
            # Step 3: Analyze component errors
            component_analysis = self.analyzer.analyze_component_failures(errors)
            
            # Extract primary issue component
            primary_issue_component = component_analysis.get("root_cause_component")
            if primary_issue_component:
                # Set it for visualizations
                self.visualizer.set_primary_issue_component(primary_issue_component)
                results["primary_issue_component"] = primary_issue_component
            
            # Store analysis in results
            results["component_analysis"] = component_analysis
            results["metrics"].update(component_analysis.get("metrics", {}))
            
            # Step 4: Generate error propagation visualization
            if self._is_feature_enabled("ENABLE_ERROR_PROPAGATION", True):
                logging.info("Generating error propagation diagram")
                # Use threading to handle errors without stopping pipeline
                propagation_error = [None]
                propagation_result = [None]
                
                def generate_propagation():
                    try:
                        # Get the error graph
                        error_graph = component_analysis.get("error_graph")
                        
                        # Generate the visualization
                        propagation_result[0] = self.visualizer.generate_error_propagation_diagram(
                            output_dir, test_id, error_graph
                        )
                    except Exception as e:
                        propagation_error[0] = e
                
                # Run in thread to isolate potential tkinter issues
                thread = threading.Thread(target=generate_propagation)
                thread.daemon = True
                thread.start()
                thread.join(timeout=30)  # Wait up to 30 seconds
                
                # Check results
                if propagation_error[0]:
                    logging.error(f"Error in error propagation visualization: {str(propagation_error[0])}")
                    traceback.print_exc()
                    raise propagation_error[0]
                
                if propagation_result[0]:
                    results["analysis_files"]["error_propagation"] = propagation_result[0]
            
            # Step 5: Generate component error visualization
            if self._is_feature_enabled("ENABLE_COMPONENT_DISTRIBUTION", True):
                logging.info("Generating component error distribution")
                # Extract component summary
                component_summary = component_analysis.get("component_summary", [])
                
                # Generate the visualization
                error_chart_path = self.visualizer.generate_component_error_distribution(
                    output_dir, 
                    test_id,
                    component_summary,
                    None,  # Clusters not needed
                    primary_issue_component
                )
                
                if error_chart_path:
                    # Store both paths for backward compatibility
                    results["analysis_files"]["component_distribution"] = error_chart_path
                    results["analysis_files"]["component_errors"] = error_chart_path
            
            # Step 6: Perform context-aware error clustering
            component_aware_clusters = self.clusterer.cluster_errors(errors)
            results["enhanced_clusters"] = component_aware_clusters
            
            # Extract root cause errors
            root_cause_errors = self.clusterer.get_root_cause_errors(component_aware_clusters)
            results["root_cause_errors"] = root_cause_errors
            
            # Extract causality paths
            causality_paths = self.clusterer.get_causality_paths()
            results["causality_paths"] = causality_paths
            
            # Export error graph
            error_graph_path = self.clusterer.export_error_graph(json_dir, test_id)
            if error_graph_path:
                results["analysis_files"]["error_graph"] = error_graph_path
            
            # Save component analysis to JSON
            component_analysis_path = get_output_path(
                output_dir,
                test_id,
                get_standardized_filename(test_id, "component_analysis", "json"),
                OutputType.JSON_DATA
            )
            
            with open(component_analysis_path, 'w', encoding='utf-8') as f:
                json.dump(component_analysis, f, cls=DateTimeEncoder, indent=2)
            
            results["analysis_files"]["component_analysis"] = component_analysis_path
            
            # Save enhanced clusters to JSON
            clusters_filename = get_standardized_filename(test_id, "enhanced_clusters", "json")
            clusters_path = get_output_path(
                output_dir,
                test_id,
                clusters_filename,
                OutputType.JSON_DATA
            )
            
            with open(clusters_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "clusters": self._serialize_clusters(component_aware_clusters),
                    "root_cause_errors": [self._serialize_error(err) for err in root_cause_errors],
                    "causality_paths": [[self._serialize_error(err) for err in path] for path in causality_paths]
                }, f, cls=DateTimeEncoder, indent=2)
            
            results["analysis_files"]["enhanced_clusters"] = clusters_path
            
            # Add metrics
            results["metrics"]["enhanced_clusters"] = True
            results["metrics"]["clusters"] = len(component_aware_clusters)
            results["metrics"]["root_cause_errors"] = len(root_cause_errors)
            
            return results
            
        except Exception as e:
            logging.error(f"Error in component integration: {str(e)}")
            traceback.print_exc()
            
            # Add error information to results
            results["error"] = str(e)
            results["component_analysis"] = {}
            
            return results
    
    def _serialize_error(self, error: Dict) -> Dict:
        """
        Convert an error dict to a serializable format.
        
        Args:
            error: Error dictionary
            
        Returns:
            Serializable error dictionary
        """
        # Create a copy to avoid modifying the original
        result = copy.deepcopy(error)
        
        # Convert timestamp to string if it's a datetime
        if isinstance(result.get('timestamp'), datetime):
            result['timestamp'] = result['timestamp'].isoformat()
            
        return result
    
    def _serialize_clusters(self, clusters: Dict[int, List[Dict]]) -> Dict[str, List[Dict]]:
        """
        Convert clusters dictionary to a serializable format.
        
        Args:
            clusters: Dictionary mapping cluster IDs to lists of errors
            
        Returns:
            Serializable clusters dictionary with string keys
        """
        return {str(cluster_id): [self._serialize_error(err) for err in errors] 
                for cluster_id, errors in clusters.items()}
    
    def get_enhanced_report_data(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate additional data for enhancing test reports.
        
        Args:
            analysis_results: Results from analyze_logs
            
        Returns:
            Report enhancement data
        """
        enhancements = {
            "component_diagrams": [],
            "root_cause_component": None,
            "affected_components": [],
            "enhanced_clustering": {
                "available": False,
                "root_cause_errors": [],
                "causality_paths": []
            }
        }
        
        # Add component diagrams
        if "analysis_files" in analysis_results:
            files = analysis_results["analysis_files"]
            for key in ["component_diagram", "error_propagation", "component_distribution", "component_errors"]:
                if key in files and os.path.exists(files[key]):
                    enhancements["component_diagrams"].append({
                        "type": key,
                        "path": files[key]
                    })
        
        # Add root cause component
        if "primary_issue_component" in analysis_results:
            enhancements["root_cause_component"] = analysis_results["primary_issue_component"]
        elif "component_analysis" in analysis_results and "root_cause_component" in analysis_results["component_analysis"]:
            enhancements["root_cause_component"] = analysis_results["component_analysis"]["root_cause_component"]
        
        # Add affected components
        if "component_analysis" in analysis_results and "component_summary" in analysis_results["component_analysis"]:
            for component in analysis_results["component_analysis"]["component_summary"]:
                if component.get("id") != enhancements["root_cause_component"]:
                    enhancements["affected_components"].append(component)
        
        # Add enhanced clustering data
        if "enhanced_clusters" in analysis_results:
            enhancements["enhanced_clustering"]["available"] = True
            if "root_cause_errors" in analysis_results:
                enhancements["enhanced_clustering"]["root_cause_errors"] = analysis_results["root_cause_errors"]
            if "causality_paths" in analysis_results:
                enhancements["enhanced_clustering"]["causality_paths"] = analysis_results["causality_paths"]
        
        return enhancements

# Define DateTimeEncoder for JSON serialization
class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles datetime objects."""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)