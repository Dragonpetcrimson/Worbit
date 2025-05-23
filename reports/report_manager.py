"""
reports/report_manager.py - Orchestrates the report generation process
"""

import os
import logging
import traceback
import copy
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

from utils.path_utils import (
    get_output_path,
    OutputType,
    normalize_test_id,
    get_standardized_filename,
    sanitize_base_directory,
    cleanup_nested_directories
)
from reports.base import ReportConfig, ReportData, DateTimeEncoder, COMPONENT_FIELDS
from reports.data_preprocessor import preprocess_errors, preprocess_clusters, normalize_data
from reports.component_analyzer import build_component_analysis
from reports.json_generator import JsonReportGenerator
from reports.markdown_generator import MarkdownReportGenerator
from reports.excel_generator import ExcelReportGenerator
from reports.docx_generator import DocxReportGenerator
from reports.visualizations import VisualizationGenerator
from utils.component_verification import verify_component_preservation, count_component_fields
# Import the new json_utils module for component-aware serialization
from json_utils import serialize_with_component_awareness, serialize_to_json_file


class ReportManager:
    """Orchestrates the report generation process with enhanced component handling."""
    
    def __init__(self, config: ReportConfig):
        """Initialize the report manager."""
        self.config = config
        
        # Normalize primary_issue_component
        if self.config.primary_issue_component:
            self.config.primary_issue_component = self.config.primary_issue_component.lower()
        
        # Ensure test_id is properly formatted
        if not config.test_id.startswith("SXM-"):
            config.test_id = f"SXM-{config.test_id}"
        
        # Create output directory structures
        self.base_dir = config.output_dir
        self.json_dir = os.path.join(config.output_dir, "json") 
        self.images_dir = os.path.join(config.output_dir, "supporting_images")
        self.debug_dir = os.path.join(config.output_dir, "debug")
        
        # Create directories
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.json_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.debug_dir, exist_ok=True)
        
        # Clean up any existing nested directories to prevent accumulation
        self._cleanup_nested_directories()
        
        # Initialize report generators based on enabled flags
        self.json_generator = JsonReportGenerator(config) if config.enable_json else None
        self.visualization_generator = VisualizationGenerator(config) if config.enable_component_report else None
        self.markdown_generator = MarkdownReportGenerator(config) if config.enable_markdown else None
        self.excel_generator = ExcelReportGenerator(config) if config.enable_excel else None
        self.docx_generator = DocxReportGenerator(config) if config.enable_docx else None
    
    def _cleanup_nested_directories(self):
        """
        Clean up any existing nested directories that might have been created by previous runs.
        This prevents accumulation of errors when reports are regenerated.
        """
        cleanup_results = cleanup_nested_directories(self.base_dir)
        if sum(cleanup_results.values()) > 0:
            logging.info(f"Cleaned up {sum(cleanup_results.values())} files in nested directories")
    
    def _create_config_for_dir(self, output_dir: str) -> ReportConfig:
        """
        Create a config with a specific output directory.
        
        Args:
            output_dir: Directory to use for this config
            
        Returns:
            New config with the specified output directory
        """
        # Create a copy of the config with a new output directory
        new_config = copy.copy(self.config)
        new_config.output_dir = output_dir
        return new_config
    
    def _ensure_serializable_error_graph(self, component_analysis):
        """
        Ensure the error_graph in component_analysis is serializable.
        If it's a NetworkX graph, convert it to a serializable format.
        If it's missing, create a basic structure.
        
        Args:
            component_analysis: Component analysis data
            
        Returns:
            Updated component analysis with serializable error graph
        """
        if not component_analysis:
            return {}
            
        if "error_graph" not in component_analysis:
            # Create a basic error graph structure if missing
            component_analysis["error_graph"] = {"nodes": [], "edges": []}
            return component_analysis
            
        error_graph = component_analysis["error_graph"]
        
        # If it's already a dict with nodes and edges, assume it's serializable
        if isinstance(error_graph, dict) and "nodes" in error_graph and "edges" in error_graph:
            return component_analysis
            
        # If it's a NetworkX graph, convert it
        try:
            import networkx as nx
            if isinstance(error_graph, nx.Graph):
                component_analysis["error_graph"] = self._convert_graph_to_serializable(error_graph)
                return component_analysis
        except (ImportError, TypeError) as e:
            logging.warning(f"Could not process error graph as NetworkX graph: {str(e)}")
        
        # If it's something else, create a basic structure
        component_analysis["error_graph"] = {"nodes": [], "edges": []}
        logging.warning(f"Replaced non-serializable error graph with empty structure")
        
        return component_analysis
    
    def _convert_graph_to_serializable(self, G):
        """
        Convert a NetworkX graph to a JSON-serializable dictionary format.
        
        Args:
            G: NetworkX graph object
            
        Returns:
            Dictionary representation of the graph
        """
        import networkx as nx
        
        if not isinstance(G, nx.Graph):
            logging.warning(f"Expected a NetworkX graph, got {type(G).__name__} instead")
            return {"nodes": [], "edges": []}
        
        serializable_graph = {
            "nodes": [],
            "edges": []
        }
        
        # Convert nodes and their attributes
        for node_id in G.nodes():
            node_data = {"id": str(node_id)}
            # Add all node attributes
            node_data.update({k: str(v) if not isinstance(v, (int, float, bool, str, list, dict, type(None))) else v 
                             for k, v in G.nodes[node_id].items()})
            serializable_graph["nodes"].append(node_data)
        
        # Convert edges and their attributes
        for u, v in G.edges():
            edge_data = {
                "source": str(u),
                "target": str(v)
            }
            # Add all edge attributes
            edge_data.update({k: str(v) if not isinstance(v, (int, float, bool, str, list, dict, type(None))) else v 
                             for k, v in G.edges[u, v].items()})
            serializable_graph["edges"].append(edge_data)
        
        return serializable_graph
    
    def _enhance_component_analysis(self, component_analysis, errors, clusters, primary_issue_component):
        """
        Enhance component analysis with consistent primary_issue_component propagation.
        
        Args:
            component_analysis: Component analysis data (or None)
            errors: List of error dictionaries
            clusters: Dictionary of clustered errors
            primary_issue_component: Primary issue component
            
        Returns:
            Enhanced component analysis
        """
        if not component_analysis:
            component_analysis = {}
        
        # Create a deep copy to avoid modifying original data
        component_analysis = copy.deepcopy(component_analysis)
        
        # Set primary_issue_component consistently
        component_analysis["primary_issue_component"] = primary_issue_component
        
        # Ensure metrics exist and contain root_cause_component
        if "metrics" not in component_analysis:
            component_analysis["metrics"] = {}
        
        component_analysis["metrics"]["root_cause_component"] = primary_issue_component
        component_analysis["metrics"]["component_tagged_logs"] = len(errors)
        component_analysis["metrics"]["component_tagged_errors"] = len([err for err in errors if isinstance(err, dict) and err.get('component') != 'unknown'])
        
        # Add timestamp for tracking component data changes
        component_analysis["metrics"]["last_enhanced"] = datetime.now().isoformat()
        
        # Log component field verification
        component_field_counts = count_component_fields(errors[:20])
        logging.info(f"Component field counts during enhancement: {component_field_counts}")
        
        # Ensure component_summary exists
        if "component_summary" not in component_analysis or not component_analysis["component_summary"]:
            component_counts = {}
            for err in errors:
                if isinstance(err, dict) and 'component' in err:
                    comp = err.get('component', 'unknown')
                    if comp != 'unknown':
                        component_counts[comp] = component_counts.get(comp, 0) + 1
            
            # Build a basic component_summary
            if component_counts:
                component_analysis["component_summary"] = []
                for comp, count in sorted(component_counts.items(), key=lambda x: x[1], reverse=True):
                    component_analysis["component_summary"].append({
                        "id": comp,
                        "name": comp.upper(),
                        "error_count": count,
                        "percentage": (count / len(errors) * 100) if len(errors) > 0 else 0,
                        "is_primary": comp == primary_issue_component
                    })
        else:
            # Update existing component_summary to ensure is_primary flag is correctly set
            for comp_info in component_analysis["component_summary"]:
                if isinstance(comp_info, dict) and "id" in comp_info:
                    comp_info["is_primary"] = comp_info["id"] == primary_issue_component
        
        # Ensure relationships exist
        if "relationships" not in component_analysis or not component_analysis["relationships"]:
            # Create basic relationships if needed
            if "component_summary" in component_analysis and len(component_analysis["component_summary"]) > 1:
                component_analysis["relationships"] = []
                comps = [c["id"] for c in component_analysis["component_summary"] if isinstance(c, dict) and "id" in c]
                
                # Create a basic relationship from primary component to others
                if primary_issue_component in comps:
                    # Add relationships from primary component to others
                    for comp in comps:
                        if comp != primary_issue_component:
                            component_analysis["relationships"].append({
                                "source": primary_issue_component,
                                "target": comp,
                                "type": "affected"
                            })
        
        # Ensure error_graph exists and is serializable
        component_analysis = self._ensure_serializable_error_graph(component_analysis)
        
        return component_analysis
    
    def _propagate_primary_component(self, errors, primary_component):
        """
        Propagate primary_issue_component to all errors.
        
        Args:
            errors: List of error dictionaries
            primary_component: Primary issue component
        
        Returns:
            Updated errors list
        """
        errors = copy.deepcopy(errors)
        
        initial_count = 0
        final_count = 0
        
        # Count initial presence
        for err in errors:
            if isinstance(err, dict) and 'primary_issue_component' in err and err['primary_issue_component'] == primary_component:
                initial_count += 1
        
        # Update all errors
        for err in errors:
            if isinstance(err, dict):
                err['primary_issue_component'] = primary_component
                # Also ensure component_source exists for consistent tracking
                if 'component' in err and 'component_source' not in err:
                    err['component_source'] = 'default'
        
        # Count final presence
        for err in errors:
            if isinstance(err, dict) and 'primary_issue_component' in err and err['primary_issue_component'] == primary_component:
                final_count += 1
        
        # Log propagation stats
        logging.info(f"Primary component propagation: {initial_count} -> {final_count} errors")
        
        return errors
    
    def _propagate_primary_component_to_clusters(self, clusters, primary_component):
        """
        Propagate primary_issue_component to all errors in clusters.
        
        Args:
            clusters: Dictionary of clustered errors
            primary_component: Primary issue component
        
        Returns:
            Updated clusters dictionary
        """
        clusters = copy.deepcopy(clusters)
        
        initial_count = 0
        final_count = 0
        
        # Count initial presence
        for cluster_id, errors in clusters.items():
            for err in errors:
                if isinstance(err, dict) and 'primary_issue_component' in err and err['primary_issue_component'] == primary_component:
                    initial_count += 1
        
        # Update all errors in clusters
        for cluster_id, errors in clusters.items():
            for err in errors:
                if isinstance(err, dict):
                    err['primary_issue_component'] = primary_component
                    # Also ensure component_source exists for consistent tracking
                    if 'component' in err and 'component_source' not in err:
                        err['component_source'] = 'default'
        
        # Count final presence
        for cluster_id, errors in clusters.items():
            for err in errors:
                if isinstance(err, dict) and 'primary_issue_component' in err and err['primary_issue_component'] == primary_component:
                    final_count += 1
        
        # Log propagation stats
        logging.info(f"Primary component propagation in clusters: {initial_count} -> {final_count} errors")
        
        return clusters
    
    def _verify_component_consistency(self, errors, primary_component):
        """
        Verify that component information is consistent throughout the error list.
        
        Args:
            errors: List of error dictionaries
            primary_component: Primary issue component
            
        Returns:
            Boolean indicating whether component information is consistent
        """
        if not errors:
            return True
            
        consistent = True
        for i, error in enumerate(errors[:20]):  # Check a sample
            if not isinstance(error, dict):
                continue
                
            # Check essential component fields
            for field in ['component', 'component_source', 'primary_issue_component']:
                if field not in error:
                    logging.warning(f"Error {i} missing {field}")
                    consistent = False
            
            # Check primary_issue_component consistency
            if 'primary_issue_component' in error and error['primary_issue_component'] != primary_component:
                logging.warning(f"Error {i} has inconsistent primary_issue_component: {error['primary_issue_component']} vs {primary_component}")
                consistent = False
        
        # Log consistency status
        if consistent:
            logging.info("Component information is consistent")
        else:
            logging.warning("Component information is inconsistent")
        
        return consistent
    
    def write_json_report(self, data: Dict, filename: str) -> str:
        """
        Write JSON report with component-preserving encoding.
        
        Args:
            data: Data to serialize
            filename: Output filename
            
        Returns:
            Path to the written file
        """
        # Use path utilities for consistent file placement
        output_path = get_output_path(
            self.config.output_dir,
            self.config.test_id,
            filename,
            OutputType.JSON_DATA
        )
        
        # Use the new helper function for serialization
        serialize_to_json_file(
            data, 
            output_path, 
            primary_issue_component=self.config.primary_issue_component,
            indent=2
        )
        
        # Verification - Check component distribution in serialized data
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
                
            # Extract and log component counts in errors from the serialized data
            if "errors" in loaded_data and isinstance(loaded_data["errors"], list):
                component_counts = {}
                for err in loaded_data["errors"][:20]:  # Sample first 20
                    if isinstance(err, dict) and "component" in err:
                        comp = err["component"]
                        component_counts[comp] = component_counts.get(comp, 0) + 1
                
                if component_counts:
                    logging.info(f"Component distribution in serialized JSON: {component_counts}")
        except Exception as e:
            logging.warning(f"Could not verify serialized JSON: {str(e)}")
            
        return output_path

    def generate_reports(self, data: ReportData) -> Dict[str, Any]:
        """
        Generate all reports with enhanced component preservation.
        
        Args:
            data: Report data
            
        Returns:
            Dictionary with report paths and metadata
        """
        results = {
            "primary_issue_component": self.config.primary_issue_component,
            "reports": {}
        }
        
        try:
            # STEP 0: Log initial component distribution for verification
            initial_component_distribution = {}
            for err in data.errors[:20]:  # Sample for logging
                if isinstance(err, dict) and 'component' in err:
                    comp = err.get('component', 'unknown')
                    initial_component_distribution[comp] = initial_component_distribution.get(comp, 0) + 1
            logging.info(f"Initial component distribution: {initial_component_distribution}")
            
            # STEP 1: Preprocess errors with enhanced component handling
            processed_errors, primary_issue_component = preprocess_errors(
                data.errors, 
                self.config.primary_issue_component,
                data.component_diagnostic
            )
            
            # Update primary_issue_component if it was derived from the data
            if primary_issue_component != self.config.primary_issue_component:
                logging.info(f"Updating primary_issue_component from '{self.config.primary_issue_component}' to '{primary_issue_component}'")
                self.config.primary_issue_component = primary_issue_component
                results["primary_issue_component"] = primary_issue_component
            
            # STEP 2: Preprocess clusters with consistent component handling
            processed_clusters = preprocess_clusters(
                data.clusters,
                self.config.primary_issue_component,
                data.component_diagnostic
            )
            
            # STEP 3: Verify component consistency after preprocessing
            preprocessed_component_distribution = {}
            for err in processed_errors[:20]:  # Sample for logging
                if isinstance(err, dict) and 'component' in err:
                    comp = err.get('component', 'unknown')
                    preprocessed_component_distribution[comp] = preprocessed_component_distribution.get(comp, 0) + 1
            logging.info(f"Component distribution after preprocessing: {preprocessed_component_distribution}")
            
            # STEP 4: Build or update component analysis
            component_analysis = build_component_analysis(
                processed_errors,
                self.config.primary_issue_component,
                data.component_analysis
            )
            
            # Ensure primary_issue_component is consistently set in component_analysis
            if "primary_issue_component" not in component_analysis or component_analysis["primary_issue_component"] != self.config.primary_issue_component:
                component_analysis["primary_issue_component"] = self.config.primary_issue_component
            
            # STEP 5: Normalize timestamps and validate components
            normalized_errors, normalized_clusters = normalize_data(
                processed_errors,
                processed_clusters,
                self.config.primary_issue_component
            )
            
            # STEP 6: Verify component consistency after normalization
            normalized_component_distribution = {}
            for err in normalized_errors[:20]:  # Sample for logging
                if isinstance(err, dict) and 'component' in err:
                    comp = err.get('component', 'unknown')
                    normalized_component_distribution[comp] = normalized_component_distribution.get(comp, 0) + 1
            logging.info(f"Component distribution after normalization: {normalized_component_distribution}")
            
            # STEP 7: Enhanced Component Information Propagation
            component_analysis = self._enhance_component_analysis(
                component_analysis,
                normalized_errors,
                normalized_clusters,
                self.config.primary_issue_component
            )
            
            # Propagate primary_issue_component to all errors and clusters
            normalized_errors = self._propagate_primary_component(normalized_errors, self.config.primary_issue_component)
            normalized_clusters = self._propagate_primary_component_to_clusters(normalized_clusters, self.config.primary_issue_component)
            
            # STEP 8: Verify component consistency after propagation
            self._verify_component_consistency(normalized_errors, self.config.primary_issue_component)
            
            # STEP 9: Generate component report (deprecated)
            if False:
                pass
            
            # STEP 10: Generate visualizations
            if self.visualization_generator:
                # Ensure step_to_logs data is preserved if available
                if hasattr(data, "step_to_logs") and data.step_to_logs:
                    step_to_logs = data.step_to_logs
                    step_dict = getattr(data, "step_dict", {})
                else:
                    step_to_logs = None
                    step_dict = None
                
                # Create new report data with enhanced component information
                visualization_data = ReportData(
                    errors=normalized_errors,
                    summary=data.summary,
                    clusters=normalized_clusters,
                    ocr_data=data.ocr_data,
                    background_text=data.background_text,
                    scenario_text=data.scenario_text,
                    ymir_flag=data.ymir_flag,
                    component_analysis=component_analysis,
                    component_diagnostic=data.component_diagnostic
                )
                
                # Add step data if available
                if step_to_logs:
                    visualization_data.step_to_logs = step_to_logs
                    visualization_data.step_dict = step_dict
                
                visualization_paths = self.visualization_generator.generate(visualization_data)
                
                results["reports"]["visualizations"] = visualization_paths
                
                # Add visualization paths to component_analysis
                if visualization_paths:
                    component_analysis["visualization_paths"] = visualization_paths
                    
                    # Update analysis_files with correct paths for HTML references
                    if "analysis_files" not in component_analysis:
                        component_analysis["analysis_files"] = {}
                    
                    sanitized_base = sanitize_base_directory(self.base_dir, "json")
                    for viz_type, viz_path in visualization_paths.items():
                        if viz_path and os.path.exists(viz_path):
                            sanitized_path = get_output_path(
                                sanitized_base,
                                self.config.test_id,
                                os.path.basename(viz_path),
                                OutputType.JSON_DATA,
                                create_dirs=False,
                            )
                            component_analysis["analysis_files"][viz_type] = sanitized_path
            
            # Create report data object with processed data
            report_data = ReportData(
                errors=normalized_errors,
                summary=data.summary,
                clusters=normalized_clusters,
                ocr_data=data.ocr_data,
                background_text=data.background_text,
                scenario_text=data.scenario_text,
                ymir_flag=data.ymir_flag,
                component_analysis=component_analysis,
                component_diagnostic=data.component_diagnostic
            )
            
            # Ensure step data is preserved if available
            if hasattr(data, "step_to_logs") and data.step_to_logs:
                report_data.step_to_logs = data.step_to_logs
                report_data.step_dict = getattr(data, "step_dict", {})
            
            # STEP 11: Generate reports using path utilities
            if self.json_generator:
                # Instead of using the JSON generator's method directly, use our write_json_report
                # This ensures we use the component-aware serialization
                json_path = self.json_generator.generate(report_data)
                results["reports"]["json"] = json_path
                
                # Verify component preservation in JSON
                if json_path and os.path.exists(json_path):
                    sample_error = normalized_errors[0] if normalized_errors else None
                    if sample_error:
                        try:
                            with open(json_path, 'r', encoding='utf-8') as f:
                                json_data = json.load(f)
                                if 'errors' in json_data and json_data['errors']:
                                    json_error = json_data['errors'][0]
                                    if not verify_component_preservation(sample_error, json_error):
                                        logging.warning("Component information not preserved in JSON serialization")
                                    else:
                                        logging.info("Component information preserved in JSON serialization")
                        except Exception as e:
                            logging.warning(f"Error checking JSON component preservation: {str(e)}")
            
            if self.markdown_generator:
                markdown_path = self.markdown_generator.generate(report_data)
                results["reports"]["markdown"] = markdown_path
            
            if self.excel_generator:
                excel_path = self.excel_generator.generate(report_data)
                results["reports"]["excel"] = excel_path
            
            if self.docx_generator:
                docx_path = self.docx_generator.generate(report_data)
                results["reports"]["docx"] = docx_path
            
            # STEP 12: Calculate final component distribution
            final_component_distribution = {}
            for err in normalized_errors[:20]:  # Sample for logging
                if isinstance(err, dict) and 'component' in err:
                    comp = err.get('component', 'unknown')
                    final_component_distribution[comp] = final_component_distribution.get(comp, 0) + 1
            logging.info(f"Final component distribution: {final_component_distribution}")
            
            # STEP 13: Save component preservation diagnostic info
            diagnostic_filename = get_standardized_filename(self.config.test_id, "component_preservation", "json")
            diagnostic_path = get_output_path(
                self.base_dir,
                self.config.test_id,
                diagnostic_filename,
                OutputType.JSON_DATA
            )
            
            diagnostic_data = {
                "initial": {"component_counts": initial_component_distribution},
                "preprocessed": {"component_counts": preprocessed_component_distribution},
                "normalized": {"component_counts": normalized_component_distribution},
                "final": {"component_counts": final_component_distribution}
            }
            
            # Use the new serialize_to_json_file method for diagnostic info
            diagnostic_path = serialize_to_json_file(
                {
                    "test_id": self.config.test_id,
                    "timestamp": datetime.now().isoformat(),
                    "primary_issue_component": self.config.primary_issue_component,
                    "stage_info": diagnostic_data,
                    "component_field_counts": count_component_fields(normalized_errors),
                    "component_fields_preserved": True
                },
                diagnostic_path,
                primary_issue_component=self.config.primary_issue_component
            )
            
            if diagnostic_path:
                results["reports"]["component_diagnostic"] = diagnostic_path
            
            # After all reports are generated, verify and fix directory structure
            from utils.path_validator import fix_directory_structure
            
            logging.info(f"Verifying directory structure for {self.config.test_id}")
            issues = fix_directory_structure(self.base_dir, self.config.test_id)
            
            # Log issues and fixes
            if issues.get("json_dir_images", []) or issues.get("images_dir_json", []) or issues.get("nested_directories", []) or issues.get("misplaced_visualizations", []):
                logging.warning(f"Found issues with directory structure: {len(issues.get('json_dir_images', [])) + len(issues.get('images_dir_json', [])) + len(issues.get('nested_directories', [])) + len(issues.get('misplaced_visualizations', []))} issues")
            if issues.get("fixed_files", []):
                logging.info(f"Fixed {len(issues.get('fixed_files', []))} files with directory structure issues")
            
            # Add information about verification to results
            results["directory_validation"] = {
                "issues_found": len(issues.get("json_dir_images", [])) + 
                               len(issues.get("images_dir_json", [])) + 
                               len(issues.get("nested_directories", [])) + 
                               len(issues.get("misplaced_visualizations", [])),
                "files_fixed": len(issues.get("fixed_files", []))
            }
            
            # Add component preservation metrics to results
            results["component_preservation"] = {
                "primary_component_consistent": self._verify_component_consistency(normalized_errors, self.config.primary_issue_component),
                "distribution_preserved": initial_component_distribution.keys() == final_component_distribution.keys()
            }
            
            return results
            
        except Exception as e:
            logging.error(f"Error generating reports: {str(e)}")
            traceback.print_exc()
            
            # Return minimal information in case of error
            results["error"] = str(e)
            return results
    
    def _save_component_preservation_diagnostic(self, errors: List[Dict], diagnostic_path: str = None, stage_info: Dict = None) -> str:
        """
        Save component preservation diagnostic information.
        
        Args:
            errors: List of error dictionaries
            diagnostic_path: Path to save the diagnostic file (optional)
            stage_info: Information about component distribution at different stages
            
        Returns:
            Path to the diagnostic file
        """
        try:
            # If no path provided, use default path in json directory
            if not diagnostic_path:
                diagnostic_filename = get_standardized_filename(self.config.test_id, "component_preservation", "json")
                diagnostic_path = get_output_path(
                    self.base_dir,
                    self.config.test_id,
                    diagnostic_filename,
                    OutputType.JSON_DATA
                )
            
            # Calculate component distribution statistics
            final_component_counts = {}
            component_source_counts = {}
            
            for err in errors:
                if isinstance(err, dict) and 'component' in err:
                    comp = err.get('component', 'unknown')
                    final_component_counts[comp] = final_component_counts.get(comp, 0) + 1
                    
                    # Track component_source field to verify preservation
                    if 'component_source' in err:
                        source = err.get('component_source')
                        component_source_counts[source] = component_source_counts.get(source, 0) + 1
            
            # Calculate alignment metrics
            source_alignment = True
            for err in errors[:20]:
                if isinstance(err, dict) and 'component' in err and 'source_component' in err:
                    if err['component'] != 'unknown' and err['source_component'] != 'unknown':
                        if err['component'] != err['source_component']:
                            source_alignment = False
                            break
            
            # Calculate primary component presence
            primary_component_presence = sum(
                1 for err in errors 
                if isinstance(err, dict) and 'primary_issue_component' in err 
                and err['primary_issue_component'] == self.config.primary_issue_component
            )
            
            # Calculate component field counts
            component_field_counts = count_component_fields(errors)
            
            preservation_stats = {
                "test_id": self.config.test_id,
                "timestamp": datetime.now().isoformat(),
                "primary_issue_component": self.config.primary_issue_component,
                "stage_info": stage_info or {},
                "component_counts": final_component_counts,
                "component_source_counts": component_source_counts,
                "source_component_alignment": source_alignment,
                "primary_component_presence": primary_component_presence,
                "component_field_counts": component_field_counts,
                "component_fields_preserved": True
            }
            
            # Use the new serialize_to_json_file method
            diagnostic_path = serialize_to_json_file(
                preservation_stats,
                diagnostic_path,
                primary_issue_component=self.config.primary_issue_component
            )
                
            logging.info(f"Saved component preservation diagnostic to: {diagnostic_path}")
            return diagnostic_path
        except Exception as e:
            logging.error(f"Error saving component preservation diagnostic: {str(e)}")
            return ""