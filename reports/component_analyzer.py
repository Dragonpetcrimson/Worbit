"""
reports/component_analyzer.py - Component analysis logic
"""

import logging
import copy
import os
import json
import networkx as nx
from typing import Dict, List, Any, Optional
from collections import defaultdict
from datetime import datetime

from reports.base import COMPONENT_FIELDS, DateTimeEncoder
from reports.data_preprocessor import count_components
from utils.path_utils import (
    get_output_path,
    OutputType,
    normalize_test_id,
    get_standardized_filename,
    sanitize_base_directory
)

def build_component_analysis(errors: List[Dict], 
                          primary_issue_component: str, 
                          existing_analysis: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Build comprehensive component analysis data structure.
    Enhanced to ensure component data is always available for visualization.
    
    Args:
        errors: List of error dictionaries
        primary_issue_component: Primary issue component
        existing_analysis: Optional existing analysis to enhance
        
    Returns:
        Component analysis data structure
    """
    # Start with existing analysis if provided
    component_analysis = copy.deepcopy(existing_analysis) if existing_analysis else {}
    
    # Ensure primary_issue_component is set consistently
    component_analysis["primary_issue_component"] = primary_issue_component
    component_analysis["root_cause_component"] = primary_issue_component
    
    # Component information with descriptions - simplified version of the schema
    component_info = {
        "soa": {
            "name": "SOA",
            "description": "SiriusXM application built on Android",
            "related_to": ["phoebe", "mimosa", "charles"],
        },
        "mimosa": {
            "name": "Mimosa",
            "description": "Provides fake testing data (Satellite/IP channel)",
            "related_to": ["soa", "lapetus"],
        },
        "charles": {
            "name": "Charles",
            "description": "Proxy for live data",
            "related_to": ["soa", "phoebe"],
        },
        "phoebe": {
            "name": "Phoebe",
            "description": "Proxy to run data to SOA",
            "related_to": ["soa", "arecibo", "lapetus", "charles"],
        },
        "arecibo": {
            "name": "Arecibo",
            "description": "Monitors traffic from Phoebe",
            "related_to": ["phoebe"],
        },
        "translator": {
            "name": "Translator",
            "description": "Translates commands between test framework and SOA",
            "related_to": ["soa", "smite"],
        },
        "telesto": {
            "name": "Telesto",
            "description": "Coordinates components",
            "related_to": ["mimosa", "phoebe", "lapetus", "arecibo"],
        },
        "lapetus": {
            "name": "Lapetus",
            "description": "API to add channel and categories",
            "related_to": ["phoebe", "telesto", "mimosa"],
        },
        "ip_traffic": {
            "name": "IP Traffic",
            "description": "Network traffic and HTTP communication",
            "related_to": ["charles", "soa", "phoebe"],
        },
        "android": {
            "name": "Android",
            "description": "Android system and app interactions",
            "related_to": ["soa"],
        }
    }
    
    # Ensure primary_issue_component has an entry in component_info if it's not "unknown"
    if primary_issue_component != "unknown" and primary_issue_component not in component_info:
        component_info[primary_issue_component] = {
            "name": primary_issue_component.upper(),
            "description": f"Automatically detected {primary_issue_component} component",
            "related_to": []
        }
    
    # If no component summary exists, create it from errors
    if "component_summary" not in component_analysis or not component_analysis["component_summary"]:
        # Count errors by component
        component_counts = count_components(errors)
        
        # Convert to component_summary format
        component_summary = []
        for comp_id, count in component_counts.items():
            if comp_id != "unknown" and count > 0:
                info = component_info.get(comp_id, {})
                component_summary.append({
                    "id": comp_id,
                    "name": info.get("name", comp_id.upper()),
                    "description": info.get("description", ""),
                    "error_count": count,
                    "related_to": info.get("related_to", [])
                })
            elif comp_id == "unknown" and count > 0:
                # Include unknown component only if it's the only one or significant
                if len(component_counts) == 1 or count > 5:
                    component_summary.append({
                        "id": "unknown",
                        "name": "UNKNOWN",
                        "description": "Component not identified",
                        "error_count": count,
                        "related_to": []
                    })
        
        # If still empty (shouldn't happen), create a placeholder component
        if not component_summary and errors:
            component_summary.append({
                "id": primary_issue_component if primary_issue_component != "unknown" else "placeholder",
                "name": (primary_issue_component if primary_issue_component != "unknown" else "PLACEHOLDER").upper(),
                "description": "Placeholder component for visualization",
                "error_count": len(errors),
                "related_to": []
            })
            
        component_analysis["component_summary"] = component_summary
    
    # If no relationships exist, create default ones based on component summary
    if "relationships" not in component_analysis or not component_analysis["relationships"]:
        # Get component IDs from summary
        components = [c.get("id") for c in component_analysis.get("component_summary", []) 
                     if c.get("id") != "unknown"]
        
        # Create basic relationships from primary component to others
        relationships = []
        
        # Only create relationships if we have multiple components
        if len(components) > 1 and primary_issue_component != "unknown":
            for comp_id in components:
                if comp_id != primary_issue_component and comp_id != "unknown":
                    relationships.append({
                        "source": primary_issue_component,
                        "target": comp_id,
                        "description": "Auto-generated relationship",
                        "weight": 1.0
                    })
        # If primary_issue_component is unknown or not in components, use first component as source
        elif components and primary_issue_component == "unknown":
            source_comp = components[0]
            for comp_id in components[1:]:
                relationships.append({
                    "source": source_comp,
                    "target": comp_id,
                    "description": "Auto-generated relationship",
                    "weight": 1.0
                })
        # Even with a single component, create a self-loop for visualization
        elif len(components) == 1:
            comp_id = components[0]
            relationships.append({
                "source": comp_id,
                "target": comp_id,
                "description": "Self-reference (auto-generated)",
                "weight": 1.0
            })
        
        component_analysis["relationships"] = relationships
    
    # If error_graph is missing, create a placeholder one
    if "error_graph" not in component_analysis and len(errors) > 0:
        component_analysis["error_graph"] = build_error_graph(errors, primary_issue_component)
    
    # Add metrics if they don't exist
    if "metrics" not in component_analysis:
        component_counts = count_components(errors)
        component_analysis["metrics"] = {
            "component_tagged_logs": len(errors),
            "component_tagged_errors": sum(1 for err in errors if isinstance(err, dict) and err.get('component') != 'unknown'),
            "component_error_counts": component_counts,
            "components_with_issues": list(component_counts.keys()),
            "root_cause_component": primary_issue_component
        }
    
    # Preserve component_info for reference
    component_analysis["component_info"] = component_info
    
    # Ensure component_error_counts exists for backward compatibility
    if "component_error_counts" not in component_analysis:
        component_analysis["component_error_counts"] = count_components(errors)
    
    logging.info(f"Component analysis built with {len(component_analysis.get('component_summary', []))} components, "
                f"{len(component_analysis.get('relationships', []))} relationships, "
                f"primary: {primary_issue_component}")
    
    return component_analysis

def build_error_graph(errors: List[Dict], primary_issue_component: str) -> Dict[str, Any]:
    """
    Build an error propagation graph for visualization.
    
    Args:
        errors: List of error dictionaries
        primary_issue_component: Primary issue component
        
    Returns:
        Serializable dictionary representation of error graph
    """
    try:
        # Create a simple NetworkX DiGraph for error propagation
        G = nx.DiGraph()
        
        # Add representative errors as nodes (up to 5)
        representative_errors = []
        
        # First add errors from primary component if available
        primary_errors = [err for err in errors if isinstance(err, dict) 
                         and err.get('component') == primary_issue_component][:3]
        representative_errors.extend(primary_errors)
        
        # Then add errors from other components to reach up to 5 total
        other_errors = [err for err in errors if isinstance(err, dict) 
                       and err.get('component') != primary_issue_component 
                       and err.get('component') != 'unknown']
        representative_errors.extend(other_errors[:5-len(representative_errors)])
        
        # If still fewer than 5, add unknown component errors
        unknown_errors = [err for err in errors if isinstance(err, dict) 
                        and err.get('component') == 'unknown']
        representative_errors.extend(unknown_errors[:5-len(representative_errors)])
        
        # Add nodes for each error
        for i, error in enumerate(representative_errors):
            if isinstance(error, dict):
                error_id = f"error_{i}"
                G.add_node(error_id, 
                         text=error.get('text', 'Error message unavailable'),
                         component=error.get('component', 'unknown'),
                         severity=error.get('severity', 'Medium'))
                
                # Mark primary component errors as root causes
                if error.get('component') == primary_issue_component:
                    G.nodes[error_id]['is_root_cause'] = True
        
        # Add some basic edges if we have multiple nodes
        node_ids = list(G.nodes())
        if len(node_ids) > 1:
            # If we have primary component errors, make them roots
            primary_nodes = [node for node in node_ids 
                           if G.nodes[node].get('component') == primary_issue_component]
            
            if primary_nodes:
                # Connect primary nodes to others
                for primary_node in primary_nodes:
                    for node in node_ids:
                        if node != primary_node:
                            G.add_edge(primary_node, node, weight=1.0)
            else:
                # If no primary nodes, just connect first to others
                for i in range(1, len(node_ids)):
                    G.add_edge(node_ids[0], node_ids[i], weight=1.0)
        
        # Convert to serializable dictionary format
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
        
    except Exception as e:
        logging.error(f"Error building error graph: {str(e)}")
        # Return minimal graph if error occurs
        return {
            "nodes": [{"id": "error_0", "text": "Error building graph", "component": "unknown", "severity": "Medium"}],
            "edges": []
        }

def get_visualization_path(output_dir, test_id, visualization_type, extension="png"):
    """
    Get a standardized path for a visualization file.
    
    Args:
        output_dir: Output directory
        test_id: Test ID
        visualization_type: Type of visualization (component, errors, etc.)
        extension: File extension (default: png)
        
    Returns:
        Standardized path for the visualization
    """
    # Sanitize output directory to prevent nested directories
    output_dir = sanitize_base_directory(output_dir)
    
    # Use path utilities consistently
    return get_output_path(
        output_dir,
        test_id,
        get_standardized_filename(test_id, visualization_type, extension),
        OutputType.PRIMARY_REPORT
    )

def get_path_reference(path, base_dir, reference_type="html"):
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

def generate_component_report(output_dir: str, 
                           test_id: str, 
                           component_analysis: Dict[str, Any], 
                           primary_issue_component: str = "unknown") -> str:
    """
    Generate component analysis report.
    
    Args:
        output_dir: Directory to write the report to
        test_id: Test identifier
        component_analysis: Component analysis data
        primary_issue_component: Primary issue component
        
    Returns:
        Path to the generated report
    """
    try:
        # First, fix any existing directory structure issues
        from utils.path_validator import fix_directory_structure
        fix_directory_structure(output_dir, test_id)
        
        # Sanitize output directory
        output_dir = sanitize_base_directory(output_dir)
        
        # Ensure component analysis has required data for visualization
        component_analysis = build_component_analysis(
            [], # Empty errors as component_analysis should already have data
            primary_issue_component,
            component_analysis
        )
        
        # Save component analysis to JSON file in json subdirectory
        component_analysis_filename = get_standardized_filename(test_id, "component_analysis", "json")
        component_analysis_path = get_output_path(
            output_dir,
            test_id,
            component_analysis_filename,
            OutputType.JSON_DATA  # This will ensure it goes to json/ subdirectory
        )
        
        logging.info(f"Saving component analysis to: {component_analysis_path}")
        
        with open(component_analysis_path, 'w', encoding='utf-8') as f:
            json.dump(component_analysis, f, cls=DateTimeEncoder, indent=2)

        # Standardize any visualization paths in component_analysis to ensure HTML reference consistency
        if "analysis_files" in component_analysis:
            for key, file_path in list(component_analysis["analysis_files"].items()):
                # Replace image paths with standardized HTML references
                if file_path and os.path.exists(file_path):
                    component_analysis["analysis_files"][key] = get_path_reference(file_path, output_dir, "html")

        # Try to use the full component report generator
        try:
            from component_report import generate_component_report as gen_comp_report
            
            # Prepare analysis results for the report generator
            analysis_results = {
                "analysis_files": component_analysis.get("analysis_files", {}) | {"component_analysis": component_analysis_path},
                "primary_issue_component": primary_issue_component
            }
            
            # Generate the report using sanitized output directory
            report_path = gen_comp_report(
                output_dir=output_dir,
                test_id=test_id,
                analysis_results=analysis_results
            )
            
            logging.info(f"Generated component analysis report: {report_path}")
            
            # Verify report path exists
            if not os.path.exists(report_path):
                logging.warning(f"Generated report not found at expected path: {report_path}")
                
                # Create fallback path if necessary
                report_filename = get_standardized_filename(test_id, "component_report", "html")
                fallback_path = get_output_path(
                    output_dir,
                    test_id,
                    report_filename,
                    OutputType.PRIMARY_REPORT
                )
                
                # If actual file exists at fallback path, use that
                if os.path.exists(fallback_path) and fallback_path != report_path:
                    logging.info(f"Found report at fallback path: {fallback_path}")
                    report_path = fallback_path
            
            return report_path
            
        except ImportError:
            # Create a simple HTML report as fallback
            logging.info("Component report module not available, creating simple report")
            
            # Collect component data
            components_with_counts = component_analysis.get("component_error_counts", {})
            component_info = component_analysis.get("component_info", {})
            component_summary = component_analysis.get("component_summary", [])
            
            # Get visualization paths and standardize references
            viz_files = {}
            for key, file_path in component_analysis.get("analysis_files", {}).items():
                if file_path and os.path.exists(file_path):
                    viz_files[key] = get_path_reference(file_path, output_dir, "html")
            
            # Create HTML report in primary directory using standard path
            report_filename = get_standardized_filename(test_id, "component_report", "html")
            simple_report_path = get_output_path(
                output_dir,
                test_id,
                report_filename,
                OutputType.PRIMARY_REPORT
            )
            
            html_content = f'''
            <!DOCTYPE html>
            <html>
            <head>
                <title>Component Analysis for {test_id}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2 {{ color: #333; }}
                    table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    tr:nth-child(even) {{ background-color: #f9f9f9; }}
                    .primary {{ background-color: #ffdddd; font-weight: bold; }}
                    .note {{ background-color: #ffffdd; padding: 10px; border-left: 4px solid #ffcc00; margin: 15px 0; }}
                    .visualization {{ margin: 20px 0; }}
                </style>
            </head>
            <body>
                <h1>Component Analysis for {test_id}</h1>
                
                <h2>Primary Issue Component</h2>
                <p><strong>{primary_issue_component.upper()}</strong>: 
                {component_info.get(primary_issue_component, {}).get("description", "Not available")}</p>
            '''
            
            # Add visualizations if available
            for viz_type, viz_title in [
                ("component_relationships", "Component Relationship Diagram"),
                ("component_distribution", "Component Error Distribution"),
                ("error_propagation", "Error Propagation Diagram")
            ]:
                if viz_files.get(viz_type):
                    image_path = viz_files.get(viz_type)
                    html_content += f'''
                    <div class="visualization">
                        <h2>{viz_title}</h2>
                        <img src="{image_path}" alt="{viz_title}">
                    </div>
                    '''
            
            html_content += '''
                <h2>Component Error Distribution</h2>
            '''
            
            # Add note if component summary is empty or has few items
            if not component_summary or len(component_summary) < 2:
                html_content += '''
                <div class="note">
                    <p><strong>Note:</strong> Limited component data available. 
                    Component relationships and error propagation visualizations will be limited.</p>
                </div>
                '''
            
            html_content += '''
                <table>
                    <tr>
                        <th>Component</th>
                        <th>Error Count</th>
                        <th>Description</th>
                    </tr>
            '''
            
            # Sort components by error count
            for comp, count in sorted(components_with_counts.items(), key=lambda x: x[1], reverse=True):
                if comp == "unknown" and len(components_with_counts) > 1:
                    continue
                    
                info = component_info.get(comp, {})
                row_class = "primary" if comp == primary_issue_component else ""
                
                html_content += f'''
                    <tr class="{row_class}">
                        <td>{comp.upper()}</td>
                        <td>{count}</td>
                        <td>{info.get("description", "")}</td>
                    </tr>
                '''
            
            # Add relationships section if available
            relationships = component_analysis.get("relationships", [])
            if relationships:
                html_content += '''
                <h2>Component Relationships</h2>
                <table>
                    <tr>
                        <th>Source</th>
                        <th>Target</th>
                        <th>Description</th>
                    </tr>
                '''
                
                for rel in relationships:
                    source = rel.get("source", "unknown").upper()
                    target = rel.get("target", "unknown").upper()
                    desc = rel.get("description", "")
                    
                    html_content += f'''
                    <tr>
                        <td>{source}</td>
                        <td>{target}</td>
                        <td>{desc}</td>
                    </tr>
                    '''
                
                html_content += '''
                </table>
                '''
            
            # Add generation timestamp
            html_content += f'''
                <p><em>Report generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>
            </body>
            </html>
            '''
            
            # Write the HTML file
            with open(simple_report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logging.info(f"Generated simple component report: {simple_report_path}")
            return simple_report_path
            
    except Exception as e:
        logging.error(f"Error generating component report: {str(e)}")
        return ""