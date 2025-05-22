import os
import re
import json
import logging
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta
import networkx as nx
from collections import defaultdict

try:
    from json_utils import (
        serialize_with_component_awareness,
        serialize_to_json_file,
        DateTimeEncoder
    )
    JSON_UTILS_AVAILABLE = True
except ImportError:
    # If not available, define a basic DateTimeEncoder for fallback
    JSON_UTILS_AVAILABLE = False
    
    class DateTimeEncoder(json.JSONEncoder):
        """Basic encoder that handles datetime objects."""
        def default(self, obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            return super().default(obj)

class ComponentAnalyzer:
    """
    Analyzer for identifying components and their relationships in log entries.
    """
    
    def __init__(self, component_schema_path: str):
        """
        Initialize the component analyzer with the schema.
        
        Args:
            component_schema_path: Path to the component schema JSON file
        """
        self.component_schema = self._load_schema(component_schema_path)
        self.component_patterns = self._extract_component_patterns()
        self.component_log_sources = self._extract_log_sources()
        self.component_relationships = self._build_component_graph()
        self.root_cause_component = "unknown"
        
    def _ensure_datetime(self, timestamp_value):
        """
        Ensure a timestamp is a datetime object.
        
        Args:
            timestamp_value: A timestamp which could be a string or datetime object
            
        Returns:
            datetime object or None if conversion fails
        """
        if timestamp_value is None:
            return None
        if isinstance(timestamp_value, datetime):
            return timestamp_value
        if isinstance(timestamp_value, str):
            try:
                return datetime.fromisoformat(timestamp_value)
            except ValueError:
                pass
            formats = [
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d %H:%M:%S.%f",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M:%S.%f",
                "%a %b %d %H:%M:%S %Y",
                "%d/%m/%Y %H:%M:%S",
                "%m/%d/%Y %H:%M:%S"
            ]
            for fmt in formats:
                try:
                    return datetime.strptime(timestamp_value, fmt)
                except ValueError:
                    continue
            match = re.search(r'(\d{2}:\d{2}:\d{2}(?:\.\d+)?)', timestamp_value)
            if match:
                time_str = match.group(1)
                try:
                    today = datetime.now().strftime("%Y-%m-%d")
                    return datetime.strptime(f"{today} {time_str}", "%Y-%m-%d %H:%M:%S.%f" if "." in time_str else "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    pass
        logging.warning(f"Could not convert timestamp: {timestamp_value} ({type(timestamp_value)})")
        return None
        
    def _load_schema(self, schema_path: str) -> Dict:
        """Load the component schema from a JSON file."""
        try:
            if not os.path.exists(schema_path):
                logging.warning(f"Component schema not found at {os.path.abspath(schema_path)}")
                alternate_paths = [
                    os.path.join(os.path.dirname(os.path.dirname(schema_path)), 'components', 'schemas', 'component_schema.json'),
                    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(schema_path))), 'components', 'schemas', 'component_schema.json')
                ]
                for alt_path in alternate_paths:
                    if os.path.exists(alt_path):
                        logging.info(f"Found component schema at alternate location: {alt_path}")
                        schema_path = alt_path
                        break
            if os.path.exists(schema_path):
                with open(schema_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                return {"components": [], "dataFlows": []}
        except Exception as e:
            logging.error(f"Error loading component schema: {str(e)}")
            return {"components": [], "dataFlows": []}
            
    def _extract_component_patterns(self) -> Dict[str, List[str]]:
        """Extract error patterns for each component."""
        patterns = {}
        for component in self.component_schema.get("components", []):
            component_id = component.get("id")
            if component_id:
                patterns[component_id] = component.get("errorPatterns", [])
        return patterns
        
    def _extract_log_sources(self) -> Dict[str, List[str]]:
        """Map file patterns to components."""
        log_sources = {}
        for component in self.component_schema.get("components", []):
            component_id = component.get("id")
            if component_id:
                log_sources[component_id] = component.get("logSources", [])
        return log_sources
    
    def _build_component_graph(self) -> nx.DiGraph:
        """Build a directed graph of component relationships."""
        G = nx.DiGraph()
        for component in self.component_schema.get("components", []):
            component_id = component.get("id")
            if component_id:
                G.add_node(component_id, **component)
        for flow in self.component_schema.get("dataFlows", []):
            source = flow.get("source")
            target = flow.get("target")
            if source and target:
                G.add_edge(source, target, **flow)
        return G
    
    def identify_component_from_line(self, line: str) -> str:
        """
        Identify component based on line content.
        Simple implementation that doesn't rely on complex patterns.
        
        Args:
            line: A line of log text
            
        Returns:
            Component ID or "unknown" if no match found
        """
        if not line:
            return "unknown"
        return "unknown"
    
    def identify_component_from_log_file(self, log_file_path: str) -> str:
        """
        Identify component based on filename pattern using a simplified approach.
        
        Args:
            log_file_path: Filename to analyze
            
        Returns:
            Component ID based on filename
        """
        if not log_file_path:
            return "unknown"
        
        filename = log_file_path.lower()
        
        # Special cases
        if 'app_debug.log' in filename:
            return 'soa'
        elif '.har' in filename or '.chlsj' in filename:
            return 'ip_traffic'
        
        # Check schema-defined mappings first
        for component_id, sources in self.component_log_sources.items():
            for source_pattern in sources:
                if source_pattern.lower() == filename:
                    return component_id
                if '*' in source_pattern:
                    pattern_part = source_pattern.strip('*').lower()
                    if pattern_part and pattern_part in filename:
                        return component_id
        
        # Standard case: use the base filename without extension
        base_name = os.path.basename(filename)
        component_name = os.path.splitext(base_name)[0]
        return component_name
    
    def identify_component_from_log_entry(self, log_entry: Any) -> str:
        """
        Identify which component generated this log entry.
        Uses simplified filename-based approach.
        
        Args:
            log_entry: A log entry object with file attribute
            
        Returns:
            Component ID or "unknown" if no match found
        """
        file_path = None
        if hasattr(log_entry, 'file'):
            file_path = log_entry.file
        elif isinstance(log_entry, dict) and 'file' in log_entry:
            file_path = log_entry['file']
            
        if file_path:
            component_id = self.identify_component_from_log_file(file_path)
            if component_id:
                return component_id
        
        return "unknown"
    
    def enrich_log_entries_with_components(self, log_entries: List[Any]) -> List[Any]:
        """
        Enrich log entries with component information.
        
        Args:
            log_entries: List of log entry objects
            
        Returns:
            Enriched log entries
        """
        logging.info(f"Enriching {len(log_entries)} log entries with component information")
        if not log_entries:
            logging.warning("No log entries to enrich")
            return log_entries
        
        component_counts = defaultdict(int)
        
        for entry in log_entries:
            component_id = self.identify_component_from_log_entry(entry)
            component_counts[component_id] += 1
            
            component_source = "filename"
            
            if isinstance(entry, dict):
                entry['component'] = component_id
                entry['component_source'] = component_source
            else:
                entry.__dict__['component'] = component_id
                entry.__dict__['component_source'] = component_source
        
        total_entries = len(log_entries)
        for component_id, count in sorted(component_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_entries) * 100 if total_entries > 0 else 0
            logging.info(f"Component {component_id}: {count} entries ({percentage:.1f}%)")
                
        return log_entries
    
    def analyze_component_failures(self, errors: List[Any]) -> Dict[str, Any]:
        """
        Analyze component failures and their relationships.
        
        Args:
            errors: List of error objects
            
        Returns:
            Analysis results containing component statistics and relationships
        """
        error_counts = defaultdict(int)
        severity_counts = defaultdict(lambda: defaultdict(int))
        
        for error in errors:
            component = getattr(error, 'component', None) or error.get('component', 'unknown')
            severity = getattr(error, 'severity', None) or error.get('severity', 'Medium')
            
            error_counts[component] += 1
            severity_counts[component][severity] += 1
        
        components_with_issues = [
            component for component, count in error_counts.items() 
            if count > 0 and component != 'unknown'
        ]
        
        self.root_cause_component = self._identify_root_cause_component(errors)
        causality_graph = self._build_causality_graph(errors)
        
        propagation_paths = []
        if self.root_cause_component and self.root_cause_component != 'unknown':
            for target in components_with_issues:
                if target != self.root_cause_component:
                    try:
                        paths = list(nx.all_simple_paths(
                            self.component_relationships, 
                            self.root_cause_component, 
                            target
                        ))
                        for path in paths:
                            propagation_paths.append(path)
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        pass
        
        # Collect results with explicit component information
        results = {
            "component_error_counts": dict(error_counts),
            "severity_by_component": {k: dict(v) for k, v in severity_counts.items()},
            "components_with_issues": components_with_issues,
            "root_cause_component": self.root_cause_component,
            "primary_issue_component": self.root_cause_component,  # Add for component preservation
            "propagation_paths": propagation_paths,
            "causality_graph": self._graph_to_dict(causality_graph)
        }
        
        return results
    
    def _identify_root_cause_component(self, errors: List[Any]) -> Optional[str]:
        """
        Attempt to identify the root cause component based on error timing and relationships.
        
        Args:
            errors: List of error objects
            
        Returns:
            Component ID of the likely root cause or None
        """
        errors_by_component = defaultdict(list)
        
        for error in errors:
            component = getattr(error, 'component', None) or error.get('component', 'unknown')
            errors_by_component[component].append(error)
        
        if not errors_by_component or list(errors_by_component.keys()) == ['unknown']:
            return None
            
        if len(errors_by_component) == 1:
            return list(errors_by_component.keys())[0]
        
        earliest_error_time = {}
        for component, component_errors in errors_by_component.items():
            timestamps = []
            for error in component_errors:
                ts = getattr(error, 'timestamp', None) or error.get('timestamp')
                if ts:
                    dt_ts = self._ensure_datetime(ts)
                    if dt_ts:
                        timestamps.append(dt_ts)
            if timestamps:
                earliest_error_time[component] = min(timestamps)
        
        if earliest_error_time:
            earliest_component = min(earliest_error_time.items(), key=lambda x: x[1])[0]
            return earliest_component
        
        severity_scores = {}
        for component, component_errors in errors_by_component.items():
            score = 0
            for error in component_errors:
                severity = getattr(error, 'severity', None) or error.get('severity', 'Medium')
                if severity == 'High':
                    score += 3
                elif severity == 'Medium':
                    score += 1
            severity_scores[component] = score
            
        if severity_scores:
            return max(severity_scores.items(), key=lambda x: x[1])[0]
        return None
    
    def _build_causality_graph(self, errors: List[Any], 
                             max_time_delta: timedelta = timedelta(seconds=10)) -> nx.DiGraph:
        """
        Build a directed graph of potential cause-effect relationships between errors.
        
        Args:
            errors: List of error objects
            max_time_delta: Maximum time difference to consider for causality
            
        Returns:
            NetworkX DiGraph of error causality
        """
        G = nx.DiGraph()
        for i, error in enumerate(errors):
            error_id = getattr(error, 'id', None) or error.get('id', f"error_{i}")
            attrs = {}
            for attr in ['component', 'severity', 'text', 'timestamp', 'file', 'line_number']:
                value = getattr(error, attr, None) or error.get(attr)
                if value is not None:
                    attrs[attr] = value
            G.add_node(error_id, **attrs)
        
        sorted_errors = []
        for i, error in enumerate(errors):
            ts = getattr(error, 'timestamp', None) or error.get('timestamp')
            error_id = getattr(error, 'id', None) or error.get('id', f"error_{i}")
            if ts:
                dt_ts = self._ensure_datetime(ts)
                if dt_ts:
                    sorted_errors.append((error_id, error, dt_ts))
        sorted_errors.sort(key=lambda x: x[2])
        
        for i in range(len(sorted_errors) - 1):
            for j in range(i + 1, min(i + 10, len(sorted_errors))):
                error1_id, error1, ts1 = sorted_errors[i]
                error2_id, error2, ts2 = sorted_errors[j]
                if (ts2 - ts1) <= max_time_delta:
                    comp1 = getattr(error1, 'component', None) or error1.get('component', 'unknown')
                    comp2 = getattr(error2, 'component', None) or error2.get('component', 'unknown')
                    if self._are_components_related(comp1, comp2):
                        weight = self._calculate_causality_weight(error1, error2, ts1, ts2)
                        G.add_edge(error1_id, error2_id, weight=weight, time_delta=(ts2-ts1).total_seconds())
        return G
    
    def _are_components_related(self, comp1: str, comp2: str) -> bool:
        """
        Check if two components have a direct or indirect relationship.
        
        Args:
            comp1: First component ID
            comp2: Second component ID
            
        Returns:
            True if components are related, False otherwise
        """
        if comp1 == 'unknown' or comp2 == 'unknown':
            return False
        if self.component_relationships.has_edge(comp1, comp2):
            return True
        try:
            paths = list(nx.all_simple_paths(
                self.component_relationships, comp1, comp2, cutoff=2
            ))
            return len(paths) > 0
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return False
    
    def _calculate_causality_weight(self, error1: Any, error2: Any, 
                                  ts1: datetime, ts2: datetime) -> float:
        """
        Calculate a weight representing the likelihood of causality between errors.
        
        Args:
            error1: First error
            error2: Second error
            ts1: Timestamp of first error
            ts2: Timestamp of second error
            
        Returns:
            Causality weight (0.0 to 1.0)
        """
        weight = 1.0
        time_diff = (ts2 - ts1).total_seconds()
        if time_diff < 1:
            weight *= 2.0
        elif time_diff < 5:
            weight *= 1.5
        sev1 = getattr(error1, 'severity', None) or error1.get('severity', 'Medium')
        sev2 = getattr(error2, 'severity', None) or error2.get('severity', 'Medium')
        if sev1 == 'High':
            weight *= 1.5
        comp1 = getattr(error1, 'component', None) or error1.get('component', 'unknown')
        comp2 = getattr(error2, 'component', None) or error2.get('component', 'unknown')
        if self.component_relationships.has_edge(comp1, comp2):
            weight *= 2.0
        return weight
    
    def _graph_to_dict(self, G: nx.Graph) -> Dict:
        """
        Convert a NetworkX graph to a serializable dictionary.
        Ensures component information is preserved.
        
        Args:
            G: NetworkX graph to convert
            
        Returns:
            Dictionary representation of the graph
        """
        if not G:
            return {"nodes": [], "edges": []}
            
        result = {
            "nodes": [],
            "edges": []
        }
        
        # Process nodes with component preservation
        for node in G.nodes():
            node_data = {"id": node}
            # Add all node attributes
            node_data.update(G.nodes[node])
            # Ensure component info is preserved
            if "component" in G.nodes[node]:
                node_data["component"] = G.nodes[node]["component"]
            result["nodes"].append(node_data)
        
        # Process edges with component information
        for u, v in G.edges():
            edge_data = {"source": u, "target": v}
            # Add all edge attributes
            edge_data.update(G.edges[u, v])
            result["edges"].append(edge_data)
        
        return result
    
    def get_component_info(self, component_id: str) -> Dict:
        """
        Get information about a specific component from schema.
        
        Args:
            component_id: Component identifier
            
        Returns:
            Component information dictionary
        """
        for component in self.component_schema.get("components", []):
            if component.get("id") == component_id:
                return component
        return {"id": component_id, "name": component_id, "description": "Unknown component"}
        
    def export_error_graph(self, output_path: str, test_id: str = "Unknown") -> str:
        """
        Export error graph as a JSON file for visualization.
        
        Args:
            output_path: Directory to save the JSON file
            test_id: Test ID for filename
            
        Returns:
            Path to the exported JSON file
        """
        if not hasattr(self, 'error_graph') or not self.error_graph:
            logging.warning("No error graph to export")
            return None
            
        try:
            # Convert graph to dictionary
            graph_dict = self._graph_to_dict(self.error_graph)
            
            # Add metadata
            graph_dict["metadata"] = {
                "test_id": test_id,
                "timestamp": datetime.now().isoformat(),
                "node_count": len(self.error_graph.nodes()),
                "edge_count": len(self.error_graph.edges())
            }
            
            # Add component information explicitly
            graph_dict["primary_issue_component"] = self.root_cause_component
            
            # Determine output path
            from utils.path_utils import get_output_path, OutputType, get_standardized_filename
            json_path = get_output_path(
                output_path,
                test_id,
                get_standardized_filename(test_id, "error_graph", "json"),
                OutputType.JSON_DATA
            )
            
            # Serialize with component awareness
            if JSON_UTILS_AVAILABLE:
                if 'serialize_to_json_file' in globals() or 'serialize_to_json_file' in locals():
                    # Use new helper function
                    serialize_to_json_file(
                        graph_dict, 
                        json_path, 
                        primary_issue_component=self.root_cause_component
                    )
                else:
                    # Fall back to basic serialization
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(graph_dict, f, cls=DateTimeEncoder, indent=2)
            else:
                # Use lambda factory approach for component preservation
                with open(json_path, 'w', encoding='utf-8') as f:
                    try:
                        from reports.base import ComponentAwareEncoder
                        json.dump(
                            graph_dict, 
                            f, 
                            cls=lambda *args, **kwargs: ComponentAwareEncoder(
                                *args, primary_issue_component=self.root_cause_component, **kwargs
                            ), 
                            indent=2
                        )
                    except ImportError:
                        json.dump(graph_dict, f, cls=DateTimeEncoder, indent=2)
                        
            # Verify component preservation
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    loaded_data = json.load(f)
                    
                # Check if primary_issue_component was preserved
                if "primary_issue_component" in graph_dict and "primary_issue_component" in loaded_data:
                    if graph_dict["primary_issue_component"] != loaded_data["primary_issue_component"]:
                        logging.warning(
                            f"Component information not preserved: primary_issue_component "
                            f"'{graph_dict['primary_issue_component']}' -> '{loaded_data['primary_issue_component']}'"
                        )
            except Exception as verify_error:
                logging.warning(f"Could not verify component preservation: {str(verify_error)}")
                
            return json_path
                
        except Exception as e:
            logging.error(f"Error exporting error graph: {str(e)}")
            return None
            
    def save_component_data(self, output_path: str, test_id: str = "Unknown") -> str:
        """
        Save component data to a JSON file.
        
        Args:
            output_path: Directory to save the file
            test_id: Test ID for filename
            
        Returns:
            Path to the saved file
        """
        # Create component data dictionary
        component_data = {
            "components": [
                self.get_component_info(comp_id) 
                for comp_id in self.component_relationships.nodes()
            ],
            "relationships": [
                {
                    "source": u,
                    "target": v,
                    **self.component_relationships.edges[u, v]
                }
                for u, v in self.component_relationships.edges()
            ],
            "test_id": test_id,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add component information explicitly
        component_data["primary_issue_component"] = self.root_cause_component
        
        # Determine output path
        from utils.path_utils import get_output_path, OutputType, get_standardized_filename
        json_path = get_output_path(
            output_path,
            test_id,
            get_standardized_filename(test_id, "component_data", "json"),
            OutputType.JSON_DATA
        )
        
        # Serialize with component awareness
        if JSON_UTILS_AVAILABLE:
            if 'serialize_to_json_file' in globals() or 'serialize_to_json_file' in locals():
                serialize_to_json_file(
                    component_data, 
                    json_path, 
                    primary_issue_component=self.root_cause_component
                )
            else:
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(component_data, f, cls=DateTimeEncoder, indent=2)
        else:
            with open(json_path, 'w', encoding='utf-8') as f:
                try:
                    from reports.base import ComponentAwareEncoder
                    json.dump(
                        component_data, 
                        f, 
                        cls=lambda *args, **kwargs: ComponentAwareEncoder(
                            *args, primary_issue_component=self.root_cause_component, **kwargs
                        ), 
                        indent=2
                    )
                except ImportError:
                    json.dump(component_data, f, cls=DateTimeEncoder, indent=2)
                    
        return json_path
"""
Append the following function to the end of your existing direct_component_analyzer.py file:
"""

def assign_components_and_relationships(errors: List[Dict]) -> Tuple[List[Dict], List[Dict], str]:
    """
    Optimized main function to assign components to errors and identify relationships.
    Uses caching to avoid redundant processing and eliminates deep copying.
    
    Args:
        errors: List of error dictionaries
        
    Returns:
        Tuple of (updated errors, component summary, primary issue component)
    """
    # Create an analyzer instance
    analyzer = ComponentAnalyzer()
    
    # Make a copy of errors to avoid modifying the original
    processed_errors = copy.deepcopy(errors)
    
    # Assign components to each error
    for error in processed_errors:
        analyzer.assign_component_to_error(error)
    
    # Identify the primary component responsible for issues
    primary_issue_component = analyzer.identify_primary_component()
    
    # Generate component summary for reporting
    component_summary = analyzer.generate_component_summary()
    
    # Add root cause information to the first error if available
    if processed_errors:
        analyzer.add_root_cause_info(processed_errors)
    
    logging.info(f"Component analysis complete. Primary issue component: {primary_issue_component}")
    logging.info(f"Component distribution: {analyzer._component_counts}")
    
    return processed_errors, component_summary, primary_issue_component