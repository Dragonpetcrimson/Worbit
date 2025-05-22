# context_aware_clusterer.py
import os
import numpy as np
import re
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from datetime import datetime, timedelta
import networkx as nx
import json

class ContextAwareClusterer:
    """
    Enhanced error clustering that takes into account component relationships,
    temporal sequences, and cause-effect relationships.
    """
    
    def __init__(self, component_schema_path: str):
        """
        Initialize the clusterer with component schema.
        
        Args:
            component_schema_path: Path to component schema JSON
        """
        self.component_schema = self._load_schema(component_schema_path)
        self.component_graph = self._build_component_graph()
        self.error_graph = None  # Will be built during clustering
    
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
            
        # If it's already a datetime object, return it
        if isinstance(timestamp_value, datetime):
            return timestamp_value
            
        # If it's a string, try to convert it
        if isinstance(timestamp_value, str):
            # Try standard ISO format first
            try:
                return datetime.fromisoformat(timestamp_value)
            except ValueError:
                pass
                
            # Try common formats
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
                    
            # Try to extract timestamp using regex
            # Look for patterns like HH:MM:SS or HH:MM:SS.microseconds
            match = re.search(r'(\d{2}:\d{2}:\d{2}(?:\.\d+)?)', timestamp_value)
            if match:
                time_str = match.group(1)
                try:
                    # Assume today's date with the extracted time
                    today = datetime.now().strftime("%Y-%m-%d")
                    return datetime.strptime(f"{today} {time_str}", "%Y-%m-%d %H:%M:%S.%f" if "." in time_str else "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    pass
        
        # For any other type or if all conversions fail
        logging.warning(f"Could not convert timestamp: {timestamp_value} ({type(timestamp_value)})")
        return None
    
    def _load_schema(self, schema_path: str) -> Dict:
        """Load the component schema from a JSON file."""
        try:
            with open(schema_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error loading component schema: {str(e)}")
            # Return a minimal schema to prevent crashes
            return {"components": [], "dataFlows": []}
    
    def _build_component_graph(self) -> nx.DiGraph:
        """Build a directed graph of component relationships."""
        G = nx.DiGraph()
        
        # Add all components as nodes
        for component in self.component_schema.get("components", []):
            component_id = component.get("id")
            if component_id:
                G.add_node(component_id, **component)
        
        # Add edges based on dataFlows
        for flow in self.component_schema.get("dataFlows", []):
            source = flow.get("source")
            target = flow.get("target")
            if source and target:
                G.add_edge(source, target, **flow)
                
        return G
        
    def cluster_errors(self, errors: List[Dict], num_clusters: Optional[int] = None) -> Dict[int, List[Dict]]:
        """
        Cluster errors with awareness of component relationships and temporal sequence.
        
        Args:
            errors: List of error objects
            num_clusters: Optional number of clusters (auto-determined if None)
            
        Returns:
            Dictionary mapping cluster IDs to lists of errors
        """
        if not errors:
            return {}
            
        # Extract error texts for TF-IDF
        texts = []
        for error in errors:
            # Get error text, normalizing format
            text = error.get('text', '')
            if not text and 'message' in error:
                text = error.get('message', '')
            
            # Clean and normalize the text
            text = self._normalize_error_text(text)
            texts.append(text)
        
        # Get component information if available
        components = []
        for error in errors:
            components.append(error.get('component', 'unknown'))
        
        # Vectorize error texts
        tfidf_matrix = self._vectorize_errors(texts)
        
        # Auto-determine number of clusters if not specified
        if num_clusters is None or num_clusters > len(errors):
            num_clusters = self._determine_optimal_clusters(tfidf_matrix, len(errors))
            
        # Apply KMeans clustering
        if tfidf_matrix.shape[0] > num_clusters:
            kmeans = KMeans(
                n_clusters=num_clusters,
                random_state=42,
                n_init=10
            )
            labels = kmeans.fit_predict(tfidf_matrix)
        else:
            # If too few samples, assign each to its own cluster
            labels = list(range(len(errors)))
        
        # Group errors by cluster
        clusters = defaultdict(list)
        for idx, label in enumerate(labels):
            clusters[int(label)].append(errors[idx])
        
        # Enhance clusters with component and temporal information
        enhanced_clusters = self._enhance_clusters(dict(clusters), components)
        
        # Build error relationship graph
        self.error_graph = self._build_error_graph(enhanced_clusters, errors)
        
        return enhanced_clusters
    
    def _normalize_error_text(self, text: str) -> str:
        """
        Normalize error text for better clustering by removing variable parts.
        
        Args:
            text: Original error text
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
            
        # Replace specific patterns with placeholders
        # 1. Replace timestamps
        text = re.sub(r'\d{2}:\d{2}:\d{2}', 'TIMESTAMP', text)
        
        # 2. Replace IDs, UUIDs
        text = re.sub(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', 'UUID', text)
        
        # 3. Replace file paths but keep the file name
        text = re.sub(r'(?:[a-zA-Z]:\\|\/)((?:[^\\\/]+\\|\/)+)([^\\\/]+)', r'PATH/\2', text)
        
        # 4. Replace memory addresses
        text = re.sub(r'0x[0-9a-f]+', 'MEMORY_ADDR', text)
        
        # 5. Replace sequence numbers
        text = re.sub(r'\b\d+\b', 'NUM', text)
        
        # 6. Lower case
        text = text.lower()
        
        return text
    
    def _vectorize_errors(self, texts: List[str]) -> np.ndarray:
        """
        Convert error texts to TF-IDF vectors.
        
        Args:
            texts: List of error texts
            
        Returns:
            TF-IDF matrix
        """
        vectorizer = TfidfVectorizer(
            stop_words='english',
            lowercase=True,
            min_df=1,
            max_df=0.9,
            ngram_range=(1, 2)
        )
        
        try:
            return vectorizer.fit_transform(texts)
        except Exception as e:
            logging.error(f"Error during vectorization: {str(e)}")
            # Return a dummy matrix to avoid crashes
            return np.zeros((len(texts), 1))
    
    def _determine_optimal_clusters(self, matrix: np.ndarray, num_errors: int) -> int:
        """
        Determine the optimal number of clusters based on dataset size.
        
        Args:
            matrix: TF-IDF matrix
            num_errors: Number of errors
            
        Returns:
            Recommended number of clusters
        """
        # Heuristic: sqrt of number of samples, capped between 2 and 8
        k = min(max(2, int(np.sqrt(num_errors))), 8)
        
        # Safely access matrix dimensions for mocks
        try:
            n_features = int(getattr(matrix, 'shape', (0, 0))[1])
        except Exception:
            n_features = 0

        if n_features < 10:  # Very few features
            k = min(k, 3)  # Reduce clusters for low feature count
            
        # CRITICAL FIX: Ensure we don't have more clusters than samples
        try:
            n_samples = int(getattr(matrix, 'shape', (num_errors, 0))[0])
        except Exception:
            n_samples = num_errors
        k = min(k, n_samples)
        
        logging.info(f"Context-aware clustering: Using {k} clusters for {num_errors} errors")
        return k
    
    def _enhance_clusters(self, clusters: Dict[int, List[Dict]], 
                        components: List[str]) -> Dict[int, List[Dict]]:
        """
        Enhance clusters with component and temporal information.
        
        Args:
            clusters: Dictionary of initial clusters
            components: List of component IDs corresponding to errors
            
        Returns:
            Enhanced clusters
        """
        # Step 1: Analyze component distribution in each cluster
        component_distribution = {}
        for cluster_id, errors in clusters.items():
            comp_counts = defaultdict(int)
            for i, error in enumerate(errors):
                comp = components[i] if i < len(components) else 'unknown'
                comp_counts[comp] += 1
            
            # Determine primary component for cluster
            if comp_counts:
                primary_component = max(comp_counts.items(), key=lambda x: x[1])[0]
            else:
                primary_component = 'unknown'
                
            component_distribution[cluster_id] = {
                'primary_component': primary_component,
                'distribution': dict(comp_counts)
            }
        
        # Step 2: Identify related clusters based on component relationships
        related_clusters = defaultdict(set)
        for c1 in clusters.keys():
            for c2 in clusters.keys():
                if c1 != c2:
                    comp1 = component_distribution[c1]['primary_component']
                    comp2 = component_distribution[c2]['primary_component']
                    
                    # Check if components are related
                    if self._are_components_related(comp1, comp2):
                        related_clusters[c1].add(c2)
        
        # Step 3: Analyze temporal relationships between clusters
        temporal_relationships = self._analyze_temporal_relationships(clusters)
        
        # Step 4: Determine root cause vs. symptom clusters
        root_vs_symptom = self._classify_root_vs_symptom(
            clusters, 
            related_clusters, 
            temporal_relationships
        )
        
        # Step 5: Tag errors with enhanced information
        for cluster_id, errors in clusters.items():
            for error in errors:
                error['cluster_id'] = cluster_id
                error['primary_component'] = component_distribution[cluster_id]['primary_component']
                error['is_root_cause'] = cluster_id in root_vs_symptom['root_cause_clusters']
                error['related_clusters'] = list(related_clusters[cluster_id])
        
        return clusters
    
    def _are_components_related(self, comp1: str, comp2: str) -> bool:
        """Check if two components have a relationship in the component graph."""
        if comp1 == 'unknown' or comp2 == 'unknown':
            return False
            
        # Check for direct connection
        if self.component_graph.has_edge(comp1, comp2) or self.component_graph.has_edge(comp2, comp1):
            return True
            
        # Check for path with max length 2
        try:
            paths1 = list(nx.all_simple_paths(
                self.component_graph, comp1, comp2, cutoff=2
            ))
            paths2 = list(nx.all_simple_paths(
                self.component_graph, comp2, comp1, cutoff=2
            ))
            return len(paths1) > 0 or len(paths2) > 0
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return False
    
    def _analyze_temporal_relationships(
        self, 
        clusters: Dict[int, List[Dict]]
    ) -> Dict[str, Any]:
        """
        Analyze temporal relationships between errors in different clusters.
        
        Args:
            clusters: Dictionary of clusters
            
        Returns:
            Dictionary with temporal relationship data
        """
        # Extract timestamps for each cluster
        cluster_timestamps = {}
        for cluster_id, errors in clusters.items():
            timestamps = []
            for error in errors:
                ts = error.get('timestamp')
                if ts:
                    # Convert to datetime if it's a string
                    dt_ts = self._ensure_datetime(ts)
                    if dt_ts:
                        timestamps.append(dt_ts)
            
            if timestamps:
                cluster_timestamps[cluster_id] = {
                    'earliest': min(timestamps),
                    'latest': max(timestamps),
                    'count': len(timestamps)
                }
        
        # Identify sequence relationships (which clusters tend to happen after others)
        sequence_graph = nx.DiGraph()
        for c1 in cluster_timestamps:
            for c2 in cluster_timestamps:
                if c1 != c2:
                    # If c1's earliest error is before c2's earliest error
                    if cluster_timestamps[c1]['earliest'] < cluster_timestamps[c2]['earliest']:
                        # Add an edge from c1 to c2
                        weight = (cluster_timestamps[c2]['earliest'] - 
                                 cluster_timestamps[c1]['earliest']).total_seconds()
                        sequence_graph.add_edge(c1, c2, weight=weight)
        
        # Identify initial clusters (those that happen first)
        initial_clusters = []
        if sequence_graph:
            # Clusters with highest in-degree to out-degree ratio are likely initial
            for node in sequence_graph.nodes():
                in_degree = sequence_graph.in_degree(node)
                out_degree = max(1, sequence_graph.out_degree(node))  # Avoid division by zero
                if in_degree == 0 and out_degree > 0:
                    initial_clusters.append(node)
        
        return {
            'cluster_timestamps': cluster_timestamps,
            'sequence_graph': sequence_graph,
            'initial_clusters': initial_clusters
        }
    
    def _classify_root_vs_symptom(
        self,
        clusters: Dict[int, List[Dict]],
        related_clusters: Dict[int, Set[int]],
        temporal_relationships: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Classify clusters as either root causes or symptoms.
        
        Args:
            clusters: Dictionary of clusters
            related_clusters: Related clusters based on component relationships
            temporal_relationships: Temporal relationship data
            
        Returns:
            Classification results
        """
        # Start with initial clusters from temporal analysis
        initial_clusters = set(temporal_relationships.get('initial_clusters', []))
        
        # Analyze severity distribution
        severity_scores = {}
        for cluster_id, errors in clusters.items():
            high_count = sum(1 for e in errors if e.get('severity') == 'High')
            medium_count = sum(1 for e in errors if e.get('severity') == 'Medium')
            low_count = sum(1 for e in errors if e.get('severity') == 'Low')
            
            # Calculate weighted score
            score = (high_count * 3) + (medium_count * 2) + low_count
            severity_scores[cluster_id] = score
        
        # Combine temporal, relational, and severity information
        root_cause_scores = {}
        for cluster_id in clusters:
            # Initialize score
            score = 0
            
            # Higher score for initial clusters
            if cluster_id in initial_clusters:
                score += 3
                
            # Higher score for clusters with many related clusters (suggesting a trigger)
            rel_cluster_count = len(related_clusters.get(cluster_id, set()))
            score += min(rel_cluster_count, 3)  # Cap at 3 points
            
            # Higher score for more severe clusters
            severity = severity_scores.get(cluster_id, 0)
            normalized_severity = min(severity / 3, 3) if severity > 0 else 0  # Cap at 3 points
            score += normalized_severity
            
            root_cause_scores[cluster_id] = score
        
        # Determine threshold for root cause classification
        threshold = max(root_cause_scores.values()) * 0.7 if root_cause_scores else 0
        
        # Classify clusters
        root_cause_clusters = [
            c for c, score in root_cause_scores.items() 
            if score >= threshold
        ]
        
        # Everything else is a symptom
        symptom_clusters = [
            c for c in clusters.keys() 
            if c not in root_cause_clusters
        ]
        
        return {
            'root_cause_clusters': root_cause_clusters,
            'symptom_clusters': symptom_clusters,
            'root_cause_scores': root_cause_scores
        }
    
    def _build_error_graph(
        self, 
        clusters: Dict[int, List[Dict]], 
        errors: List[Dict]
    ) -> nx.DiGraph:
        """
        Build a directed graph of error relationships.
        
        Args:
            clusters: Dictionary of clusters
            errors: Original list of errors
            
        Returns:
            NetworkX DiGraph of error relationships
        """
        G = nx.DiGraph()
        
        # Add all errors as nodes
        for error in errors:
            error_id = error.get('id', f"{error.get('file', 'unknown')}:{error.get('line_number', 0)}")
            G.add_node(error_id, **error)
        
        # Sort errors by timestamp
        timed_errors = []
        for error in errors:
            ts = error.get('timestamp')
            if ts:
                # Convert to datetime if it's a string
                dt_ts = self._ensure_datetime(ts)
                if dt_ts:
                    error_id = error.get('id', f"{error.get('file', 'unknown')}:{error.get('line_number', 0)}")
                    cluster_id = error.get('cluster_id')
                    timed_errors.append((error_id, error, dt_ts, cluster_id))
        
        # Sort by timestamp
        timed_errors.sort(key=lambda x: x[2])
        
        # Add edges based on temporal sequence and causality likelihood
        for i in range(len(timed_errors) - 1):
            error1_id, error1, ts1, cluster1 = timed_errors[i]
            
            # Look ahead to errors that occurred within 30 seconds
            for j in range(i + 1, len(timed_errors)):
                error2_id, error2, ts2, cluster2 = timed_errors[j]
                
                # If too far apart in time, stop looking ahead
                if (ts2 - ts1).total_seconds() > 30:
                    break
                    
                # Calculate causality weight
                weight = self._calculate_causality_weight(error1, error2, ts1, ts2, cluster1, cluster2)
                
                # Only add edge if weight is significant
                if weight > 0.3:
                    G.add_edge(error1_id, error2_id, weight=weight)
        
        return G
    
    def _calculate_causality_weight(
        self, 
        error1: Dict, 
        error2: Dict, 
        ts1: datetime, 
        ts2: datetime,
        cluster1: Optional[int],
        cluster2: Optional[int]
    ) -> float:
        """
        Calculate a weight representing the likelihood of causality.
        
        Args:
            error1: First error
            error2: Second error
            ts1: Timestamp of first error
            ts2: Timestamp of second error
            cluster1: Cluster ID of first error
            cluster2: Cluster ID of second error
            
        Returns:
            Causality weight (0.0 to 1.0)
        """
        # Start with base weight
        weight = 0.5
        
        # Adjust based on time proximity (closer = higher weight)
        time_diff = (ts2 - ts1).total_seconds()
        if time_diff < 1:
            weight += 0.3  # Very close in time
        elif time_diff < 5:
            weight += 0.2  # Fairly close
        elif time_diff > 15:
            weight -= 0.2  # Further apart
            
        # Adjust based on severity (high severity more likely to cause others)
        sev1 = error1.get('severity', 'Medium')
        sev2 = error2.get('severity', 'Medium')
        
        if sev1 == 'High' and sev2 != 'High':
            weight += 0.2  # High severity error causing lower severity
        elif sev1 != 'High' and sev2 == 'High':
            weight -= 0.1  # Lower severity unlikely to cause high severity
            
        # Adjust based on component relationship
        comp1 = error1.get('component', 'unknown')
        comp2 = error2.get('component', 'unknown')
        
        if comp1 == comp2:
            weight += 0.1  # Same component
        elif self._are_components_related(comp1, comp2):
            if self.component_graph.has_edge(comp1, comp2):
                weight += 0.2  # Direct data flow relationship
            else:
                weight += 0.1  # Indirect relationship
        else:
            weight -= 0.2  # Unrelated components
            
        # Adjust based on cluster relationship
        if cluster1 is not None and cluster2 is not None:
            if cluster1 == cluster2:
                weight += 0.1  # Same cluster
                
        # Ensure weight is between 0 and 1
        return max(0.0, min(1.0, weight))
    
    def get_root_cause_errors(self, clusters: Dict[int, List[Dict]]) -> List[Dict]:
        """
        Get the list of errors identified as potential root causes.
        
        Args:
            clusters: Dictionary of clustered errors
            
        Returns:
            List of root cause errors
        """
        root_cause_errors = []
        
        for cluster_id, errors in clusters.items():
            for error in errors:
                if error.get('is_root_cause', False):
                    root_cause_errors.append(error)
        
        # Sort by severity
        return sorted(
            root_cause_errors,
            key=lambda e: {'High': 0, 'Medium': 1, 'Low': 2}.get(e.get('severity', 'Low'), 3)
        )
    
    def get_causality_paths(self) -> List[List[Dict]]:
        """
        Get potential causality paths from the error graph.
        
        Returns:
            List of error paths representing causal chains
        """
        if not self.error_graph:
            return []
            
        # Find source nodes (potential initial causes)
        sources = [n for n in self.error_graph.nodes() 
                 if self.error_graph.in_degree(n) == 0 and self.error_graph.out_degree(n) > 0]
        
        # If no clear sources, use nodes with highest severity
        if not sources:
            sources = []
            for node in self.error_graph.nodes():
                severity = self.error_graph.nodes[node].get('severity', 'Low')
                if severity == 'High' and self.error_graph.out_degree(node) > 0:
                    sources.append(node)
        
        # If still no sources, just use nodes with highest out_degree
        if not sources:
            sources = sorted(
                self.error_graph.nodes(),
                key=lambda n: self.error_graph.out_degree(n),
                reverse=True
            )[:3]  # Top 3 nodes with highest out-degree
        
        # Find all paths from source nodes
        paths = []
        for source in sources:
            # Find all nodes reachable from this source
            for target in self.error_graph.nodes():
                if target != source and self.error_graph.has_node(target):
                    try:
                        # Find simple paths with max length of 5
                        simple_paths = list(nx.all_simple_paths(
                            self.error_graph, source, target, cutoff=5
                        ))
                        for path in simple_paths:
                            # Convert node IDs to error objects
                            error_path = [self.error_graph.nodes[n] for n in path]
                            paths.append(error_path)
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        # No path found or node not in graph
                        pass
        
        # Sort paths by combined weight
        weighted_paths = []
        for path in paths:
            total_weight = 0
            for i in range(len(path) - 1):
                error1 = path[i]
                error2 = path[i+1]
                error1_id = error1.get('id', f"{error1.get('file', 'unknown')}:{error1.get('line_number', 0)}")
                error2_id = error2.get('id', f"{error2.get('file', 'unknown')}:{error2.get('line_number', 0)}")
                
                if self.error_graph.has_edge(error1_id, error2_id):
                    weight = self.error_graph.edges[error1_id, error2_id].get('weight', 0)
                    total_weight += weight
            
            weighted_paths.append((path, total_weight))
        
        # Sort by weight and return top 10 paths
        return [p[0] for p in sorted(weighted_paths, key=lambda x: x[1], reverse=True)[:10]]
    
    def export_error_graph(self, output_path: str, test_id: str = "Unknown") -> str:
        """
        Export error graph as a JSON file for visualization.
        
        Args:
            output_path: Directory to save the JSON file
            test_id: Test ID for filename
            
        Returns:
            Path to the exported JSON file
        """
        if not self.error_graph:
            return None
            
        os.makedirs(output_path, exist_ok=True)
        json_path = os.path.join(output_path, f"{test_id}_error_graph.json")
        
        # Convert graph to dictionary
        graph_dict = {
            "nodes": [],
            "links": []
        }
        
        # Add nodes
        for node in self.error_graph.nodes():
            node_data = self.error_graph.nodes[node].copy()
            # Convert non-serializable objects
            for key, value in list(node_data.items()):
                if not isinstance(value, (str, int, float, bool, list, dict, type(None))):
                    node_data[key] = str(value)
            
            # Add required visualization fields
            node_data["id"] = node
            graph_dict["nodes"].append(node_data)
        
        # Add links
        for u, v, data in self.error_graph.edges(data=True):
            link_data = data.copy()
            link_data["source"] = u
            link_data["target"] = v
            graph_dict["links"].append(link_data)
        
        # Save to JSON
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(graph_dict, f, indent=2)
            return json_path
        except Exception as e:
            logging.error(f"Error exporting error graph: {str(e)}")
            return None
