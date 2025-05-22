# error_clusterer.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
import logging
from typing import List, Dict, Optional
import re
import copy

# Import component utilities for preservation
from components.component_utils import (
    extract_component_fields, 
    apply_component_fields
)

def _determine_optimal_clusters(matrix, num_errors, user_specified=None):
    """
    Determine the optimal number of clusters based on dataset size.
    Improved to prevent overly imbalanced clustering.
    
    Args:
        matrix: TF-IDF matrix
        num_errors: Number of errors
        user_specified: User-specified number of clusters
        
    Returns:
        Recommended number of clusters
    """
    # If user specified a number, use it as a maximum
    max_clusters = user_specified if user_specified is not None else 8
    
    # Heuristic: sqrt of number of samples, capped between 2 and max_clusters
    k = min(max(2, int(np.sqrt(num_errors/2))), max_clusters)  # Divide by 2 to prefer fewer clusters
    
    # Adjust based on matrix density and uniqueness
    if matrix.shape[1] < 10:  # Very few features
        k = min(k, 3)  # Reduce clusters for low feature count
    
    # Ensure we don't request more clusters than we have samples
    k = min(k, matrix.shape[0])
    
    logging.info(f"Determined optimal cluster count: {k} (from {num_errors} errors)")
    return k

def _normalize_error_text(text):
    """Normalize text for better clustering."""
    # Replace specific variable patterns with placeholders
    # IDs, UUIDs, timestamps, file paths, line numbers, etc.
    text = re.sub(r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b', 'UUID', text)
    text = re.sub(r'\d{2}:\d{2}:\d{2}', 'TIMESTAMP', text)
    text = re.sub(r'0x[0-9a-f]+', 'MEMORY_ADDR', text)
    
    # Keep class and method names but replace line numbers
    text = re.sub(r'(\.java|\.py|\.js|\.ts):\d+', r'\1:LINE', text)
    
    return text

def perform_error_clustering(errors: List[Dict], num_clusters: int = None) -> Dict[int, List[Dict]]:
    """
    Cluster errors based on similarity with enhanced component preservation.
    
    Args:
        errors: List of error dictionaries
        num_clusters: Number of clusters to create (auto-determined if None)
        
    Returns:
        Dictionary mapping cluster IDs to lists of errors
    """
    # Early return for empty input
    if not errors:
        logging.warning("No errors to cluster")
        return {}

    # Store original component information
    original_component_fields = {}
    for i, error in enumerate(errors):
        original_component_fields[i] = extract_component_fields(error)

    # Extract error texts for TF-IDF
    texts = []
    for error in errors:
        # Get error text, normalizing format
        text = error.get("text", "")
        if text:
            texts.append(_normalize_error_text(text))
        else:
            texts.append("NO_TEXT")

    # Configure vectorizer
    vectorizer = TfidfVectorizer(
        stop_words='english',
        min_df=1,
        max_df=0.9,
        ngram_range=(1, 2)
    )

    # Perform clustering within try/except block
    try:
        # Transform text to feature vectors
        feature_matrix = vectorizer.fit_transform(texts)
        
        # Determine cluster count if not specified
        if num_clusters is None or num_clusters > len(errors):
            num_clusters = _determine_optimal_clusters(feature_matrix, len(errors), num_clusters)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(feature_matrix)
        
        # Group errors by cluster label with component preservation
        clusters = {}
        for idx, label in enumerate(labels):
            cluster_id = int(label)
            
            if cluster_id not in clusters:
                clusters[cluster_id] = []
                
            # Get original error
            error = errors[idx]
            
            # Create a copy to avoid modifying the original
            error_copy = copy.deepcopy(error)
            
            # Ensure all component fields are preserved
            if idx in original_component_fields:
                apply_component_fields(error_copy, original_component_fields[idx])
                
            # Add to cluster
            clusters[cluster_id].append(error_copy)
        
        # Balance extremely unbalanced clusters
        cluster_sizes = [len(cluster) for cluster in clusters.values()]
        if len(cluster_sizes) > 1:
            avg_size = sum(cluster_sizes) / len(cluster_sizes)
            max_size = max(cluster_sizes)
            
            # Split large clusters if significantly larger than average
            if max_size > 3 * avg_size and max_size >= 10:
                logging.warning(f"Detected imbalanced clustering. Largest cluster ({max_size}) > 3x avg ({avg_size:.1f})")
                
                # Find the largest cluster and split it
                largest_cluster_id = max(clusters, key=lambda k: len(clusters[k]))
                new_cluster_id = max(clusters.keys()) + 1
                cluster_to_split = clusters[largest_cluster_id]
                
                # Simple split at midpoint with component preservation
                midpoint = len(cluster_to_split) // 2
                clusters[new_cluster_id] = cluster_to_split[midpoint:]
                clusters[largest_cluster_id] = cluster_to_split[:midpoint]
                
                logging.info(f"Split cluster {largest_cluster_id} into two clusters to improve balance")
        
        # Log result and return
        logging.info(f"Grouped {len(errors)} errors into {len(clusters)} clusters with component preservation")
        return clusters

    except Exception as e:
        # Handle any exceptions and return fallback dictionary
        logging.error(f"Error during clustering: {str(e)}")
        
        # Even in error case, ensure component preservation
        fallback_clusters = {0: []}
        for idx, error in enumerate(errors):
            # Create a copy with preserved component fields
            error_copy = copy.deepcopy(error)
            if idx in original_component_fields:
                apply_component_fields(error_copy, original_component_fields[idx])
            fallback_clusters[0].append(error_copy)
            
        return fallback_clusters

# For backward compatibility
cluster_errors = perform_error_clustering