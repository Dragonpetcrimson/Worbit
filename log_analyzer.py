# log_analyzer.py - Enhanced with better error filtering, HAR support, and component analysis
import os
import re
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import logging
import sys

# Explicitly import the cluster_errors function - used in analyze_error_clusters
from error_clusterer import cluster_errors

if getattr(sys, 'frozen', False):
    # Running as bundled .exe
    base_path = sys._MEIPASS
else:
    # Running in development
    base_path = os.path.abspath(".")

component_schema_path = os.path.join(base_path, 'components', 'schemas', 'component_schema.json')

component_analyzer = None
try:
    from components.component_analyzer import ComponentAnalyzer
    if os.path.exists(component_schema_path):
        component_analyzer = ComponentAnalyzer(component_schema_path)
        logging.info("Component analyzer initialized successfully")
    else:
        logging.warning(f"Component schema not found at {component_schema_path}")
except ImportError as e:
    logging.warning(f"Component analyzer module not available: {str(e)}")

def extract_timestamp(line):
    """Extract timestamp from various log line formats."""
    timestamp_patterns = [
        r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}[.,]\d+)',
        r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})',
        r'(\d{2}:\d{2}:\d{2}[.,]\d+)',
        r'(\d{2}:\d{2}:\d{2})',
        r'(\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2})',
        r'(\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})'
    ]
    for pattern in timestamp_patterns:
        match = re.search(pattern, line)
        if match:
            timestamp_str = match.group(1)
            # Try to convert string to datetime object
            try:
                # Try different formats based on the matched pattern
                for fmt in [
                    '%Y-%m-%d %H:%M:%S.%f',
                    '%Y-%m-%d %H:%M:%S,%f', 
                    '%Y-%m-%d %H:%M:%S',
                    '%H:%M:%S.%f',
                    '%H:%M:%S',
                    '%m/%d/%Y %H:%M:%S',
                    '%b %d %H:%M:%S'
                ]:
                    try:
                        return datetime.strptime(timestamp_str, fmt)
                    except ValueError:
                        continue
            except Exception:
                # If conversion fails, return the string
                pass
            return timestamp_str
    return "No timestamp"

def determine_severity(error_text):
    """Determine error severity based on text content."""
    high_severity = ['fatal', 'crash', 'exception', 'null', 'undefined', 'failure', 'timeout']
    medium_severity = ['error', 'warning', 'failed', 'not found']
    error_text_lower = error_text.lower()
    if any(word in error_text_lower for word in high_severity):
        return "High"
    elif any(word in error_text_lower for word in medium_severity):
        return "Medium"
    else:
        return "Low"

def is_false_positive(line):
    """Check if an error line is likely a false positive."""
    false_positive_patterns = [
        r'INFO:.*error',
        r'DEBUG:.*error',
        r'expected error in test case',
        r'simulated error for testing',
        r'intentional error condition',
        r'error handling disabled',
        r'error checking skipped',
        r'error pages configured correctly',
        r'successfully handled error',
        r'error rate is 0',
        r'no errors found'
    ]
    line_lower = line.lower()
    if re.match(r'^\s*\[?(INFO|DEBUG)', line) and 'error' in line_lower:
        return True
    for pattern in false_positive_patterns:
        if re.search(pattern, line, re.IGNORECASE):
            return True
    return False

def identify_component_from_filename(filename):
    """
    Identify component based solely on filename, with special cases for known files.
    
    Args:
        filename: Filename to analyze
        
    Returns:
        Component identifier string and source as a tuple
    """
    if not filename:
        return 'unknown', 'default'
    
    filename = os.path.basename(filename).lower()
    
    # Special cases
    if 'app_debug.log' in filename:
        return 'soa', 'filename_special'
    elif '.har' in filename or '.chlsj' in filename:
        return 'ip_traffic', 'filename_special'
    
    # Use filename without extension as component
    component_name = os.path.splitext(filename)[0]
    return component_name, 'filename'

def parse_logs(log_paths: List[str], context_lines: int = 3) -> List[Dict]:
    """
    Parse log files to extract errors and their context, with improved SOA vs Android disambiguation.
    
    Args:
        log_paths: List of paths to log files
        context_lines: Number of preceding lines to include as context
        
    Returns:
        List of error dictionaries with file, line number, text, severity, and context
    """
    results = []

    for path in log_paths:
        try:
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            lines = None

            for encoding in encodings:
                try:
                    with open(path, 'r', encoding=encoding, errors='replace') as f:
                        lines = f.readlines()
                    break
                except UnicodeDecodeError:
                    continue

            if not lines:
                logging.warning(f"Could not read {path} with any of the supported encodings")
                continue

            # Special handling for HAR files
            if path.endswith('.har'):
                try:
                    har_errors = parse_har_file(path, lines)
                    results.extend(har_errors)
                except Exception as e:
                    logging.error(f"Failed to parse HAR file {path}: {str(e)}")
                continue  # Skip line-by-line scan

            # Non-HAR log processing
            context_buffer = []
            buffer_size = context_lines + 1
            
            # Identify component from filename
            component, component_source = identify_component_from_filename(path)

            for i, line in enumerate(lines):
                if len(context_buffer) >= buffer_size:
                    context_buffer.pop(0)
                context_buffer.append((i + 1, line.strip()))

                if any(indicator in line.lower() for indicator in ['error', 'failed', 'not found', 'exception', 'failure']):
                    if is_false_positive(line):
                        continue
                    if any(fp in line.lower() for fp in ['info:', 'debug:', 'successfully']):
                        continue

                    severity = determine_severity(line)
                    timestamp = extract_timestamp(line)
                    context = [f"Line {ctx_num}: {ctx_line}" for ctx_num, ctx_line in context_buffer[:-1]]
                    
                    entry = {
                        'file': path,  # Full path for component analysis
                        'line_num': i + 1,
                        'text': line.strip(),
                        'severity': severity,
                        'timestamp': timestamp,
                        'context': context,
                        'is_error': True,
                        'component': component,
                        'component_source': component_source
                    }
                    results.append(entry)

        except Exception as e:
            logging.error(f"Failed to parse {path}: {str(e)}")

    # Enrich errors with component information if analyzer is available
    if component_analyzer is not None:
        try:
            enriched_results = component_analyzer.enrich_log_entries_with_components(results)
            logging.info(f"Enriched {len(results)} errors with component information")
            return enriched_results
        except Exception as e:
            logging.error(f"Error enriching logs with component info: {str(e)}")
    
    # Return original results if component enrichment fails or isn't available
    return results

def parse_har_file(path: str, lines: List[str]) -> List[Dict]:
    """
    Parse HAR file to extract HTTP errors.
    
    Args:
        path: Path to the HAR file
        lines: Lines read from the HAR file
        
    Returns:
        List of error dictionaries extracted from the HAR file
    """
    results = []
    try:
        har_data = json.loads(''.join(lines))
        
        # Get component info from filename
        component, component_source = identify_component_from_filename(path)
        
        for entry in har_data.get('log', {}).get('entries', []):
            request = entry.get('request', {})
            response = entry.get('response', {})
            status = response.get('status', 0)
            url = request.get('url', '')
            method = request.get('method', '')

            # Consider all 4xx and 5xx as errors
            if status >= 400:
                # Try to parse the timestamp
                timestamp = entry.get('startedDateTime', 'No timestamp')
                try:
                    # HAR timestamps are ISO format
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                except Exception:
                    pass

                results.append({
                    'file': path,  # Full path for component analysis
                    'line_num': 0,
                    'text': f"HTTP {status} Error: {method} {url}",
                    'severity': determine_severity(f"HTTP {status}"),
                    'timestamp': timestamp,
                    'context': [],
                    'is_error': True,
                    'component': component,
                    'component_source': component_source
                })
    except Exception as e:
        logging.error(f"Error parsing HAR file {path}: {str(e)}")
    
    return results

def parse_log_entries(log_path: str) -> List[Dict]:
    """
    Parse a log file to extract all entries (not just errors).
    Used for component relationship analysis.
    
    Args:
        log_path: Path to the log file
        
    Returns:
        List of log entry dictionaries
    """
    entries = []
    
    try:
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        lines = None

        for encoding in encodings:
            try:
                with open(log_path, 'r', encoding=encoding, errors='replace') as f:
                    lines = f.readlines()
                break
            except UnicodeDecodeError:
                continue

        if not lines:
            return entries

        # Identify component from filename
        component, component_source = identify_component_from_filename(log_path)
        
        # Special handling for HAR files
        if log_path.endswith('.har'):
            try:
                har_data = json.loads(''.join(lines))
                for entry in har_data.get('log', {}).get('entries', []):
                    request = entry.get('request', {})
                    response = entry.get('response', {})
                    status = response.get('status', 0)
                    url = request.get('url', '')
                    method = request.get('method', '')
                    
                    # Try to parse the timestamp
                    timestamp = entry.get('startedDateTime', 'No timestamp')
                    try:
                        timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    except Exception:
                        pass
                    
                    entries.append({
                        'file': log_path,
                        'line_number': 0,
                        'text': f"{method} {url} - HTTP {status}",
                        'timestamp': timestamp,
                        'is_error': status >= 400,
                        'format_name': 'har',
                        'component': component,
                        'component_source': component_source
                    })
            except Exception as e:
                logging.error(f"Failed to parse HAR file {log_path}: {str(e)}")
            return entries

        # Non-HAR log processing
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            timestamp = extract_timestamp(line)
            
            # Determine if this is an error line
            is_error = any(indicator in line.lower() for indicator in 
                         ['error', 'failed', 'not found', 'exception', 'failure'])
            
            # Skip false positives
            if is_error and is_false_positive(line):
                is_error = False
                
            entries.append({
                'file': log_path,
                'line_number': i + 1,
                'text': line,
                'timestamp': timestamp,
                'is_error': is_error,
                'format_name': 'log',
                'component': component,
                'component_source': component_source
            })
    
    except Exception as e:
        logging.error(f"Failed to parse entries from {log_path}: {str(e)}")
    
    return entries

def analyze_error_clusters(errors, num_clusters=None):
    """
    Analyze errors by clustering them and extracting common patterns.
    
    Args:
        errors: List of error dictionaries
        num_clusters: Optional number of clusters to create
        
    Returns:
        Dictionary with cluster analysis results
    """
    if not errors:
        logging.info("No errors to analyze")
        return {"clusters": {}, "stats": {"total_errors": 0, "cluster_count": 0}}
    
    # Call the cluster_errors function and store the result in 'clusters' variable
    # This is the only correct variable name - never using 'cluster_errors' as a variable
    clusters = cluster_errors(errors, num_clusters)
    
    # Prepare cluster statistics
    stats = {
        "total_errors": len(errors),
        "cluster_count": len(clusters),
        "avg_errors_per_cluster": len(errors) / max(1, len(clusters)),
        "largest_cluster": max([len(cluster) for cluster in clusters.values()]) if clusters else 0
    }
    
    # Enrich clusters with additional analysis
    enriched_clusters = {}
    for cluster_id, cluster_items in clusters.items():
        # Calculate distribution of components in this cluster
        component_counts = {}
        for error in cluster_items:
            component = error.get('component', 'unknown')
            component_counts[component] = component_counts.get(component, 0) + 1
        
        # Find most common words in this cluster
        all_text = " ".join([error.get('text', '') for error in cluster_items])
        common_words = extract_common_words(all_text)
        
        enriched_clusters[cluster_id] = {
            "errors": cluster_items,
            "size": len(cluster_items),
            "component_distribution": component_counts,
            "common_patterns": common_words
        }
    
    return {
        "clusters": enriched_clusters,
        "stats": stats
    }

def extract_common_words(text, top_n=10):
    """Extract most common meaningful words from text."""
    # Simple implementation - in real code this would be more sophisticated
    words = re.findall(r'\b\w{3,}\b', text.lower())
    word_counts = {}
    
    # Very basic stopwords list
    stopwords = {'the', 'and', 'for', 'with', 'not', 'this', 'from', 'that'}
    
    for word in words:
        if word not in stopwords:
            word_counts[word] = word_counts.get(word, 0) + 1
    
    # Sort by count (descending) and return top N
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    return sorted_words[:top_n]