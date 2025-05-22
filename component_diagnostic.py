# component_diagnostic.py
import os
import logging
import json
import re
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='component_diagnostic.log',
    filemode='w'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

# Enhanced component model
COMPONENT_MODEL = {
    # Application components
    'soa': {
        'name': 'SOA',
        'description': 'SiriusXM application built on Android',
        'patterns': [
            r'com\.siriusxm',
            r'siriusxm\.soa',
            r'sxm:s:',
            r'SiriusXM',  # General application name
            r'SXM'        # Abbreviated application name
        ]
    },
    # Infrastructure components
    'mimosa': {
        'name': 'Mimosa',
        'description': 'Provides fake testing data (Satellite/IP channel)',
        'patterns': [
            r'mimosa',
            r'data.*?simula',
            r'simula.*?data',
            r'fake.*?channel'
        ]
    },
    'phoebe': {
        'name': 'Phoebe',
        'description': 'Proxy to run data to SOA',
        'patterns': [
            r'phoebe',
            r'proxy.*?data',
            r'proxy.*?soa'
        ]
    },
    'translator': {
        'name': 'Translator',
        'description': 'Translates commands between test framework and SOA',
        'patterns': [
            r'translator',
            r'SiriusXm\.Translator',
            r'\[TX\]',
            r'\[RX\]',
            r'command.*?test'
        ]
    },
    # System components
    'android': {
        'name': 'Android',
        'description': 'Android system logs',
        'patterns': [
            r'ActivityManager',
            r'PackageManager',
            r'dalvik',
            r'BluetoothManager',
            r'System\.err'
        ]
    },
    # Monitoring components
    'ip_traffic': {
        'name': 'IP Traffic',
        'description': 'Network traffic monitoring (via Charles or other tools)',
        'patterns': [
            r'http[s]?://',
            r'GET ',
            r'POST ',
            r'HTTP/[0-9]',
            r'Status: [0-9]{3}',
            r'Connection: '
        ]
    }
}

def identify_component_from_filename(filename):
    """Identify component based solely on filename."""
    filename = os.path.basename(filename).lower()
    
    if 'app_debug' in filename:
        return 'android'  # Changed from 'soa' to 'android'
    elif 'appium' in filename:
        return 'android'  # Also likely Android system logs
    elif 'phoebe' in filename:
        return 'phoebe'
    elif 'arecibo' in filename:
        return 'arecibo'
    elif 'translator' in filename:
        return 'translator'
    elif 'mimosa' in filename:
        return 'mimosa'
    elif 'charles' in filename or 'har' in filename or 'ip.' in filename:
        return 'ip_traffic'  # Changed from 'charles' to 'ip_traffic'
    elif 'telesto' in filename:
        return 'telesto'
    elif 'lapetus' in filename:
        return 'lapetus'
    elif any(img_ext in filename for img_ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']):
        return 'screenshot'  # New component type for images
    else:
        return 'unknown'

def analyze_file_content(file_path, expected_component):
    """
    Analyze file content to refine component identification.
    Returns dictionary with content analysis results.
    """
    # Check if file exists
    if not os.path.exists(file_path):
        return {
            'status': 'error',
            'message': f"File not found: {file_path}",
            'component': expected_component
        }
    
    # Skip binary files and very large files
    try:
        file_size = os.path.getsize(file_path)
        if file_size > 10 * 1024 * 1024:  # Skip files larger than 10MB
            return {
                'status': 'skipped',
                'message': f"File too large ({file_size / 1024 / 1024:.1f} MB)",
                'component': expected_component
            }
        
        # Try to read the file
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            # Read first 50 lines for analysis
            lines = [line.strip() for line in f.readlines()[:50]]
    except Exception as e:
        return {
            'status': 'error',
            'message': f"Error reading file: {str(e)}",
            'component': expected_component
        }
    
    # Count pattern matches for each component
    component_scores = defaultdict(int)
    
    for line in lines:
        for component, info in COMPONENT_MODEL.items():
            for pattern in info['patterns']:
                if re.search(pattern, line, re.IGNORECASE):
                    component_scores[component] += 1
    
    # Calculate component distribution
    total_matches = sum(component_scores.values())
    component_distribution = {}
    
    if total_matches > 0:
        for component, count in component_scores.items():
            component_distribution[component] = count / total_matches
    
    # Determine most likely component
    if component_scores:
        most_likely = max(component_scores.items(), key=lambda x: x[1])[0]
    else:
        most_likely = expected_component
    
    # Special handling for Android logs
    if expected_component == 'android' and 'soa' in component_scores:
        # If we have significant SOA matches in Android logs, it's likely SOA-related
        if component_scores['soa'] > 2:
            most_likely = 'soa'
    
    return {
        'status': 'analyzed',
        'component': most_likely,
        'expected_component': expected_component,
        'component_scores': dict(component_scores),
        'component_distribution': component_distribution,
        'sample_lines': lines[:10]  # Include first 10 lines for reference
    }

def extract_errors_from_file(file_path, component):
    """
    Extract error entries from a file and assign components.
    Returns list of error objects.
    """
    errors = []
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            for i, line in enumerate(f):
                # Simple error detection - look for common error indicators
                lower_line = line.lower()
                if any(indicator in lower_line for indicator in ['error', 'exception', 'fail', 'crash']):
                    # Skip false positives
                    if any(fp in lower_line for fp in ['info:', 'debug:', 'successfully']):
                        continue
                    
                    # Create error object
                    error = {
                        'file': os.path.basename(file_path),
                        'line_num': i + 1,
                        'text': line.strip(),
                        'component': component,
                        'source_component': component
                    }
                    
                    # Refine component based on line content
                    line_component = identify_component_from_line(line)
                    if line_component != 'unknown' and line_component != component:
                        error['component'] = line_component
                        error['component_source'] = 'content'
                    else:
                        error['component_source'] = 'file'
                    
                    errors.append(error)
    except Exception as e:
        logging.error(f"Error processing file {file_path}: {str(e)}")
    
    return errors

def identify_component_from_line(line):
    """Identify component based on line content."""
    for component, info in COMPONENT_MODEL.items():
        for pattern in info['patterns']:
            if re.search(pattern, line, re.IGNORECASE):
                return component
    return 'unknown'

def scan_directory(dir_path):
    """Scan directory for log files and identify components."""
    if not os.path.exists(dir_path):
        logging.error(f"Directory not found: {dir_path}")
        return {}
    
    component_counts = defaultdict(int)
    files_by_component = defaultdict(list)
    file_analysis = {}
    all_errors = []
    
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        if os.path.isfile(file_path):
            # Initial component based on filename
            file_component = identify_component_from_filename(filename)
            
            # Analyze content to refine component identification
            content_analysis = analyze_file_content(file_path, file_component)
            refined_component = content_analysis['component']
            
            # Update counts and lists
            component_counts[refined_component] += 1
            files_by_component[refined_component].append(filename)
            file_analysis[filename] = content_analysis
            
            # Extract errors from the file
            if refined_component != 'screenshot':  # Skip images
                errors = extract_errors_from_file(file_path, refined_component)
                all_errors.extend(errors)
    
    return {
        'counts': dict(component_counts),
        'files': {k: sorted(v) for k, v in files_by_component.items()},
        'file_analysis': file_analysis,
        'errors': all_errors
    }

def create_report(test_id, base_dir='./logs'):
    """Create comprehensive component identification report."""
    logging.info(f"=== Component Diagnostic for {test_id} ===")
    
    # Check test directory
    test_dir = os.path.join(base_dir, test_id)
    if not os.path.exists(test_dir):
        logging.error(f"Test directory not found: {test_dir}")
        return None
    
    # Scan directory
    result = scan_directory(test_dir)
    logging.info(f"Found components: {result['counts']}")
    logging.info(f"Extracted {len(result['errors'])} errors")
    
    # Analyze error components
    error_components = {}
    for error in result['errors']:
        component = error['component']
        error_components[component] = error_components.get(component, 0) + 1
    
    logging.info(f"Error components: {error_components}")
    
    # Create report
    report = {
        'test_id': test_id,
        'component_counts': result['counts'],
        'files_by_component': result['files'],
        'file_analysis': result['file_analysis'],
        'error_components': error_components,
        'errors_sample': result['errors'][:20] if result['errors'] else []
    }
    
    # Write report to file
    output_file = f"{test_id}_component_diagnostic.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    logging.info(f"Report saved to {output_file}")
    return report

def main():
    """Run diagnostic on test ID from command line."""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python component_diagnostic.py <test_id>")
        print("Example: python component_diagnostic.py SXM-2094922")
        return
    
    test_id = sys.argv[1]
    create_report(test_id)
    print("\nRun complete! Check component_diagnostic.log for details.")

if __name__ == "__main__":
    main()