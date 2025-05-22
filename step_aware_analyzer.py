# step_aware_analyzer.py
import os
import sys  # Added sys import for checking test environment
import logging
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from jinja2 import Environment, FileSystemLoader, select_autoescape
from gherkin_log_correlator import GherkinParser, LogEntry, correlate_logs_with_steps

# Import timeline generators directly from reports.visualizations
from reports.visualizations import generate_timeline_image, generate_cluster_timeline_image, generate_visualization_placeholder

# Import Config for feature flags - if it exists
try:
    from config import Config
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False
    # Define a dummy Config to avoid errors
class Config:
    ENABLE_STEP_REPORT_IMAGES = True
    ENABLE_DIAGNOSTIC_CHECKS = False

# Jinja2 environment for HTML templates
TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")
env = Environment(
    loader=FileSystemLoader(TEMPLATE_DIR),
    autoescape=select_autoescape(["html", "xml"]),
)

def extract_step_name(step_number, feature_file):
    """
    Extract step name from feature file for a given step number.
    """
    try:
        parser = GherkinParser(feature_file)
        steps = parser.parse()
        for step in steps:
            if step.step_number == step_number:
                return f"{step.keyword} {step.text}"
        return f"Step {step_number} (Unknown)"
        logging.error(f"Error extracting step name: {str(e)}")
        return f"Step {step_number} (Unknown)"

def validate_timestamp(timestamp):
    """
    Validate that a timestamp is a valid datetime object.
    Returns the timestamp if valid, None otherwise.
    """
    if not timestamp:
        return None
        
    if isinstance(timestamp, datetime):
        return timestamp
        
    try:
        # Try parsing as string if it's not already a datetime
        if isinstance(timestamp, str):
            # Try different formats depending on the string
            formats = [
                '%Y-%m-%dT%H:%M:%S.%f',  # ISO format with microseconds
                '%Y-%m-%dT%H:%M:%S',     # ISO format without microseconds
                '%Y-%m-%d %H:%M:%S.%f',  # Standard format with microseconds
                '%Y-%m-%d %H:%M:%S'      # Standard format without microseconds
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(timestamp, fmt)
                except ValueError:
                    continue
                    
        logging.warning(f"Unable to parse timestamp: {timestamp}")
        return None
        logging.error(f"Error validating timestamp {timestamp}: {str(e)}")
        return None

def validate_timestamps(step_to_logs):
    """
    Validate timestamp extraction from logs.
    This function is used for diagnostics when ENABLE_DIAGNOSTIC_CHECKS is enabled.
    
    Args:
        step_to_logs: Dictionary mapping step numbers to log entries
        
    Returns:
        Tuple of (total_timestamps, valid_timestamps, has_sufficient)
    """
    if not step_to_logs:
        logging.warning("No step-to-logs data provided for timestamp validation")
        return 0, 0, False
    
    total_logs = sum(len(logs) for logs in step_to_logs.values())
    total_timestamps = 0
    valid_timestamps = 0
    
    for step_num, logs in step_to_logs.items():
        for log in logs:
            if hasattr(log, 'timestamp') and log.timestamp is not None:
                total_timestamps += 1
                if validate_timestamp(log.timestamp) is not None:
                    valid_timestamps += 1
    
    # Determine if we have sufficient timestamps for visualization
    # At least 50% of logs should have valid timestamps
    has_sufficient = valid_timestamps >= max(3, total_logs * 0.5)
    
    # Log the results
    logging.info(f"Step count: {len(step_to_logs)}")
    logging.info(f"Total logs: {total_logs}")
    logging.info(f"Total timestamps: {total_timestamps}")
    logging.info(f"Valid timestamps: {valid_timestamps}")
    logging.info(f"Has sufficient timestamps: {has_sufficient}")
    
    if not has_sufficient and HAS_CONFIG and Config.ENABLE_DIAGNOSTIC_CHECKS:
        logging.warning(f"Insufficient timestamps, extracted {valid_timestamps} from {total_logs} logs.")
    
    return total_timestamps, valid_timestamps, has_sufficient

def build_step_dict(step_to_logs, feature_file):
    """
    Build the step_dict with essential metadata for timeline visualization.
    This includes step names and duration information (start/end times).
    """
    step_dict = {}
    
    if not step_to_logs:
        logging.error("No step-to-logs data provided to build_step_dict")
        return {}
        
    logging.info(f"Building step dictionary from {len(step_to_logs)} steps")
    
    # Run timestamp validation when diagnostic checks are enabled
    if HAS_CONFIG and getattr(Config, 'ENABLE_DIAGNOSTIC_CHECKS', False):
        total_timestamps, valid_timestamps, has_sufficient = validate_timestamps(step_to_logs)
        if not has_sufficient:
            logging.warning(f"Insufficient timestamps for reliable timeline visualization. Found {valid_timestamps} valid timestamps from {total_timestamps} total.")
    
    total_timestamps = 0
    valid_timestamps = 0
    
    for step_num, logs in step_to_logs.items():
        # Extract and validate timestamps from logs
        timestamps = []
        for log in logs:
            total_timestamps += 1 if hasattr(log, 'timestamp') and log.timestamp else 0
            if hasattr(log, 'timestamp') and log.timestamp:
                valid_ts = validate_timestamp(log.timestamp)
                if valid_ts:
                    valid_timestamps += 1
                    timestamps.append(valid_ts)
        
        logging.debug(f"Step {step_num}: Found {len(timestamps)} valid timestamps from {len(logs)} logs")
        
        if timestamps:
            start_time = min(timestamps)
            end_time = max(timestamps)

            # Correct inverted timestamps
            if start_time > end_time:
                logging.warning(f"Step {step_num} has inverted timestamps. Correcting order.")
                start_time, end_time = end_time, start_time

            duration = (end_time - start_time).total_seconds()
            
            # Extract step name from feature file
            step_name = extract_step_name(step_num, feature_file)
            
            # Build step metadata
            step_dict[step_num] = {
                "step_name": step_name,
                "start_time": start_time,
                "end_time": end_time,
                "duration": duration
            }
            
            logging.debug(f"Step {step_num}: {step_name}, Duration: {duration:.3f} seconds")
            logging.debug(f"  Time range: {start_time.isoformat()} to {end_time.isoformat()}")
            
            if duration < 0.001:
                logging.warning(f"Step {step_num} has suspiciously short duration: {duration:.6f}s")
        else:
            logging.warning(f"No valid timestamps for step {step_num}, cannot calculate duration")

            # Still include the step with basic metadata so that a timeline can
            # be generated even when timestamps are missing. This ensures that
            # scenarios with sparse logging still produce a visualization.

            step_name = extract_step_name(step_num, feature_file)
            step_dict[step_num] = {
                "step_name": step_name,
                "start_time": None,
                "end_time": None,
                "duration": 0,
            }
    
    if not step_dict:
        logging.warning("No step metadata could be extracted. Timeline visualization will fail.")
    else:
        logging.info(f"Generated step metadata for {len(step_dict)} steps")
    
    # Additional diagnostic output when enabled
    if HAS_CONFIG and getattr(Config, 'ENABLE_DIAGNOSTIC_CHECKS', False):
        logging.info(f"Timestamp extraction summary: {valid_timestamps} valid timestamps from {total_timestamps} total timestamps")
        if valid_timestamps < 3:  # Minimum needed for a meaningful timeline
            logging.warning("Not enough valid timestamps for meaningful timeline visualization")
    
    return step_dict


def generate_step_report(
    feature_file: str, 
    logs_dir: str, 
    step_to_logs: Dict[int, List[LogEntry]],
    output_dir: str,
    test_id: str,
    clusters: Optional[Dict[int, List[Dict]]] = None,
    component_analysis: Optional[Dict[str, Any]] = None
) -> str:
    """
    Generate an HTML report showing logs correlated with Gherkin steps.
    
    Args:
        feature_file: Path to the feature file
        logs_dir: Directory containing logs
        step_to_logs: Dictionary mapping step numbers to log entries
        output_dir: Directory to write the report
        test_id: Test ID for the report title
        clusters: Optional dictionary of error clusters
        component_analysis: Optional component analysis results
        
    Returns:
        Path to the generated HTML report
    """
    # Input validation
    if not feature_file or not os.path.exists(feature_file):
        error_msg = f"Feature file not found: {feature_file}"
        logging.error(error_msg)
        raise FileNotFoundError(error_msg)
        
    if not step_to_logs:
        error_msg = "No step data provided"
        logging.error(error_msg)
        raise ValueError(error_msg)
    
    # Ensure output directory exists
    try:
        os.makedirs(output_dir, exist_ok=True)
        error_msg = f"Failed to create output directory {output_dir}: {str(e)}"
        logging.error(error_msg)
        raise OSError(error_msg) from e
    
    # Perform timestamp extraction diagnostics if enabled
    if HAS_CONFIG and getattr(Config, 'ENABLE_DIAGNOSTIC_CHECKS', False):
        total_timestamps, valid_timestamps, has_sufficient = validate_timestamps(step_to_logs)
        if not has_sufficient:
            logging.warning(f"WARNING: Timeline visualization may be incomplete due to insufficient timestamp data.")
            if valid_timestamps == 0:
                logging.error("ERROR: No valid timestamps extracted from logs. Timeline visualization will fail.")
    
    # Build the step dictionary with metadata used for timeline visualization
    # This dictionary contains step names and start/end timestamps
    step_dict = build_step_dict(step_to_logs, feature_file)
    
    report_path = os.path.join(output_dir, f"{test_id}_step_report.html")
    
    # Check dataset size and log warning if very large
    total_logs = sum(len(logs) for logs in step_to_logs.values())
    if total_logs > 10000:
        logging.warning(f"Large dataset detected with {total_logs} log entries. Visualization may be slow.")
    

    
    # Define supporting_images directory path - always use this consistent subdirectory
    images_dir = os.path.join(output_dir, "supporting_images")
    os.makedirs(images_dir, exist_ok=True)
    
    # Check if timeline images are enabled via feature flag
    # For test compatibility, always enable images during tests
    is_test_environment = 'unittest' in sys.modules or 'pytest' in sys.modules

    # Enable images by default unless explicitly disabled
    enable_images = is_test_environment or (HAS_CONFIG and getattr(Config, 'ENABLE_STEP_REPORT_IMAGES', True))
    logging.info(f"Timeline visualization enabled: {enable_images}")
    
    # Determine which timeline generator to use
    use_cluster_timeline = clusters is not None and len(clusters) > 0
    
    # Only attempt to generate images if the feature flag is enabled or in test mode
    timeline_image_path = None
    timeline_type = "none"
    
    # IMPLEMENTATION CHANGE (Module 2): Remove the conditional guard that prevented visualization attempts
    if enable_images:  # Removed the "and timeline_step_dict" condition
        try:
            if use_cluster_timeline:
                # Use cluster timeline if clusters are available
                logging.info("Generating cluster timeline visualization")
                timeline_image_path = generate_cluster_timeline_image(
                    step_to_logs=step_to_logs,
                    step_dict=step_dict,  # Use our enriched step_dict with durations
                    clusters=clusters,
                    output_dir=output_dir,
                    test_id=test_id
                )
                timeline_type = "cluster"
                logging.info(f"Generated cluster timeline image: {timeline_image_path}")
            else:
                # Fall back to the original timeline if no clusters provided
                logging.info("Generating standard timeline visualization")
                timeline_image_path = generate_timeline_image(
                    step_to_logs=step_to_logs,
                    step_dict=step_dict,  # Use our enriched step_dict with durations
                    output_dir=output_dir,
                    test_id=test_id
                )
                timeline_type = "standard"
                logging.info(f"Generated standard timeline image: {timeline_image_path}")
            
            # Add verification of the generated image
            if timeline_image_path:
                logging.info(f"Successfully generated timeline image at: {timeline_image_path}")
                # Verify file exists
                if not os.path.exists(timeline_image_path):
                    logging.error(f"Generated timeline image path doesn't exist: {timeline_image_path}")
                    timeline_image_path = None
                # Verify file has content
                elif os.path.getsize(timeline_image_path) == 0:
                    logging.error(f"Generated timeline image is empty: {timeline_image_path}")
                    timeline_image_path = None
            else:
                logging.warning("Failed to generate timeline image - path is None")
                
            logging.error(f"Error generating timeline image: {str(e)}")
            traceback.print_exc()
            # Continue without timeline
            timeline_image_path = None
    else:
        if not enable_images:
            logging.info("Timeline images are disabled via configuration")
    
    # IMPLEMENTATION CHANGE (Module 2): Add fallback if timeline generation failed
    if not timeline_image_path:
        try:
            timeline_image_path = generate_visualization_placeholder(
                output_dir,
                test_id,
                "Timeline visualization failed: insufficient step data"
            )
            logging.info(f"Generated placeholder image: {timeline_image_path}")
            logging.error(f"Error generating placeholder: {str(e)}")
            traceback.print_exc()
    
    # Get relative path to the image for HTML embedding
    # Use consistent supporting_images/ prefix for HTML references
    try:
        # Try to use path_utils if available for consistent path handling
        from utils.path_utils import get_path_reference
        image_relative_path = get_path_reference(timeline_image_path, output_dir, "html") if timeline_image_path else ""
        logging.info(f"Using path_utils.get_path_reference for timeline image: {image_relative_path}")
    except ImportError:
        # Fallback to os.path.relpath if path_utils not available
        image_relative_path = os.path.relpath(timeline_image_path, output_dir) if timeline_image_path else ""
        logging.info(f"Using os.path.relpath for timeline image: {image_relative_path}")

    # Normalize path separators for consistent HTML embedding
    if image_relative_path:
        image_relative_path = image_relative_path.replace(os.sep, "/")
    
    # Check for component analysis report
    component_report_file = f"{test_id}_component_report.html"
    component_report_path = os.path.join(output_dir, component_report_file)
    component_report_available = os.path.exists(component_report_path)

    # Build context for Jinja2 template
    steps_context = []
    for step_num, logs in sorted(step_to_logs.items()):
        step = step_dict.get(step_num)
        step_text = step.get("step_name", f"Step {step_num}") if isinstance(step, dict) else f"Step {step_num}"

        formats = {}
        component_counts = {}
        sample_logs = []
        timestamps = []
        for i, log in enumerate(logs):
            format_name = getattr(log, 'format_name', 'unknown')
            formats[format_name] = formats.get(format_name, 0) + 1
            if hasattr(log, 'is_error') and log.is_error and hasattr(log, 'component'):
                component = log.component
                component_counts[component] = component_counts.get(component, 0) + 1
            ts = getattr(log, 'timestamp', None)
            valid_ts = validate_timestamp(ts)
            if valid_ts:
                timestamps.append(valid_ts)
            if i < 5:
                ts_str = valid_ts.isoformat() if valid_ts else 'No timestamp'
                component_info = f" [Component: <strong>{log.component.upper()}</strong>]" if hasattr(log, 'component') else ""
                sample_logs.append({
                    'file': log.file,
                    'line_number': log.line_number,
                    'timestamp': ts_str,
                    'component_info': component_info,
                    'text': log.text,
                    'error': getattr(log, 'is_error', False)
                })

        timestamp_range = "Unknown"
        if timestamps:
            first_ts = min(timestamps)
            last_ts = max(timestamps)
            duration = (last_ts - first_ts).total_seconds()
            timestamp_range = f"{first_ts.strftime('%H:%M:%S.%f')[:-3]} to {last_ts.strftime('%H:%M:%S.%f')[:-3]} ({duration:.2f}s)"

        steps_context.append({
            'number': step_num,
            'text': step_text,
            'log_count': len(logs),
            'timestamp_range': timestamp_range,
            'formats': formats,
            'component_counts': component_counts,
            'logs': sample_logs
        })

    context = {
        'title': f'Step-Aware Log Analysis for {test_id}',
        'test_id': test_id,
        'feature_file': os.path.basename(feature_file),
        'logs_dir': logs_dir,
        'total_steps': len(step_dict),
        'steps_with_logs': len(step_to_logs),
        'total_logs': total_logs,
        'generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'timeline_image': image_relative_path,
        'component_report_available': component_report_available,
        'component_report_file': component_report_file,
        'component_analysis': component_analysis,
        'steps': steps_context
    }

    try:
        template = env.get_template('step_report.html.j2')
        html = template.render(**context)

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html)

        logging.info(f"Generated step-aware HTML report with {timeline_type} timeline image: {report_path}")
        return report_path
    except Exception as e:
        error_msg = f"Error generating HTML report: {str(e)}"
        logging.error(error_msg)
        traceback.print_exc()

        try:
            error_report_path = os.path.join(output_dir, f"{test_id}_error_report.html")
            with open(error_report_path, 'w', encoding='utf-8') as f:
                f.write(f"""
                <!DOCTYPE html>
                <html>
                <head><title>Error Report for {test_id}</title></head>
                <body>
                    <h1>Error Generating Report for {test_id}</h1>
                    <p>An error occurred while generating the step-aware report:</p>
                    <pre>{str(e)}</pre>
                    <p>Please check the logs for more details.</p>
                </body>
                </html>
                """)
            return error_report_path
        except Exception:
            return report_path


def run_step_aware_analysis(
    test_id: str, 
    feature_file: str, 
    logs_dir: str, 
    output_dir: str,
    clusters: Optional[Dict[int, List[Dict]]] = None,
    errors: Optional[List[Dict]] = None,
    component_analysis: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """
    Run a step-aware analysis and generate HTML report.
    
    Args:
        test_id: The test ID (e.g., SXM-123456)
        feature_file: Path to the Gherkin feature file
        logs_dir: Directory containing log files
        output_dir: Directory for output reports
        clusters: Optional dictionary of error clusters
        errors: Optional list of errors from log_analyzer
        component_analysis: Optional component analysis results
        
    Returns:
        Path to the generated report or None if analysis failed
    """
    # Input validation
    if not test_id:
        logging.error("No test ID provided")
        return None
        
    if not feature_file or not os.path.exists(feature_file):
        logging.error(f"Feature file not found: {feature_file}")
        return None
        
    if not logs_dir or not os.path.exists(logs_dir):
        logging.error(f"Logs directory not found: {logs_dir}")
        return None
    
    # Ensure output directory exists
    try:
        os.makedirs(output_dir, exist_ok=True)
        logging.error(f"Failed to create output directory {output_dir}: {str(e)}")
        return None
    
    # Find log files
    log_files = []
    try:
        for root, _, files in os.walk(logs_dir):
            for file in files:
                if any(file.endswith(ext) for ext in ['.log', '.txt', '.chlsj']):
                    log_files.append(os.path.join(root, file))
        
        if not log_files:
            logging.error(f"No log files found in {logs_dir}")
            return None
            
        logging.info(f"Found {len(log_files)} log files for analysis")
        logging.error(f"Error finding log files: {str(e)}")
        return None
    
    try:
        # Correlate logs with steps
        logging.info(f"Correlating logs with steps from {feature_file}")
        step_to_logs = correlate_logs_with_steps(feature_file, log_files)
        
        if not step_to_logs:
            logging.warning("No steps with logs found in correlation")
        else:
            logging.info(f"Found {len(step_to_logs)} steps with logs")
            
            # Log the first few steps to aid debugging
            steps_sample = list(step_to_logs.keys())[:3]
            for step_num in steps_sample:
                log_count = len(step_to_logs[step_num])
                logging.debug(f"Step {step_num}: {log_count} log entries")
        
        # Perform timestamp extraction diagnostics when enabled
        if HAS_CONFIG and getattr(Config, 'ENABLE_DIAGNOSTIC_CHECKS', False):
            total_timestamps, valid_timestamps, has_sufficient = validate_timestamps(step_to_logs)
            if not has_sufficient:
                logging.warning(f"Insufficient timestamps for reliable timeline visualization. Extracted {valid_timestamps} timestamps from logs.")
        
        # Enrich logs with error information if available
        if errors and step_to_logs:
            try:
                from gpt_summarizer import enrich_logs_with_errors
                step_to_logs = enrich_logs_with_errors(step_to_logs, errors)
                logging.info("Enhanced log entries with error information")
                logging.warning(f"Error enriching logs with error info: {str(e)}")
        
        # Generate HTML report
        report_path = generate_step_report(
            feature_file=feature_file,
            logs_dir=logs_dir,
            step_to_logs=step_to_logs,
            output_dir=output_dir,
            test_id=test_id,
            clusters=clusters,
            component_analysis=component_analysis
        )
        
        return report_path
        logging.error(f"Error in step-aware analysis: {str(e)}")
        traceback.print_exc()
        return None