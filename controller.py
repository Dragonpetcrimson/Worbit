import os
import sys
import traceback
import argparse
from typing import Dict, List, Optional, Tuple, Any
from log_segmenter import collect_all_supported_files
from log_analyzer import parse_logs
from ocr_processor import extract_ocr_data
# Fixed import to avoid naming conflict
from error_clusterer import perform_error_clustering  # Fixed import to use renamed function
from gpt_summarizer import generate_summary_from_clusters, enrich_logs_with_errors, build_gpt_prompt
from reports import write_reports
from reports.docx_generator import generate_bug_document
from components.direct_component_analyzer import assign_components_and_relationships
from log_analyzer import parse_logs, parse_log_entries
from config import Config
import logging
import warnings
import copy
from sklearn.exceptions import ConvergenceWarning
from reports.component_report import generate_component_report
# Import new utilities
from utils.path_utils import (
    normalize_test_id, 
    setup_output_directories, 
    ensure_test_dirs,
    get_output_path, 
    OutputType,
    get_standardized_filename,
    sanitize_base_directory
)

warnings.filterwarnings("ignore", category=ConvergenceWarning, message="Number of distinct clusters")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

handler = logging.StreamHandler(sys.stdout)
# Ensure we output in UTF-8:
handler.stream = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
logging.basicConfig(level=logging.INFO, handlers=[handler])
# Import Gherkin log correlation modules with improved error handling
try:
    from gherkin_log_correlator import GherkinParser, correlate_logs_with_steps
    from step_aware_analyzer import validate_timestamps
    GHERKIN_AVAILABLE = True
except ImportError as e:
    GHERKIN_AVAILABLE = False
    logging.warning(f"Gherkin log correlation modules not available - step-aware analysis disabled: {str(e)}")

# Cluster timeline generator is not available; fallback implementations are used.
CLUSTER_TIMELINE_AVAILABLE = False

def generate_visualization_placeholder(output_dir, test_id, message):
        """
        Generate a placeholder image with informative text.
        
        Args:
            output_dir: Output directory
            test_id: Test ID
            message: Message to display in the placeholder
            
        Returns:
            Path to the generated placeholder image
        """
    try:
        import matplotlib.pyplot as plt
        # Sanitize output directory to prevent nested directories
        output_dir = sanitize_base_directory(output_dir, "supporting_images")
        
        # Get standardized path for visualization placeholder
        placeholder_path = get_output_path(
            output_dir,
            test_id,
            get_standardized_filename(test_id, "visualization_placeholder", "png"),
            OutputType.PRIMARY_REPORT
        )
        
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, message, ha='center', va='center', fontsize=14, wrap=True)
        plt.axis('off')
        plt.savefig(placeholder_path, dpi=100)
        plt.close()
        
        return placeholder_path
    except Exception as e:
        logging.error(f"Error creating visualization placeholder: {str(e)}")
        return None
    
# Define fallback function for cluster timeline
def generate_cluster_timeline_image(step_to_logs, step_dict, clusters, output_dir, test_id):
    """Fallback function when cluster timeline visualization is not available"""
    return generate_visualization_placeholder(
        output_dir, 
        test_id, 
        "Cluster timeline visualization is not available"
    )

# Define fallback function for timeline validation
def validate_timeline_in_report(output_dir, test_id):
    """Fallback function for timeline validation when visualization is not available"""
    logging.warning("Timeline validation not available - visualization modules not loaded")
    return False
    
logging.warning("Cluster timeline generator not available - using placeholder implementation")

# Try to import the component integration module with enhanced error handling
try:
    from components.component_integration import ComponentIntegration
    COMPONENT_INTEGRATION_AVAILABLE = True
except ImportError as e:
    COMPONENT_INTEGRATION_AVAILABLE = False
    logging.warning(f"Component integration module not available - will use direct component mapping: {str(e)}")

# Enhanced visualization generation with robust fallback mechanism
def generate_with_fallback(generation_func, output_dir, test_id, *args, **kwargs):
    """
    Generate visualization with automatic fallback to placeholder.
    
    Args:
        generation_func: Visualization generation function
        output_dir: Output directory
        test_id: Test ID
        *args, **kwargs: Additional arguments for the generation function
        
    Returns:
        Path to the generated visualization or placeholder
    """
    try:
        # Attempt to generate the requested visualization
        viz_path = generation_func(output_dir, test_id, *args, **kwargs)
        
        # Check if generation returned None or if file doesn't exist
        if viz_path is None or not os.path.exists(viz_path):
            # Get function name for the message
            func_name = getattr(generation_func, "__name__", "Visualization")
            return generate_visualization_placeholder(
                output_dir, 
                test_id,
                f"{func_name} generation failed"
            )
        
        return viz_path
    except Exception as e:
        # Get function name for error message
        func_name = getattr(generation_func, "__name__", "Visualization")
        logging.error(f"Error in {func_name} generation: {str(e)}")
        
        # Generate informative placeholder
        return generate_visualization_placeholder(
            output_dir, 
            test_id, 
            f"Error generating {func_name}: {str(e)}"
        )

def find_feature_file(base_dir, test_id):
    """
    Find a feature file related to the test ID.
    Returns the path to the feature file or None if not found.
    """
    if not os.path.exists(base_dir):
        logging.warning(f"Feature file directory does not exist: {base_dir}")
        return None
        
    normalized = test_id.lower().replace("-", "_")
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".feature"):
                path = os.path.join(root, file)
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read().lower()
                        if normalized in content or normalized in file.lower():
                            return path
                except Exception as e:
                    logging.warning(f"Error reading feature file {path}: {str(e)}")
    
    logging.info(f"No feature file found for {test_id} in {base_dir}")
    return None

def extract_background_and_scenario(feature_file):
    """
    Extract background and scenario sections from a feature file.
    Returns a tuple of (background_text, scenario_text).
    """
    if not feature_file or not os.path.exists(feature_file):
        return "", ""
        
    background_lines, scenario_lines = [], []
    current = None
    
    try:
        with open(feature_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                stripped = line.strip()
                lower = stripped.lower()
                if lower.startswith("background:"):
                    current = "background"
                elif lower.startswith("scenario:"):
                    current = "scenario"
                if current == "background":
                    background_lines.append(stripped)
                elif current == "scenario":
                    scenario_lines.append(stripped)
        return "\n".join(background_lines), "\n".join(scenario_lines)
    except Exception as e:
        logging.error(f"Error extracting background/scenario from {feature_file}: {str(e)}")
        return "", ""

def run_gherkin_correlation(feature_file, log_files, output_dir, test_id, errors=None, error_clusters=None, component_analysis=None):
    """
    Run the Gherkin log correlation and return correlated logs.
    Returns a tuple of (report_path, step_to_logs). The report path is always
    None as step-aware reports have been removed.
    """
    if not GHERKIN_AVAILABLE:
        logging.warning("Skipping Gherkin correlation - modules not available")
        return None, None
    
    if not feature_file or not os.path.exists(feature_file):
        logging.warning(f"Skipping Gherkin correlation - feature file not found: {feature_file}")
        return None, None
    
    if not log_files:
        logging.warning("Skipping Gherkin correlation - no log files provided")
        return None, None
        
    try:
        # Create output directories using the new utility function
        output_paths = setup_output_directories(output_dir, test_id)
        
        # Correlate logs with steps
        logging.info("Running Gherkin log correlation")
        step_to_logs = correlate_logs_with_steps(feature_file, log_files)
        
        if not step_to_logs:
            logging.warning("Gherkin correlation found no steps with logs")
            return None, None
        
        # Enrich logs with error information if available
        if errors:
            try:
                step_to_logs = enrich_logs_with_errors(step_to_logs, errors)
                logging.info("Enhanced log entries with error information")
            except Exception as e:
                logging.warning(f"Error enriching logs with error information: {str(e)}")
        
        # Enhanced cluster visualization handling with fallback
        try:
            report_path = None
            logging.info("Gherkin correlation completed")
            
            # Log cluster visualization status in a more resilient way
            # Check if the feature is available rather than if clusters exist
            if CLUSTER_TIMELINE_AVAILABLE:
                logging.info("Report includes cluster timeline visualization")
            else:
                logging.info("Report includes placeholder for cluster visualization (feature unavailable)")
                
            return report_path, step_to_logs
        except Exception as e:
            logging.error(f"Error generating step report: {str(e)}")
            traceback.print_exc()
            return None, step_to_logs
            
    except Exception as e:
        logging.error(f"Error in Gherkin correlation: {str(e)}")
        traceback.print_exc()
        return None, None

# New function for diagnostic checks
def run_diagnostics(test_id: str) -> Dict[str, Any]:
    """
    Run diagnostic checks on a test ID.
    
    Args:
        test_id: Test ID to diagnose
    
    Returns:
        Dictionary with diagnostic results
    """
    if not test_id:
        return {"error": "No test ID provided for diagnostics"}
    
    # Enable diagnostic checks
    Config.ENABLE_DIAGNOSTIC_CHECKS = True
    
    # Normalize test ID
    test_id = normalize_test_id(test_id)
    logging.info(f"Running diagnostics for {test_id}")
    
    # Ensure required directories exist
    try:
        dirs = ensure_test_dirs(test_id)
        logging.info(f"Test directories created or verified: {dirs}")
    except Exception as e:
        logging.error(f"Error creating test directories: {str(e)}")
        return {"error": f"Directory setup failed: {str(e)}"}
    
    # Check logs directory
    logs_dir = os.path.join(Config.LOG_BASE_DIR, test_id)
    if not os.path.exists(logs_dir):
        logging.warning(f"Logs directory does not exist: {logs_dir}")
        return {"error": f"Logs directory not found: {logs_dir}"}
    
    # Collect log files
    log_results = {}
    try:
        logs, images = collect_all_supported_files(logs_dir)
        log_results["log_count"] = len(logs)
        log_results["image_count"] = len(images)
    except Exception as e:
        logging.error(f"Error collecting files: {str(e)}")
        log_results["error"] = f"File collection failed: {str(e)}"
    
    # Check for output directory and reports
    output_dir = os.path.join(Config.OUTPUT_BASE_DIR, test_id)
    output_results = {"output_dir_exists": os.path.exists(output_dir)}
    
    if output_results["output_dir_exists"]:
        # Check for HTML report
        component_report_path = os.path.join(output_dir, f"{test_id}_component_report.html")
        output_results["component_report_exists"] = os.path.exists(component_report_path)
        
        # Validate timeline in HTML report if it exists
        if output_results["component_report_exists"]:
            try:
                timeline_valid = validate_timeline_in_report(output_dir, test_id)
                output_results["timeline_valid"] = timeline_valid
            except Exception as e:
                logging.error(f"Error validating timeline: {str(e)}")
                output_results["timeline_error"] = str(e)
    
    # Check Gherkin correlation modules
    gherkin_results = {"modules_available": GHERKIN_AVAILABLE}
    if gherkin_results["modules_available"]:
        # Try to find a feature file
        ymir_dir = r"C:\\gitrepos\\ymir\\mimosa\\bdd"
        if os.path.exists(ymir_dir):
            feature_file = find_feature_file(ymir_dir, test_id)
            gherkin_results["feature_file"] = feature_file
            
            # Check timestamps if feature file exists
            if feature_file and logs:
                try:
                    # Validate timestamps in logs
                    timestamp_results = validate_timestamps(logs)
                    gherkin_results["timestamp_results"] = timestamp_results
                except Exception as e:
                    logging.error(f"Error validating timestamps: {str(e)}")
                    gherkin_results["timestamp_error"] = str(e)
    
    # Check component integration module
    component_results = {"module_available": COMPONENT_INTEGRATION_AVAILABLE}
    if component_results["module_available"]:
        component_schema_path = os.path.join('components', 'schemas', 'component_schema.json')
        component_results["schema_exists"] = os.path.exists(component_schema_path)
    
    # Combine all results
    diagnostic_results = {
        "test_id": test_id,
        "logs": log_results,
        "output": output_results,
        "gherkin": gherkin_results,
        "components": component_results
    }
    
    # Print diagnostic summary
    print("\n=== DIAGNOSTIC SUMMARY ===")
    print(f"Test ID: {test_id}")
    print(f"Logs directory: {logs_dir} - {'FOUND' if os.path.exists(logs_dir) else 'MISSING'}")
    print(f"Log files: {log_results.get('log_count', 0)}")
    print(f"Image files: {log_results.get('image_count', 0)}")
    print(f"Output directory: {output_dir} - {'FOUND' if output_results['output_dir_exists'] else 'MISSING'}")
    
    if output_results["output_dir_exists"] and output_results.get("component_report_exists", False):
        print(f"Component report: FOUND")
        if "timeline_valid" in output_results:
            print(f"Timeline in report: {'VALID' if output_results['timeline_valid'] else 'MISSING'}")
    else:
        print("Component report: MISSING")
    
    print(f"Gherkin modules: {'AVAILABLE' if gherkin_results['modules_available'] else 'MISSING'}")
    if gherkin_results["modules_available"]:
        print(f"Feature file: {'FOUND' if gherkin_results.get('feature_file') else 'MISSING'}")
        
        # Print timestamp results if available
        if "timestamp_results" in gherkin_results:
            ts_results = gherkin_results["timestamp_results"]
            print(f"Total logs: {ts_results.get('total_logs', 0)}")
            print(f"Valid timestamps: {ts_results.get('valid_timestamps', 0)}")
            print(f"Timestamp sufficiency: {'SUFFICIENT' if ts_results.get('sufficient', False) else 'INSUFFICIENT'}")
    
    print(f"Component integration: {'AVAILABLE' if component_results['module_available'] else 'MISSING'}")
    if component_results['module_available']:
        print(f"Component schema: {'FOUND' if component_results.get('schema_exists') else 'MISSING'}")
    
    print("=========================")
    
    # Restore diagnostic checks setting
    Config.ENABLE_DIAGNOSTIC_CHECKS = False
    
    return diagnostic_results

def run_pipeline(test_id: str, gpt_model: str = None, enable_ocr: bool = None, test_type: str = "ymir"):
    """
    Run the log analysis pipeline programmatically.
    Returns a tuple of (result_message, step_report_path).
    """
    if not test_id:
        error_msg = "Error: Empty test ID provided"
        logging.error(error_msg)
        return error_msg, None
        
    # Set defaults from config if not provided
    if gpt_model is None:
        gpt_model = Config.DEFAULT_MODEL
    if enable_ocr is None:
        enable_ocr = Config.ENABLE_OCR

    # Normalize test ID using the utility function
    test_id = normalize_test_id(test_id)
    logging.info(f"Starting analysis for {test_id} with model {gpt_model}")
    logging.info(f"Test type selected: {test_type}")

    # Ensure base directories exist
    os.makedirs(Config.LOG_BASE_DIR, exist_ok=True)
    os.makedirs(Config.OUTPUT_BASE_DIR, exist_ok=True)

    input_dir = os.path.join(Config.LOG_BASE_DIR, test_id)
    output_dir = os.path.join(Config.OUTPUT_BASE_DIR, test_id)
    
    # Setup output directories using the utility function
    output_paths = setup_output_directories(output_dir, test_id)

    if not os.path.exists(input_dir):
        error_msg = f"Error: Input directory {input_dir} does not exist"
        logging.error(error_msg)
        return error_msg, None

    # Initialize variables
    background_text, scenario_text = "", ""
    ymir_flag = False
    feature_file = None

    # Find feature file based on test type
    if test_type == "ymir":
        ymir_dir = r"C:\\gitrepos\\ymir\\mimosa\\bdd"
        if os.path.exists(ymir_dir):
            feature_file = find_feature_file(ymir_dir, test_id)
            if feature_file:
                background_text, scenario_text = extract_background_and_scenario(feature_file)
                ymir_flag = True
                logging.info(f"Found Ymir feature file: {feature_file}")
            else:
                logging.warning(f"No Ymir feature file found for {test_id}")
        else:
            logging.warning(f"Ymir directory not found: {ymir_dir}")

    elif test_type == "installer":
        installer_base = r"C:\\Program Files (x86)\\Sirius XM Radio\\SMITE"
        selected_folder = os.environ.get("INSTALLER_FOLDER")
        if selected_folder:
            mimosa_path = os.path.join(installer_base, selected_folder, "mimosa")
            if os.path.exists(mimosa_path):
                feature_file = find_feature_file(mimosa_path, test_id)
                if feature_file:
                    background_text, scenario_text = extract_background_and_scenario(feature_file)
                    ymir_flag = True
                    logging.info(f"Found Installer feature file: {feature_file}")
                else:
                    logging.warning(f"No Installer feature file found for {test_id}")
            else:
                logging.warning(f"Mimosa path not found: {mimosa_path}")
        else:
            logging.warning("No installer folder selected")
    
    # Collect log files and images
    try:
        logs, images = collect_all_supported_files(input_dir)
        har_logs = [f for f in logs if f.endswith(".har")]
        logging.info(f"Found {len(logs)} log files ({len(har_logs)} HAR logs) and {len(images)} image files")
        
        if not logs:
            warning_msg = f"Warning: No log files found in {input_dir}"
            logging.warning(warning_msg)
    except Exception as e:
        error_msg = f"Error collecting files: {str(e)}"
        logging.error(error_msg)
        return error_msg, None

    # Log file names for debugging
    logging.info(f"Log file names: {[os.path.basename(f) for f in logs[:5]]}")
    
    # Extract errors from logs
    try:
        errors = parse_logs(logs)
        logging.info(f"Extracted {len(errors)} errors from logs")
        
        # Debug component distribution after parsing
        component_check = {}
        for err in errors[:20]:
            comp = err.get('component', 'unknown')
            component_check[comp] = component_check.get(comp, 0) + 1
        logging.info(f"Initial component distribution: {component_check}")
    except Exception as e:
        error_msg = f"Error parsing logs: {str(e)}"
        logging.error(error_msg)
        return error_msg, None

    # Extract OCR data from images
    ocr_data = []
    if enable_ocr and images:
        try:
            ocr_data = extract_ocr_data(images)
            logging.info(f"Extracted OCR data from {len(ocr_data)} images")
        except Exception as e:
            logging.warning(f"Error extracting OCR data: {str(e)}")

    # Apply direct component mapping to errors - CRITICAL for primary_issue_component
    try:
        logging.info("Applying direct component mapping for errors")
        # Create a deep copy of errors to prevent reference issues
        errors_for_component_analysis = copy.deepcopy(errors)
        errors_with_components, component_summary, primary_issue_component = assign_components_and_relationships(errors_for_component_analysis)
        
        # Update original errors with component information
        for i, original_error in enumerate(errors):
            if i < len(errors_with_components):
                # Transfer component information
                if 'component' in errors_with_components[i]:
                    original_error['component'] = errors_with_components[i]['component']
                if 'component_source' in errors_with_components[i]:
                    original_error['component_source'] = errors_with_components[i]['component_source']
        
        # Log the primary_issue_component immediately after determination - CRITICAL
        logging.info(f"Direct component analysis identified '{primary_issue_component}' as the primary issue component")
        
        # Debug component distribution after direct mapping
        component_check = {}
        for err in errors[:20]:
            comp = err.get('component', 'unknown')
            component_check[comp] = component_check.get(comp, 0) + 1
        logging.info(f"Component distribution after direct mapping: {component_check}")
    except Exception as e:
        logging.error(f"Error applying direct component mapping: {str(e)}")
        component_summary = []
        primary_issue_component = "unknown"
        
    # Double-check error components
    logging.info(f"Sample error components: {[e.get('component', 'unknown') for e in errors[:5]]}")

    # Cluster errors
    try:
        # Determine number of clusters adaptively based on error count
        error_count = len(errors)
        # Don't use a fixed value from Config, calculate it dynamically
        if error_count <= 5:
            num_clusters = min(2, error_count)  # Very small datasets get 1-2 clusters
        elif error_count <= 20:
            num_clusters = min(3, error_count)  # Small datasets get 2-3 clusters
        elif error_count <= 50:
            num_clusters = min(5, error_count)  # Medium datasets get 3-5 clusters
        else:
            num_clusters = 8  # Large datasets get max 8 clusters
        
        logging.info(f"Using dynamic cluster count: {num_clusters} for {error_count} errors")
        
        # Make a deep copy of errors to avoid modifying original components
        errors_for_clustering = copy.deepcopy(errors)
        
        # Use the renamed function
        error_clusters = perform_error_clustering(errors_for_clustering, num_clusters=num_clusters)
        
        logging.info(f"Grouped errors into {len(error_clusters)} clusters")
        
        # Ensure the components from original errors are preserved in the clustered version
        # This prevents double-assignment and component information loss
        enhanced_clusters = {}
        
        for cluster_id, errors_in_cluster in error_clusters.items():
            # Create a copy of the cluster errors to avoid modifying the original
            enhanced_cluster_errors = []
            
            for cluster_error in errors_in_cluster:
                # Find matching original error to get its component
                matching_original = None
                
                # Try to match by file and line number first (most reliable)
                for original_error in errors:
                    if (original_error.get('file') == cluster_error.get('file') and 
                        original_error.get('line_num') == cluster_error.get('line_num') and
                        original_error.get('text') == cluster_error.get('text')):
                        matching_original = original_error
                        break
                
                # If found a match, copy the component information
                if matching_original and 'component' in matching_original:
                    # Make a new copy to avoid modifying the original
                    enhanced_error = cluster_error.copy()
                    enhanced_error['component'] = matching_original['component']
                    if 'component_source' in matching_original:
                        enhanced_error['component_source'] = matching_original['component_source']
                    # Explicitly propagate primary_issue_component to all cluster errors
                    enhanced_error['primary_issue_component'] = primary_issue_component
                    enhanced_cluster_errors.append(enhanced_error)
                else:
                    # If no match found, keep the error as is but ensure it has primary_issue_component
                    enhanced_error = cluster_error.copy()
                    enhanced_error['primary_issue_component'] = primary_issue_component
                    enhanced_cluster_errors.append(enhanced_error)
            
            enhanced_clusters[cluster_id] = enhanced_cluster_errors
                
        error_clusters = enhanced_clusters
        
        # Log cluster components for debugging
        for cluster_id, cluster_errors in list(enhanced_clusters.items())[:2]:
            logging.info(f"Cluster {cluster_id} components: {[e.get('component', 'unknown') for e in cluster_errors[:3]]}")
    except Exception as e:
        error_msg = f"Error clustering errors: {str(e)}"
        logging.error(error_msg)
        # Continue with manually created clusters that preserve component info
        # Create a deep copy of errors to avoid reference issues
        errors_for_cluster = copy.deepcopy(errors)
        # Ensure primary_issue_component is set in all errors
        for error in errors_for_cluster:
            error['primary_issue_component'] = primary_issue_component
        error_clusters = {0: errors_for_cluster}  # Put all errors in one cluster to preserve component info
        logging.info(f"Created a single fallback cluster with {len(errors)} errors")
        # Double-check component assignment
        logging.info(f"Components in error fallback cluster: {[e.get('component', 'unknown') for e in errors_for_cluster[:5]]}")      
    
    # Initialize component_analysis_results
    component_analysis_results = None
    
    # Define component_schema_path before using it in the conditional
    component_schema_path = os.path.join('components', 'schemas', 'component_schema.json')
    
    # Try to use the component integration module if available
    if COMPONENT_INTEGRATION_AVAILABLE and os.path.exists(component_schema_path):
        try:
            integrator = ComponentIntegration(component_schema_path)
            # Collect all log entries from the logs
            log_entries = []
            for log_file in logs:
                try:
                    entries = parse_log_entries(log_file)
                    log_entries.extend(entries)
                except Exception as log_err:
                    logging.warning(f"Error parsing log entries from {log_file}: {str(log_err)}")
            
            # Enhanced error handling for component analysis
            try:
                component_analysis_results = integrator.analyze_logs(
                    log_entries, errors, output_paths["json"], output_paths["test_id"]
                )
                logging.info("Component relationship analysis completed using integration module")
            except Exception as comp_err:
                logging.error(f"Error in component integration analyze_logs: {str(comp_err)}")
                component_analysis_results = None
            
            # Create fallback if component_analysis_results is None
            if component_analysis_results is None:
                component_analysis_results = {
                    "primary_issue_component": primary_issue_component,
                    "component_summary": component_summary,
                    "metrics": {
                        "root_cause_component": primary_issue_component
                    }
                }
                logging.info("Created fallback component analysis due to integration module failure")
            
            # CRITICAL: Explicitly inject primary_issue_component into component_analysis_results
            component_analysis_results["primary_issue_component"] = primary_issue_component
            
            # Also inject into metrics if they exist
            if "metrics" in component_analysis_results:
                component_analysis_results["metrics"]["root_cause_component"] = primary_issue_component
                
                # Ensure components_with_issues includes the primary component
                if "components_with_issues" in component_analysis_results["metrics"]:
                    if primary_issue_component not in component_analysis_results["metrics"]["components_with_issues"]:
                        component_analysis_results["metrics"]["components_with_issues"].insert(0, primary_issue_component)
            
            logging.info(f"Injected primary_issue_component '{primary_issue_component}' into component analysis results")
            
        except Exception as e:
            logging.error(f"Error in component integration analysis: {str(e)}")
            # Fall back to direct component mapping results
            # CRITICAL: Create component_analysis_results with primary_issue_component
            component_analysis_results = {
                "primary_issue_component": primary_issue_component,
                "component_summary": component_summary
            }
            logging.info(f"Created fallback component analysis with primary_issue_component '{primary_issue_component}'")
    else:
        # Create a comprehensive component analysis result from direct mapping
        # Count components directly from errors for accurate metrics
        component_counts = {}
        for err in errors:
            if isinstance(err, dict) and 'component' in err:
                comp = err.get('component')
                if comp and comp != 'unknown':
                    component_counts[comp] = component_counts.get(comp, 0) + 1
        
        # CRITICAL: Create component_analysis_results with primary_issue_component
        component_analysis_results = {
            "primary_issue_component": primary_issue_component,  # Explicitly set
            "component_summary": component_summary,
            "component_error_counts": component_counts,
            "metrics": {
                "component_tagged_logs": len(errors),
                "component_tagged_errors": len(errors),
                "component_error_counts": component_counts,
                "components_with_issues": list(component_counts.keys()),
                "root_cause_component": primary_issue_component  # Explicitly set
            }
        }
        logging.info(f"Created direct component mapping results with primary_issue_component '{primary_issue_component}'")
    
    # Special check for Android - if we have Android errors but primary is unknown
    android_errors = [err for err in errors if isinstance(err, dict) and 
                      err.get('component') == 'android']
    if android_errors and primary_issue_component == "unknown":
        # Only set Android as primary if there are no other significant components
        other_components = [err.get('component') for err in errors if isinstance(err, dict) and 
                           err.get('component') not in ['android', 'unknown']]
        
        if not other_components:
            primary_issue_component = "android"
            # Update component_analysis_results
            if component_analysis_results:
                component_analysis_results["primary_issue_component"] = primary_issue_component
                if "metrics" in component_analysis_results:
                    component_analysis_results["metrics"]["root_cause_component"] = primary_issue_component
            
            logging.info(f"Found {len(android_errors)} android errors - updated primary_issue_component to 'android'")
    
    # CRITICAL: Final check and log for primary_issue_component before passing to downstream functions
    logging.info(f"Primary issue component identified as: {primary_issue_component}")
    
    # Run Gherkin log correlation if we have a feature file
    step_report = None
    step_to_logs = None
    if feature_file and GHERKIN_AVAILABLE and logs:
        # Enhanced error handling for step-aware analysis
        try:
            # Pass clusters to enable cluster visualization
            step_report, step_to_logs = run_gherkin_correlation(
                feature_file, logs, output_paths["base"], output_paths["test_id"], errors, error_clusters, component_analysis_results
            )
            if step_report:
                logging.info("Successfully completed step-aware correlation analysis")
        except Exception as e:
            logging.error(f"Error in step-aware correlation: {str(e)}")
            step_report, step_to_logs = None, None

    # Generate summary with GPT
    use_gpt = gpt_model.lower() != "none"
    try:
        summary = generate_summary_from_clusters(
            error_clusters, ocr_data, output_paths["test_id"],
            scenario_text=scenario_text,
            use_gpt=use_gpt,
            model=gpt_model,
            step_to_logs=step_to_logs,
            feature_file=feature_file,
            component_analysis=component_analysis_results 
        )
    except Exception as e:
        logging.error(f"Error generating summary: {str(e)}")
        summary = f"Error generating summary: {str(e)}\n\nPlease review the logs and errors manually."

    # CRITICAL: Log primary_issue_component once more before writing reports
    logging.info(f"PRIMARY ISSUE COMPONENT BEING PASSED TO REPORTS: {primary_issue_component}")
    
    # Defensive check to ensure primary_issue_component is not None before passing to functions
    if primary_issue_component is None:
        primary_issue_component = "unknown"
        logging.warning("Primary issue component was None, defaulting to 'unknown'")
    
    # Generate component report with enhanced error handling
    try:
        component_report_path = None
    except Exception as e:
        logging.warning(f"Failed to generate component report: {str(e)}")
        component_report_path = None
    
    # Write reports - updated to use the new output path structure
    try:
        # Verify component information before writing reports
        component_check = {}
        for err in errors[:20]:
            comp = err.get('component', 'unknown')
            component_check[comp] = component_check.get(comp, 0) + 1
        logging.info(f"Final component distribution before reports: {component_check}")
        
        # Update the write_reports call - remove json_dir and images_dir parameters
        report_results = write_reports(
            output_dir=output_paths["base"],
            test_id=output_paths["test_id"],
            summary=summary,
            errors=errors,
            ocr_data=ocr_data,
            clusters=error_clusters,
            ymir_flag=ymir_flag,
            background_text=background_text,
            scenario_text=scenario_text,
            component_analysis=component_analysis_results,
            primary_issue_component=primary_issue_component,
            component_report_path=component_report_path
        )
        logging.info(f"Wrote reports to {output_paths['base']}")
        
        # Check if reports returned primary component different from our analysis
        if isinstance(report_results, dict) and 'primary_issue_component' in report_results:
            if report_results['primary_issue_component'] != primary_issue_component:
                logging.warning(f"Report manager changed primary_issue_component from '{primary_issue_component}' to '{report_results['primary_issue_component']}'")
                primary_issue_component = report_results['primary_issue_component']
    
    except Exception as e:
        logging.error(f"Error writing reports: {str(e)}")

    # Generate docx bug report
    try:
        # Use the new utility function for file path with enhanced error handling
        docx_path = generate_with_fallback(
            generate_bug_document,
            output_paths["base"],
            output_paths["test_id"],
            summary,
            errors=errors,
            ocr_data=ocr_data,
            clusters=error_clusters,
            background_text=background_text,
            scenario_text=scenario_text,
            component_analysis=component_analysis_results,
            primary_issue_component=primary_issue_component,
            component_report_path=component_report_path
        )
        
        if docx_path:
            logging.info(f"Generated bug report document: {docx_path}")
        else:
            logging.warning("Bug report document generation returned None")
    except Exception as e:
        logging.error(f"Error generating bug report document: {str(e)}")

    # Return results
    result_message = f"Analysis complete for {output_paths['test_id']}. Found {len(errors)} errors across {len(logs)} log files."
    if primary_issue_component != "unknown":
        result_message += f" Primary issue component: {primary_issue_component.upper()}."
    logging.info(result_message)
    return result_message, step_report

def run_pipeline_interactive():
    """Interactive command-line interface for the log analyzer."""
    print("🔍 QA Log Analyzer (V50 Modular)")

    try:
        Config.setup_logging()
        Config.validate()
    except Exception as e:
        print(f"❌ Configuration error: {str(e)}")
        return

    # Get test ID with validation
    while True:
        test_input = input("Enter Test ID (e.g. SXM-1234567 or just 1234567): ").strip()
        if not test_input:
            print("❌ Test ID cannot be empty. Please enter a valid Test ID.")
            continue
        test_id = normalize_test_id(test_input)
        break

    # Select test type
    print("\nSelect test type:")
    print("1. Ymir Test")
    print("2. Installer Test")
    
    test_type = None
    while test_type is None:
        choice = input("Choice (1 or 2): ").strip()
        if choice == '1':
            test_type = "ymir"
            ymir_dir = r"C:\\gitrepos\\ymir\\mimosa\\bdd"
            if not os.path.exists(ymir_dir):
                print(f"⚠️ Warning: Ymir directory not found: {ymir_dir}")
                print("   Feature files may not be located.")
        elif choice == '2':
            test_type = "installer"
            installer_base = r"C:\\Program Files (x86)\\Sirius XM Radio\\SMITE"
            if not os.path.exists(installer_base):
                print(f"⚠️ Warning: Installer directory not found: {installer_base}")
            else:
                folders = [f for f in os.listdir(installer_base) if os.path.isdir(os.path.join(installer_base, f))]
                if not folders:
                    print("❌ No installer folders found.")
                else:
                    print("\nAvailable Installer Folders:")
                    for idx, folder in enumerate(folders, 1):
                        print(f"{idx}. {folder}")
                    
                    folder_selected = False
                    while not folder_selected:
                        choice_folder = input("Select installation folder (enter number): ").strip()
                        if not choice_folder.isdigit():
                            print("❌ Please enter a valid number.")
                            continue
                            
                        folder_idx = int(choice_folder)
                        if 1 <= folder_idx <= len(folders):
                            selected_folder = folders[folder_idx - 1]
                            os.environ["INSTALLER_FOLDER"] = selected_folder
                            folder_selected = True
                        else:
                            print(f"❌ Please enter a number between 1 and {len(folders)}.")
        else:
            print("❌ Invalid choice. Please enter 1 or 2.")

    # Choose GPT model
    print("\nChoose GPT Model:")
    print("1. GPT-4 (accurate, slower)")
    print("2. GPT-3.5 (faster, cheaper)")
    print("3. None (offline mode)")
    
    gpt_model = None
    while gpt_model is None:
        model_choice = input("Choice (1, 2, or 3): ").strip()
        if model_choice == '1':
            gpt_model = "gpt-4"
        elif model_choice == '2':
            gpt_model = "gpt-3.5-turbo"
        elif model_choice == '3':
            gpt_model = "none"
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")

    # Validate input directory exists
    input_dir = os.path.join(Config.LOG_BASE_DIR, test_id)
    if not os.path.exists(input_dir):
        print(f"❌ Error: Input directory {input_dir} does not exist")
        print(f"   Please ensure that logs for {test_id} are in {Config.LOG_BASE_DIR}")
        return
    
    #Silence warning date time
    logging.getLogger().setLevel(logging.ERROR)  # Only show ERROR level messages, not WARNING

    # Run the pipeline
    try:
        result, step_report = run_pipeline(test_id, gpt_model=gpt_model, test_type=test_type)
        print(f"\n✅ {result}")
    except Exception as e:
        print(f"\n❌ Error running analysis: {str(e)}")
        traceback.print_exc()
        return

    # Report on output files
    output_dir = os.path.join(Config.OUTPUT_BASE_DIR, test_id)
    if os.path.exists(output_dir):
        print(f"\nOutput files available in: {output_dir}")
        print("Files generated:")
        docx_file = None
        cluster_timeline_file = None
        component_analysis_file = None
        
        # Check root directory files
        for file in os.listdir(output_dir):
            if os.path.isfile(os.path.join(output_dir, file)):
                print(f" - {file}")
                if file.endswith("_bug_report.docx"):
                    docx_file = os.path.join(output_dir, file)
                if file.endswith("_component_report.html"):
                    component_report_path = os.path.join(output_dir, file)
        
        # Check JSON directory
        json_dir = os.path.join(output_dir, "json")
        if os.path.exists(json_dir):
            print("\nJSON files:")
            for file in os.listdir(json_dir):
                print(f" - json/{file}")
                if file.endswith("_component_analysis.json"):
                    component_analysis_file = os.path.join(json_dir, file)
        
        # Check images directory
        images_dir = os.path.join(output_dir, "supporting_images")
        if os.path.exists(images_dir):
            print("\nSupporting images:")
            for file in os.listdir(images_dir):
                print(f" - supporting_images/{file}")
                if file.endswith("_cluster_timeline.png"):
                    cluster_timeline_file = os.path.join(images_dir, file)

        if docx_file and os.path.exists(docx_file):
            print(f"\n🔍 Bug report template created: {docx_file}")
            print("   This document is formatted for easy submission to Jira.")
        
        if cluster_timeline_file and os.path.exists(cluster_timeline_file):
            print(f"\n📊 Cluster timeline visualization created: {cluster_timeline_file}")
            print("   This shows how error clusters occur throughout the test execution.")
        
        if component_analysis_file and os.path.exists(component_analysis_file):
            print(f"\n🧩 Component analysis generated: {component_analysis_file}")
            print("   This analysis identifies the root cause component and affected relationships.")
        
        if step_report and os.path.exists(step_report):
            print(f"\n🧩 Step-aware analysis report generated: {step_report}")
            print("   This report shows logs correlated with Gherkin test steps.")
            if CLUSTER_TIMELINE_AVAILABLE:
                print("   The visualization groups errors by clusters for better readability.")

def diagnose_output_structure(test_id: str):
    """
    Run diagnostics on the output directory structure.
    
    Args:
        test_id: Test ID to diagnose
        
    Returns:
        Diagnostic results
    """
    from utils.path_validator import validate_file_structure, check_html_references, print_validation_results
    
    test_id = normalize_test_id(test_id)
    output_dir = os.path.join(Config.OUTPUT_BASE_DIR, test_id)
    
    if not os.path.exists(output_dir):
        return f"Output directory does not exist: {output_dir}"
    
    # Print validation results
    print_validation_results(output_dir, test_id)
    
    # Return validation data
    structure_issues = validate_file_structure(output_dir, test_id)
    
    # Enhance error handling for HTML checks
    html_issues = {}
    component_report_path = os.path.join(output_dir, f"{test_id}_component_report.html")
    if os.path.exists(component_report_path):
        try:
            html_issues = check_html_references(component_report_path)
        except Exception as e:
            html_issues = {"error": f"Error checking HTML references: {str(e)}"}
    else:
        html_issues = {"error": f"Component report not found: {component_report_path}"}
    
    # Combine results
    return {
        "test_id": test_id,
        "output_dir": output_dir,
        "structure_issues": structure_issues,
        "html_issues": html_issues
    }

# Modified main entry point to use argparse
if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="QA Log Analyzer")
    parser.add_argument("--test-id", "-t", help="Test ID (e.g. SXM-1234567)")
    parser.add_argument("--diagnose", "-d", action="store_true", help="Run pre-flight diagnostics")
    parser.add_argument("--continue-after-diagnose", "-c", action="store_true", help="Continue analysis after diagnostics")
    parser.add_argument("--type", choices=["ymir", "installer"], default="ymir", help="Test type (ymir or installer)")
    parser.add_argument("--model", choices=["gpt-4", "gpt-3.5-turbo", "none"], default=None, help="GPT model to use")
    parser.add_argument("--ocr", action="store_true", help="Enable OCR processing")
    args = parser.parse_args()
    
    try:
        Config.setup_logging()
        Config.validate()
        
        # Handle diagnostic mode
        if args.diagnose:
            # Get test ID from arguments or prompt
            test_id = args.test_id
            if not test_id:
                test_id = input("Enter Test ID for diagnostics: ").strip()
                if not test_id:
                    print("❌ Test ID cannot be empty. Exiting.")
                    sys.exit(1)
            
            # Run diagnostics
            diagnostic_results = run_diagnostics(test_id)
            
            # Exit if not continuing
            if not args.continue_after_diagnose:
                sys.exit(0)
        
        # If we're here, either no diagnostics were requested,
        # or we're continuing after diagnostics
        
        # Check if we should use interactive mode
        if not args.test_id and not args.diagnose:
            # No args provided, run interactive mode
            run_pipeline_interactive()
        else:
            # Get test ID from arguments or prompt
            test_id = args.test_id
            if not test_id:
                test_id = input("Enter Test ID: ").strip()
                if not test_id:
                    print("❌ Test ID cannot be empty. Exiting.")
                    sys.exit(1)
            
            # Get model from arguments or config
            gpt_model = args.model
            
            # Get OCR setting from arguments or config
            enable_ocr = args.ocr if args.ocr else Config.ENABLE_OCR
            
            # Run the pipeline
            result, step_report = run_pipeline(
                test_id, 
                gpt_model=gpt_model, 
                enable_ocr=enable_ocr, 
                test_type=args.type
            )
            print(f"\n✅ {result}")
    except KeyboardInterrupt:
        print("\n\nAnalysis cancelled by user.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        traceback.print_exc()
        sys.exit(1)