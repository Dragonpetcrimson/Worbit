# Orbit: Technical Guide

**Version:** May 2025  
**Audience:** Developers, SDETs, Automation Engineers

## Architecture Overview

Orbit is a modular log analysis system designed to automate root cause analysis of test failures. It's built with a component-based architecture that separates concerns for maintainability, testability, and extensibility.

The system processes log files, screenshots, and test metadata to produce multi-format reports that explain why tests failed and recommend solutions. A key strength is its ability to identify which components in the system are causing issues and how errors propagate through component relationships.

## Module Structure

```
Orbit/
├── controller.py               # Pipeline orchestration and user interface
├── config.py                   # Configuration management
├── log_segmenter.py            # Log file collection and type detection
├── log_analyzer.py             # Log parsing and error extraction
├── error_clusterer.py          # Error similarity analysis and clustering
├── ocr_processor.py            # Screenshot text extraction
├── direct_component_analyzer.py # Direct component mapping and analysis
├── gpt_summarizer.py           # AI root cause analysis
├── gherkin_log_correlator.py   # Gherkin feature file parsing
├── step_aware_analyzer.py      # Step-aware analysis and reporting
├── batch_processor.py          # Batch processing functionality
├── secure_api_key.py           # Secure API key handling
├── json_utils.py               # JSON serialization utilities
├── components/                 # Enhanced component analysis system
│   ├── component_analyzer.py   # Component relationship analysis
│   ├── component_integration.py # Integration layer for components
│   ├── component_visualizer.py # Component visualization generation
│   ├── context_aware_clusterer.py # Enhanced error clustering
│   └── schemas/
│       └── component_schema.json # Component relationship definitions
├── reports/                    # Report generation framework
│   ├── __init__.py             # Package initialization
│   ├── base.py                 # Common utilities and base classes
│   ├── report_manager.py       # Report generation orchestrator
│   ├── data_preprocessor.py    # Data normalization and validation
│   ├── component_analyzer.py   # Component analysis for reports
│   ├── component_report.py     # Component report generation
│   ├── json_generator.py       # JSON report generation
│   ├── markdown_generator.py   # Markdown report generation
│   ├── excel_generator.py      # Excel report generation
│   ├── docx_generator.py       # Bug report document generation
│   └── visualizations.py       # Visualization generation
├── utils/                      # Utility modules
│   ├── path_utils.py           # Centralized path handling
│   └── path_validator.py       # Path validation utilities
└── Build_Script.py             # Executable build script
```

## Core Components

### Controller (`controller.py`)

The controller orchestrates the entire analysis pipeline. It:

- Handles user input through command-line interface
- Manages program flow and pipeline execution
- Collects test metadata and locates feature files
- Calls processing components in sequence
- Handles error conditions with graceful degradation
- Provides feedback to the user

The key functions include:

```python
def run_pipeline(test_id: str, gpt_model: str = None, enable_ocr: bool = None, test_type: str = "ymir") -> Tuple[str, Optional[str]]:
    """Run the log analysis pipeline programmatically."""
    
def run_pipeline_interactive():
    """Interactive command-line interface for the log analyzer."""
    
def run_gherkin_correlation(feature_file, log_files, output_dir, test_id, errors=None, error_clusters=None, component_analysis=None) -> Tuple[Optional[str], Optional[Dict]]:
    """Run the Gherkin log correlation and generate step-aware report with cluster visualization."""
    
def diagnose_output_structure(test_id: str) -> Dict[str, Any]:
    """Run diagnostics on the output directory structure."""
```

### Configuration (`config.py`)

The configuration module manages system settings and environment variables:

```python
class Config:
    """Configuration settings for the log analyzer."""
    
    # Base directories
    LOG_BASE_DIR: str  # Directory containing log files
    OUTPUT_BASE_DIR: str  # Directory for output reports
    
    # API settings
    OPENAI_API_KEY: str  # Authentication key for OpenAI API
    
    # Analysis settings
    ENABLE_OCR: bool  # Whether to enable OCR
    
    # Model settings
    DEFAULT_MODEL: str  # Default GPT model to use
    
    # Log settings
    LOG_LEVEL: str  # Logging verbosity
    LOG_FILE: str  # File to write logs
    
    # Feature flags for visualization generation
    ENABLE_CLUSTER_TIMELINE: bool
    ENABLE_COMPONENT_DISTRIBUTION: bool 
    ENABLE_ERROR_PROPAGATION: bool
    ENABLE_STEP_REPORT_IMAGES: bool
    ENABLE_COMPONENT_REPORT_IMAGES: bool
    
    @classmethod
    def setup_logging(cls):
        """Set up logging configuration with proper UTF-8 handling."""
        
    @classmethod
    def validate(cls):
        """Validate configuration settings."""
```

### Log Segmenter (`log_segmenter.py`)

This module handles the collection and categorization of files:

- Recursively scans the input directory
- Collects all supported log file types (`.log`, `.txt`, `.chlsj`, `.har`)
- Gathers all screenshot files (`.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`)
- Returns organized file lists for processing

```python
# Supported file extensions
SUPPORTED_LOG_EXTENSIONS = ('.log', '.txt', '.chlsj', '.har')
SUPPORTED_IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

def collect_log_files(base_dir: str) -> List[str]:
    """Recursively collects all log files from a directory tree."""
    
def collect_image_files(base_dir: str) -> List[str]:
    """Recursively collects all image files from a directory tree."""
    
def collect_all_supported_files(base_dir: str) -> Tuple[List[str], List[str]]:
    """Collects both logs and images for processing."""
```

### Log Analyzer (`log_analyzer.py`)

Responsible for parsing log files and extracting error information:

```python
def parse_logs(log_paths: List[str], context_lines: int = 3) -> List[Dict]:
    """Parse log files to extract errors and their context."""
    
def parse_log_entries(log_path: str) -> List[Dict]:
    """Parse a log file to extract all entries (not just errors)."""
    
def parse_har_file(path: str, lines: List[str]) -> List[Dict]:
    """Parse HAR file to extract HTTP errors."""
    
def identify_component_from_filename(filename: str) -> Tuple[str, str]:
    """Identify component based on filename pattern using a simplified approach."""
    
def determine_severity(error_text: str) -> str:
    """Determine error severity based on text content."""
    
def is_false_positive(line: str) -> bool:
    """Check if an error line is likely a false positive."""
```

### Error Clusterer (`error_clusterer.py`)

Groups similar errors together using Natural Language Processing:

```python
def perform_error_clustering(errors: List[Dict], num_clusters: int = None) -> Dict[int, List[Dict]]:
    """Cluster errors based on similarity."""
    
def _normalize_error_text(text: str) -> str:
    """Normalize error text for better clustering by removing variable parts."""
    
def _vectorize_errors(texts: List[str]) -> np.ndarray:
    """Convert error texts to TF-IDF vectors for clustering."""
    
def _determine_optimal_clusters(matrix: np.ndarray, num_errors: int, 
                             user_specified: Optional[int] = None) -> int:
    """Determine the optimal number of clusters based on dataset size."""
```

### OCR Processor (`ocr_processor.py`)

Extracts text from screenshots to capture errors not present in logs:

```python
def extract_ocr_data(image_paths: List[str], min_length: int = 15) -> List[Dict]:
    """Performs OCR on each image. Skips blank or too-short results."""
```

### Direct Component Analyzer (`direct_component_analyzer.py`)

Efficiently maps errors to system components and identifies primary issue component:

```python
def identify_component_from_filename(filename: str) -> Tuple[str, str]:
    """Identify component based on filename pattern."""
    
class ComponentCache:
    """A caching mechanism for component identification to improve performance."""
    
class ComponentAnalyzer:
    """Optimized Component Analyzer that performs component analysis in a single pass."""
    
    def assign_component_to_error(self, error: Dict) -> None:
        """Assign component to a single error based on analysis rules."""
        
    def identify_primary_component(self) -> str:
        """Identify the primary component with issues based on component counts."""
        
    def generate_component_summary(self) -> List[Dict]:
        """Generate summary of components for error report."""
        
def assign_components_and_relationships(errors: List[Dict]) -> Tuple[List[Dict], List[Dict], str]:
    """Optimized main function to assign components to errors and identify relationships."""
```

### Enhanced Component Analysis (`components/` package)

Provides deeper component relationship analysis:

#### Component Analyzer (`components/component_analyzer.py`)

```python
class ComponentAnalyzer:
    """Analyzer for identifying components and their relationships in log entries."""
    
    def identify_component_from_line(self, line: str) -> str:
        """Identify component based on line content."""
    
    def identify_component_from_log_file(self, log_file_path: str) -> str:
        """Identify component based on filename pattern using schema patterns."""
    
    def enrich_log_entries_with_components(self, log_entries: List[Any]) -> List[Any]:
        """Enrich log entries with component information."""
    
    def analyze_component_failures(self, errors: List[Any]) -> Dict[str, Any]:
        """Analyze component failures and their relationships."""
    
    def _identify_root_cause_component(self, errors: List[Any]) -> Optional[str]:
        """Attempt to identify the root cause component based on error timing and relationships."""
    
    def _build_causality_graph(self, errors: List[Any], 
                             max_time_delta: timedelta = timedelta(seconds=10)) -> nx.DiGraph:
        """Build a directed graph of potential cause-effect relationships between errors."""
```

#### Component Visualizer (`components/component_visualizer.py`)

```python
class ComponentVisualizer:
    """Generates visualizations of component relationships and error propagation."""
    
    def generate_component_relationship_diagram(self, output_dir: str, test_id: str = None) -> str:
        """Generate a basic component relationship diagram."""
    
    def generate_error_propagation_diagram(self, output_dir: str, component_errors: Dict[str, int],
        root_cause_component: Optional[str] = None, propagation_paths: List[List[str]] = None,
        test_id: str = "Unknown") -> str:
        """Generate a diagram showing error propagation through components."""
    
    def generate_component_error_heatmap(self, output_dir: str, error_analysis: Dict[str, Any],
        test_id: str = "Unknown") -> str:
        """Generate a heatmap showing error counts across components."""
```

#### Context-Aware Clusterer (`components/context_aware_clusterer.py`)

```python
class ContextAwareClusterer:
    """Enhanced error clustering that takes into account component relationships,
    temporal sequences, and cause-effect relationships."""
    
    def cluster_errors(self, errors: List[Dict], num_clusters: Optional[int] = None) -> Dict[int, List[Dict]]:
        """Cluster errors with awareness of component relationships and temporal sequence."""
    
    def _enhance_clusters(self, clusters: Dict[int, List[Dict]], components: List[str]) -> Dict[int, List[Dict]]:
        """Enhance clusters with component and temporal information."""
    
    def _classify_root_vs_symptom(self, clusters: Dict[int, List[Dict]],
        related_clusters: Dict[int, Set[int]], temporal_relationships: Dict[str, Any]) -> Dict[str, Any]:
        """Classify clusters as either root causes or symptoms."""
    
    def _build_error_graph(self, clusters: Dict[int, List[Dict]], errors: List[Dict]) -> nx.DiGraph:
        """Build a directed graph of error relationships."""
    
    def get_root_cause_errors(self, clusters: Dict[int, List[Dict]]) -> List[Dict]:
        """Get the list of errors identified as potential root causes."""
    
    def get_causality_paths(self) -> List[List[Dict]]:
        """Get potential causality paths from the error graph."""
```

#### Component Integration (`components/component_integration.py`)

```python
class ComponentIntegration:
    """Integration layer for component relationship analysis, visualizations,
    and enhanced error clustering."""
    
    def analyze_logs(self, log_entries: List[Any], errors: List[Any], output_dir: str, test_id: str) -> Dict[str, Any]:
        """Perform comprehensive component-aware analysis with simplified component identification."""
    
    def get_enhanced_report_data(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate additional data for enhancing test reports."""
```

### GPT Summarizer (`gpt_summarizer.py`)

Leverages OpenAI's GPT models to analyze errors and determine root cause:

```python
def build_clustered_prompt(test_id: str, clusters: Dict[int, List[Dict]], ocr_data: List[Dict], 
                          scenario_text: str = "", limited: bool = False) -> str:
    """Build a prompt for GPT with clustered error information, OCR data, and scenario context."""
    
def build_gpt_prompt(test_id: str, errors: List[Dict], component_summary: List[Dict], 
                    primary_issue_component: str, clusters: Dict[int, List[Dict]] = None,
                    ocr_data: List[Dict] = None, scenario_text: str = "",
                    limited: bool = False) -> str:
    """Build an enhanced GPT prompt with component relationship information."""
    
def sanitize_text_for_api(text: str) -> str:
    """Sanitize text before sending to API to remove potentially sensitive information."""
    
def enhance_prompt_with_component_data(prompt: str, component_analysis: Dict[str, Any]) -> str:
    """Enhance the GPT prompt with component relationship data."""
    
def send_to_openai_chat(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """Send a prompt to OpenAI's Chat API and return the response."""
    
def fallback_summary(errors: List[Dict], clusters: Dict[int, List[Dict]], 
                    component_summary: List[Dict] = None, 
                    primary_issue_component: str = "unknown") -> str:
    """Generate a basic summary when GPT is not available."""
    
def generate_summary_from_clusters(clusters: Dict[int, List[Dict]], ocr_data: List[Dict],
    test_id: str, scenario_text: str = "", use_gpt: bool = True,
    model: str = DEFAULT_MODEL, step_to_logs: Optional[Dict[int, List[Any]]] = None,
    feature_file: Optional[str] = None, component_analysis: Optional[Dict[str, Any]] = None) -> str:
    """Generate a summary from clustered errors using GPT with enhanced component analysis."""
```

### Path Utilities (`utils/path_utils.py`)

Provides standardized path handling for consistent file placement:

```python
class OutputType(Enum):
    """Enumeration of output file types with their destinations"""
    PRIMARY_REPORT = "primary"  # Goes in root directory (Excel, DOCX, HTML)
    JSON_DATA = "json"          # Goes in json/ subdirectory
    VISUALIZATION = "image"     # Goes in supporting_images/ subdirectory
    DEBUGGING = "debug"         # Goes in debug/ subdirectory (optional)

def normalize_test_id(test_id: str) -> str:
    """Normalize test ID to standard SXM-#### format."""
    
def get_output_path(base_dir: str, test_id: str, filename: str, 
                  output_type: OutputType = OutputType.PRIMARY_REPORT,
                  create_dirs: bool = True) -> str:
    """Get standardized output path based on file type."""
    
def setup_output_directories(base_dir: str, test_id: str) -> Dict[str, str]:
    """Create standard output directory structure."""
    
def get_standardized_filename(test_id: str, file_type: str, extension: str) -> str:
    """Create standardized filename with test ID prefix."""
```

### Path Validator (`utils/path_validator.py`)

Validates correct file placement within the output structure:

```python
def validate_file_structure(base_dir: str, test_id: str) -> Dict[str, List[str]]:
    """Validate that files are in their proper locations."""
    
def check_html_references(html_file: str) -> Dict[str, List[str]]:
    """Check HTML file for correct references to supporting files."""
    
def print_validation_results(base_dir: str, test_id: str):
    """Run validation and print results in a user-friendly format."""
```

### Gherkin Log Correlation (`gherkin_log_correlator.py`)

Correlates log entries with steps in Gherkin feature files:

```python
@dataclass
class GherkinStep:
    """Represents a single step in a Gherkin feature file."""
    
class LogEntry:
    """Represents a parsed log entry with metadata."""
    
class LogFormatAdapter:
    """Base class for log format adapters."""
    
class GherkinParser:
    """Parser for Gherkin feature files."""
    
class GherkinLogCorrelator:
    """Correlates Gherkin steps with log entries."""
    
    def analyze(self) -> Dict[int, List[LogEntry]]:
        """Analyze logs and correlate them with Gherkin steps."""
        
def correlate_logs_with_steps(feature_file_path: str, log_file_paths: List[str]) -> Dict[int, List[LogEntry]]:
    """Correlate log entries with Gherkin steps."""
```

### Step-Aware Analyzer (`step_aware_analyzer.py`)

Generates step-aware reports with visualizations for Gherkin tests:

```python
def generate_step_report(feature_file: str, logs_dir: str, step_to_logs: Dict[int, List[LogEntry]],
    output_dir: str, test_id: str, clusters: Optional[Dict[int, List[Dict]]] = None,
    component_analysis: Optional[Dict[str, Any]] = None) -> str:
    """Generate an HTML report showing logs correlated with Gherkin steps."""
    
def run_step_aware_analysis(test_id: str, feature_file: str, logs_dir: str, output_dir: str,
    clusters: Optional[Dict[int, List[Dict]]] = None, errors: Optional[List[Dict]] = None,
    component_analysis: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Run a step-aware analysis and generate HTML report."""
```

### Report Generation System (`reports/` package)

Generates comprehensive reports in various formats:

#### Report Manager (`reports/report_manager.py`)

```python
class ReportManager:
    """Orchestrates the report generation process."""
    
    def generate_reports(self, data: ReportData) -> Dict[str, Any]:
        """Generate all reports."""
```

#### Base Classes (`reports/base.py`)

```python
class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles datetime objects."""
    
class ComponentAwareEncoder(DateTimeEncoder):
    """Enhanced JSON encoder that carefully preserves component information during serialization."""
    
class ReportConfig:
    """Configuration settings for report generation."""
    
class ReportData:
    """Container for data used in report generation."""
    
class ReportGenerator:
    """Base class for report generators."""
```

#### Excel Generator (`reports/excel_generator.py`)

```python
class ExcelReportGenerator(ReportGenerator):
    """Generator for Excel reports."""
```

#### DOCX Generator (`reports/docx_generator.py`)

```python
class DocxReportGenerator(ReportGenerator):
    """Generator for DOCX reports."""
    
def generate_bug_document(output_dir: str, test_id: str, summary: str, errors: List[Dict],
    ocr_data: List[Dict], clusters: Dict[int, List[Dict]], background_text: str = "",
    scenario_text: str = "", component_analysis: Optional[Dict[str, Any]] = None,
    primary_issue_component: str = "unknown", component_report_path: Optional[str] = None) -> str:
    """Generate a DOCX file formatted for Jira bug submission."""
```

#### JSON Generator (`reports/json_generator.py`)

```python
class JsonReportGenerator(ReportGenerator):
    """Generator for JSON reports."""
```

#### Markdown Generator (`reports/markdown_generator.py`)

```python
class MarkdownReportGenerator(ReportGenerator):
    """Generator for Markdown reports."""
```

#### Component Report (`reports/component_analyzer.py`)

```python
def build_component_analysis(errors: List[Dict], primary_issue_component: str, 
                          existing_analysis: Optional[Dict] = None) -> Dict[str, Any]:
    """Build comprehensive component analysis data structure."""
    
def generate_component_report(output_dir: str, test_id: str,
                           component_analysis: Dict[str, Any],
                           primary_issue_component: str) -> str:
    """Generate component analysis report."""
```
HTML reports are rendered using Jinja2 templates stored in `reports/templates`.
```
```

#### Visualizations (`reports/visualizations.py`)

```python
def generate_timeline_image(step_to_logs, step_dict, output_dir, test_id) -> str:
    """Generate a PNG image of the timeline visualization."""
    
def generate_cluster_timeline_image(step_to_logs, step_dict, clusters, output_dir, test_id) -> str:
    """Generate a cluster timeline image."""
    
def generate_component_visualization(output_dir, test_id, components=None, relationships=None, primary_component=None) -> str:
    """Generate a visualization of component relationships."""
    
def generate_component_error_distribution(output_dir, test_id, component_summary=None, clusters=None, primary_component=None) -> str:
    """Generate a visualization of error distribution across components."""
    
def generate_error_propagation_diagram(output_dir, test_id, error_graph=None) -> str:
    """Generate a visualization of error propagation paths."""
```

### Batch Processing (`batch_processor.py`)

Provides batch processing for analyzing multiple tests:

```python
def find_test_folders() -> List[str]:
    """Automatically discover all SXM-* folders in the logs directory."""
    
def process_single_test(test_id: str) -> Dict[str, Any]:
    """Process a single test case and return results."""
    
def process_batch(test_ids: List[str], parallel: bool = False) -> Dict[str, Dict]:
    """Process multiple test IDs either sequentially or in parallel."""
    
def generate_batch_report(results: Dict[str, Dict], output_file: str = None) -> str:
    """Generate a summary report for the batch processing results."""
```

### Build Script (`Build_Script.py`)

Creates a standalone executable package:

```python
def get_version():
    """Get version from file or generate based on date."""
    
def backup_folder(folder_name):
    """Backup an existing folder with timestamp."""
    
def check_pyinstaller():
    """Check if PyInstaller is installed, install if needed."""
    
def build_executable():
    """Build the executable using PyInstaller."""
    
def create_distribution_folder():
    """Create a distribution folder with executable and documentation."""
    
def zip_distribution():
    """Create ZIP archive of the distribution folder."""
```

## Data Flow

The system implements a pipeline architecture with these key stages:

1. **Collection and Preprocessing**:
   - User provides test ID (or batch of IDs)
   - Log files and images are collected
   - Logs are parsed to extract errors
   - OCR processes images to extract text

2. **Component Identification**:
   - Initial component identification during log parsing
   - Enhanced identification with direct_component_analyzer
   - Comprehensive component relationship analysis (if available)
   - Primary issue component determination

3. **Error Analysis**:
   - Error clustering by similarity
   - Context-aware clustering with component knowledge
   - Component relationship mapping
   - Error propagation analysis

4. **AI-Powered Summary**:
   - Component-enhanced prompt construction
   - GPT analysis of errors and components
   - Summary generation with root cause identification
   - Fallback to rule-based summary if GPT unavailable

5. **Report Generation**:
   - Multi-format report generation (Excel, DOCX, Markdown, JSON)
   - Visualization creation
   - Component relationship diagrams
   - Timeline generation

This pipeline uses standardized paths and consistent component information throughout.

## Output Directory Structure

```
output/
+-- SXM-123456/                        # Test-specific directory
|   |-- SXM-123456_log_analysis.xlsx   # Main Excel report
|   |-- SXM-123456_bug_report.docx     # Bug report document
|   |-- SXM-123456_log_analysis.md     # Markdown report
|   |-- SXM-123456_component_report.html # Component analysis report
|   |
|   +-- json/                          # JSON data subdirectory
|   |   |-- SXM-123456_log_analysis.json # Main analysis data
|   |   |-- SXM-123456_component_analysis.json # Component analysis data
|   |   |-- SXM-123456_error_graph.json # Error relationship graph
|   |   +-- SXM-123456_component_preservation.json # Component preservation data
|   |
|   +-- supporting_images/             # Visualizations subdirectory
|   |   |-- SXM-123456_timeline.png    # Standard timeline visualization
|   |   |-- SXM-123456_cluster_timeline.png # Cluster timeline visualization
|   |   |-- SXM-123456_component_relationships.png # Component relationship diagram
|   |   |-- SXM-123456_error_propagation.png # Error propagation diagram
|   |   +-- SXM-123456_component_distribution.png # Component error distribution
|   |
|   +-- debug/                         # Debug information
|       |-- SXM-123456_timeline_debug.txt # Timeline generation debug log
|       +-- SXM-123456_component_debug.txt # Component analysis debug log
```

## Component Identification and Preservation

The system employs a sophisticated mechanism for identifying components and preserving component information throughout processing:

### Component Fields

```python
# Critical component fields that must be preserved
COMPONENT_FIELDS = {
    'component',              # The identified component
    'component_source',       # How the component was identified
    'source_component',       # Original component before processing
    'root_cause_component',   # Component identified as root cause
    'primary_issue_component',# Primary component for reporting
    'affected_components',    # Related components affected
    'expected_component',     # Expected component for validation
    'component_scores',       # Confidence scores for components
    'component_distribution', # Distribution of components
    'parent_component',       # Parent in component hierarchy
    'child_components'        # Children in component hierarchy
}
```

### Component Schema

Component relationships are defined in `component_schema.json`:

```json
{
  "components": [
    {
      "id": "soa",
      "name": "SOA",
      "description": "SiriusXM application built on Android",
      "type": "application",
      "logSources": ["adb", "appium*.log", "app_debug.log"],
      "receives_from": ["phoebe", "mimosa"],
      "sends_to": ["charles"],
      "parent": "android",
      "errorPatterns": [
        "(?i)exception.*?com\\.siriusxm",
        "(?i)soa.*?error",
        "(?i)failed to load.*?channel"
      ]
    }
  ],
  "dataFlows": [
    {
      "source": "mimosa",
      "target": "soa",
      "description": "Fake test data (direct)",
      "dataType": "test_signals"
    }
  ]
}
```

### Component Visualization

The system generates several component-related visualizations:

1. **Component Relationship Diagram**: Shows how different components interact with each other
2. **Error Propagation Diagram**: Visualizes how errors propagate through the component structure
3. **Component Error Heatmap**: Shows which components have the most errors

## Path Handling

File paths are managed through a standardized system:

```python
from utils.path_utils import get_output_path, OutputType, get_standardized_filename

# Example usage:
image_path = get_output_path(
    output_dir, 
    test_id, 
    get_standardized_filename(test_id, "component_distribution", "png"),
    OutputType.VISUALIZATION
)
```

This ensures files are consistently placed in the correct subdirectories based on their type.

## Advanced Technical Usage

### Programmatic Integration

```python
from controller import run_pipeline
from config import Config

# Configure environment
import os
os.environ["ENABLE_OCR"] = "True"
os.environ["LOG_LEVEL"] = "INFO"

# Set up logging
Config.setup_logging()
Config.validate()

# Run analysis
result, step_report = run_pipeline(
    test_id="SXM-1234567",
    gpt_model="gpt-3.5-turbo",
    enable_ocr=True,
    test_type="ymir"
)

# Access results
print(f"Analysis completed: {result}")
if step_report:
    print(f"Step report generated at: {step_report}")
```

### Batch Processing

For analyzing multiple tests efficiently:

```python
from batch_processor import process_batch, find_test_folders, generate_batch_report

# Find all test folders
test_ids = find_test_folders()
print(f"Found {len(test_ids)} test IDs: {test_ids}")

# Process in parallel for better performance
results = process_batch(test_ids, parallel=True)

# Generate and save a summary report
report = generate_batch_report(results, "batch_analysis_report.txt")
print(report)
```

### Component Analysis Enhancement

To customize component analysis:

```python
# Define custom component relationships
from components.component_integration import ComponentIntegration

# Initialize with custom schema
integrator = ComponentIntegration("path/to/custom_schema.json")

# Perform custom analysis
component_analysis = integrator.analyze_logs(
    log_entries=log_entries, 
    errors=errors, 
    output_dir=output_dir, 
    test_id=test_id
)

# Extract root cause component
root_cause = component_analysis.get("metrics", {}).get("root_cause_component", "unknown")
print(f"Root cause component: {root_cause}")
```

### Custom Report Generation

To generate specific reports:

```python
from reports.report_manager import ReportManager
from reports.base import ReportConfig, ReportData

# Configure report generation
config = ReportConfig(
    output_dir="path/to/output",
    test_id="SXM-1234567",
    primary_issue_component="soa",
    enable_excel=True,
    enable_markdown=True,
    enable_json=True,
    enable_docx=True,
    enable_component_report=True
)

# Set up report data
data = ReportData(
    errors=errors,
    summary="AI-generated summary of the issue...",
    clusters=error_clusters,
    ocr_data=ocr_results,
    background_text="Gherkin background...",
    scenario_text="Gherkin scenario...",
    ymir_flag=True,
    component_analysis=component_analysis
)

# Generate reports
manager = ReportManager(config)
results = manager.generate_reports(data)

# Output paths
for report_type, path in results.items():
    print(f"{report_type}: {path}")
```

## Troubleshooting Advanced Issues

### 1. Component Identification Issues

If component identification is inconsistent or producing unexpected results:

```python
# Enable diagnostic logging
import logging
logging.getLogger('components').setLevel(logging.DEBUG)

# Verify component identification directly
from direct_component_analyzer import identify_component_from_filename
print(identify_component_from_filename("app_debug.log"))  # Should return ('soa', 'filename_special')

# Check component schema loading
from components.component_analyzer import ComponentAnalyzer
analyzer = ComponentAnalyzer("components/schemas/component_schema.json")
print(analyzer.component_schema)
```

### 2. Path-Related Issues

For issues with file placement or HTML references:

```python
# Run path validation
from utils.path_validator import print_validation_results
print_validation_results("output/SXM-1234567", "SXM-1234567")

# Check HTML references
from utils.path_validator import check_html_references
issues = check_html_references("output/SXM-1234567/SXM-1234567_component_report.html")
print(issues)
```

### 3. Report Generation Issues

For report generation problems:

```python
# Run diagnostic analysis on output structure
from controller import diagnose_output_structure
diagnosis = diagnose_output_structure("SXM-1234567")
print(diagnosis)

# Check component preservation
import json
with open("output/SXM-1234567/json/SXM-1234567_component_preservation.json", "r") as f:
    preservation_data = json.load(f)
print(preservation_data)
```

### 4. GPT Integration Issues

For problems with AI-powered analysis:

```python
# Check API key configuration
from secure_api_key import get_openai_api_key
api_key = get_openai_api_key()
print(f"API key found: {'Yes' if api_key else 'No'}")

# Test prompt sanitization
from gpt_summarizer import sanitize_text_for_api
sample_text = "Error connecting to https://username:password@api.example.com"
sanitized = sanitize_text_for_api(sample_text)
print(f"Sanitized: {sanitized}")

# Use fallback summary generation
from gpt_summarizer import fallback_summary
summary = fallback_summary(errors, clusters, component_summary, primary_issue_component)
print(summary)
```

### 5. Memory Issues with Large Log Sets

For handling large log files:

```python
# Process logs in chunks
def process_large_logs(log_files, chunk_size=1000):
    all_errors = []
    for log_file in log_files:
        with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
        
        # Process in chunks
        for i in range(0, len(lines), chunk_size):
            chunk = lines[i:i+chunk_size]
            # Process this chunk
            errors = process_chunk(log_file, chunk)
            all_errors.extend(errors)
    
    return all_errors
```

## Component-Aware Development Best Practices

1. **Component Identification Conservation**
   - Preserve component information throughout processing
   - Use consistent field names from `COMPONENT_FIELDS`
   - Track how component was identified with `component_source`

2. **Component Relationship Awareness**
   - Understand the component schema relationships
   - Preserve relationship information in analysis
   - Use the component graph for relationship lookups

3. **Primary Component Consistency**
   - Maintain consistency of `primary_issue_component`
   - Propagate `primary_issue_component` to all data structures
   - Verify `primary_issue_component` before generating reports

4. **Path Handling Best Practices**
   - Always use `get_output_path()` for file paths
   - Use correct `OutputType` for each file
   - Use `get_standardized_filename()` for consistent naming

5. **HTML References**
   - Always use `supporting_images/` prefix in HTML
   - Use relative paths for better portability
   - Validate HTML references with `check_html_references()`

## Extending Orbit

Orbit's modular architecture makes it easy to extend:

### Adding New Report Formats

Create a new generator class in the `reports` package:

```python
from reports.base import ReportGenerator

class MyCustomReportGenerator(ReportGenerator):
    """Generator for custom reports."""
    
    def generate(self, data: ReportData) -> str:
        """Generate a custom report."""
        # Implementation...
        return output_path
```

Then register it in `ReportManager.__init__()`.

### Supporting New Component Types

Extend the component schema with new component definitions:

```json
{
  "components": [
    {
      "id": "my_component",
      "name": "My Component",
      "description": "New component in the system",
      "type": "custom_type",
      "logSources": ["my_component*.log"],
      "receives_from": ["existing_component"],
      "sends_to": [],
      "parent": "platform",
      "errorPatterns": [
        "(?i)my_component.*?error",
        "(?i)failed to.*?my service"
      ]
    }
  ]
}
```

### Creating Custom Visualizations

Implement a new visualization function in `reports/visualizations.py`:

```python
def generate_my_custom_visualization(output_dir, test_id, data) -> str:
    """Generate a custom visualization."""
    # Implementation using matplotlib, networkx, etc.
    return output_path
```

Then call it from the report generation process.

## Conclusion

The Orbit Analyzer provides a comprehensive framework for log analysis, component identification, and report generation. By understanding the core modules and their interactions, developers can effectively maintain, extend, and optimize the system to meet specific requirements.

Key architectural strengths include:
- Modular design with clear separation of concerns
- Standardized path handling for consistent file organization
- Component-aware analysis for deep system understanding
- Visualization capabilities for intuitive error analysis
- Multi-format reporting for different stakeholder needs

For further customization or specific integration requirements, refer to the in-code documentation and the extensive module interfaces outlined in this guide.