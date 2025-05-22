# Orbit Analyzer - Environment and Configuration Documentation

## 1. Environment Variables and Configuration

### Environment Variables

| Variable | Purpose | Default | Required |
|----------|---------|---------|----------|
| `OPENAI_API_KEY` | Authentication key for OpenAI API | None | Yes (for GPT features) |
| `LOG_BASE_DIR` | Directory containing log files | `./logs` | No |
| `OUTPUT_BASE_DIR` | Directory for output reports | `./output` | No |
| `LOG_LEVEL` | Logging verbosity | `INFO` | No |
| `LOG_FILE` | File to write logs | `orbit_analyzer.log` | No |
| `DEFAULT_MODEL` | Default GPT model to use | `gpt-3.5-turbo` | No |
| `ENABLE_OCR` | Whether to enable OCR | `True` | No |
| `INSTALLER_FOLDER` | Selected folder for installer tests | None | No (for installer tests) |
| `PYTHONIOENCODING` | Ensures proper encoding for logs and output | `utf-8` | No (set automatically) |

### Configuration Files

| File | Purpose | Format |
|------|---------|--------|
| `config.py` | Core configuration settings and environment variable handling | Python |
| `.env` | Environment variable definitions | Key-value pairs |
| `component_schema.json` | Component relationship definitions | JSON |
| `Requirements.txt` | Python package dependencies | Text |

### Configuration Class

The main configuration is managed through the `Config` class in `config.py`:

```python
class Config:
    """Configuration settings for the log analyzer."""
    
    # Base directories
    LOG_BASE_DIR = os.getenv("LOG_BASE_DIR", "./logs")
    OUTPUT_BASE_DIR = os.getenv("OUTPUT_BASE_DIR", "./output")

    # API settings
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    
    # Analysis settings
    ENABLE_OCR = os.getenv("ENABLE_OCR", "True").lower() in ("true", "1", "yes")
    
    # Model settings
    DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-3.5-turbo")
    
    # Log settings
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.getenv("LOG_FILE", "orbit_analyzer.log")
    
    # Flag to track if logging has been initialized
    _logging_initialized = False
    
    # Visualization enablement flags
    ENABLE_CLUSTER_TIMELINE = False
    ENABLE_COMPONENT_DISTRIBUTION = True
    ENABLE_COMPONENT_RELATIONSHIPS = True
    ENABLE_ERROR_PROPAGATION = False
    ENABLE_STEP_REPORT_IMAGES = False
    ENABLE_COMPONENT_REPORT_IMAGES = True
    
    @classmethod
    def setup_logging(cls):
        """Set up logging configuration with proper UTF-8 handling."""
        # Check if logging is already configured to prevent duplicates
        if cls._logging_initialized:
            # Logging was already configured, don't do it again - silently return
            return
            
        # Set PYTHONIOENCODING environment variable to utf-8
        os.environ["PYTHONIOENCODING"] = "utf-8"
        
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        log_level = getattr(logging, cls.LOG_LEVEL.upper(), logging.INFO)
        
        # Get root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        
        # Clear existing handlers to avoid duplicates
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Create a UTF-8 enabled file handler
        file_handler = logging.FileHandler(cls.LOG_FILE, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter(log_format))
        
        # Create a UTF-8 enabled console handler
        console_handler = UTF8LoggingHandler()
        console_handler.setFormatter(logging.Formatter(log_format))
        
        # Add handlers to the root logger
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        # Mark logging as initialized
        cls._logging_initialized = True
        
        logging.info(f"Logging initialized at {datetime.now().isoformat()} with UTF-8 encoding")
    
    @classmethod
    def validate(cls):
        """Validate configuration settings."""
        # Check if base directories exist, create if they don't
        os.makedirs(cls.LOG_BASE_DIR, exist_ok=True)
        os.makedirs(cls.OUTPUT_BASE_DIR, exist_ok=True)
        
        # Check OpenAI API key if using GPT models
        if not cls.OPENAI_API_KEY and cls.DEFAULT_MODEL.lower() != "none":
            logging.warning(f"OpenAI API key not set. GPT analysis will not be available.")
            
        # Validate model settings
        valid_models = ["gpt-4", "gpt-3.5-turbo", "none"]
        if cls.DEFAULT_MODEL.lower() not in valid_models:
            logging.warning(f"Unknown model: {cls.DEFAULT_MODEL}. Using gpt-3.5-turbo as fallback.")
            cls.DEFAULT_MODEL = "gpt-3.5-turbo"

    @classmethod
    def configure_matplotlib(cls):
        """
        Configure matplotlib settings for consistent visualization.
        This method configures matplotlib to use a non-GUI backend for thread safety.
        """
        try:
            import matplotlib
            # Force Agg backend for headless environments and thread safety
            matplotlib.use('Agg', force=True)
            
            import matplotlib.pyplot as plt
            # Configure global settings
            plt.rcParams['figure.max_open_warning'] = 50  # Prevent warnings for many figures
            plt.rcParams['font.size'] = 10  # Readable default font size
            plt.rcParams['figure.dpi'] = 100  # Default DPI
            
            logging.info("Matplotlib configured with Agg backend for thread safety")
            return True
        except Exception as e:
            logging.error(f"Error configuring matplotlib: {str(e)}")
            return False
```

### UTF-8 Logging Handler

The system implements a specialized logging handler to ensure proper UTF-8 encoding:

```python
class UTF8LoggingHandler(logging.StreamHandler):
    """Logging handler that ensures UTF-8 encoding for all log messages."""
    def __init__(self, stream=None):
        # If no stream is provided, use stdout with UTF-8 encoding
        if stream is None:
            # Create a new stream with UTF-8 encoding instead of modifying an existing one
            stream = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
        super().__init__(stream)
    
    def emit(self, record):
        """Emit a log record with UTF-8 encoding."""
        try:
            msg = self.format(record)
            # Ensure message is properly encoded as UTF-8
            if isinstance(msg, bytes):
                msg = msg.decode('utf-8', errors='replace')
            self.stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)
```

### Feature Flags

The system implements several feature flags in the `Config` class to control visualization and report generation:

```python
# Visualization enablement flags
ENABLE_CLUSTER_TIMELINE = False         # Enable cluster timeline visualization
ENABLE_COMPONENT_DISTRIBUTION = True    # Enable component distribution charts
ENABLE_COMPONENT_RELATIONSHIPS = True   # Enable component relationship diagrams
ENABLE_ERROR_PROPAGATION = False        # Enable error propagation visualization
ENABLE_STEP_REPORT_IMAGES = False       # Enable step report images
ENABLE_COMPONENT_REPORT_IMAGES = True   # Enable all component report visualizations
```

Additionally, runtime feature detection is implemented:

```python
# Try to import the component integration module
try:
    from components.component_integration import ComponentIntegration
    COMPONENT_INTEGRATION_AVAILABLE = True
except ImportError as e:
    COMPONENT_INTEGRATION_AVAILABLE = False
    logging.warning(f"Component integration module not available - will use direct component mapping: {str(e)}")
```

### Thread-Safe Feature Flag Checking

The visualization system provides a thread-safe mechanism for checking feature flags:

```python
# Thread-local storage for thread safety
import threading
_visualization_local = threading.local()

def _is_feature_enabled(feature_name, default=False):
    """
    Check if a feature is enabled with thread-safe fallback.
    
    Args:
        feature_name: Name of the feature flag in Config
        default: Default value if flag doesn't exist
        
    Returns:
        Boolean indicating if feature is enabled
    """
    # Use thread-local cache if available
    if not hasattr(_visualization_local, 'feature_cache'):
        _visualization_local.feature_cache = {}
    
    # Check cache first
    if feature_name in _visualization_local.feature_cache:
        return _visualization_local.feature_cache[feature_name]
    
    # Get from config
    from config import Config
    result = getattr(Config, feature_name, default)
    
    # Cache for future use
    _visualization_local.feature_cache[feature_name] = result
    
    return result
```

## 2. Path Handling and Output Structure

### Path Utilities

The system uses standardized path handling through the `utils/path_utils.py` module:

```python
class OutputType(Enum):
    """Enumeration of output file types with their destinations"""
    PRIMARY_REPORT = "primary"  # Goes in root directory (Excel, DOCX, HTML)
    JSON_DATA = "json"          # Goes in json/ subdirectory
    VISUALIZATION = "image"     # Goes in supporting_images/ subdirectory
    DEBUGGING = "debug"         # Goes in debug/ subdirectory (optional)

def normalize_test_id(test_id: str) -> str:
    """Normalize test ID to standard SXM-#### format"""
    if not isinstance(test_id, str):
        test_id = str(test_id)  # Convert non-string test_ids to strings
    test_id = test_id.strip()
    return test_id if test_id.upper().startswith("SXM-") else f"SXM-{test_id}"

def get_output_path(
    base_dir: str, 
    test_id: str, 
    filename: str, 
    output_type: OutputType = OutputType.PRIMARY_REPORT,
    create_dirs: bool = True
) -> str:
    """Get standardized output path based on file type"""
    # Implementation details...

def setup_output_directories(base_dir: str, test_id: str) -> Dict[str, str]:
    """Create standard output directory structure"""
    # Implementation details...

def get_standardized_filename(test_id: str, file_type: str, extension: str) -> str:
    """Create standardized filename with test ID prefix"""
    # Implementation details...
```

### Directory Structure

The system creates a standardized directory structure for each test:

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
|   |   +-- SXM-123456_component_preservation.json # Component preservation data
|   |
|   +-- supporting_images/             # Visualizations subdirectory
|   |   |-- SXM-123456_timeline.png    # Standard timeline visualization
|   |   |-- SXM-123456_cluster_timeline.png # Cluster timeline visualization
|   |   |-- SXM-123456_component_errors.png # Component error distribution visualization
|   |   |-- SXM-123456_component_distribution.png # Alias for component_errors.png
|   |   |-- SXM-123456_component_relationships.png # Component relationship diagram
|   |   |-- SXM-123456_error_propagation.png # Error propagation visualization
|   |
|   +-- debug/                         # Debug information
|       |-- SXM-123456_timeline_debug.txt # Timeline generation debug log
```

### Path Validation

The system includes a path validator to ensure correct file placement:

```python
def validate_file_structure(base_dir: str, test_id: str) -> Dict[str, List[str]]:
    """Validate that files are in their proper locations."""
    issues = {
        "json_dir_images": [],
        "images_dir_json": [],
        "nested_directories": [],
        "expected_but_missing": []
    }
    # Implementation details...
    return issues

def check_html_references(html_file: str) -> Dict[str, List[str]]:
    """Check HTML file for correct references to supporting files."""
    # Implementation details...
    return issues
```

## 3. Visualization System

### Visualization Architecture

The Orbit Analyzer visualization system has been redesigned to remove dependencies on PyGraphviz, providing a more reliable experience across different environments. The new system uses a layered approach:

```
┌─────────────────────────────────┐
│ Visualization Interface Layer   │
│ (reports/visualizations.py)     │
└───────────────┬─────────────────┘
                │
┌───────────────▼─────────────────┐
│ Component Visualization Layer   │
│ (components/component_visualizer.py) │
└───────────────┬─────────────────┘
                │
┌───────────────▼─────────────────┐
│ Graph Layout & Rendering Layer  │
│ (NetworkX with multi-level fallbacks) │
└───────────────┬─────────────────┘
                │
┌───────────────▼─────────────────┐
│ Matplotlib Backend Layer        │
│ (Thread-safe Agg renderer)      │
└─────────────────────────────────┘
```

### Visualization Components

The key components of the visualization system include:

1. **VisualizationGenerator** - High-level class in `reports/visualizations.py` that orchestrates visualization creation
2. **ComponentVisualizer** - Core visualization renderer in `components/component_visualizer.py`
3. **Layout System** - Multi-level fallback mechanism for graph layout without PyGraphviz

### Backend Configuration

The visualization system uses the non-interactive "Agg" backend for matplotlib to ensure thread safety and compatibility in all environments:

```python
def _configure_matplotlib_backend():
    """Configure matplotlib to work in any environment."""
    # Force Agg backend to avoid tkinter thread issues completely
    import matplotlib
    matplotlib.use('Agg', force=True)
    
    import matplotlib.pyplot as plt
    
    # Configure global settings
    plt.rcParams['figure.max_open_warning'] = 50  # Prevent warnings for many figures
    plt.rcParams['font.size'] = 10  # Readable default font size
    
    return plt
```

### Graph Layout System

The visualization system uses a multi-layered fallback approach for graph layouts:

```python
def _get_graph_layout(self, G):
    """
    Get a layout for the graph using available algorithms with robust fallbacks.
    Completely removes dependency on PyGraphviz while maintaining visualization quality.
    """
    # Check graph size to optimize layout approach
    node_count = G.number_of_nodes()
    if node_count == 0:
        return {}
    
    # For very small graphs, spring layout is sufficient and fast
    if node_count <= 3:
        return nx.spring_layout(G, seed=42)
    
    # First attempt: Try pydot (part of NetworkX)
    try:
        import pydot
        from networkx.drawing.nx_pydot import graphviz_layout
        # Silence warning messages from Pydot
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return graphviz_layout(G, prog='dot')
    except (ImportError, Exception) as e:
        logging.debug(f"Pydot layout unavailable ({str(e)}), trying next option")
    
    # Second attempt: Try spectral layout (good for component relationships)
    try:
        # Spectral layout works well for tree-like structures
        return nx.spectral_layout(G)
    except Exception as e:
        logging.debug(f"Spectral layout failed ({str(e)}), trying next option")
    
    # Third attempt: Try shell layout (good for visualizing hierarchies)
    try:
        # Group nodes by type or relationships
        groups = []
        seen = set()
        
        # Create groups based on node types or importance
        type_groups = defaultdict(list)
        for node in G.nodes():
            node_type = G.nodes[node].get("type", "unknown")
            type_groups[node_type].append(node)
            
        # Add groups in order of importance
        for group_type in ["application", "proxy", "platform", "unknown"]:
            if group_type in type_groups and type_groups[group_type]:
                groups.append(type_groups[group_type])
                seen.update(type_groups[group_type])
        
        # Add any remaining nodes
        remaining = [node for node in G.nodes() if node not in seen]
        if remaining:
            groups.append(remaining)
            
        # Only use shell layout if we have valid groups
        if groups:
            return nx.shell_layout(G, groups)
    except Exception as e:
        logging.debug(f"Shell layout failed ({str(e)}), falling back to spring layout")
    
    # Final fallback: Enhanced spring layout with optimized parameters
    return nx.spring_layout(
        G, 
        k=0.3 + (0.1 / max(node_count, 1)),  # Dynamic spacing based on node count
        iterations=100,                      # More iterations for better layout
        seed=42                              # Consistent layout between runs
    )
```

### Memory Management

All visualizations include proper memory management to prevent memory leaks:

```python
def _save_figure_with_cleanup(self, fig, image_path, dpi=100):
    """
    Save figure and ensure proper cleanup to prevent memory leaks.
    
    Args:
        fig: Matplotlib figure
        image_path: Path to save the image
        dpi: Resolution in dots per inch
        
    Returns:
        Path to the saved image
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        
        # Save figure with specified DPI
        fig.savefig(image_path, bbox_inches='tight', dpi=dpi)
        return image_path
    finally:
        # Always close figure to free memory, even if save fails
        plt.close(fig)
```

## 4. OpenAI API Integration

### API Key Management

The system handles OpenAI API keys securely through multiple methods:

```python
def get_openai_api_key():
    """Get the OpenAI API key from environment variable or keyring."""
    # First try to get from environment variable
    api_key = os.environ.get("OPENAI_API_KEY", "")
    
    # If not found in environment, try loading from .env file
    if not api_key:
        try:
            dotenv.load_dotenv()
            api_key = os.environ.get("OPENAI_API_KEY", "")
        except Exception:
            pass
    
    # If still not found, try keyring
    if not api_key:
        try:
            keyring_key = keyring.get_password("orbit_analyzer", "openai_api_key")
            if keyring_key:
                api_key = keyring_key
        except Exception as e:
            logging.warning(f"Could not access system keyring: {str(e)}")
    
    # If no key found, log a warning
    if not api_key:
        logging.warning("OpenAI API key not found. GPT-based analysis will not be available.")
        return ""  # Return empty string instead of None
    
    return api_key
```

### API Request Handling

The system sends requests to OpenAI's API with privacy-focused headers:

```python
def send_to_openai_chat(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """
    Send a prompt to OpenAI's Chat API and return the response.
    Uses secure API key handling and ensures privacy compliance.
    """
    # Get API key using secure method
    api_key = get_openai_api_key()
    if not api_key:
        return "Error: OpenAI API key not found. Please set OPENAI_API_KEY environment variable or configure it in the system keyring."
        
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # Define messages for the conversation
    messages = [
        {"role": "system", "content": "You are a test automation expert who analyzes logs and provides clear, concise explanations of test failures."},
        {"role": "user", "content": prompt}
    ]
    
    # Create the payload without the metadata field
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE
    }
    
    # Add privacy-related headers instead of metadata
    headers["OpenAI-Beta"] = "optout=train"  # Optional: Signal not to use for training
    
    # Make request and handle response
    # Implementation details...
```

### Model Selection

The system supports multiple GPT models and an offline mode:

```python
# In controller.py's interactive mode
print("\nChoose GPT Model:")
print("1. GPT-4 (accurate, slower)")
print("2. GPT-3.5 (faster, cheaper)")
print("3. None (offline mode)")

model_choice = input("Choice (1, 2, or 3): ").strip()
if model_choice == '1':
    gpt_model = "gpt-4"
elif model_choice == '2':
    gpt_model = "gpt-3.5-turbo"
elif model_choice == '3':
    gpt_model = "none"
```

### Fallback Mechanism

The system includes fallback summary generation when GPT is unavailable:

```python
def fallback_summary(errors: List[Dict], clusters: Dict[int, List[Dict]], 
                    component_summary: List[Dict] = None, 
                    primary_issue_component: str = "unknown") -> str:
    """
    Generate a basic summary when GPT is not available.
    """
    # Implementation details...
```

## 5. Component Analysis and Identification

### Component Schema

Components are defined in `component_schema.json`:

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
        "(?i)failed to load.*?channel",
        "(?i)null pointer.*?siriusxm",
        "(?i)\\[HTTP\\]",
        "(?i)AppiumDriver",
        "(?i)EspressoDriver",
        "(?i)siriusxm",
        "(?i)com\\.siriusxm",
        "(?i)sxm:s:",
        "(?i)^sxm\\b"
      ]
    },
    // Other components...
  ],
  "dataFlows": [
    {
      "source": "mimosa",
      "target": "soa",
      "description": "Fake test data (direct)",
      "dataType": "test_signals"
    },
    // Other data flows...
  ]
}
```

### Component Identification

The system uses multiple methods to identify components:

```python
# 1. Filename-based identification (direct_component_analyzer.py)
def identify_component_from_filename(filename: str) -> Tuple[str, str]:
    """
    Identify component based on filename pattern using a simplified approach.
    """
    if not filename:
        return 'unknown', 'default'
    
    filename = filename.lower()
    
    # Special cases
    if 'app_debug.log' in filename:
        return 'soa', 'filename_special'
    elif '.har' in filename or '.chlsj' in filename:
        return 'ip_traffic', 'filename_special'
    
    # Standard case: use the base filename without extension
    base_name = os.path.basename(filename)
    component_name = os.path.splitext(base_name)[0]
    return component_name, 'filename'

# 2. Schema-based identification (component_analyzer.py)
def identify_component_from_log_file(self, log_file_path: str) -> str:
    """
    Identify component based on filename pattern using a simplified approach.
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
```

### Component Relationship Analysis

The system builds a graph of component relationships:

```python
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
```

### Component Integration

The system integrates component information with log analysis:

```python
def analyze_logs(self, log_entries: List[Any], errors: List[Any], output_dir: str, test_id: str) -> Dict[str, Any]:
    """
    Perform comprehensive component-aware analysis with simplified component identification.
    """
    results = {
        "test_id": test_id,
        "timestamp": datetime.now().isoformat(),
        "analysis_files": {},
        "metrics": {}
    }
    
    # Step 1: Enrich logs with component information
    log_entries = self.analyzer.enrich_log_entries_with_components(log_entries)
    errors = self.analyzer.enrich_log_entries_with_components(errors)
    
    # Step 2: Generate baseline component relationship diagram
    # Step 3: Analyze component errors
    # Step 4: Generate error propagation visualization
    # Step 5: Generate component error heatmap
    # Step 6: Perform context-aware error clustering
    
    # Implementation details...
    
    return results
```

## 6. Error Clustering and Analysis

### Error Clustering

The system clusters similar errors using TF-IDF and K-means:

```python
def perform_error_clustering(errors: List[Dict], num_clusters: int = None) -> Dict[int, List[Dict]]:
    """
    Cluster errors based on similarity.
    
    Args:
        errors: List of error dictionaries.
        num_clusters: Number of clusters to create (auto-determined if None).
        
    Returns:
        Dictionary mapping cluster IDs to lists of errors.
    """
    # Early return for empty input
    if not errors:
        logging.warning("No errors to cluster")
        return {}

    # Extract error texts for TF-IDF
    texts = []
    for error in errors:
        # Get error text, normalizing format
        text = error.get("text", "")
        if text:
            texts.append(_preprocess_text(text))
        else:
            texts.append("NO_TEXT")

    # Configure vectorizer
    vectorizer = TfidfVectorizer(
        stop_words='english',
        min_df=1,
        max_df=0.9,
        ngram_range=(1, 2)
    )

    # Perform clustering
    # Implementation details...
```

### Dynamic Cluster Sizing

The system determines optimal cluster count based on dataset size:

```python
def _determine_optimal_clusters(matrix, num_errors, user_specified=None):
    """
    Determine the optimal number of clusters based on dataset size.
    Improved to prevent overly imbalanced clustering.
    """
    # If user specified a number, use it as a maximum
    max_clusters = user_specified if user_specified is not None else 8
    
    # Heuristic: sqrt of number of samples, capped between 2 and max_clusters
    k = min(max(2, int(np.sqrt(num_errors/2))), max_clusters)  # Divide by 2 to prefer fewer clusters
    
    # Adjust based on matrix density and uniqueness
    if matrix.shape[1] < 10:  # Very few features
        k = min(k, 3)  # Reduce clusters for low feature count
        
    # CRITICAL FIX: Ensure we don't have more clusters than samples
    k = min(k, matrix.shape[0])
    
    logging.info(f"Context-aware clustering: Using {k} clusters for {num_errors} errors")
    return k
```

### Context-Aware Clustering

The system enhances clustering with component and temporal information:

```python
def _enhance_clusters(self, clusters: Dict[int, List[Dict]], 
                    components: List[str]) -> Dict[int, List[Dict]]:
    """
    Enhance clusters with component and temporal information.
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
    # Step 3: Analyze temporal relationships between clusters
    # Step 4: Determine root cause vs. symptom clusters
    # Step 5: Tag errors with enhanced information
    
    # Implementation details...
    
    return clusters
```

## 7. Report Generation

### Report Manager

The system uses a centralized `ReportManager` to orchestrate report generation:

```python
class ReportManager:
    """Orchestrates the report generation process."""
    
    def __init__(self, config: ReportConfig):
        """Initialize the report manager."""
        self.config = config
        
        # Ensure test_id is properly formatted
        if not config.test_id.startswith("SXM-"):
            config.test_id = f"SXM-{config.test_id}"
        
        # Create output directory structures
        self.base_dir = config.output_dir
        self.json_dir = os.path.join(config.output_dir, "json") 
        self.images_dir = os.path.join(config.output_dir, "supporting_images")
        
        # Create directories
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.json_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        
        # Initialize report generators
        self.json_generator = JsonReportGenerator(self._create_config_for_dir(self.json_dir)) if config.enable_json else None
        self.visualization_generator = VisualizationGenerator(self._create_config_for_dir(self.images_dir)) if config.enable_component_report else None
        self.markdown_generator = MarkdownReportGenerator(config) if config.enable_markdown else None
        self.excel_generator = ExcelReportGenerator(config) if config.enable_excel else None
        self.docx_generator = DocxReportGenerator(config) if config.enable_docx else None
```

### Report Base Classes

The system uses consistent base classes for reports:

```python
class ReportConfig:
    """Configuration settings for report generation."""
    
    def __init__(self, 
                output_dir: str, 
                test_id: str, 
                primary_issue_component: str = "unknown",
                enable_excel: bool = True,
                enable_markdown: bool = True,
                enable_json: bool = True,
                enable_docx: bool = True,
                enable_component_report: bool = True):
        """Initialize report configuration."""
        # Implementation details...

class ReportData:
    """Container for data used in report generation."""
    
    def __init__(self,
                errors: List[Dict],
                summary: str,
                clusters: Dict[int, List[Dict]],
                ocr_data: List[Dict] = None,
                background_text: str = "",
                scenario_text: str = "",
                ymir_flag: bool = False,
                component_analysis: Dict[str, Any] = None,
                component_diagnostic: Dict[str, Any] = None):
        """Initialize report data."""
        # Implementation details...

class ReportGenerator:
    """Base class for report generators."""
    
    def __init__(self, config: ReportConfig):
        """Initialize a report generator."""
        self.config = config
    
    def generate(self, data: ReportData) -> str:
        """Generate a report."""
        raise NotImplementedError("Subclasses must implement generate()")
```

### Component Information Preservation

The system includes a custom JSON encoder to preserve component information:

```python
class ComponentAwareEncoder(DateTimeEncoder):
    """
    Enhanced JSON encoder that carefully preserves component information during serialization.
    This encoder ensures that component fields retain their original values without overriding
    and properly handles nested structures to prevent component information loss.
    """
    
    def __init__(self, *args, primary_issue_component=None, **kwargs):
        """Initialize encoder with optional primary_issue_component reference."""
        super().__init__(*args, **kwargs)
        self.primary_issue_component = primary_issue_component
        self.component_fields = COMPONENT_FIELDS
    
    def default(self, obj):
        """
        Enhanced encoding that preserves component information without modification.
        """
        # Implementation details...
```

## 8. Batch Processing

### Batch Processor

The system includes a batch processor for analyzing multiple tests:

```python
def process_batch(test_ids: List[str], parallel: bool = False) -> Dict[str, Dict]:
    """
    Process multiple test IDs either sequentially or in parallel.
    Returns a dictionary of results keyed by test ID.
    """
    if not test_ids:
        logging.warning("No test IDs provided for batch processing")
        return {}
        
    results = {}
    
    if parallel:
        # Use multiprocessing for parallel execution
        import multiprocessing as mp
        with mp.Pool() as pool:
            result_list = pool.map(process_single_test, test_ids)
            for result in result_list:
                results[result["test_id"]] = result
    else:
        # Process sequentially
        for test_id in test_ids:
            results[test_id] = process_single_test(test_id)
            
    return results

def find_test_folders() -> List[str]:
    """
    Automatically discover all SXM-* folders in the logs directory.
    Returns a list of test IDs (folder names)
    """
    test_ids = []
    log_dir = Config.LOG_BASE_DIR
    
    if not os.path.exists(log_dir):
        logging.warning(f"Logs directory {log_dir} does not exist")
        return []
        
    for item in os.listdir(log_dir):
        folder_path = os.path.join(log_dir, item)
        if os.path.isdir(folder_path) and item.startswith("SXM-"):
            test_ids.append(item)
            
    return sorted(test_ids)
```

### Command-Line Interface

The batch processor supports command-line arguments:

```python
def main():
    """Main function to run the batch processor from command line"""
    parser = argparse.ArgumentParser(description="Log Analyzer Batch Processor")
    parser.add_argument("--tests", nargs="+", help="List of test IDs to process")
    parser.add_argument("--all", action="store_true", help="Process all tests in logs directory")
    parser.add_argument("--parallel", action="store_true", help="Process tests in parallel")
    parser.add_argument("--report", help="Output file for batch report")
    
    args = parser.parse_args()
    
    # Initialize configuration
    Config.setup_logging()
    Config.validate()
    
    # Determine which tests to process
    if args.all:
        test_ids = find_test_folders()
        if not test_ids:
            logging.error("No test folders found in logs directory")
            return
    elif args.tests:
        test_ids = args.tests
    else:
        logging.error("No tests specified. Use --tests or --all")
        parser.print_help()
        return
    
    logging.info(f"Starting batch processing of {len(test_ids)} tests")
    results = process_batch(test_ids, args.parallel)
    
    # Generate and print report
    report = generate_batch_report(results, args.report)
    print("\n" + report)
    
    # Return success if all tests passed
    return all(r.get("status") == "success" for r in results.values())
```

## 9. Environment Setup Instructions

### Development Environment Setup

#### Prerequisites

1. **Python Environment**
   - Python 3.8 or higher
   - Virtual environment recommended

2. **Required System Components**
   - Tesseract OCR (for image processing)
   - Microsoft Office (optional, for DOCX validation)

#### Step-by-step Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/orbit-analyzer.git
   cd orbit-analyzer
   ```

2. **Create and activate virtual environment**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   - Create `.env` file with the following variables:
     ```
     OPENAI_API_KEY=your_key_here
     LOG_BASE_DIR=./logs
     OUTPUT_BASE_DIR=./output
     LOG_LEVEL=INFO
     DEFAULT_MODEL=gpt-3.5-turbo
     ENABLE_OCR=true
     ```

5. **Create directory structure**
   ```bash
   mkdir -p logs output
   ```

### Required Dependencies

The system requires the following Python packages (from Requirements.txt):

```
# Core functionality
openai>=1.0.0
pandas>=1.3.0
numpy>=1.20.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
pillow>=8.0.0
requests>=2.25.0

# OCR
pytesseract>=0.3.8

# Reporting and Document Generation
openpyxl>=3.0.0
python-docx>=0.8.11

# Environment and Configuration
python-dotenv>=1.0.0
keyring>=24.0.0

# Testing and Coverage
coverage>=6.0.0
networkx>=1.0.0

# Optional layout enhancement
pydot>=1.4.2; sys_platform != "win32"  # Optional for better layouts, exclude on Windows
pydot-ng>=2.0.0; sys_platform == "win32"  # Windows-compatible alternative
```

### Running the Application

#### Interactive Mode

```bash
python controller.py
```

This will prompt for:
1. Test ID (e.g., SXM-1234567)
2. Test type (Ymir or Installer)
3. GPT model selection

#### Command Line Mode

```bash
# Run analysis for a specific test
python controller.py SXM-1234567 --model gpt-3.5-turbo --ocr true

# Run batch processing
python batch_processor.py --tests SXM-1234567 SXM-2345678
python batch_processor.py --all --parallel
```

### Common Issues and Troubleshooting

#### API Key Issues
```
Error: OpenAI API key not found. GPT-based analysis will not be available.
```
**Solution**: Set the `OPENAI_API_KEY` environment variable or add it to `.env` file.

#### Directory Permission Issues
```
PermissionError: [Errno 13] Permission denied: 'logs/...'
```
**Solution**: Ensure directories exist and you have write permissions.

#### Excel File Access Issues
```
PermissionError: [Errno 13] Permission denied: '...xlsx'
```
**Solution**: Close any open Excel files before running analysis.

#### Encoding Issues
```
UnicodeEncodeError: 'charmap' codec can't encode character...
```
**Solution**: The system uses UTF-8 logging, but you can explicitly set:
```bash
set PYTHONIOENCODING=utf-8  # Windows
export PYTHONIOENCODING=utf-8  # macOS/Linux
```

#### Visualization Issues

```
MainThread is not in main loop
```
**Solution**: The visualization system now uses the 'Agg' backend for matplotlib which avoids GUI thread issues. If you encounter this error, ensure your code is using `_configure_matplotlib_backend()` before creating any visualizations.

```python
# Configure matplotlib backend before creating any figures
_configure_matplotlib_backend()

# Now create visualizations safely
fig = plt.figure(figsize=(10, 6))
```

#### Memory Leaks

```
MemoryError or application slowdown after many visualizations
```
**Solution**: Always use the `_save_figure_with_cleanup()` function to ensure figures are properly closed:

```python
try:
    # Create and configure figure
    fig = plt.figure(figsize=(10, 6))
    # ... plotting code ...
    
    # Save with cleanup
    return _save_figure_with_cleanup(fig, image_path)
finally:
    # Ensure cleanup on error
    plt.close('all')
```

## 10. Conclusion

The Orbit Analyzer provides a comprehensive environment for analyzing test logs, identifying component relationships, and generating detailed reports. The configuration system allows for flexible deployment in various environments while maintaining consistent behavior.

Key capabilities include:
- Standardized path handling and directory structure
- Secure API key management with multiple fallback options
- Component-aware analysis with relationship mapping
- Advanced error clustering with dynamic sizing
- Thread-safe visualization generation with multi-level fallbacks
- Comprehensive report generation in multiple formats
- Parallel batch processing for efficient analysis

This document covers the essential configuration aspects of the system, providing guidance for developers, testers, and system administrators working with the Orbit Analyzer.
