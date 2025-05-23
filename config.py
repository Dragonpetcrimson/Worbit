# config.py
import os
import logging
from datetime import datetime
import sys
import io

# Configure logging with proper UTF-8 encoding
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

class Config:
    """Configuration settings for the log analyzer.
    
    This class provides centralized configuration settings, including:
    - Directory paths for logs and output
    - API settings
    - Feature flags for various functionalities
    - Visualization settings
    
    When ENABLE_DIAGNOSTIC_CHECKS is true, run pre-flight diagnostics before generating reports.
    """
    
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
    # Defaults are set to provide the most useful visualizations while maintaining backwards compatibility
    ENABLE_CLUSTER_TIMELINE = os.getenv("ENABLE_CLUSTER_TIMELINE", "False").lower() in ("true", "1", "yes")
    ENABLE_COMPONENT_DISTRIBUTION = os.getenv("ENABLE_COMPONENT_DISTRIBUTION", "True").lower() in ("true", "1", "yes")
    # Disable component relationship diagrams by default
    ENABLE_COMPONENT_RELATIONSHIPS = os.getenv("ENABLE_COMPONENT_RELATIONSHIPS", "False").lower() in ("true", "1", "yes")
    ENABLE_ERROR_PROPAGATION = os.getenv("ENABLE_ERROR_PROPAGATION", "False").lower() in ("true", "1", "yes")
    
    # Step report and timeline visualizations
    # This flag controls whether timeline images are generated for step reports
    # Default to True to ensure timeline images are always generated
    ENABLE_STEP_REPORT_IMAGES = os.getenv("ENABLE_STEP_REPORT_IMAGES", "True").lower() in ("true", "1", "yes")
    
    # Component report visualizations
    ENABLE_COMPONENT_REPORT_IMAGES = os.getenv("ENABLE_COMPONENT_REPORT_IMAGES", "True").lower() in ("true", "1", "yes")
    
    # Visualization placeholder settings
    # When enabled, placeholder visualizations will be generated for invalid/empty data
    # Default to False so placeholders are only created when explicitly enabled
    ENABLE_VISUALIZATION_PLACEHOLDERS = os.getenv("ENABLE_VISUALIZATION_PLACEHOLDERS", "False").lower() in ("true", "1", "yes")
    
    # Diagnostic check enablement flag
    # When enabled, run pre-flight diagnostics before generating reports
    ENABLE_DIAGNOSTIC_CHECKS = os.getenv("ENABLE_DIAGNOSTIC_CHECKS", "False").lower() in ("true", "1", "yes")
    
    # Visualization settings
    VISUALIZATION_DPI = int(os.getenv("VISUALIZATION_DPI", "100"))
    VISUALIZATION_FORMAT = os.getenv("VISUALIZATION_FORMAT", "png")
    
    @classmethod
    def setup_logging(cls):
        """
        Set up logging configuration with proper UTF-8 handling.
        Ensures all log messages use UTF-8 encoding to handle special characters.
        Prevents multiple logging initialization.
        """
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
        
        # Validate visualization settings
        cls.validate_visualization_settings()
    
    @classmethod
    def validate_visualization_settings(cls):
        """Validate visualization-specific settings."""
        # Validate DPI setting
        try:
            dpi = int(cls.VISUALIZATION_DPI)
            if dpi < 50 or dpi > 300:
                logging.warning(f"Unusual VISUALIZATION_DPI value: {dpi}. Should be between 50-300. Using default of 100.")
                cls.VISUALIZATION_DPI = 100
        except (ValueError, TypeError):
            logging.warning(f"Invalid VISUALIZATION_DPI value: {cls.VISUALIZATION_DPI}. Using default of 100.")
            cls.VISUALIZATION_DPI = 100
        
        # Validate format setting
        valid_formats = ["png", "svg", "jpg", "pdf"]
        if cls.VISUALIZATION_FORMAT.lower() not in valid_formats:
            logging.warning(f"Unsupported VISUALIZATION_FORMAT: {cls.VISUALIZATION_FORMAT}. Using default of png.")
            cls.VISUALIZATION_FORMAT = "png"
        
        # Ensure consistency of feature flags
        # If step report images are disabled but placeholders are enabled, log a warning
        if not cls.ENABLE_STEP_REPORT_IMAGES and cls.ENABLE_VISUALIZATION_PLACEHOLDERS:
            logging.info("ENABLE_STEP_REPORT_IMAGES is disabled, but ENABLE_VISUALIZATION_PLACEHOLDERS is enabled. " +
                        "Placeholders will be generated for step report visualizations.")
    
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
            plt.rcParams['figure.dpi'] = cls.VISUALIZATION_DPI  # Use configured DPI
            
            # Additional settings for better visualization
            plt.rcParams['axes.grid'] = True  # Enable grid by default
            plt.rcParams['savefig.bbox'] = 'tight'  # Tight bounding box for saved figures
            plt.rcParams['savefig.dpi'] = cls.VISUALIZATION_DPI  # Consistent DPI for saved figures
            
            logging.info(f"Matplotlib configured with Agg backend for thread safety (DPI: {cls.VISUALIZATION_DPI}, Format: {cls.VISUALIZATION_FORMAT})")
            return True
        except Exception as e:
            logging.error(f"Error configuring matplotlib: {str(e)}")
            return False
    
    @classmethod
    def is_visualization_enabled(cls, visualization_type):
        """
        Check if a specific visualization type is enabled.
        
        Args:
            visualization_type: Type of visualization to check (e.g., 'step_report', 'component_distribution')
            
        Returns:
            Boolean indicating if the visualization is enabled
        """
        visualization_map = {
            'cluster_timeline': cls.ENABLE_CLUSTER_TIMELINE,
            'component_distribution': cls.ENABLE_COMPONENT_DISTRIBUTION,
            'component_relationships': cls.ENABLE_COMPONENT_RELATIONSHIPS,
            'error_propagation': cls.ENABLE_ERROR_PROPAGATION,
            'step_report': cls.ENABLE_STEP_REPORT_IMAGES,
            'component_report': cls.ENABLE_COMPONENT_REPORT_IMAGES,
            'timeline': cls.ENABLE_STEP_REPORT_IMAGES  # Timeline is part of step report
        }
        
        # Default to True for unknown visualization types to maintain backward compatibility
        return visualization_map.get(visualization_type.lower(), True)
    
    @classmethod
    def should_generate_placeholder(cls, visualization_type=None):
        """
        Check if a placeholder should be generated for a failed visualization.
        
        Args:
            visualization_type: Optional type of visualization to check specific rules
            
        Returns:
            Boolean indicating if a placeholder should be generated
        """
        # Base check on the global setting
        should_generate = cls.ENABLE_VISUALIZATION_PLACEHOLDERS
        
        # Add type-specific logic if needed in the future
        if visualization_type == 'timeline' and not cls.ENABLE_STEP_REPORT_IMAGES:
            # Don't generate placeholders for timeline if step report images are disabled
            should_generate = False
        
        return should_generate

# Function to determine optimal cluster count based on error count
def determine_optimal_cluster_count(error_count):
    """
    Calculate the optimal number of clusters based on the dataset size.
    
    Args:
        error_count: Number of errors
        
    Returns:
        Recommended number of clusters based on a heuristic formula
    """
    if error_count <= 5:
        return min(2, error_count)  # Very small datasets
    elif error_count <= 20:
        return min(3, error_count)  # Small datasets
    elif error_count <= 50:
        return min(5, error_count)  # Medium datasets
    else:
        return min(8, error_count)  # Large datasets

# Initialize logging with UTF-8 support when this module is imported
# This ensures all logging throughout the application uses the correct encoding
Config.setup_logging()