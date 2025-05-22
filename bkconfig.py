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
    
    # Visualization enablement flags - Existing flags preserved
    ENABLE_CLUSTER_TIMELINE = False
    ENABLE_COMPONENT_DISTRIBUTION = True
    ENABLE_COMPONENT_RELATIONSHIPS = True
    ENABLE_ERROR_PROPAGATION = False
    ENABLE_STEP_REPORT_IMAGES = True
    ENABLE_COMPONENT_REPORT_IMAGES = True
    
    # New visualization placeholder setting (default: disabled)
    # When disabled, no placeholder visualizations will be generated for invalid/empty data
    ENABLE_VISUALIZATION_PLACEHOLDERS = False
    
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