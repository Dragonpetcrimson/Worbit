"""
Components Package - Package initialization

This package provides functionality for component identification, tracking,
and relationship analysis throughout the Orbit Analyzer system.
"""

# This file makes the directory a Python package
# Import key modules to make them directly available from the package
from components.component_model import (
    ComponentInfo, 
    ComponentRegistry,
    get_component_registry,
    create_component_info
)

from components.component_utils import (
    extract_component_fields,
    apply_component_fields,
    preserve_component_fields,
    verify_component_preservation,
    COMPONENT_FIELDS
)

# Version information
__version__ = '1.0.0'