"""
tests/utils/test_validator.py - Test for the path validator utility
"""

import os
import sys

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.path_validator import print_validation_results

# Define output base directory - adjust this path to match your environment
# This should point to the "output" folder in your Orbit repository
OUTPUT_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../output'))

# Try to import Config as a fallback
try:
    from config import Config
    if not os.path.exists(OUTPUT_BASE_DIR):
        OUTPUT_BASE_DIR = Config.OUTPUT_BASE_DIR
except ImportError:
    # Keep using the hardcoded path if Config is not available
    pass

# Take a test ID from command line or use a default
test_id = sys.argv[1] if len(sys.argv) > 1 else "SXM-1782734"

# Normalize test ID
if not test_id.startswith("SXM-"):
    test_id = f"SXM-{test_id}"

# Validate the output directory
output_dir = os.path.join(OUTPUT_BASE_DIR, test_id)
print(f"Looking for output directory: {output_dir}")

if os.path.exists(output_dir):
    print_validation_results(output_dir, test_id)
else:
    print(f"Output directory not found: {output_dir}")
    print("Please run the pipeline first with:")
    print(f"  python controller.py  (and enter {test_id} when prompted)")