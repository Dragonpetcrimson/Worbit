# step_aware_analyzer.py
import os
import sys  # Added sys import for checking test environment
import logging
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from gherkin_log_correlator import GherkinParser, LogEntry

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
    except Exception as e:
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
    except Exception as e:
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


