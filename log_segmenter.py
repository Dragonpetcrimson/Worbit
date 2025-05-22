# log_segmenter.py
import os
from typing import List, Tuple

SUPPORTED_LOG_EXTENSIONS = ('.log', '.txt', '.chlsj', '.har') 
SUPPORTED_IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

def collect_log_files(base_dir: str) -> List[str]:
    """
    Recursively collects all log files from a directory tree.
    """
    log_files = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(SUPPORTED_LOG_EXTENSIONS):
                log_files.append(os.path.join(root, file))
    return log_files

def collect_image_files(base_dir: str) -> List[str]:
    """
    Recursively collects all image files from a directory tree.
    """
    image_files = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(SUPPORTED_IMAGE_EXTENSIONS):
                image_files.append(os.path.join(root, file))
    return image_files

def collect_all_supported_files(base_dir: str) -> Tuple[List[str], List[str]]:
    """
    Collects both logs and images for processing.
    Returns (logs, images)
    """
    return collect_log_files(base_dir), collect_image_files(base_dir)
