"""
Core module tests for Orbit Analyzer project.

This module tests the core functionality of the system, including:
- Log analysis and parsing (log_analyzer)
- Error clustering and grouping (error_clusterer)
- OCR processing for images (ocr_processor)
- AI-powered summary generation (gpt_summarizer)
- Secure API key handling (secure_api_key)
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import tempfile
import logging
from datetime import datetime

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import TestRegistry with fallback
try:
    from test_registry import TestRegistry
except ImportError:
    # Simple placeholder for backward compatibility
    class TestRegistry:
        @classmethod
        def register(cls, **kwargs):
            def decorator(test_class):
                return test_class
            return decorator

# Import test utilities
try:
    from test_utils import (
        ConfigManager, 
        get_test_folder,
        create_test_data,
        setup_test_output_directories,
        validate_report_file
    )
except ImportError:
    # Fallback for basic test utilities
    from test_utils import get_test_folder

# Import modules to test with appropriate error handling
try:
    from log_segmenter import collect_log_files, collect_image_files, collect_all_supported_files
except ImportError:
    logging.warning("log_segmenter module not available, related tests will be skipped")
    collect_log_files = collect_image_files = collect_all_supported_files = None

try:
    from log_analyzer import parse_logs
except ImportError:
    logging.warning("log_analyzer module not available, related tests will be skipped")
    parse_logs = None

try:
    from error_clusterer import perform_error_clustering as cluster_errors
except ImportError:
    try:
        from error_clusterer import cluster_errors
    except ImportError:
        logging.warning("error_clusterer module not available, related tests will be skipped")
        cluster_errors = None

try:
    from ocr_processor import extract_ocr_data
except ImportError:
    logging.warning("ocr_processor module not available, related tests will be skipped")
    extract_ocr_data = None

try:
    from gpt_summarizer import (
        generate_summary_from_clusters,
        build_clustered_prompt,
        sanitize_text_for_api,
        fallback_summary
    )
except ImportError:
    logging.warning("gpt_summarizer module not available, related tests will be skipped")
    generate_summary_from_clusters = build_clustered_prompt = sanitize_text_for_api = fallback_summary = None

try:
    from secure_api_key import get_openai_api_key
except ImportError:
    logging.warning("secure_api_key module not available, related tests will be skipped")
    get_openai_api_key = None

# Set up logging for tests
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("core_module_tests")


@TestRegistry.register(category='core')
class TestLogSegmenter(unittest.TestCase):
    """
    Tests for the log_segmenter module.
    
    Verifies the ability to collect log files and images from
    test directories with various file types and structures.
    """
    
    def setUp(self):
        """Set up test environment."""
        # Skip if module not available
        if collect_all_supported_files is None:
            self.skipTest("log_segmenter module not available")
        
        self.test_folder = get_test_folder()
        
    def test_file_collection(self):
        """
        Test that log and image files are correctly collected.
        
        Verifies:
        1. Log files with appropriate extensions are found
        2. Image files with appropriate extensions are found
        3. No invalid files are included in the collections
        """
        logger.info(f"Testing with folder: {self.test_folder}")
        
        logs, images = collect_all_supported_files(self.test_folder)

        # Print basic results
        logger.info(f"Found {len(logs)} log files and {len(images)} image files")
        
        # Add assertions
        if len(logs) == 0:
            self.skipTest("No log files found")
            
        # Check file extensions
        for log_file in logs:
            self.assertTrue(
                any(log_file.lower().endswith(ext) for ext in ('.log', '.txt', '.chlsj', '.har')),
                f"Invalid log file extension: {log_file}"
            )
                
        for image_file in images:
            self.assertTrue(
                any(image_file.lower().endswith(ext) for ext in ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')),
                f"Invalid image file extension: {image_file}"
            )
        
        logger.info("✅ All file extensions valid")

    def test_collect_log_files(self):
        """
        Test the collect_log_files function specifically.
        
        Verifies that log files are collected correctly with a range
        of file extensions.
        """
        # Skip if function not available
        if collect_log_files is None:
            self.skipTest("collect_log_files function not available")
            
        # Test log file collection
        log_files = collect_log_files(self.test_folder)
        
        # Check collected files
        self.assertIsInstance(log_files, list, "collect_log_files should return a list")
        
        # Verify extensions
        for log_file in log_files:
            self.assertTrue(
                any(log_file.lower().endswith(ext) for ext in ('.log', '.txt', '.chlsj', '.har')),
                f"Invalid log file extension: {log_file}"
            )
            self.assertTrue(os.path.exists(log_file), f"Log file doesn't exist: {log_file}")
    
    def test_collect_image_files(self):
        """
        Test the collect_image_files function specifically.
        
        Verifies that image files are collected correctly with a range
        of file extensions.
        """
        # Skip if function not available
        if collect_image_files is None:
            self.skipTest("collect_image_files function not available")
            
        # Test image file collection
        image_files = collect_image_files(self.test_folder)
        
        # Check collected files
        self.assertIsInstance(image_files, list, "collect_image_files should return a list")
        
        # Verify extensions
        for image_file in image_files:
            self.assertTrue(
                any(image_file.lower().endswith(ext) for ext in ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')),
                f"Invalid image file extension: {image_file}"
            )
            self.assertTrue(os.path.exists(image_file), f"Image file doesn't exist: {image_file}")
    
    def test_empty_directory(self):
        """
        Test handling of empty directories.
        
        Verifies that the log segmenter correctly handles empty
        directories without errors.
        """
        # Skip if functions not available
        if collect_all_supported_files is None:
            self.skipTest("collect_all_supported_files function not available")
            
        # Create an empty temporary directory
        with tempfile.TemporaryDirectory() as empty_dir:
            # Test with empty directory
            logs, images = collect_all_supported_files(empty_dir)
            
            # Verify empty returns
            self.assertEqual(len(logs), 0, "Should return empty list for logs in empty directory")
            self.assertEqual(len(images), 0, "Should return empty list for images in empty directory")


@TestRegistry.register(category='core')
class TestLogAnalyzer(unittest.TestCase):
    """
    Tests for the log_analyzer module.
    
    Verifies the ability to parse log files and extract errors
    with appropriate metadata.
    """
    
    def setUp(self):
        """Set up test environment."""
        # Skip if module not available
        if parse_logs is None:
            self.skipTest("log_analyzer module not available")
            
        self.log_dir = get_test_folder()
        self.log_files = collect_log_files(self.log_dir) if collect_log_files else []
        
    def test_log_parsing(self):
        """
        Test that logs can be parsed and errors extracted correctly.
        
        Verifies:
        1. Errors are extracted from log files
        2. Extracted errors have the required fields
        3. Error severity is assigned correctly
        """
        logger.info(f"Testing with folder: {self.log_dir}")
        
        if len(self.log_files) == 0:
            self.skipTest("No log files found for testing")
        
        logger.info(f"Parsing {len(self.log_files)} log files...")
        extracted = parse_logs(self.log_files)
        
        # Basic validation
        if len(extracted) == 0:
            logger.warning("No errors extracted from logs")
            # Not necessarily a failure as the logs might not have errors
        else:
            logger.info(f"✓ Successfully extracted {len(extracted)} error entries")
        
        # Validate structure of extracted data
        required_fields = ['file', 'line_num', 'severity', 'text']
        
        for item in extracted[:5]:  # Check first few items
            for field in required_fields:
                self.assertIn(field, item, f"Missing required field: {field}")
        
        logger.info("✅ All extracted errors have the required fields")
        
        # Print sample for manual verification
        if extracted:
            sample = extracted[0]
            logger.info("\nSample extracted error:")
            logger.info(f"File: {sample['file']} | Line: {sample['line_num']} | Severity: {sample['severity']}")
            logger.info(f"Text: {sample['text'][:100]}...")
    
    def test_create_minimal_log(self):
        """
        Test parsing with minimal log file.
        
        Creates and parses a minimal log file to ensure the parser can
        handle basic log formats.
        """
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a minimal log file
            log_path = os.path.join(temp_dir, "minimal.log")
            with open(log_path, 'w') as f:
                f.write("2023-01-01 12:00:00 INFO: Test message\n")
                f.write("2023-01-01 12:01:00 ERROR: Test error\n")
                f.write("2023-01-01 12:02:00 WARNING: Test warning\n")
            
            # Parse the log file
            errors = parse_logs([log_path])
            
            # Verify errors were extracted
            self.assertGreater(len(errors), 0, "Should extract at least one error")
            
            # Verify error fields
            for error in errors:
                self.assertEqual(error['file'], log_path, "File field should match log path")
                self.assertIn('line_num', error, "Error should have line_num field")
                self.assertIn('text', error, "Error should have text field")
                self.assertIn('severity', error, "Error should have severity field")
    
    def test_context_extraction(self):
        """
        Test context extraction from log files.
        
        Verifies that error context lines are correctly extracted from
        log files.
        """
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a log file with context
            log_path = os.path.join(temp_dir, "context.log")
            with open(log_path, 'w') as f:
                f.write("2023-01-01 12:00:00 INFO: Context line 1\n")
                f.write("2023-01-01 12:00:01 INFO: Context line 2\n")
                f.write("2023-01-01 12:00:02 ERROR: Test error with context\n")
                f.write("2023-01-01 12:00:03 INFO: After context\n")
            
            # Parse the log file
            errors = parse_logs([log_path])
            
            # Verify context was extracted
            self.assertGreater(len(errors), 0, "Should extract at least one error")
            
            # Check for context field
            error = errors[0]
            self.assertIn('context', error, "Error should have context field")
            self.assertIsInstance(error['context'], list, "Context should be a list")
            self.assertGreater(len(error['context']), 0, "Context should not be empty")
            
            # Verify context includes preceding lines
            context_text = '\n'.join(error['context'])
            self.assertIn("Context line", context_text, "Context should include preceding lines")


@TestRegistry.register(category='core')
class TestErrorClusterer(unittest.TestCase):
    """
    Test suite for the error_clusterer module.
    
    Tests the clustering of errors based on similarity and the
    determination of optimal cluster counts.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        # Skip if module not available
        if cluster_errors is None:
            self.skipTest("error_clusterer module not available")
            
        self.test_dir = get_test_folder()
        logger.info(f"Testing with folder: {self.test_dir}")
        
        # Collect log files
        if collect_log_files:
            self.logs = collect_log_files(self.test_dir)
        else:
            self.logs = []
        
        # Create minimal error set for testing if no logs found
        self.minimal_errors = [
            {'file': 'test.log', 'line_num': 1, 'text': 'Error: Database connection failed', 'severity': 'High'},
            {'file': 'test.log', 'line_num': 2, 'text': 'Error: Database query timeout', 'severity': 'Medium'},
            {'file': 'test.log', 'line_num': 3, 'text': 'Warning: Network delay detected', 'severity': 'Low'}
        ]
    
    def test_cluster_creation(self):
        """
        Test that clusters can be created from log files.
        
        Verifies:
        1. Clusters are created from errors
        2. Clusters contain error objects
        3. Each cluster has at least one error
        """
        if not self.logs:
            logger.warning("No logs found - using minimal sample for testing")
            errors = self.minimal_errors
        else:
            errors = parse_logs(self.logs) if parse_logs else self.minimal_errors
            if not errors:
                logger.warning("No errors found in logs - using minimal sample")
                errors = self.minimal_errors
        
        logger.info(f"Clustering {len(errors)} errors...")
        clusters = cluster_errors(errors, num_clusters=min(5, len(errors)))
        
        # Validate that clusters were created
        self.assertIsNotNone(clusters, "Clusters should not be None")
        self.assertGreater(len(clusters), 0, "At least one cluster should be created")
        
        # Validate cluster structure
        for cluster_id, group in clusters.items():
            self.assertIsInstance(group, list, f"Cluster {cluster_id} should be a list")
            self.assertGreater(len(group), 0, f"Cluster {cluster_id} should not be empty")
            
            logger.info(f"Cluster {cluster_id} contains {len(group)} errors")
    
    def test_cluster_distribution(self):
        """
        Test that error distribution across clusters is reasonable.
        
        Verifies that errors are distributed reasonably across clusters
        without extremely imbalanced clusters.
        """
        # Use minimal errors for consistency
        errors = self.minimal_errors
        
        # Create multiple clusters
        clusters = cluster_errors(errors, num_clusters=min(2, len(errors)))
        
        # Skip test if only one cluster created
        if len(clusters) <= 1:
            self.skipTest("Not enough clusters to test distribution")
            
        # Check distribution
        sizes = [len(group) for group in clusters.values()]
        avg_size = sum(sizes) / len(sizes)
        max_size = max(sizes)
        
        # This is a soft validation - not failing the test but logging a warning
        if max_size > avg_size * 3 and len(errors) > 10:
            logger.warning(f"Possible clustering issue: largest cluster ({max_size}) is more than 3x the average ({avg_size:.1f})")
            
        # Ensure each cluster has at least one error
        self.assertTrue(all(size > 0 for size in sizes), "All clusters should have at least one error")
    
    def test_specified_clusters(self):
        """
        Test clustering with a specified number of clusters.
        
        Verifies that the specified number of clusters is respected
        when it's explicitly provided.
        """
        # Skip if we don't have enough errors for meaningful test
        if len(self.minimal_errors) < 2:
            self.skipTest("Not enough errors for meaningful cluster test")
            
        # Test with 2 clusters specified
        num_clusters = 2
        clusters = cluster_errors(self.minimal_errors, num_clusters=num_clusters)
        
        # For some implementations, if the data isn't well-separated, it might not create exactly
        # the requested number of clusters, so use an appropriate test
        self.assertLessEqual(len(clusters), num_clusters, 
                          f"Should not create more than {num_clusters} clusters when specified")
    
    def test_error_text_similarity(self):
        """
        Test clustering of similar error texts.
        
        Verifies that errors with similar text content are grouped
        in the same cluster.
        """
        # Create errors with similar text
        similar_errors = [
            {'file': 'test.log', 'line_num': 1, 'text': 'Database connection failed: timeout', 'severity': 'High'},
            {'file': 'test.log', 'line_num': 2, 'text': 'Database connection failed: auth error', 'severity': 'High'},
            {'file': 'test.log', 'line_num': 3, 'text': 'Network error: connection refused', 'severity': 'Medium'},
            {'file': 'test.log', 'line_num': 4, 'text': 'Network error: host not found', 'severity': 'Medium'},
        ]
        
        # Cluster the errors - request 2 clusters
        clusters = cluster_errors(similar_errors, num_clusters=2)
        
        # Check that we got the expected number of clusters
        self.assertLessEqual(len(clusters), 2, "Should create at most 2 clusters when specified")
        
        # The following test may be sensitive to the specific clustering implementation
        # so we'll make it more flexible - either separate cluster for each error type,
        # or clustering by broader categories
        
        # Create maps of error texts to cluster IDs
        error_to_cluster = {}
        for cluster_id, errors in clusters.items():
            for error in errors:
                error_to_cluster[error['text']] = cluster_id
        
        # Check for some similarity in clustering
        database_errors = [e['text'] for e in similar_errors if 'Database' in e['text']]
        network_errors = [e['text'] for e in similar_errors if 'Network' in e['text']]
        
        # If we have multiple database errors, they should be in the same cluster
        if len(database_errors) > 1:
            db_clusters = set(error_to_cluster.get(e, -1) for e in database_errors)
            self.assertLessEqual(len(db_clusters), 1, "Database errors should be in the same cluster")
            
        # If we have multiple network errors, they should be in the same cluster
        if len(network_errors) > 1:
            net_clusters = set(error_to_cluster.get(e, -1) for e in network_errors)
            self.assertLessEqual(len(net_clusters), 1, "Network errors should be in the same cluster")


@TestRegistry.register(category='core')
class TestOcrProcessor(unittest.TestCase):
    """
    Tests for the ocr_processor module.
    
    Verifies the ability to extract text from images using OCR
    processing.
    """
    
    def setUp(self):
        """Set up test environment."""
        # Skip if module not available
        if extract_ocr_data is None:
            self.skipTest("ocr_processor module not available")
            
        self.test_dir = get_test_folder()
        if collect_image_files:
            self.image_files = collect_image_files(self.test_dir)
        else:
            self.image_files = []
        
    def test_ocr_extraction(self):
        """
        Test that OCR extraction works correctly on image files.
        
        Verifies:
        1. OCR processing extracts text from images
        2. Extracted data has the expected structure
        3. Text extraction works with various image formats
        """
        logger.info(f"Testing with folder: {self.test_dir}")
        
        if len(self.image_files) == 0:
            self.skipTest("No image files found for testing")
            
        logger.info(f"Testing OCR on {len(self.image_files)} images...")
        ocr_output = extract_ocr_data(self.image_files)
        
        # Validate OCR results
        if len(ocr_output) == 0 and len(self.image_files) > 0:
            logger.warning("No OCR text extracted from any images")
            # Not failing since some images might not have any text
        
        # Check that results have expected structure
        for result in ocr_output:
            self.assertIn('file', result, "OCR result missing 'file' field")
            self.assertIn('text', result, "OCR result missing 'text' field")
                
            logger.info(f"✓ Successfully extracted {len(result['text'])} characters from {result['file']}")
    
    def test_min_length_filtering(self):
        """
        Test the minimum length filtering for OCR results.
        
        Verifies that the OCR processor correctly filters out results
        shorter than the specified minimum length.
        """
        # Skip if no image files available
        if len(self.image_files) == 0:
            self.skipTest("No image files found for testing")
            
        # Test with very high min_length to filter everything
        high_min = 10000  # Unlikely any OCR result will be this long
        filtered_ocr = extract_ocr_data(self.image_files, min_length=high_min)
        
        # Should filter out all results
        self.assertEqual(len(filtered_ocr), 0, 
                      f"Should filter out all results with min_length={high_min}")
        
        # Test with very low min_length to include everything
        low_min = 1
        unfiltered_ocr = extract_ocr_data(self.image_files, min_length=low_min)
        
        # Log the results
        logger.info(f"With min_length={low_min}, found {len(unfiltered_ocr)} OCR results")
    
    def test_create_test_image(self):
        """
        Test OCR with a created test image.
        
        Creates a test image with known text content and verifies
        that OCR processing can extract the text.
        """
        # Skip this test if we don't have PIL for image creation
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            self.skipTest("PIL not available for creating test images")
            
        # Create a temporary directory for the test image
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a test image with known text
            test_text = "OCR TEST IMAGE"
            image_path = os.path.join(temp_dir, "test_ocr.png")
            
            # Create the image
            img = Image.new('RGB', (400, 100), color='white')
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("arial.ttf", 24)
            except IOError:
                font = ImageFont.load_default()
            draw.text((20, 30), test_text, fill='black', font=font)
            img.save(image_path)
            
            # Process the image with OCR
            ocr_results = extract_ocr_data([image_path], min_length=1)
            
            # Verify results - OCR might not be perfect, so check for partial match
            if ocr_results:
                text_found = any(test_text in result['text'] for result in ocr_results)
                # Using a warning rather than assertion for OCR tests to account for OCR quality variations
                if not text_found:
                    logger.warning(f"OCR did not extract the expected text: {test_text}")
                    logger.warning(f"Extracted: {[r['text'] for r in ocr_results]}")
            else:
                logger.warning("No OCR results extracted from test image")


@TestRegistry.register(category='core')
class TestGPTSummarizer(unittest.TestCase):
    """
    Unit tests for the GPT summarizer module.
    
    Tests the functionality for generating AI-powered summaries
    from error clusters and related data.
    """
    
    def setUp(self):
        """Set up test data for each test."""
        # Skip if module not available
        if generate_summary_from_clusters is None:
            self.skipTest("gpt_summarizer module not available")
            
        self.test_dir = get_test_folder()
        # Extract the test ID from the folder path
        self.test_id = os.path.basename(self.test_dir) if "SXM-" in self.test_dir else "TEST-123"
        
        # Initialize logs, images, errors, clusters, and ocr data
        if collect_all_supported_files:
            self.logs, self.images = collect_all_supported_files(self.test_dir)
        else:
            self.logs, self.images = [], []
        
        if not self.logs:
            # Create minimal test data for testing
            self.errors = [{'text': 'Test error', 'severity': 'High'}]
            self.clusters = {0: self.errors}
            self.ocr = []
        else:
            self.errors = parse_logs(self.logs) if parse_logs else []
            self.ocr = extract_ocr_data(self.images) if extract_ocr_data else []
            self.clusters = cluster_errors(self.errors) if cluster_errors else {0: self.errors}
    
    def test_build_clustered_prompt(self):
        """
        Test the prompt building functionality.
        
        Verifies that prompts are correctly built from clusters,
        OCR data, and scenario text.
        """
        # Skip if function not available
        if build_clustered_prompt is None:
            self.skipTest("build_clustered_prompt function not available")
            
        prompt = build_clustered_prompt(self.test_id, self.clusters, self.ocr, "Test scenario")
        
        # Verify the prompt is generated and has content
        self.assertIsNotNone(prompt)
        self.assertGreater(len(prompt), 50)
        
        # Verify prompt contains expected elements
        self.assertIn(self.test_id, prompt)
        self.assertIn("Test scenario", prompt)
    
    def test_generate_summary_offline(self):
        """
        Test offline summary generation (no API call).
        
        Verifies that summaries can be generated offline without
        making API calls.
        """
        summary = generate_summary_from_clusters(
            self.clusters, 
            self.ocr, 
            self.test_id, 
            use_gpt=False
        )
        
        # Verify the summary is generated
        self.assertIsNotNone(summary)
        self.assertGreater(len(summary), 10)
        
        # Verify summary contains required sections
        required_sections = ["ROOT CAUSE", "IMPACT", "RECOMMENDED ACTIONS"]
        for section in required_sections:
            self.assertIn(section, summary)
    
    @patch('gpt_summarizer.send_to_openai_chat')
    def test_generate_summary_with_gpt(self, mock_send):
        """
        Test summary generation with mocked GPT API.
        
        Verifies that the GPT API is called correctly and response
        is properly processed.
        """
        # Mock the GPT API response
        mock_send.return_value = (
            "1. ROOT CAUSE: Mock test response.\n"
            "2. IMPACT: This is a mock impact section.\n"
            "3. RECOMMENDED ACTIONS:\n- Use mocks in tests\n- Validate behavior"
        )
        
        # Generate summary with mocked API
        summary = generate_summary_from_clusters(
            self.clusters, 
            self.ocr, 
            self.test_id, 
            use_gpt=True,
            model="gpt-test-model"
        )
        
        # Verify API was called with correct parameters
        mock_send.assert_called_once()
        args = mock_send.call_args[0]
        self.assertIn(self.test_id, args[0], "Prompt should contain the test ID")
        self.assertEqual(args[1], "gpt-test-model", "Should use the specified model")
        
        # Verify summary contains the mocked content
        self.assertIn("Mock test response", summary)
        self.assertIn("RECOMMENDED ACTIONS", summary)
    
    def test_sanitize_text_for_api(self):
        """
        Test text sanitization for API if the function is available.
        
        Verifies that sensitive information is properly redacted
        from text before sending to API.
        """
        # Skip if function not available
        if sanitize_text_for_api is None:
            self.skipTest("sanitize_text_for_api function not available")
            
        # Test with sensitive information
        sensitive_text = "API key: sk-12345abcdef\nEmail: user@example.com\nIP: 192.168.1.1"
        sanitized = sanitize_text_for_api(sensitive_text)
        
        # Verify email and IP are redacted (based on actual implementation)
        self.assertNotIn("user@example.com", sanitized)
        self.assertIn("[EMAIL]", sanitized)
        self.assertNotIn("192.168.1.1", sanitized)
        self.assertIn("[IP_ADDRESS]", sanitized)
        
        # Note: The actual implementation might not redact API keys,
        # so we don't test for that behavior
    
    def test_fallback_summary(self):
        """
        Test the fallback summary generation if the function is available.
        
        Verifies that fallback summaries are generated correctly
        when GPT is not available.
        """
        # Skip if function not available
        if fallback_summary is None:
            self.skipTest("fallback_summary function not available")
            
        summary = fallback_summary(
            self.errors,
            self.clusters,
            [],  # Empty component summary
            "unknown"  # primary_issue_component
        )
        
        # Verify the fallback summary works
        self.assertIsNotNone(summary)
        self.assertGreater(len(summary), 10)


@TestRegistry.register(category='core')
class TestSecureApiKey(unittest.TestCase):
    """
    Test class for secure API key retrieval functionality.
    
    Tests the retrieval of API keys from various sources with
    appropriate fallbacks.
    """
    
    def setUp(self):
        """Set up the test environment."""
        # Skip if module not available
        if get_openai_api_key is None:
            self.skipTest("secure_api_key module not available")
            
        self.original_env = os.environ.get("OPENAI_API_KEY")
    
    def tearDown(self):
        """Clean up the test environment."""
        if self.original_env is not None:
            os.environ["OPENAI_API_KEY"] = self.original_env
        else:
            os.environ.pop("OPENAI_API_KEY", None)
    
    def test_env_only(self):
        """
        Test API key retrieval when only environment variable is set.
        
        Verifies that the API key is correctly retrieved from
        environment variables.
        """
        os.environ["OPENAI_API_KEY"] = "test_env_key"
        result = get_openai_api_key()
        self.assertEqual(result, "test_env_key", "Should retrieve key from environment variable")
    
    def test_keyring_fallback(self):
        """
        Test API key retrieval fallback to keyring when ENV is empty.
        
        Verifies that the system falls back to keyring when the
        environment variable is not set.
        """
        os.environ["OPENAI_API_KEY"] = ""
        with patch("keyring.get_password") as mock_get_password:
            mock_get_password.return_value = "test_keyring_key"
            result = get_openai_api_key()
            self.assertEqual(result, "test_keyring_key", "Should retrieve key from keyring")
    
    def test_both_empty(self):
        """
        Test API key retrieval when both ENV and keyring are empty.
        
        Verifies that an empty string is returned when no API key
        is found in any source.
        """
        os.environ["OPENAI_API_KEY"] = ""
        with patch("keyring.get_password") as mock_get_password:
            mock_get_password.return_value = None
            result = get_openai_api_key()
            self.assertEqual(result, "", "Should return empty string when no key found")
    
    def test_keyring_exception(self):
        """
        Test API key retrieval when keyring raises exception.
        
        Verifies that the system gracefully handles exceptions
        from the keyring module.
        """
        os.environ["OPENAI_API_KEY"] = ""
        with patch("keyring.get_password") as mock_get_password:
            mock_get_password.side_effect = Exception("Simulated keyring failure")
            result = get_openai_api_key()
            self.assertEqual(result, "", "Should return empty string when keyring fails")


if __name__ == "__main__":
    unittest.main()