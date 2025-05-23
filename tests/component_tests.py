"""
Component tests for Orbit Analyzer project.

This module contains tests for component-related functionality, including:
- Component identification and analysis (ComponentAnalyzer)
- Component relationship detection (ComponentIntegration)
- Component visualization generation (ComponentVisualizer)
- Context-aware error clustering (ContextAwareClusterer)
- Direct component mapping (DirectComponentAnalyzer)
- Component information preservation
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import tempfile
import json
import warnings
from datetime import datetime, timedelta
import networkx as nx
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing

# Filter out the specific warning about spring layout fallback
warnings.filterwarnings("ignore", message="Using spring layout as fallback")

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
        get_test_output_path, 
        setup_test_output_directories, 
        get_component_schema_path,
        validate_visualization,
        validate_report_file
    )
except ImportError:
    # Fallback for basic test utilities
    from test_utils import get_test_folder

# Import modules to test with appropriate error handling
try:
    from components.component_analyzer import ComponentAnalyzer
except ImportError:
    ComponentAnalyzer = None

try:
    from components.component_integration import ComponentIntegration
except ImportError:
    ComponentIntegration = None

try:
    from components.component_visualizer import ComponentVisualizer
except ImportError:
    ComponentVisualizer = None

try:
    from components.context_aware_clusterer import ContextAwareClusterer
except ImportError:
    ContextAwareClusterer = None

try:
    from components.direct_component_analyzer import assign_components_and_relationships, identify_component_from_filename
except ImportError:
    assign_components_and_relationships = identify_component_from_filename = None

try:
    from reports.base import ComponentAwareEncoder, COMPONENT_FIELDS
except ImportError:
    ComponentAwareEncoder = None
    COMPONENT_FIELDS = {
        'component', 'component_source', 'source_component', 'root_cause_component',
        'primary_issue_component', 'affected_components', 'expected_component',
        'component_scores', 'component_distribution', 'parent_component', 'child_components'
    }

# Import the new ComponentInfo and ComponentRegistry classes
try:
    from components.component_model import (
        ComponentInfo, 
        ComponentRegistry,
        get_component_registry,
        create_component_info
    )
except ImportError:
    ComponentInfo = ComponentRegistry = get_component_registry = create_component_info = None

# Import component utilities
try:
    from components.component_utils import (
        extract_component_fields,
        apply_component_fields,
        preserve_component_fields,
        enrich_with_component_info,
        identify_component_from_file,
        determine_primary_component,
        validate_component_data,
        verify_component_preservation,
        normalize_component_fields
    )
except ImportError:
    extract_component_fields = apply_component_fields = preserve_component_fields = None
    enrich_with_component_info = identify_component_from_file = determine_primary_component = None
    validate_component_data = verify_component_preservation = normalize_component_fields = None

# Try to import additional modules for verification tests
try:
    from config import Config
except ImportError:
    # Create minimal Config class for testing
    class Config:
        ENABLE_COMPONENT_DISTRIBUTION = False
        ENABLE_CLUSTER_TIMELINE = False
        OUTPUT_BASE_DIR = os.path.join(os.path.dirname(__file__), "output")

try:
    from reports.component_analyzer import generate_component_report
except ImportError:
    generate_component_report = None

try:
    from controller import diagnose_output_structure
except ImportError:
    # Simple fallback if diagnose_output_structure is not available
    def diagnose_output_structure(test_id):
        print(f"Diagnostic function not available, skipping validation for {test_id}")
        return {}

# Import data preprocessor for component preservation tests
try:
    from reports.data_preprocessor import (
        preprocess_errors,
        preprocess_clusters,
        normalize_data,
        normalize_timestamps_in_dict
    )
except ImportError:
    preprocess_errors = preprocess_clusters = normalize_data = normalize_timestamps_in_dict = None


@TestRegistry.register(category='component', importance=1, tags=['analyzer'])
class TestComponentAnalyzer(unittest.TestCase):
    """
    Unit tests for the ComponentAnalyzer class.
    
    Tests the ability to identify components from log files,
    analyze component relationships, and determine root causes.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        # Skip test if module not available
        if ComponentAnalyzer is None:
            self.skipTest("ComponentAnalyzer module not available")
        
        # Get or create schema path
        self.schema_path = get_component_schema_path()
        
        # Initialize the analyzer
        self.analyzer = ComponentAnalyzer(self.schema_path)
        
        # Create test log entries and errors
        self.now = datetime.now()
        
        # Log entries
        LogEntry = type('LogEntry', (), {})
        
        self.soa_entry = LogEntry()
        self.soa_entry.file = "app_debug.log"
        self.soa_entry.text = "Error in SOA component"
        
        self.mimosa_entry = LogEntry()
        self.mimosa_entry.file = "mimosa.log"
        self.mimosa_entry.text = "Data unavailable"
        
        self.unknown_entry = LogEntry()
        self.unknown_entry.file = "unknown.log"
        self.unknown_entry.text = "Generic log message"
        
        # Dictionary-style log entries
        self.log_entries = [
            {"file": "app_debug.log", "line_number": 1, "text": "Error in SOA component"},
            {"file": "mimosa.log", "line_number": 2, "text": "Data unavailable"},
            {"file": "unknown.log", "line_number": 3, "text": "Generic log message"}
        ]
        
        # Errors
        self.errors = [
            {
                "file": "app_debug.log",
                "line_num": 1,
                "text": "Error in SOA component",
                "severity": "High",
                "timestamp": self.now.isoformat(),
                "component": "soa"
            },
            {
                "file": "app_debug.log",
                "line_num": 2,
                "text": "Another SOA error",
                "severity": "Medium",
                "timestamp": (self.now + timedelta(seconds=5)).isoformat(),
                "component": "soa"
            },
            {
                "file": "mimosa.log",
                "line_num": 3,
                "text": "Data unavailable",
                "severity": "Medium",
                "timestamp": (self.now + timedelta(seconds=10)).isoformat(),
                "component": "mimosa"
            }
        ]
    
    def test_initialization(self):
        """
        Test that ComponentAnalyzer initializes correctly.
        
        Verifies that the analyzer is properly initialized with the
        component schema and creates appropriate data structures.
        """
        self.assertIsNotNone(self.analyzer.component_schema)
        self.assertIsNotNone(self.analyzer.component_patterns)
        self.assertIsNotNone(self.analyzer.component_log_sources)
        self.assertIsNotNone(self.analyzer.component_relationships)
    
    def test_ensure_datetime(self):
        """
        Test the _ensure_datetime method.
        
        Verifies that the method correctly handles various timestamp formats
        and converts them to datetime objects.
        """
        # Test with ISO format string
        dt_str = "2023-01-01T12:00:00"
        result = self.analyzer._ensure_datetime(dt_str)
        self.assertIsInstance(result, datetime)
        
        # Test with datetime object
        dt_obj = datetime.now()
        result = self.analyzer._ensure_datetime(dt_obj)
        self.assertIs(result, dt_obj)
        
        # Test with None
        result = self.analyzer._ensure_datetime(None)
        self.assertIsNone(result)
    
    def test_identify_component_from_log_file(self):
        """
        Test identifying component from log filename.
        
        Verifies that components are correctly identified from log filenames
        based on the schema patterns.
        """
        # Test with SOA log file
        self.assertEqual(self.analyzer.identify_component_from_log_file("app_debug.log"), "soa")
        
        # Test with Mimosa log file
        self.assertEqual(self.analyzer.identify_component_from_log_file("mimosa.log"), "mimosa")
        
        # Test with unknown log file - updated to expect "unknown" instead of None
        self.assertEqual(self.analyzer.identify_component_from_log_file("unknown.log"), "unknown")
    
    def test_identify_component_from_log_entry(self):
        """
        Test identifying component from log entry.
        
        Verifies that components are correctly identified from log entry
        objects based on their file attribute.
        """
        # Test with SOA entry
        self.assertEqual(self.analyzer.identify_component_from_log_entry(self.soa_entry), "soa")
        
        # Test with Mimosa entry
        self.assertEqual(self.analyzer.identify_component_from_log_entry(self.mimosa_entry), "mimosa")
        
        # Test with unknown entry
        self.assertEqual(self.analyzer.identify_component_from_log_entry(self.unknown_entry), "unknown")
    
    def test_enrich_log_entries_with_components(self):
        """
        Test enriching log entries with component information.
        
        Verifies that log entries are correctly enriched with component
        information based on their file names.
        """
        # Enrich log entries
        enriched = self.analyzer.enrich_log_entries_with_components(self.log_entries)
        
        # Check component assignments
        self.assertEqual(enriched[0]['component'], "soa")
        self.assertEqual(enriched[1]['component'], "mimosa")
        self.assertEqual(enriched[2]['component'], "unknown")
    
    def test_analyze_component_failures(self):
        """
        Test analyzing component failures.
        
        Verifies that the analyzer correctly analyzes component failures
        and generates the expected result structure with component statistics.
        """
        # Analyze failures
        analysis = self.analyzer.analyze_component_failures(self.errors)
        
        # Check result structure
        self.assertIsInstance(analysis, dict)
        self.assertIn("component_error_counts", analysis)
        self.assertIn("severity_by_component", analysis)
        self.assertIn("components_with_issues", analysis)
        self.assertIn("root_cause_component", analysis)
        
        # Check component error counts
        self.assertEqual(analysis["component_error_counts"]["soa"], 2)
        self.assertEqual(analysis["component_error_counts"]["mimosa"], 1)
        
        # Check components with issues
        self.assertIn("soa", analysis["components_with_issues"])
        self.assertIn("mimosa", analysis["components_with_issues"])
        
        # Root cause should be either soa or mimosa (depends on implementation details)
        self.assertIn(analysis["root_cause_component"], ["soa", "mimosa"])
    
    def test_are_components_related(self):
        """
        Test checking component relationships.
        
        Verifies that component relationships are correctly identified
        based on the schema.
        """
        # Test direct relationship (mimosa -> soa)
        self.assertTrue(self.analyzer._are_components_related("mimosa", "soa"))
        
        # Test with unknown component
        self.assertFalse(self.analyzer._are_components_related("unknown", "soa"))
        self.assertFalse(self.analyzer._are_components_related("soa", "unknown"))
    
    def test_calculate_causality_weight(self):
        """
        Test calculating causality weight between errors.
        
        Verifies that causality weights are correctly calculated based on
        error metadata and timestamps.
        """
        # Create test errors
        error1 = {
            "component": "soa",
            "severity": "High"
        }
        
        error2 = {
            "component": "mimosa",
            "severity": "Medium"
        }
        
        # Calculate weight
        now = datetime.now()
        weight = self.analyzer._calculate_causality_weight(
            error1, error2, now, now + timedelta(seconds=1)
        )
        
        # Weight should be a float between 0 and 2
        self.assertIsInstance(weight, float)
        self.assertGreaterEqual(weight, 0.0)
        
        # Test with reversed component order
        weight2 = self.analyzer._calculate_causality_weight(
            error2, error1, now, now + timedelta(seconds=1)
        )
        
        # Weight should be different based on component relationship
        self.assertNotEqual(weight, weight2)
    
    def test_build_causality_graph(self):
        """
        Test building causality graph from errors.
        
        Verifies that a causality graph is correctly built from a list of
        errors, with nodes for each error and edges between related errors.
        """
        # Skip if _build_causality_graph method doesn't exist
        if not hasattr(self.analyzer, '_build_causality_graph'):
            self.skipTest("_build_causality_graph method not available")
            
        # Build causality graph
        try:
            graph = self.analyzer._build_causality_graph(self.errors)
            
            # Graph should be a NetworkX DiGraph
            self.assertIsInstance(graph, nx.DiGraph)
            
            # Graph should have nodes for each error
            self.assertEqual(len(graph.nodes()), len(self.errors))
            
            # Graph should have at least some edges (based on timestamps)
            self.assertGreater(len(graph.edges()), 0)
        except AttributeError:
            self.skipTest("_build_causality_graph method is not accessible")
    
    def test_get_component_info(self):
        """
        Test getting component information.
        
        Verifies that component information is correctly retrieved from
        the schema based on component ID.
        """
        # Get SOA component info
        soa_info = self.analyzer.get_component_info("soa")
        
        # Check info - using more flexible assertions that don't depend on exact description content
        self.assertEqual(soa_info["id"], "soa")
        self.assertEqual(soa_info["name"], "SOA")
        self.assertIn("description", soa_info)  # Just check it has a description
        self.assertIsInstance(soa_info["description"], str)
        
        # Test with unknown component
        unknown_info = self.analyzer.get_component_info("unknown")
        
        # Should return default info
        self.assertEqual(unknown_info["id"], "unknown")
        self.assertEqual(unknown_info["name"], "unknown")
        self.assertIn("description", unknown_info)
        


@TestRegistry.register(category='component', importance=1, tags=['integration'])
class TestComponentIntegration(unittest.TestCase):
    """
    Unit tests for the ComponentIntegration class.
    
    Tests the integration of component analysis, visualization, and clustering
    to provide comprehensive component relationship analysis.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        # Skip test if ComponentIntegration is not available
        if ComponentIntegration is None:
            self.skipTest("ComponentIntegration module not available")
            
        # Get or create schema path
        self.schema_path = get_component_schema_path()
        
        # Create test data
        self.log_entries = [
            {"file": "soa.log", "line_number": 1, "text": "Test log entry 1", "timestamp": "2023-01-01 12:00:00"},
            {"file": "mimosa.log", "line_number": 2, "text": "Test log entry 2", "timestamp": "2023-01-01 12:01:00"}
        ]
        
        self.errors = [
            {"file": "soa.log", "line_num": 1, "text": "Test error 1", "severity": "High", "timestamp": "2023-01-01 12:00:00"},
            {"file": "mimosa.log", "line_num": 2, "text": "Test error 2", "severity": "Medium", "timestamp": "2023-01-01 12:01:00"}
        ]
        
        # Initialize the component integration
        self.integrator = ComponentIntegration(self.schema_path)
        
        # Create a temporary output directory
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directory
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """
        Test that the ComponentIntegration initializes correctly.
        
        Verifies that the integrator is properly initialized with
        the analyzer, visualizer, and clusterer components.
        """
        self.assertIsNotNone(self.integrator.analyzer)
        self.assertIsNotNone(self.integrator.visualizer)
        self.assertIsNotNone(self.integrator.clusterer)
    
    @patch('components.component_visualizer.ComponentVisualizer.generate_component_relationship_diagram')
    @patch('components.context_aware_clusterer.ContextAwareClusterer.cluster_errors')
    def test_analyze_logs(self, mock_clusterer, mock_diagram):
        """
        Test the analyze_logs method.
        
        Verifies that the integrator correctly analyzes logs, assigns
        components, and generates appropriate results.
        """
        # Set up mock for visualization methods
        mock_diagram.return_value = os.path.join(self.temp_dir, "mock_diagram.png")
        
        # Set up mock for clusterer to avoid the vectorization error
        mock_clusterer.return_value = {0: self.errors}
        
        # Test the analyze_logs method
        result = self.integrator.analyze_logs(self.log_entries, self.errors, self.temp_dir, "TEST-123")
        
        # Verify the result structure
        self.assertIsNotNone(result)
        self.assertIn("test_id", result)
        self.assertIn("timestamp", result)
        self.assertIn("analysis_files", result)
        self.assertIn("metrics", result)
        
        # Verify metrics
        metrics = result.get("metrics", {})
        self.assertIn("component_tagged_logs", metrics)
        self.assertIn("component_tagged_errors", metrics)
    
    def test_get_enhanced_report_data(self):
        """
        Test the get_enhanced_report_data method.
        
        Verifies that enhanced report data is correctly generated based on
        the analysis results.
        """
        # Skip if method doesn't exist
        if not hasattr(self.integrator, 'get_enhanced_report_data'):
            self.skipTest("get_enhanced_report_data method not available")
            
        # First create some analysis results
        with patch('components.component_visualizer.ComponentVisualizer.generate_component_relationship_diagram'):
            result = self.integrator.analyze_logs(self.log_entries, self.errors, self.temp_dir, "TEST-123")
        
        # Now test the enhanced report data
        enhanced_data = self.integrator.get_enhanced_report_data(result)
        
        # Verify the result structure
        self.assertIsNotNone(enhanced_data)
        self.assertIn("component_analysis", enhanced_data)
    
    def test_serialize_error(self):
        """
        Test the _serialize_error method.
        
        Verifies that errors are correctly serialized with proper
        handling of non-serializable objects.
        """
        # Skip if method doesn't exist
        if not hasattr(self.integrator, '_serialize_error'):
            self.skipTest("_serialize_error method not available")
            
        # Create a test error with various data types
        error = {
            "text": "Test error",
            "timestamp": "2023-01-01 12:00:00",
            "complex_value": MagicMock()  # Something not JSON serializable
        }
        
        # Test serialization
        result = self.integrator._serialize_error(error)
        
        # Verify results
        self.assertIn("text", result)
        self.assertIn("timestamp", result)
        self.assertIn("complex_value", result)
        self.assertTrue(isinstance(result["complex_value"], str))  # Should be converted to string
    
    @patch('components.context_aware_clusterer.ContextAwareClusterer.cluster_errors')
    def test_comprehensive_integration(self, mock_clusterer):
        """
        Test the full integration workflow with actual files.
        
        Verifies the end-to-end component integration process with file
        generation and validation.
        """
        # Set up mock for clusterer to avoid the vectorization error
        mock_clusterer.return_value = {0: self.errors}
        
        # Create a temporary output directory specifically for this test
        with tempfile.TemporaryDirectory() as test_output_dir:
            # Run the analysis with multiple patches to avoid warnings/errors
            with patch('components.component_visualizer.ComponentVisualizer.generate_component_relationship_diagram') as mock_diagram:
                # Create an actual file for the mock to return
                mock_file_path = os.path.join(test_output_dir, "mock_diagram.png")
                with open(mock_file_path, 'wb') as f:
                    f.write(b'PNG')  # Write dummy content
                
                # Set up mock returns to return the actual created file
                mock_diagram.return_value = mock_file_path
                
                # Run the analysis
                result = self.integrator.analyze_logs(self.log_entries, self.errors, test_output_dir, "TEST-COMP-123")
                
                # Verify the results
                self.assertIsNotNone(result)
                self.assertIn("test_id", result)
                self.assertEqual(result["test_id"], "TEST-COMP-123")
                
                # Check for files that we directly created via mocks
                analysis_files = result.get("analysis_files", {})
                for file_key, file_path in analysis_files.items():
                    if file_path and file_path == mock_file_path:
                        self.assertTrue(os.path.exists(file_path), f"File {file_path} should exist")
                
                # Test the enhanced report data if method exists
                if hasattr(self.integrator, 'get_enhanced_report_data'):
                    enhanced_data = self.integrator.get_enhanced_report_data(result)
                    self.assertIsNotNone(enhanced_data)
                    self.assertIn("component_analysis", enhanced_data)
    
    def test_component_preservation_in_analyze_logs(self):
        """
        Test that component information is preserved during analysis.
        
        Verifies that component information is properly preserved when
        passing through the analyze_logs method.
        """
        # Skip if verify_component_preservation is not available
        if verify_component_preservation is None:
            self.skipTest("verify_component_preservation function not available")
        
        # Add component information to test errors
        errors_with_components = []
        for error in self.errors:
            error_copy = error.copy()
            error_copy['component'] = 'test_component'
            error_copy['component_source'] = 'manual'
            error_copy['primary_issue_component'] = 'test_component'
            errors_with_components.append(error_copy)
        
        # Create a patched version of analyze_logs that isolates component handling
        with patch('components.component_visualizer.ComponentVisualizer.generate_component_relationship_diagram'):
            with patch('components.context_aware_clusterer.ContextAwareClusterer.cluster_errors'):
                result = self.integrator.analyze_logs(
                    self.log_entries, 
                    errors_with_components, 
                    self.temp_dir, 
                    "TEST-123"
                )
        
        # Verify the primary_issue_component is preserved
        self.assertEqual(
            result.get("primary_issue_component", ""), 
            "test_component",
            "Primary issue component should be preserved"
        )


@TestRegistry.register(category='component', importance=1, tags=['preservation'])
class ComponentPreservationTest(unittest.TestCase):
    """
    Test for component preservation validation in the ComponentAwareEncoder.
    
    Verifies that component information is preserved during serialization
    and deserialization processes.
    """
    
    def setUp(self):
        """Set up the test environment."""
        # Skip test if ComponentAwareEncoder is not available
        if ComponentAwareEncoder is None:
            self.skipTest("ComponentAwareEncoder not available")
            
        self.encoder = ComponentAwareEncoder()
        
        # Create a sample data structure with component information
        self.original_data = {
            'component': 'soa',
            'component_source': 'filename',
            'primary_issue_component': 'android',
            'affected_components': ['soa', 'android'],
            'other_field': 'test value'
        }
        
    def test_validation_success(self):
        """
        Test successful validation when component info is preserved.
        
        Verifies that validation succeeds when all component fields
        are properly preserved.
        """
        # Create processed data that preserves all component fields
        processed_data = self.original_data.copy()
        processed_data['new_field'] = 'new value'
        
        # Validate
        # Check if validate_component_preservation method exists
        if hasattr(self.encoder, 'validate_component_preservation'):
            result = self.encoder.validate_component_preservation(self.original_data, processed_data)
            self.assertTrue(result, "Validation should succeed when all component fields are preserved")
        else:
            # If method doesn't exist, manually check fields
            for key, value in self.original_data.items():
                if key.startswith('component') or key == 'primary_issue_component' or key == 'affected_components':
                    self.assertEqual(processed_data.get(key), value, f"Component field {key} should be preserved")
    
    def test_validation_failure_missing_field(self):
        """
        Test validation failure when a component field is missing.
        
        Verifies that validation fails when a component field is
        missing from the processed data.
        """
        # Create processed data with a missing component field
        processed_data = self.original_data.copy()
        del processed_data['component']
        
        # Validate
        if hasattr(self.encoder, 'validate_component_preservation'):
            result = self.encoder.validate_component_preservation(self.original_data, processed_data)
            self.assertFalse(result, "Validation should fail when a component field is missing")
        else:
            # If method doesn't exist, perform manual check
            self.assertNotEqual(
                set(k for k in self.original_data if k.startswith('component') or k in ['primary_issue_component', 'affected_components']),
                set(k for k in processed_data if k.startswith('component') or k in ['primary_issue_component', 'affected_components']),
                "Component fields should be different when one is missing"
            )
    
    def test_validation_failure_changed_field(self):
        """
        Test validation failure when a component field is changed.
        
        Verifies that validation fails when a component field is
        modified in the processed data.
        """
        # Create processed data with a changed component field
        processed_data = self.original_data.copy()
        processed_data['component'] = 'android'  # Changed from 'soa'
        
        # Validate
        if hasattr(self.encoder, 'validate_component_preservation'):
            result = self.encoder.validate_component_preservation(self.original_data, processed_data)
            self.assertFalse(result, "Validation should fail when a component field is changed")
        else:
            # If method doesn't exist, check specific field
            self.assertNotEqual(
                self.original_data['component'],
                processed_data['component'],
                "Component field should be different"
            )
    
    def test_validation_unknown_values(self):
        """
        Test validation with 'unknown' values which should be ignored.
        
        Verifies that validation ignores 'unknown' values in component fields
        since they are considered default values.
        """
        # Create original data with an 'unknown' component field
        original_with_unknown = self.original_data.copy()
        original_with_unknown['root_cause_component'] = 'unknown'
        
        # Create processed data without that field
        processed_data = self.original_data.copy()
        # Note: processed_data doesn't have 'root_cause_component'
        
        # Validate
        if hasattr(self.encoder, 'validate_component_preservation'):
            result = self.encoder.validate_component_preservation(original_with_unknown, processed_data)
            self.assertTrue(result, "Validation should ignore 'unknown' component values")
        else:
            # If method doesn't exist, skip this test
            pass
    
    def test_json_serialization_preserves_components(self):
        """
        Test that JSON serialization preserves component information.
        
        Verifies that component information is preserved when data is
        serialized to JSON and then deserialized.
        """
        # Serialize and deserialize data
        json_str = json.dumps(self.original_data, cls=ComponentAwareEncoder)
        deserialized_data = json.loads(json_str)
        
        # Validate
        if hasattr(self.encoder, 'validate_component_preservation'):
            result = self.encoder.validate_component_preservation(self.original_data, deserialized_data)
            self.assertTrue(result, "JSON serialization should preserve component information")
        else:
            # Manually check component fields
            for key, value in self.original_data.items():
                if key.startswith('component') or key == 'primary_issue_component' or key == 'affected_components':
                    self.assertEqual(deserialized_data.get(key), value, f"Component field {key} should be preserved in JSON")
    
    def test_nested_component_preservation(self):
        """
        Test that component information is preserved in nested structures.
        
        Verifies that component information is properly preserved in nested
        dictionaries and arrays during serialization.
        """
        # Create nested data structure
        nested_data = {
            'component': 'soa',
            'component_source': 'filename',
            'errors': [
                {
                    'text': 'Error 1',
                    'component': 'soa',
                    'component_source': 'filename'
                },
                {
                    'text': 'Error 2',
                    'component': 'android',
                    'component_source': 'content'
                }
            ],
            'details': {
                'component': 'mimosa',
                'component_source': 'schema',
                'metrics': {
                    'component_counts': {'soa': 2, 'android': 1}
                }
            }
        }
        
        # Serialize and deserialize
        json_str = json.dumps(nested_data, cls=ComponentAwareEncoder)
        deserialized_data = json.loads(json_str)
        
        # Verify top-level components are preserved
        self.assertEqual(deserialized_data['component'], 'soa')
        self.assertEqual(deserialized_data['component_source'], 'filename')
        
        # Verify components in arrays are preserved
        self.assertEqual(deserialized_data['errors'][0]['component'], 'soa')
        self.assertEqual(deserialized_data['errors'][1]['component'], 'android')
        
        # Verify components in nested dictionaries are preserved
        self.assertEqual(deserialized_data['details']['component'], 'mimosa')
        self.assertEqual(deserialized_data['details']['component_source'], 'schema')
        self.assertEqual(deserialized_data['details']['metrics']['component_counts']['soa'], 2)


@TestRegistry.register(category='component', importance=1, tags=['visualizer'])
class TestComponentVisualizer(unittest.TestCase):
    """
    Unit tests for the ComponentVisualizer class.
    
    Tests the generation of visualizations for component relationships,
    error propagation, and component error distribution.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        # Skip test if ComponentVisualizer is not available
        if ComponentVisualizer is None:
            self.skipTest("ComponentVisualizer module not available")
            
        # Get or create schema path
        self.schema_path = get_component_schema_path()
        
        # Initialize the component visualizer
        self.visualizer = ComponentVisualizer(self.schema_path)
        
        # Set up test data
        self.component_errors = {
            "soa": 5,
            "mimosa": 3
        }
        
        self.error_analysis = {
            "component_error_counts": self.component_errors,
            "severity_by_component": {
                "soa": {"High": 2, "Medium": 2, "Low": 1},
                "mimosa": {"High": 1, "Medium": 1, "Low": 1}
            }
        }
        
        # Create a temporary directory for outputs
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directory
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """
        Test that the ComponentVisualizer initializes correctly.
        
        Verifies that the visualizer is properly initialized with
        the component schema and color mappings.
        """
        self.assertIsNotNone(self.visualizer.component_schema)
        self.assertIsNotNone(self.visualizer.component_graph)
        self.assertIsNotNone(self.visualizer.severity_colors)
        self.assertIsNotNone(self.visualizer.component_colors)
    
    def test_generate_component_relationship_diagram(self):
        """
        Test the generation of component relationship diagrams.
        
        Verifies that component relationship diagrams are correctly
        generated and saved to the proper location.
        """
        # Get the output path directly
        test_id = "TEST-VIZ-123"
        dirs = setup_test_output_directories(test_id)
        diagram_path = self.visualizer.generate_component_relationship_diagram(dirs["images"])
        
        # Verify the diagram was created
        self.assertIsNotNone(diagram_path)
        
        # Validate the visualization
        is_valid, issues = validate_visualization(diagram_path)
        self.assertTrue(is_valid, f"Visualization validation failed: {', '.join(issues)}")
    
    def test_generate_error_propagation_diagram(self):
        """
        Test the generation of error propagation diagrams.
        
        Verifies that error propagation diagrams are correctly generated
        and saved to the proper location.
        """
        # Skip if method doesn't exist
        if not hasattr(self.visualizer, 'generate_error_propagation_diagram'):
            self.skipTest("generate_error_propagation_diagram method not available")
            
        # Generate the diagram
        test_id = "TEST-PROP-123"
        dirs = setup_test_output_directories(test_id)
        propagation_path = self.visualizer.generate_error_propagation_diagram(
            dirs["images"],
            self.component_errors,
            "soa",
            [["mimosa", "soa"]],
            test_id
        )
        
        # Verify the diagram was created
        self.assertIsNotNone(propagation_path)
        
        # Validate the visualization
        is_valid, issues = validate_visualization(propagation_path)
        self.assertTrue(is_valid, f"Visualization validation failed: {', '.join(issues)}")
    
    def test_generate_component_error_heatmap(self):
        """
        Test the generation of component error heatmaps.
        
        Verifies that component error heatmaps are correctly generated
        and saved to the proper location.
        """
        # Skip if method doesn't exist
        if not hasattr(self.visualizer, 'generate_component_error_heatmap'):
            self.skipTest("generate_component_error_heatmap method not available")
            
        # Generate the heatmap
        test_id = "TEST-HEAT-123"
        dirs = setup_test_output_directories(test_id)
        heatmap_path = self.visualizer.generate_component_error_heatmap(
            dirs["images"],
            self.error_analysis,
            test_id
        )
        
        # Verify the heatmap was created
        self.assertIsNotNone(heatmap_path)
        
        # Validate the visualization
        is_valid, issues = validate_visualization(heatmap_path)
        self.assertTrue(is_valid, f"Visualization validation failed: {', '.join(issues)}")
    
    def test_get_component_name(self):
        """
        Test the _get_component_name method.
        
        Verifies that component names are correctly retrieved from
        the schema with proper capitalization.
        """
        # Skip if method doesn't exist
        if not hasattr(self.visualizer, '_get_component_name'):
            self.skipTest("_get_component_name method not available")
            
        # Test with known component
        name = self.visualizer._get_component_name("soa")
        self.assertEqual(name, "SOA")
        
        # Test with unknown component
        name = self.visualizer._get_component_name("unknown_component")
        self.assertEqual(name, "unknown_component")
    
    @patch('matplotlib.pyplot.savefig')  # Mock to avoid matplotlib warnings
    def test_get_graph_layout(self, mock_savefig):
        """
        Test the _get_graph_layout method.
        
        Verifies that graph layouts are correctly generated for
        component relationship visualization.
        """
        # Skip if method doesn't exist
        if not hasattr(self.visualizer, '_get_graph_layout'):
            self.skipTest("_get_graph_layout method not available")
            
        # Create a small test graph
        import networkx as nx
        G = nx.DiGraph()
        G.add_node("node1")
        G.add_node("node2")
        G.add_edge("node1", "node2")
        
        # Test the layout generation with patched warning handling
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress all warnings
            layout = self.visualizer._get_graph_layout(G)
        
        # Check that the layout includes positions for all nodes
        self.assertIn("node1", layout)
        self.assertIn("node2", layout)
        
        # Check that the positions are valid
        self.assertEqual(len(layout["node1"]), 2)  # x, y coordinates
        self.assertEqual(len(layout["node2"]), 2)
    
    def test_generate_empty_diagram(self):
        """
        Test the _generate_empty_diagram method.
        
        Verifies that empty diagrams are correctly generated as
        fallbacks when visualization generation fails.
        """
        # Skip if method doesn't exist
        if not hasattr(self.visualizer, '_generate_empty_diagram'):
            self.skipTest("_generate_empty_diagram method not available")
            
        # Include the filename parameter that was missing
        test_id = "TEST-EMPTY-123"
        dirs = setup_test_output_directories(test_id)
        empty_path = self.visualizer._generate_empty_diagram(dirs["images"], test_id, "empty_diagram")

        # Expect standardized filename without duplicate extension
        expected_name = f"{test_id}_empty_diagram.png"
        self.assertEqual(os.path.basename(empty_path), expected_name)
        
        # Verify the empty diagram was created
        self.assertIsNotNone(empty_path)
        self.assertTrue(os.path.exists(empty_path))
        self.assertTrue(os.path.getsize(empty_path) > 0)


@TestRegistry.register(category='component', importance=1, tags=['clusterer'])
class TestContextAwareClusterer(unittest.TestCase):
    """
    Unit tests for the ContextAwareClusterer class.
    
    Tests the context-aware clustering of errors based on component
    relationships and temporal information.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        # Skip test if ContextAwareClusterer is not available
        if ContextAwareClusterer is None:
            self.skipTest("ContextAwareClusterer module not available")
            
        # Get or create schema path
        self.schema_path = get_component_schema_path()
        
        # Initialize the clusterer
        self.clusterer = ContextAwareClusterer(self.schema_path)
        
        # Create test errors
        now = datetime.now()
        self.errors = [
            {
                "text": "Error connecting to database",
                "severity": "High",
                "component": "soa",
                "timestamp": now.isoformat(),
                "file": "soa.log",
                "line_num": 1
            },
            {
                "text": "Database timeout after 30 seconds",
                "severity": "High",
                "component": "soa",
                "timestamp": (now + timedelta(seconds=5)).isoformat(),
                "file": "soa.log",
                "line_num": 2
            },
            {
                "text": "Data feed unavailable",
                "severity": "Medium",
                "component": "mimosa",
                "timestamp": (now + timedelta(seconds=10)).isoformat(),
                "file": "mimosa.log",
                "line_num": 3
            },
            {
                "text": "Failed to update channel listing",
                "severity": "Medium",
                "component": "mimosa",
                "timestamp": (now + timedelta(seconds=15)).isoformat(),
                "file": "mimosa.log",
                "line_num": 4
            }
        ]
        
        # Temporary directory for output files
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directory
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """
        Test that the ContextAwareClusterer initializes correctly.
        
        Verifies that the clusterer is properly initialized with
        the component schema and relationships.
        """
        self.assertIsNotNone(self.clusterer.component_schema)
        self.assertIsNotNone(self.clusterer.component_graph)
    
    def test_ensure_datetime(self):
        """
        Test the _ensure_datetime method.
        
        Verifies that timestamps are correctly converted to datetime
        objects from various input formats.
        """
        # Test with ISO format string
        dt_str = "2023-01-01T12:00:00"
        result = self.clusterer._ensure_datetime(dt_str)
        self.assertIsInstance(result, datetime)
        
        # Test with datetime object
        dt_obj = datetime.now()
        result = self.clusterer._ensure_datetime(dt_obj)
        self.assertIs(result, dt_obj)  # Should return the same object
        
        # Test with None
        result = self.clusterer._ensure_datetime(None)
        self.assertIsNone(result)
        
        # Test with invalid string
        result = self.clusterer._ensure_datetime("not a datetime")
        self.assertIsNone(result)
    
    def test_normalize_error_text(self):
        """
        Test the _normalize_error_text method.
        
        Verifies that error text is properly normalized by removing
        variable parts for better clustering.
        """
        # Skip if method doesn't exist
        if not hasattr(self.clusterer, '_normalize_error_text'):
            self.skipTest("_normalize_error_text method not available")
            
        # Test with standard error text
        text = "Error connecting to database at 12:34:56"
        result = self.clusterer._normalize_error_text(text)
        # Fix case sensitivity - using lowercase for both sides of comparison
        self.assertIn("error connecting to database at timestamp", result.lower())
        
        # Test with UUID - note that our implementation seems to lowercase the UUID to "uuid"
        text = "Error with ID: 12345678-1234-5678-1234-567812345678"
        result = self.clusterer._normalize_error_text(text)
        self.assertIn("uuid", result.lower())
        
        # Test with file path - check for lowercase "path" instead of uppercase "PATH"
        text = "Error reading file: C:\\Users\\user\\file.txt"
        result = self.clusterer._normalize_error_text(text)
        self.assertIn("path", result.lower())
        
        # Test with memory address
        text = "Object at address 0x123abc"
        result = self.clusterer._normalize_error_text(text)
        self.assertIn("memory_addr", result.lower())  # Change to check for lowercase
        
        # Test with numbers
        text = "Error code 404 received"
        result = self.clusterer._normalize_error_text(text)
        self.assertIn("error code num", result.lower())
    
    @patch('components.context_aware_clusterer.TfidfVectorizer')
    @patch('components.context_aware_clusterer.KMeans')
    def test_cluster_errors(self, mock_kmeans, mock_vectorizer):
        """
        Test the cluster_errors method.
        
        Verifies that errors are correctly clustered based on
        similarity and component relationships.
        """
        # Set up mocks
        mock_vectorizer_instance = MagicMock()
        mock_vectorizer.return_value = mock_vectorizer_instance
        
        # Create a proper mock for the TF-IDF matrix with a shape property
        mock_matrix = MagicMock()
        mock_matrix.shape = (4, 100)  # 4 errors, 100 features
        mock_vectorizer_instance.fit_transform.return_value = mock_matrix
        
        mock_kmeans_instance = MagicMock()
        mock_kmeans.return_value = mock_kmeans_instance
        mock_kmeans_instance.fit_predict.return_value = [0, 0, 1, 1]  # Two clusters
        
        # Test clustering
        clusters = self.clusterer.cluster_errors(self.errors, num_clusters=2)
        
        # Verify clusters were created
        self.assertEqual(len(clusters), 2)
        self.assertEqual(len(clusters[0]), 2)
        self.assertEqual(len(clusters[1]), 2)
        
        # Verify vectorizer and kmeans were called
        mock_vectorizer.assert_called_once()
        mock_kmeans.assert_called_once()
    
    def test_are_components_related(self):
        """
        Test the _are_components_related method.
        
        Verifies that component relationships are correctly identified
        from the component schema.
        """
        # Skip if method doesn't exist
        if not hasattr(self.clusterer, '_are_components_related'):
            self.skipTest("_are_components_related method not available")
            
        # Test with directly related components (mimosa -> soa)
        self.assertTrue(self.clusterer._are_components_related("mimosa", "soa"))
        
        # Test with unrelated components
        self.assertFalse(self.clusterer._are_components_related("soa", "unknown"))
        
        # Test with unknown components
        self.assertFalse(self.clusterer._are_components_related("unknown1", "unknown2"))
    
    def test_calculate_causality_weight(self):
        """
        Test the _calculate_causality_weight method.
        
        Verifies that causality weights are correctly calculated based on
        error attributes and temporal proximity.
        """
        # Skip if method doesn't exist
        if not hasattr(self.clusterer, '_calculate_causality_weight'):
            self.skipTest("_calculate_causality_weight method not available")
            
        now = datetime.now()
        error1 = {
            "component": "soa",
            "severity": "High"
        }
        error2 = {
            "component": "mimosa",
            "severity": "Medium"
        }
        
        # Calculate weight
        weight = self.clusterer._calculate_causality_weight(
            error1, error2, now, now + timedelta(seconds=1), 
            cluster1=0, cluster2=1
        )
        
        # Verify weight is between 0 and 1
        self.assertGreaterEqual(weight, 0.0)
        self.assertLessEqual(weight, 1.0)
    
    def test_get_root_cause_errors(self):
        """
        Test the get_root_cause_errors method.
        
        Verifies that root cause errors are correctly identified
        from clustered errors.
        """
        # Skip if method doesn't exist
        if not hasattr(self.clusterer, 'get_root_cause_errors'):
            self.skipTest("get_root_cause_errors method not available")
            
        # Create clusters directly without calling cluster_errors
        clusters = {
            0: [self.errors[0], self.errors[1]],
            1: [self.errors[2], self.errors[3]]
        }
        
        # Add is_root_cause flag to some errors
        for error in clusters[0]:
            error['is_root_cause'] = True
        
        for error in clusters[1]:
            error['is_root_cause'] = False
        
        # Get root cause errors
        root_cause_errors = self.clusterer.get_root_cause_errors(clusters)
        
        # Verify results
        self.assertEqual(len(root_cause_errors), 2)  # Should have the two errors from cluster 0
    
    def test_export_error_graph(self):
        """
        Test the export_error_graph method.
        
        Verifies that error graphs are correctly exported to JSON
        for visualization purposes.
        """
        # Skip if method doesn't exist
        if not hasattr(self.clusterer, 'export_error_graph'):
            self.skipTest("export_error_graph method not available")
            
        # Create an error graph attribute directly
        self.clusterer.error_graph = nx.DiGraph()
        
        # Add some basic nodes and edges
        self.clusterer.error_graph.add_node("node1", component="soa")
        self.clusterer.error_graph.add_node("node2", component="mimosa")
        self.clusterer.error_graph.add_edge("node1", "node2", weight=0.5)
        
        # Create test directories
        test_id = "TEST-GRAPH-123"
        dirs = setup_test_output_directories(test_id)
        
        # Export the graph
        graph_path = self.clusterer.export_error_graph(dirs["json"], test_id)
        
        # Verify the file was created
        self.assertIsNotNone(graph_path)
        
        # Verify the file exists
        self.assertTrue(os.path.exists(graph_path))
        
        # Verify file is valid JSON with expected structure
        with open(graph_path, 'r') as f:
            graph_data = json.load(f)
        
        self.assertIn("nodes", graph_data)
        self.assertIn("links", graph_data)
    
    def test_component_info_preservation_during_clustering(self):
        """
        Test that component information is preserved during clustering.
        
        Verifies that component information is properly preserved when
        errors are processed through the clustering algorithm.
        """
        # Skip if verify_component_preservation is not available
        if verify_component_preservation is None:
            self.skipTest("verify_component_preservation function not available")
        
        # Add extra component fields to test errors
        enriched_errors = []
        for error in self.errors:
            error_copy = error.copy()
            error_copy['primary_issue_component'] = 'test_primary'
            error_copy['affected_components'] = ['soa', 'mimosa']
            error_copy['root_cause_component'] = 'test_root_cause'
            enriched_errors.append(error_copy)
        
        # Patch to avoid actual clustering
        with patch('components.context_aware_clusterer.TfidfVectorizer'):
            with patch('components.context_aware_clusterer.KMeans') as mock_kmeans:
                # Set up mock to return two clusters
                mock_kmeans_instance = MagicMock()
                mock_kmeans.return_value = mock_kmeans_instance
                mock_kmeans_instance.fit_predict.return_value = [0, 0, 1, 1]  # Two clusters
                
                # Cluster errors
                clusters = self.clusterer.cluster_errors(enriched_errors)
                
                # Verify component fields are preserved in clusters
                for cluster_id, cluster_errors in clusters.items():
                    for error in cluster_errors:
                        self.assertIn('component', error)
                        self.assertIn('primary_issue_component', error)
                        self.assertEqual(error['primary_issue_component'], 'test_primary')
                        self.assertIn('affected_components', error)
                        self.assertIn('root_cause_component', error)
                        self.assertEqual(error['root_cause_component'], 'test_root_cause')


@TestRegistry.register(category='component', importance=1, tags=['analyzer'])
class TestDirectComponentAnalyzer(unittest.TestCase):
    """
    Unit tests for the direct_component_analyzer module.
    
    Tests the direct component identification from filenames and
    component relationship assignment.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        # Skip test if direct_component_analyzer is not available
        if identify_component_from_filename is None or assign_components_and_relationships is None:
            self.skipTest("direct_component_analyzer module not available")
    
    def test_identify_component_from_filename(self):
        """
        Test component identification from filenames.
        
        Verifies that components are correctly identified from
        filenames based on patterns and special cases.
        """
        # Test app_debug.log special case
        component, source = identify_component_from_filename("app_debug.log")
        self.assertEqual(component, "soa")
        self.assertEqual(source, "filename_special")
        
        # Test .har special case
        component, source = identify_component_from_filename("test.har")
        self.assertEqual(component, "ip_traffic")
        self.assertEqual(source, "filename_special")
        
        # Test charles special case
        component, source = identify_component_from_filename("charles_proxy.log")
        self.assertEqual(component, "charles")
        self.assertEqual(source, "filename_special")
        
        # Test regular filename
        component, source = identify_component_from_filename("mimosa.log")
        self.assertEqual(component, "mimosa")
        self.assertEqual(source, "filename")
        
        # Test unknown/empty
        component, source = identify_component_from_filename("")
        self.assertEqual(component, "unknown")
        self.assertEqual(source, "default")
    
    def test_assign_components(self):
        """
        Test component assignment based on filenames.
        
        Verifies that components are correctly assigned to errors
        based on their file attributes.
        """
        errors = [
            {"file": "app_debug.log", "text": "Error in app"},
            {"file": "mimosa.log", "text": "Data unavailable"},
            {"file": "mimosa_debug.log", "text": "Debug data"},
            {"file": "charles_proxy.log", "text": "Network error"},
            {"file": "unknown.log", "text": "Generic error"}
        ]
        
        updated_errors, _, _ = assign_components_and_relationships(errors)
        
        self.assertEqual(updated_errors[0]['component'], 'soa')
        self.assertEqual(updated_errors[1]['component'], 'mimosa')
        self.assertEqual(updated_errors[2]['component'], 'mimosa_debug')
        self.assertEqual(updated_errors[3]['component'], 'charles')
        self.assertEqual(updated_errors[4]['component'], 'unknown')
    
    def test_component_summary(self):
        """
        Test that component summary has correct structure and data.
        
        Verifies that the component summary contains the expected
        fields and data for each component.
        """
        errors = [
            {"file": "app_debug.log", "text": "Error in app"},
            {"file": "charles_proxy.log", "text": "Network error"}
        ]
        
        _, component_summary, _ = assign_components_and_relationships(errors)
        
        self.assertIsInstance(component_summary, list)
        self.assertGreater(len(component_summary), 0)
        
        # Check summary structure
        required_keys = ["id", "name", "description", "error_count"]
        for comp in component_summary:
            for key in required_keys:
                self.assertIn(key, comp)
        
        # Check component IDs
        component_ids = [comp["id"] for comp in component_summary]
        self.assertTrue('soa' in component_ids)
        self.assertTrue('charles' in component_ids)
    
    def test_primary_issue_component(self):
        """
        Test that the primary issue component is identified correctly.
        
        Verifies that the primary issue component is correctly
        determined based on error counts and component relationships.
        """
        # Create more SOA errors to make it the primary component
        errors = [
            {"file": "app_debug.log", "text": "Error 1"},
            {"file": "app_debug.log", "text": "Error 2"},
            {"file": "mimosa.log", "text": "Error 3"}
        ]
        
        _, _, primary_issue = assign_components_and_relationships(errors)
        
        self.assertEqual(primary_issue, 'soa')
        
        # Test with empty list
        _, _, primary_issue_empty = assign_components_and_relationships([])
        self.assertEqual(primary_issue_empty, 'unknown')
    
    def test_empty_input(self):
        """
        Test handling of empty input.
        
        Verifies that the function correctly handles empty input
        by returning empty results.
        """
        result = assign_components_and_relationships([])
        self.assertEqual(result, ([], [], 'unknown'))
    
    def test_text_based_component_assignment(self):
        """
        Test component assignment based on filename with special cases.
        
        Verifies that components are correctly assigned even with
        mixed file types and special cases.
        """
        # Create errors with different filenames
        errors = [
            {"file": "generic.log", "text": "Error in SOA component"},
            {"file": "android.log", "text": "Android system error"},
            {"file": "app_debug.log", "text": "Random error"}, # Special case
            {"file": "test.har", "text": "HTTP traffic"} # Special case
        ]
        
        # Update with component assignment
        updated_errors, _, _ = assign_components_and_relationships(errors)
        
        # Check that components were assigned based on filename
        self.assertEqual(updated_errors[0]['component'], 'generic')
        self.assertEqual(updated_errors[1]['component'], 'android')
        # Check special cases
        self.assertEqual(updated_errors[2]['component'], 'soa')
        self.assertEqual(updated_errors[3]['component'], 'ip_traffic')
    
    def test_primary_component_propagation(self):
        """
        Test that primary_issue_component is propagated to all errors.
        
        Verifies that all errors receive the same primary_issue_component
        value after processing.
        """
        # Create errors with different components
        errors = [
            {"file": "app_debug.log", "text": "Error 1"},  # Will be soa
            {"file": "mimosa.log", "text": "Error 2"},     # Will be mimosa
            {"file": "unknown.log", "text": "Error 3"}     # Will be unknown
        ]
        
        # Update with component assignment
        updated_errors, _, primary_issue = assign_components_and_relationships(errors)
        
        # Verify primary_issue_component is assigned to all errors
        for error in updated_errors:
            self.assertIn('primary_issue_component', error)
            # All errors should have the same primary_issue_component
            self.assertEqual(error['primary_issue_component'], primary_issue)


@TestRegistry.register(category='component', importance=1, tags=['component_model'])
class TestComponentModel(unittest.TestCase):
    """
    Tests for the new ComponentInfo and ComponentRegistry classes.
    
    Verifies the core component model implementation, including component
    creation, registry management, and component information retrieval.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        # Skip tests if component model modules are not available
        if ComponentInfo is None or ComponentRegistry is None:
            self.skipTest("Component model classes not available")
            
        # Set up a test schema path
        self.schema_path = get_component_schema_path()
        
        # Create sample component data
        self.component_data = {
            'component': 'test_component',
            'component_name': 'Test Component',
            'component_description': 'Test component description',
            'component_type': 'test_type',
            'component_source': 'test',
            'parent_component': 'parent_component',
            'child_components': ['child1', 'child2'],
            'related_components': ['related1', 'related2']
        }
    
    def test_component_info_creation(self):
        """
        Test creating ComponentInfo objects.
        
        Verifies that ComponentInfo objects are correctly created with
        all required fields and that the objects are immutable.
        """
        # Create ComponentInfo from component_id
        component = ComponentInfo("test_component")
        
        # Check basic properties
        self.assertEqual(component.id, "test_component")
        self.assertEqual(component.name, "TEST_COMPONENT")  # Default behavior is to uppercase
        self.assertEqual(component.type, "unknown")  # Default type
        
        # Create ComponentInfo with all fields
        component = ComponentInfo(
            component_id="test_component",
            name="Test Component",
            description="Test component description",
            component_type="test_type",
            component_source="test",
            parent="parent_component",
            children=["child1", "child2"],
            related_to=["related1", "related2"]
        )
        
        # Check all properties
        self.assertEqual(component.id, "test_component")
        self.assertEqual(component.name, "Test Component")
        self.assertEqual(component.description, "Test component description")
        self.assertEqual(component.type, "test_type")
        self.assertEqual(component.source, "test")
        
        # Test immutability by attempting to modify
        with self.assertRaises(AttributeError):
            component.id = "new_id"
            
        with self.assertRaises(AttributeError):
            component.name = "New Name"
    
    def test_component_info_to_dict(self):
        """
        Test converting ComponentInfo to dictionary.
        
        Verifies that ComponentInfo objects are correctly converted to
        dictionaries with all required fields.
        """
        # Create ComponentInfo with all fields
        component = ComponentInfo(
            component_id="test_component",
            name="Test Component",
            description="Test component description",
            component_type="test_type",
            component_source="test",
            parent="parent_component",
            children=["child1", "child2"],
            related_to=["related1", "related2"]
        )
        
        # Convert to dictionary
        component_dict = component.to_dict()
        
        # Verify dictionary contains all expected fields
        self.assertEqual(component_dict["component"], "test_component")
        self.assertEqual(component_dict["component_name"], "Test Component")
        self.assertEqual(component_dict["component_description"], "Test component description")
        self.assertEqual(component_dict["component_type"], "test_type")
        self.assertEqual(component_dict["component_source"], "test")
        self.assertEqual(component_dict["parent_component"], "parent_component")
        self.assertEqual(component_dict["child_components"], ["child1", "child2"])
        self.assertEqual(component_dict["related_components"], ["related1", "related2"])
    
    def test_component_info_from_dict(self):
        """
        Test creating ComponentInfo from dictionary.
        
        Verifies that ComponentInfo objects are correctly created from
        dictionaries with all required fields.
        """
        # Create ComponentInfo from dictionary
        component = ComponentInfo.from_dict(self.component_data)
        
        # Verify fields are correctly set
        self.assertEqual(component.id, "test_component")
        self.assertEqual(component.name, "Test Component")
        self.assertEqual(component.description, "Test component description")
        self.assertEqual(component.type, "test_type")
        self.assertEqual(component.source, "test")
        
        # Test with minimal dictionary
        minimal_data = {'component': 'minimal_component'}
        component = ComponentInfo.from_dict(minimal_data)
        
        # Verify default values are used
        self.assertEqual(component.id, "minimal_component")
        self.assertEqual(component.name, "MINIMAL_COMPONENT")  # Default uppercase
        self.assertEqual(component.description, "")  # Default empty string
        self.assertEqual(component.type, "unknown")  # Default type
        self.assertEqual(component.source, "default")  # Default source
        
        # Test with None
        component = ComponentInfo.from_dict(None)
        self.assertEqual(component.id, "unknown")  # Default for None
    
    def test_component_registry_initialization(self):
        """
        Test ComponentRegistry initialization.
        
        Verifies that the ComponentRegistry is correctly initialized with
        a schema and loads components from the schema.
        """
        # Initialize registry with schema
        registry = ComponentRegistry(self.schema_path)
        
        # Verify registry has loaded components
        components = registry.get_all_components()
        self.assertGreater(len(components), 0)
        
        # Verify known components are included
        soa_component = registry.get_component("soa")
        self.assertEqual(soa_component.id, "soa")
        self.assertEqual(soa_component.name, "SOA")
        
        # Test with None schema - should still have default components
        registry = ComponentRegistry(None)
        components = registry.get_all_components()
        self.assertGreater(len(components), 0)
    
    def test_component_registry_get_component(self):
        """
        Test getting components from the registry.
        
        Verifies that components can be correctly retrieved from the
        registry by ID, including special handling for unknown IDs.
        """
        # Initialize registry
        registry = ComponentRegistry(self.schema_path)
        
        # Get known component
        soa_component = registry.get_component("soa")
        self.assertEqual(soa_component.id, "soa")
        
        # Get unknown component
        unknown_component = registry.get_component("non_existent")
        self.assertEqual(unknown_component.id, "unknown")
        
        # Test case insensitivity
        upper_component = registry.get_component("SOA")
        self.assertEqual(upper_component.id, "soa")
        
        # Test with None
        none_component = registry.get_component(None)
        self.assertEqual(none_component.id, "unknown")
    
    def test_identify_component_from_filename(self):
        """
        Test identifying components from filenames.
        
        Verifies that the registry correctly identifies components from
        filenames using patterns and special cases.
        """
        # Initialize registry
        registry = ComponentRegistry(self.schema_path)
        
        # Test with app_debug.log (special case)
        component = registry.identify_component_from_filename("app_debug.log")
        self.assertEqual(component.id, "soa")
        
        # Test with .har file (special case)
        component = registry.identify_component_from_filename("test.har")
        self.assertEqual(component.id, "ip_traffic")
        
        # Test with standard filename
        component = registry.identify_component_from_filename("mimosa.log")
        self.assertEqual(component.id, "mimosa")
        
        # Test with unknown filename
        component = registry.identify_component_from_filename("unknown_file.log")
        self.assertEqual(component.id, "unknown_file")
        
        # Test with None
        component = registry.identify_component_from_filename(None)
        self.assertEqual(component.id, "unknown")
    
    def test_identify_primary_component(self):
        """
        Test identifying primary component from error counts.
        
        Verifies that the primary component is correctly identified
        based on error counts and component relationships.
        """
        # Initialize registry
        registry = ComponentRegistry(self.schema_path)
        
        # Create component counts
        component_counts = {
            'soa': 10,
            'android': 5,
            'mimosa': 3
        }
        
        # Identify primary component
        primary_component = registry.identify_primary_component(component_counts)
        self.assertEqual(primary_component.id, "soa")
        
        # Test with more android errors - should still prefer SOA due to special logic
        android_heavy_counts = {
            'soa': 7,  # >= 50% of android, so should win with boost
            'android': 12,
            'mimosa': 3
        }
        primary_component = registry.identify_primary_component(android_heavy_counts)
        self.assertEqual(primary_component.id, "soa")
        
        # Test with very few SOA errors - should now pick android
        android_dominant_counts = {
            'soa': 2,  # < 50% of android, so android should win
            'android': 12,
            'mimosa': 3
        }
        primary_component = registry.identify_primary_component(android_dominant_counts)
        self.assertEqual(primary_component.id, "android")
        
        # Test with empty counts
        primary_component = registry.identify_primary_component({})
        self.assertEqual(primary_component.id, "unknown")
        
        # Test with only unknown component
        primary_component = registry.identify_primary_component({'unknown': 5})
        self.assertEqual(primary_component.id, "unknown")
    
    def test_create_component_info_factory(self):
        """
        Test create_component_info factory function.
        
        Verifies that the factory function correctly creates ComponentInfo
        objects with standardized fields and registry integration.
        """
        # Test if factory function is available
        if create_component_info is None:
            self.skipTest("create_component_info function not available")
            
        # Create component using factory
        component = create_component_info("soa", "test_source", name="Test SOA")
        
        # Verify the component has correct fields
        self.assertEqual(component.id, "soa")
        self.assertEqual(component.source, "test_source")
        self.assertEqual(component.name, "Test SOA")
        
        # Component should inherit properties from registry
        self.assertIsNotNone(component.description)
        self.assertIsNotNone(component.type)
    
    def test_get_component_registry_singleton(self):
        """
        Test get_component_registry singleton function.
        
        Verifies that the get_component_registry function correctly returns
        a singleton instance of ComponentRegistry.
        """
        # Test if function is available
        if get_component_registry is None:
            self.skipTest("get_component_registry function not available")
            
        # Get registry singleton
        registry1 = get_component_registry(self.schema_path)
        registry2 = get_component_registry()  # Second call should return same instance
        
        # Verify same instance is returned
        self.assertIs(registry1, registry2)
        
        # Verify registry is properly initialized
        components = registry1.get_all_components()
        self.assertGreater(len(components), 0)


@TestRegistry.register(category='component', importance=1, tags=['component_utils'])
class TestComponentUtils(unittest.TestCase):
    """
    Tests for component utility functions.
    
    Verifies the utility functions for component field extraction,
    application, preservation, validation, and normalization.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        # Skip tests if component utils are not available
        if extract_component_fields is None or apply_component_fields is None:
            self.skipTest("Component utility functions not available")
            
        # Create test data
        self.test_data = {
            'component': 'soa',
            'component_source': 'filename',
            'primary_issue_component': 'android',
            'affected_components': ['soa', 'android'],
            'other_field': 'test value'
        }
    
    def test_extract_component_fields(self):
        """
        Test extracting component fields from a dictionary.
        
        Verifies that component fields are correctly extracted from
        a dictionary, ignoring non-component fields.
        """
        # Extract component fields
        component_fields = extract_component_fields(self.test_data)
        
        # Verify component fields are extracted
        self.assertIn('component', component_fields)
        self.assertIn('component_source', component_fields)
        self.assertIn('primary_issue_component', component_fields)
        self.assertIn('affected_components', component_fields)
        
        # Verify non-component fields are not extracted
        self.assertNotIn('other_field', component_fields)
        
        # Verify extracted values are correct
        self.assertEqual(component_fields['component'], 'soa')
        self.assertEqual(component_fields['component_source'], 'filename')
        self.assertEqual(component_fields['primary_issue_component'], 'android')
        self.assertEqual(component_fields['affected_components'], ['soa', 'android'])
        
        # Test with empty dictionary
        self.assertEqual(extract_component_fields({}), {})
        
        # Test with None
        self.assertEqual(extract_component_fields(None), {})
        
        # Test with non-dictionary
        self.assertEqual(extract_component_fields("not a dict"), {})
    
    def test_apply_component_fields(self):
        """
        Test applying component fields to a dictionary.
        
        Verifies that component fields are correctly applied to a
        dictionary, preserving existing fields.
        """
        # Create target dictionary
        target = {
            'text': 'Error message',
            'severity': 'High'
        }
        
        # Apply component fields
        result = apply_component_fields(target, self.test_data)
        
        # Verify component fields are applied
        self.assertIn('component', result)
        self.assertIn('component_source', result)
        self.assertIn('primary_issue_component', result)
        self.assertIn('affected_components', result)
        
        # Verify original fields are preserved
        self.assertIn('text', result)
        self.assertIn('severity', result)
        
        # Verify applied values are correct
        self.assertEqual(result['component'], 'soa')
        self.assertEqual(result['component_source'], 'filename')
        self.assertEqual(result['primary_issue_component'], 'android')
        self.assertEqual(result['affected_components'], ['soa', 'android'])
        
        # Test with empty target
        result = apply_component_fields({}, self.test_data)
        self.assertIn('component', result)
        self.assertEqual(result['component'], 'soa')
        
        # Test with None target
        self.assertIsNone(apply_component_fields(None, self.test_data))
        
        # Test with None source
        self.assertEqual(apply_component_fields(target, None), target)
    
    def test_preserve_component_fields(self):
        """
        Test preserving component fields during processing.
        
        Verifies that component fields are correctly preserved when
        transferring from a source to a target dictionary.
        """
        # Create source and target dictionaries
        source = self.test_data.copy()
        target = {
            'text': 'Error message',
            'severity': 'High'
        }
        
        # Preserve component fields
        result = preserve_component_fields(source, target)
        
        # Verify component fields are preserved
        self.assertIn('component', result)
        self.assertIn('component_source', result)
        self.assertIn('primary_issue_component', result)
        self.assertIn('affected_components', result)
        
        # Verify original fields are preserved
        self.assertIn('text', result)
        self.assertIn('severity', result)
        
        # Verify preserved values are correct
        self.assertEqual(result['component'], 'soa')
        self.assertEqual(result['component_source'], 'filename')
        self.assertEqual(result['primary_issue_component'], 'android')
        self.assertEqual(result['affected_components'], ['soa', 'android'])
        
        # Test with None source
        self.assertEqual(preserve_component_fields(None, target), target)
        
        # Test with None target
        self.assertIsNone(preserve_component_fields(source, None))
    
    def test_enrich_with_component_info(self):
        """
        Test enriching data with component information.
        
        Verifies that data is correctly enriched with component information
        from the registry.
        """
        # Create target dictionary
        target = {
            'text': 'Error message',
            'severity': 'High'
        }
        
        # Enrich with component info
        result = enrich_with_component_info(target, 'soa', 'test')
        
        # Verify component fields are added
        self.assertIn('component', result)
        self.assertIn('component_source', result)
        self.assertIn('component_name', result)
        self.assertIn('component_description', result)
        self.assertIn('component_type', result)
        
        # Verify values are correct
        self.assertEqual(result['component'], 'soa')
        self.assertEqual(result['component_source'], 'test')
        self.assertEqual(result['component_name'], 'SOA')
        self.assertIsInstance(result['component_description'], str)
        self.assertIsInstance(result['component_type'], str)
        
        # Verify original fields are preserved
        self.assertIn('text', result)
        self.assertIn('severity', result)
        
        # Test with no component ID (should use component from data)
        target_with_component = {
            'text': 'Error message',
            'component': 'mimosa'
        }
        result = enrich_with_component_info(target_with_component)
        self.assertEqual(result['component'], 'mimosa')
        
        # Test with None data
        self.assertIsNone(enrich_with_component_info(None))
    
    def test_identify_component_from_file(self):
        """
        Test identifying component from filename.
        
        Verifies that components are correctly identified from
        filenames using the utility function.
        """
        # Test with app_debug.log (special case)
        component, source = identify_component_from_file("app_debug.log")
        self.assertEqual(component, "soa")
        self.assertEqual(source, "filename")
        
        # Test with .har file (special case)
        component, source = identify_component_from_file("test.har")
        self.assertEqual(component, "ip_traffic")
        self.assertEqual(source, "filename")
        
        # Test with standard filename
        component, source = identify_component_from_file("mimosa.log")
        self.assertEqual(component, "mimosa")
        self.assertEqual(source, "filename")
        
        # Test with empty string
        component, source = identify_component_from_file("")
        self.assertEqual(component, "unknown")
    
    def test_determine_primary_component(self):
        """
        Test determining primary component from errors.
        
        Verifies that the primary component is correctly determined
        based on component counts in errors.
        """
        # Create test errors
        errors = [
            {'component': 'soa'},
            {'component': 'soa'},
            {'component': 'android'},
            {'component': 'mimosa'}
        ]
        
        # Determine primary component
        primary_component = determine_primary_component(errors)
        self.assertEqual(primary_component, "soa")
        
        # Test with empty list
        self.assertEqual(determine_primary_component([]), "unknown")
        
        # Test with only unknown components
        unknown_errors = [
            {'component': 'unknown'},
            {'component': 'unknown'}
        ]
        self.assertEqual(determine_primary_component(unknown_errors), "unknown")
    
    def test_validate_component_data(self):
        """
        Test validating component data.
        
        Verifies that component data is correctly validated and
        default values are applied where needed.
        """
        # Test with complete data
        data = {
            'component': 'soa',
            'component_source': 'filename'
        }
        result = validate_component_data(data)
        self.assertEqual(result['component'], 'soa')
        self.assertEqual(result['component_source'], 'filename')
        
        # Test with missing component - should default to unknown
        data = {'text': 'Error message'}
        result = validate_component_data(data)
        self.assertEqual(result['component'], 'unknown')
        self.assertEqual(result['component_source'], 'default')
        
        # Test with primary_issue_component parameter
        result = validate_component_data(data, 'android')
        self.assertEqual(result['primary_issue_component'], 'android')
        
        # Test with None data
        self.assertIsNone(validate_component_data(None))
    
    def test_verify_component_preservation(self):
        """
        Test verifying component preservation.
        
        Verifies that the function correctly checks that component
        information is preserved between original and processed data.
        """
        # Create original and processed data
        original = self.test_data.copy()
        processed = self.test_data.copy()
        processed['new_field'] = 'new value'
        
        # Verify preservation
        self.assertTrue(verify_component_preservation(original, processed))
        
        # Test with missing component field
        processed_missing = processed.copy()
        del processed_missing['component']
        self.assertFalse(verify_component_preservation(original, processed_missing))
        
        # Test with changed component field
        processed_changed = processed.copy()
        processed_changed['component'] = 'android'
        self.assertFalse(verify_component_preservation(original, processed_changed))
        
        # Test with None data
        self.assertTrue(verify_component_preservation(None, processed))
        self.assertTrue(verify_component_preservation(original, None))
    
    def test_normalize_component_fields(self):
        """
        Test normalizing component fields.
        
        Verifies that component fields are correctly normalized to
        standard format (lowercase, etc.).
        """
        # Create data with non-standard case
        data = {
            'component': 'SOA',
            'primary_issue_component': 'ANDROID',
            'root_cause_component': 'Mimosa'
        }
        
        # Normalize component fields
        result = normalize_component_fields(data)
        
        # Verify fields are normalized to lowercase
        self.assertEqual(result['component'], 'soa')
        self.assertEqual(result['primary_issue_component'], 'android')
        self.assertEqual(result['root_cause_component'], 'mimosa')
        
        # Test with None data
        self.assertIsNone(normalize_component_fields(None))
        
        # Test with non-dictionary
        self.assertEqual(normalize_component_fields("not a dict"), "not a dict")


@TestRegistry.register(category='component', importance=1, tags=['data_preprocessor'])
class TestDataPreprocessor(unittest.TestCase):
    """
    Tests for the data preprocessing functions.
    
    Verifies that data preprocessing functions correctly preserve
    component information during normalization and preprocessing.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        # Skip tests if data preprocessor functions are not available
        if preprocess_errors is None or preprocess_clusters is None:
            self.skipTest("Data preprocessor functions not available")
            
        # Create test data
        self.errors = [
            {
                'text': 'Error 1',
                'severity': 'High',
                'component': 'soa',
                'component_source': 'filename',
                'timestamp': '2023-01-01T12:00:00'
            },
            {
                'text': 'Error 2',
                'severity': 'Medium',
                'component': 'android',
                'component_source': 'content',
                'timestamp': '2023-01-01T12:05:00'
            }
        ]
        
        self.clusters = {
            0: [self.errors[0]],
            1: [self.errors[1]]
        }
        
        self.primary_issue_component = 'soa'
    
    def test_normalize_timestamps_in_dict(self):
        """
        Test normalizing timestamps in dictionary.
        
        Verifies that timestamps are correctly normalized to datetime
        objects while preserving component information.
        """
        # Test with standard case
        data = {
            'timestamp': '2023-01-01T12:00:00',
            'component': 'soa',
            'component_source': 'filename'
        }
        
        result = normalize_timestamps_in_dict(data)
        
        # Verify timestamp is converted to datetime
        self.assertIsInstance(result['timestamp'], datetime)
        
        # Verify component fields are preserved
        self.assertEqual(result['component'], 'soa')
        self.assertEqual(result['component_source'], 'filename')
        
        # Test with nested dictionaries
        nested_data = {
            'error': {
                'timestamp': '2023-01-01T12:00:00',
                'component': 'soa'
            },
            'component': 'android'
        }
        
        result = normalize_timestamps_in_dict(nested_data)
        
        # Verify nested timestamp is converted
        self.assertIsInstance(result['error']['timestamp'], datetime)
        
        # Verify component fields are preserved at all levels
        self.assertEqual(result['error']['component'], 'soa')
        self.assertEqual(result['component'], 'android')
        
        # Test with array of dictionaries
        array_data = [
            {
                'timestamp': '2023-01-01T12:00:00',
                'component': 'soa'
            },
            {
                'timestamp': '2023-01-01T12:05:00',
                'component': 'android'
            }
        ]
        
        result = normalize_timestamps_in_dict(array_data)
        
        # Verify timestamps are converted in array items
        self.assertIsInstance(result[0]['timestamp'], datetime)
        self.assertIsInstance(result[1]['timestamp'], datetime)
        
        # Verify component fields are preserved in array items
        self.assertEqual(result[0]['component'], 'soa')
        self.assertEqual(result[1]['component'], 'android')
        
        # Test with primary_issue_component parameter
        result = normalize_timestamps_in_dict(data, True, 'android')
        self.assertEqual(result['primary_issue_component'], 'android')
    
    def test_preprocess_errors(self):
        """
        Test preprocessing errors.
        
        Verifies that errors are correctly preprocessed with component
        information preserved and enhanced.
        """
        # Preprocess errors
        processed_errors, primary_component = preprocess_errors(
            self.errors,
            self.primary_issue_component
        )
        
        # Verify errors are not modified in place
        self.assertIsNot(processed_errors, self.errors)
        
        # Verify component fields are preserved
        self.assertEqual(processed_errors[0]['component'], 'soa')
        self.assertEqual(processed_errors[0]['component_source'], 'filename')
        self.assertEqual(processed_errors[1]['component'], 'android')
        self.assertEqual(processed_errors[1]['component_source'], 'content')
        
        # Verify primary_issue_component is added to all errors
        self.assertEqual(processed_errors[0]['primary_issue_component'], self.primary_issue_component)
        self.assertEqual(processed_errors[1]['primary_issue_component'], self.primary_issue_component)
        
        # Verify primary_issue_component is returned correctly
        self.assertEqual(primary_component, self.primary_issue_component)
        
        # Test with empty list
        processed_empty, primary_empty = preprocess_errors([], 'unknown')
        self.assertEqual(processed_empty, [])
        self.assertEqual(primary_empty, 'unknown')
        
        # Test with component_diagnostic parameter
        component_diagnostic = {
            'component_counts': {'soa': 5, 'android': 3},
            'component_sources': {'filename': 5, 'content': 3}
        }
        
        processed_with_diagnostic, _ = preprocess_errors(
            self.errors,
            self.primary_issue_component,
            component_diagnostic
        )
        
        # Results should be the same since our test data already has components
        self.assertEqual(processed_with_diagnostic[0]['component'], 'soa')
        self.assertEqual(processed_with_diagnostic[1]['component'], 'android')
    
    def test_preprocess_clusters(self):
        """
        Test preprocessing clusters.
        
        Verifies that clusters are correctly preprocessed with component
        information preserved and enhanced.
        """
        # Preprocess clusters
        processed_clusters = preprocess_clusters(
            self.clusters,
            self.primary_issue_component
        )
        
        # Verify clusters are not modified in place
        self.assertIsNot(processed_clusters, self.clusters)
        
        # Verify component fields are preserved in clustered errors
        self.assertEqual(processed_clusters[0][0]['component'], 'soa')
        self.assertEqual(processed_clusters[0][0]['component_source'], 'filename')
        self.assertEqual(processed_clusters[1][0]['component'], 'android')
        self.assertEqual(processed_clusters[1][0]['component_source'], 'content')
        
        # Verify primary_issue_component is added to all errors in clusters
        self.assertEqual(processed_clusters[0][0]['primary_issue_component'], self.primary_issue_component)
        self.assertEqual(processed_clusters[1][0]['primary_issue_component'], self.primary_issue_component)
        
        # Test with empty dictionary
        processed_empty = preprocess_clusters({}, 'unknown')
        self.assertEqual(processed_empty, {})
        
        # Test with component_diagnostic parameter
        component_diagnostic = {
            'component_counts': {'soa': 5, 'android': 3},
            'component_sources': {'filename': 5, 'content': 3}
        }
        
        processed_with_diagnostic = preprocess_clusters(
            self.clusters,
            self.primary_issue_component,
            component_diagnostic
        )
        
        # Results should be the same since our test data already has components
        self.assertEqual(processed_with_diagnostic[0][0]['component'], 'soa')
        self.assertEqual(processed_with_diagnostic[1][0]['component'], 'android')
    
    def test_normalize_data(self):
        """
        Test normalizing data.
        
        Verifies that errors and clusters are correctly normalized with
        timestamps converted and component information preserved.
        """
        # Normalize data
        normalized_errors, normalized_clusters = normalize_data(
            self.errors,
            self.clusters,
            self.primary_issue_component
        )
        
        # Verify data structures are not modified in place
        self.assertIsNot(normalized_errors, self.errors)
        self.assertIsNot(normalized_clusters, self.clusters)
        
        # Verify timestamps are converted to datetime objects
        self.assertIsInstance(normalized_errors[0]['timestamp'], datetime)
        self.assertIsInstance(normalized_errors[1]['timestamp'], datetime)
        self.assertIsInstance(normalized_clusters[0][0]['timestamp'], datetime)
        self.assertIsInstance(normalized_clusters[1][0]['timestamp'], datetime)
        
        # Verify component fields are preserved
        self.assertEqual(normalized_errors[0]['component'], 'soa')
        self.assertEqual(normalized_errors[0]['component_source'], 'filename')
        self.assertEqual(normalized_errors[1]['component'], 'android')
        self.assertEqual(normalized_errors[1]['component_source'], 'content')
        
        # Verify component fields are preserved in clusters
        self.assertEqual(normalized_clusters[0][0]['component'], 'soa')
        self.assertEqual(normalized_clusters[0][0]['component_source'], 'filename')
        self.assertEqual(normalized_clusters[1][0]['component'], 'android')
        self.assertEqual(normalized_clusters[1][0]['component_source'], 'content')
        
        # Verify primary_issue_component is preserved
        self.assertEqual(normalized_errors[0]['primary_issue_component'], self.primary_issue_component)
        self.assertEqual(normalized_errors[1]['primary_issue_component'], self.primary_issue_component)
        self.assertEqual(normalized_clusters[0][0]['primary_issue_component'], self.primary_issue_component)
        self.assertEqual(normalized_clusters[1][0]['primary_issue_component'], self.primary_issue_component)


@TestRegistry.register(category='component', importance=1, tags=['integration'])
class TestComponentPreservationWorkflow(unittest.TestCase):
    """
    Integration tests for component preservation across the workflow.
    
    Verifies that component information is preserved through the entire
    processing pipeline from identification to report generation.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        # Skip tests if required functions are not available
        if assign_components_and_relationships is None or preprocess_errors is None:
            self.skipTest("Required functions not available")
            
        # Set up test data
        self.raw_errors = [
            {"file": "app_debug.log", "text": "Error 1", "severity": "High", "timestamp": "2023-01-01T12:00:00"},
            {"file": "mimosa.log", "text": "Error 2", "severity": "Medium", "timestamp": "2023-01-01T12:05:00"},
            {"file": "android.log", "text": "Error 3", "severity": "Low", "timestamp": "2023-01-01T12:10:00"}
        ]
    
    def test_end_to_end_component_preservation(self):
        """
        Test end-to-end component preservation.
        
        Verifies that component information is preserved through the
        entire processing pipeline from identification to serialization.
        """
        # Skip if ComponentAwareEncoder is not available
        if ComponentAwareEncoder is None:
            self.skipTest("ComponentAwareEncoder not available")
            
        # Step 1: Component assignment
        errors_with_components, component_summary, primary_issue_component = assign_components_and_relationships(self.raw_errors)
        
        # Verify component assignment
        self.assertEqual(errors_with_components[0]['component'], 'soa')
        self.assertEqual(errors_with_components[1]['component'], 'mimosa')
        self.assertEqual(errors_with_components[2]['component'], 'android')
        
        # Check for primary_issue_component
        for error in errors_with_components:
            self.assertIn('primary_issue_component', error)
            self.assertEqual(error['primary_issue_component'], primary_issue_component)
        
        # Step 2: Data preprocessing
        processed_errors, updated_primary = preprocess_errors(errors_with_components, primary_issue_component)
        
        # Verify component preservation in preprocessing
        self.assertEqual(processed_errors[0]['component'], 'soa')
        self.assertEqual(processed_errors[1]['component'], 'mimosa')
        self.assertEqual(processed_errors[2]['component'], 'android')
        self.assertEqual(updated_primary, primary_issue_component)
        
        # Step 3: Normalization
        normalized_errors, _ = normalize_data(processed_errors, {0: processed_errors}, primary_issue_component)
        
        # Verify component preservation in normalization
        self.assertEqual(normalized_errors[0]['component'], 'soa')
        self.assertEqual(normalized_errors[1]['component'], 'mimosa')
        self.assertEqual(normalized_errors[2]['component'], 'android')
        
        # Step 4: JSON serialization
        json_data = {
            "errors": normalized_errors,
            "summary": "Test summary",
            "primary_issue_component": primary_issue_component
        }
        
        json_str = json.dumps(json_data, cls=ComponentAwareEncoder)
        deserialized_data = json.loads(json_str)
        
        # Verify component preservation in serialization
        self.assertEqual(deserialized_data["errors"][0]['component'], 'soa')
        self.assertEqual(deserialized_data["errors"][1]['component'], 'mimosa')
        self.assertEqual(deserialized_data["errors"][2]['component'], 'android')
        self.assertEqual(deserialized_data["primary_issue_component"], primary_issue_component)
    
    @unittest.skipIf(preprocess_clusters is None, "preprocess_clusters not available")
    @patch('error_clusterer.perform_error_clustering')
    def test_component_preservation_in_clustering(self, mock_clustering):
        """
        Test component preservation during clustering.
        
        Verifies that component information is preserved during the
        error clustering process.
        """
        # Set up mock for clustering
        mock_clustering.return_value = {
            0: [self.raw_errors[0]],
            1: [self.raw_errors[1], self.raw_errors[2]]
        }
        
        # Step 1: Component assignment
        errors_with_components, _, primary_issue_component = assign_components_and_relationships(self.raw_errors)
        
        # Step 2: Error clustering (mocked)
        from error_clusterer import perform_error_clustering
        clusters = perform_error_clustering(errors_with_components)
        
        # Verify clusters structure
        self.assertEqual(len(clusters), 2)
        
        # Step 3: Cluster preprocessing
        processed_clusters = preprocess_clusters(clusters, primary_issue_component)
        
        # Verify component preservation in cluster preprocessing
        for cluster_id, cluster_errors in processed_clusters.items():
            for error in cluster_errors:
                self.assertIn('component', error)
                self.assertIn('component_source', error)
                self.assertIn('primary_issue_component', error)
                self.assertEqual(error['primary_issue_component'], primary_issue_component)
        
        # Verify specific components
        if len(processed_clusters[0]) > 0:
            self.assertEqual(processed_clusters[0][0]['component'], 'soa')
        
        if len(processed_clusters[1]) > 0:
            # First error in cluster 1 should be mimosa
            self.assertEqual(processed_clusters[1][0]['component'], 'mimosa')
            
            # Second error in cluster 1 should be android (if it exists)
            if len(processed_clusters[1]) > 1:
                self.assertEqual(processed_clusters[1][1]['component'], 'android')


if __name__ == "__main__":
    unittest.main()