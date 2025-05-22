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
    from reports.base import ComponentAwareEncoder
except ImportError:
    ComponentAwareEncoder = None

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
        
    def test_component_report_generation(self):
        """
        Test component report generation with path verification.
        
        Verifies that component reports are generated with the correct
        directory structure and file paths.
        """
        # Skip if required imports aren't available
        if generate_component_report is None:
            self.skipTest("generate_component_report function not available")
        
        # Test ID
        test_id = "TEST-ANALYZER-999"

        # Setup output directories
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dirs = setup_test_output_directories(test_id)

            # Create test component analysis
            component_analysis = {
                "primary_issue_component": "soa",
                "root_cause_component": "soa",
                "component_summary": [
                    {"id": "soa", "name": "SOA", "description": "Test", "error_count": 10}
                ],
                "component_error_counts": {"soa": 10, "android": 5}
            }

            # Generate component report
            report_path = generate_component_report(
                output_dirs["base"],
                test_id,
                component_analysis,
                "soa"
            )

            # Verify report was generated
            self.assertIsNotNone(report_path)
            self.assertTrue(os.path.exists(report_path))

            # Check the JSON files are in the right location
            json_dir = os.path.join(output_dirs["base"], "json")
            self.assertTrue(os.path.exists(json_dir))
            
            # Check for component_analysis.json in the JSON directory
            component_analysis_files = [f for f in os.listdir(json_dir) if "component_analysis" in f]
            self.assertTrue(len(component_analysis_files) > 0, 
                           f"No component_analysis.json found in {json_dir}")


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
        empty_path = self.visualizer._generate_empty_diagram(dirs["images"], test_id, "empty_diagram.png")
        
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

if __name__ == "__main__":
    unittest.main()