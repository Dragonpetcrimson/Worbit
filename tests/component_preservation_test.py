import unittest
import copy
import json
import tempfile
import os
import shutil
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
    verify_component_preservation
)
from reports.base import ComponentAwareEncoder

# Import the new helper function from json_utils
from json_utils import serialize_with_component_awareness, serialize_to_json_file

class ComponentPreservationTest(unittest.TestCase):
    """Tests for component information preservation."""
    
    def setUp(self):
        """Set up test environment."""
        # Create test data
        self.test_error = {
            "text": "Error message",
            "file": "app_debug.log",
            "severity": "High",
            "component": "soa",
            "component_source": "filename",
            "primary_issue_component": "soa"
        }
        
        # Create test registry
        self.registry = get_component_registry()
        
        # Create temp directory for output
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test environment."""
        # Remove temp directory
        shutil.rmtree(self.temp_dir)
    
    def test_component_info_creation(self):
        """Test ComponentInfo creation and immutability."""
        # Create component info
        component = create_component_info("soa", "test")
        
        # Test properties
        self.assertEqual(component.id, "soa")
        self.assertEqual(component.source, "test")
        
        # Test immutability
        with self.assertRaises(AttributeError):
            component.id = "android"
            
        # Test to_dict
        data = component.to_dict()
        self.assertEqual(data["component"], "soa")
        self.assertEqual(data["component_source"], "test")
        
        # Test from_dict
        component2 = ComponentInfo.from_dict(data)
        self.assertEqual(component2.id, "soa")
        self.assertEqual(component2.source, "test")
    
    def test_component_field_extraction(self):
        """Test component field extraction."""
        # Extract fields
        fields = extract_component_fields(self.test_error)
        
        # Verify extraction
        self.assertEqual(fields["component"], "soa")
        self.assertEqual(fields["component_source"], "filename")
        self.assertEqual(fields["primary_issue_component"], "soa")
        
        # Test with empty data
        self.assertEqual(extract_component_fields(None), {})
        self.assertEqual(extract_component_fields({}), {})
        
        # Test with non-dictionary
        self.assertEqual(extract_component_fields("not a dict"), {})
    
    def test_component_field_application(self):
        """Test component field application."""
        # Create test data
        source = {
            "component": "soa",
            "component_source": "filename",
            "primary_issue_component": "soa"
        }
        
        target = {
            "text": "Error message",
            "severity": "High"
        }
        
        # Apply fields
        result = apply_component_fields(target, source)
        
        # Verify application
        self.assertEqual(result["component"], "soa")
        self.assertEqual(result["component_source"], "filename")
        self.assertEqual(result["primary_issue_component"], "soa")
        self.assertEqual(result["text"], "Error message")
        
        # Test with empty data
        self.assertEqual(apply_component_fields(None, source), None)
        self.assertEqual(apply_component_fields({}, source), {"component": "soa", "component_source": "filename", "primary_issue_component": "soa"})
        self.assertEqual(apply_component_fields(target, None), target)
        self.assertEqual(apply_component_fields(target, {}), target)
    
    def test_component_field_preservation(self):
        """Test component field preservation."""
        # Create test data
        source = copy.deepcopy(self.test_error)
        target = {
            "text": "Processed error message",
            "severity": "Medium"
        }
        
        # Preserve fields
        result = preserve_component_fields(source, target)
        
        # Verify preservation
        self.assertEqual(result["component"], "soa")
        self.assertEqual(result["component_source"], "filename")
        self.assertEqual(result["primary_issue_component"], "soa")
        self.assertEqual(result["text"], "Processed error message")
        
        # Test with empty data
        self.assertEqual(preserve_component_fields(None, target), target)
        self.assertEqual(preserve_component_fields(source, None), None)
    
    def test_component_verification(self):
        """Test component preservation verification."""
        # Create test data
        original = copy.deepcopy(self.test_error)
        modified = copy.deepcopy(self.test_error)
        
        # Verify identical data
        self.assertTrue(verify_component_preservation(original, modified))
        
        # Modify non-component field
        modified["text"] = "Changed text"
        self.assertTrue(verify_component_preservation(original, modified))
        
        # Modify component field
        modified["component"] = "android"
        self.assertFalse(verify_component_preservation(original, modified))
        
        # Test with empty data
        self.assertTrue(verify_component_preservation(None, modified))
        self.assertTrue(verify_component_preservation(original, None))
    
    def test_json_serialization_preservation(self):
        """Test component preservation during JSON serialization."""
        # Create test data
        data = {
            "test_id": "SXM-123456",
            "errors": [self.test_error],
            "clusters": {
                "0": [self.test_error]
            },
            "primary_issue_component": "soa"
        }
        
        # Create a temporary file for testing
        temp_file = os.path.join(self.temp_dir, "test.json")
        
        # Use the new helper function for serialization
        serialize_to_json_file(data, temp_file, primary_issue_component="soa")
            
        # Read back and verify
        with open(temp_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
            
        # Verify component preservation
        self.assertEqual(loaded_data["primary_issue_component"], "soa")
        self.assertEqual(loaded_data["errors"][0]["component"], "soa")
        self.assertEqual(loaded_data["errors"][0]["component_source"], "filename")
        self.assertEqual(loaded_data["clusters"]["0"][0]["component"], "soa")
    
    def test_nested_structure_preservation(self):
        """Test component preservation in nested structures."""
        # Create test data with nesting
        nested_data = {
            "test_id": "SXM-123456",
            "errors": [self.test_error],
            "clusters": {
                "0": [self.test_error]
            },
            "nested": {
                "deeper": {
                    "component": "mimosa",
                    "component_source": "special"
                }
            },
            "primary_issue_component": "soa"
        }
        
        # Create a temporary file for testing
        temp_file = os.path.join(self.temp_dir, "nested.json")
        
        # Use the new helper function for serialization
        serialize_to_json_file(nested_data, temp_file, primary_issue_component="soa")
            
        # Read back and verify
        with open(temp_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
            
        # Verify component preservation in nested structures
        self.assertEqual(loaded_data["nested"]["deeper"]["component"], "mimosa")
        self.assertEqual(loaded_data["nested"]["deeper"]["component_source"], "special")
        
    def test_component_registry_functionality(self):
        """Test ComponentRegistry functionality."""
        # Test getting component by ID
        component = self.registry.get_component("soa")
        self.assertEqual(component.id, "soa")
        self.assertEqual(component.name, "SOA")
        
        # Test getting unknown component
        unknown = self.registry.get_component("nonexistent")
        self.assertEqual(unknown.id, "unknown")
        
        # Test filename-based component identification
        component_id = self.registry.identify_component_from_filename("app_debug.log")
        self.assertEqual(component_id, "soa")
        
        # Test primary component identification with simulated error counts
        component_counts = {
            "soa": 5,
            "android": 3,
            "unknown": 1
        }
        primary = self.registry.identify_primary_component(component_counts)
        self.assertEqual(primary, "soa")
        
    def test_primary_component_propagation(self):
        """Test primary_issue_component propagation across structures."""
        # Create test structure
        errors = [
            {"text": "Error 1", "component": "soa"},
            {"text": "Error 2", "component": "android"},
            {"text": "Error 3", "component": "mimosa"}
        ]
        
        # Apply primary component
        processed_errors = []
        for error in errors:
            error_copy = copy.deepcopy(error)
            error_copy["primary_issue_component"] = "soa"
            processed_errors.append(error_copy)
            
        # Verify primary component in all errors
        for error in processed_errors:
            self.assertEqual(error["primary_issue_component"], "soa")
            
        # Test with nested clusters
        clusters = {
            "0": [{"text": "Cluster 0 Error", "component": "soa"}],
            "1": [{"text": "Cluster 1 Error", "component": "android"}]
        }
        
        # Process clusters
        processed_clusters = {}
        for cluster_id, cluster_errors in clusters.items():
            processed_clusters[cluster_id] = []
            for error in cluster_errors:
                error_copy = copy.deepcopy(error)
                error_copy["primary_issue_component"] = "soa"
                processed_clusters[cluster_id].append(error_copy)
        
        # Verify primary component in all cluster errors
        for cluster_id, cluster_errors in processed_clusters.items():
            for error in cluster_errors:
                self.assertEqual(error["primary_issue_component"], "soa")
                
    def test_combined_preservation_workflow(self):
        """Test the entire component preservation workflow."""
        # Create test data simulating entire pipeline
        original_errors = [
            {
                "text": "Error in SOA module",
                "file": "app_debug.log",
                "component": "soa",
                "component_source": "filename"
            },
            {
                "text": "Error in Android module",
                "file": "android.log",
                "component": "android",
                "component_source": "filename"
            }
        ]
        
        # Step 1: Extract original component fields
        original_fields = []
        for error in original_errors:
            original_fields.append(extract_component_fields(error))
            
        # Step 2: Process errors (simulation)
        processed_errors = []
        for error in original_errors:
            processed = copy.deepcopy(error)
            processed["severity"] = "High"  # Add additional field
            processed["primary_issue_component"] = "soa"  # Add primary component
            processed_errors.append(processed)
            
        # Step 3: Serialize to JSON
        temp_file = os.path.join(self.temp_dir, "workflow.json")
        
        # Use the new helper function for serialization
        serialize_to_json_file({"errors": processed_errors}, temp_file, primary_issue_component="soa")
            
        # Step 4: Deserialize
        with open(temp_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
            
        # Step 5: Verify preservation
        for i, original in enumerate(original_fields):
            loaded_error = loaded_data["errors"][i]
            for field, value in original.items():
                self.assertEqual(loaded_error[field], value)
            
        # Verify primary component was preserved
        for error in loaded_data["errors"]:
            self.assertEqual(error["primary_issue_component"], "soa")
            
    def test_serialize_with_component_awareness(self):
        """Test the serialize_with_component_awareness function."""
        # Create test data
        data = {
            "test_id": "SXM-123456",
            "component": "soa",
            "component_source": "filename"
        }
        
        # Create a temporary file for testing
        temp_file = os.path.join(self.temp_dir, "awareness.json")
        
        # Serialize with component awareness function
        with open(temp_file, 'w', encoding='utf-8') as f:
            serialize_with_component_awareness(data, f, primary_issue_component="soa")
            
        # Read back and verify
        with open(temp_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
            
        # Verify component preservation
        self.assertEqual(loaded_data["component"], "soa")
        self.assertEqual(loaded_data["component_source"], "filename")
        
    def test_lambda_factory_serialization(self):
        """Test the lambda factory approach for serialization."""
        # Create test data
        data = {
            "test_id": "SXM-123456",
            "component": "soa",
            "component_source": "filename"
        }
        
        # Create a temporary file for testing
        temp_file = os.path.join(self.temp_dir, "lambda.json")
        
        # Serialize using lambda factory
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, 
                   cls=lambda *a, **kw: ComponentAwareEncoder(
                       *a, primary_issue_component="soa", **kw),
                   indent=2)
            
        # Read back and verify
        with open(temp_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
            
        # Verify component preservation
        self.assertEqual(loaded_data["component"], "soa")
        self.assertEqual(loaded_data["component_source"], "filename")
        
    def test_incorrect_usage_detection(self):
        """Test that incorrect usage of ComponentAwareEncoder raises errors."""
        # Create test data
        data = {"component": "soa"}
        
        # Create a temporary file for testing
        temp_file = os.path.join(self.temp_dir, "incorrect.json")
        
        # Incorrectly try to use an instance directly
        with self.assertRaises(TypeError):
            encoder = ComponentAwareEncoder(primary_issue_component="soa")
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, cls=encoder, indent=2)

if __name__ == '__main__':
    unittest.main()