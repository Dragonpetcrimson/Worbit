# Orbit Analyzer Test Suite - Technical Guide

This document explains the purpose and implementation details of each test component in the Orbit Analyzer test suite.

## Test Architecture

The test suite is organized into two primary types:
1. **Standard Tests**: Function-based tests that validate basic functionality using a `test_[module_name]()` pattern
2. **UnitTests**: Class-based tests using the unittest framework with `TestCase` classes for more complex validations

### Test Runner
The `run_all_tests.py` script orchestrates test execution with the following capabilities:
- Selective test execution based on command-line arguments
- Combined execution of both standard and unittest-based tests
- Detailed result tracking and reporting
- Integration verification capability
- Time measurement for performance monitoring

## Core Module Tests

### log_segmenter_test.py
**Purpose**: Validates the file collection functionality that finds log and image files.  
**Key Validations**:
- Recursive directory scanning
- Correct file type identification
- Handling of various file extensions
- Fallback to generated test data when no logs are found

### ocr_processor_test.py
**Purpose**: Tests OCR text extraction from images.  
**Key Validations**:
- Image loading and processing
- Text extraction accuracy
- Handling of images with no readable text
- Minimum text length thresholds

### log_analyzer_test.py
**Purpose**: Validates the core log parsing functionality.  
**Key Validations**:
- Error detection in log files
- Timestamp extraction
- Severity determination
- Context line capture
- File, line number and structural validation

### error_clusterer_test.py
**Purpose**: Tests grouping of similar errors.  
**Key Validations**:
- TF-IDF vectorization
- K-means clustering
- Error grouping accuracy
- Cluster size and distribution validation

### gpt_summarizer_test.py 
**Purpose**: Tests the GPT prompt construction and summary generation.  
**Key Validations**:
- Prompt structure and content
- Offline summary generation
- Output format verification
- Required section presence (ROOT CAUSE, IMPACT, RECOMMENDED ACTIONS)

## Component Analysis Tests

### component_analyzer_test.py
**Purpose**: Tests the component analysis functionality.  
**Key Validations**:
- Component schema loading
- Component identification from log files
- Component relationship analysis
- Primary issue component identification

### component_integration_test.py
**Purpose**: Tests the integration of component analysis with the main pipeline.  
**Key Validations**:
- Component schema initialization
- Log analysis with component tagging
- Enhanced report data generation
- Integration with error clustering

### component_visualizer_test.py
**Purpose**: Tests the visualization of component relationships.  
**Key Validations**:
- Component relationship diagram generation
- Error propagation diagram generation
- Component error heatmap generation
- Graph layout algorithms

### context_aware_clusterer_test.py
**Purpose**: Tests context-aware error clustering.  
**Key Validations**:
- Temporal clustering of errors
- Component-based clustering
- Root cause error identification
- Causality path generation
- Error graph export

### direct_component_analyzer_test.py
**Purpose**: Tests direct component mapping and analysis.  
**Key Validations**:
- Component assignment from file names
- Component assignment from error text
- Component relationships identification
- Primary issue component determination

## Report Generation Tests

### reports_basic_test.py
**Purpose**: Tests the new modular report generation framework.  
**Key Validations**:
- Report package availability detection
- Basic functionality for all report types
- Proper file creation
- Component information integration

### docx_generator_test.py
**Purpose**: Tests the bug report document generation.  
**Key Validations**:
- Document creation with proper sections
- Content extraction from summary
- Document saving and path handling
- Mock testing when python-docx is unavailable

### json_utils_test.py
**Purpose**: Tests JSON utilities for report generation.  
**Key Validations**:
- Custom datetime encoding
- JSON serialization of complex objects
- Component preservation during serialization
- File writing and loading

## Integration and System Tests

### integration_check_test.py
**Purpose**: Verifies system integration and component availability.  
**Key Validations**:
- Module presence and importability
- Required function availability
- Directory structure validation
- Configuration validation
- Package and module relationship checks

### controller_test.py
**Purpose**: Tests the main orchestration module.  
**Key Validations**:
- Pipeline execution with various configurations
- Feature file handling
- Path normalization
- Exception handling and graceful degradation

### gherkin_log_correlator_test.py
**Purpose**: Tests the correlation of logs with Gherkin steps.  
**Key Validations**:
- Feature file parsing
- Step extraction
- Log to step correlation
- Specialized log adapter functionality
- Step transition identification

## Advanced Tests

### gpt_mock_test.py
**Purpose**: Tests the GPT API integration using mock responses.  
**Implementation Details**:
- Uses unittest.mock to intercept API calls
- Verifies correct API parameters
- Validates processing of responses
- Tests error handling

### parameterized_tests.py
**Purpose**: Tests error clustering with multiple configurations.  
**Implementation Details**:
- Creates diverse test data with distinct error types
- Tests various cluster sizes
- Verifies clustering output
- Uses subTest for parameterized execution

### performance_tests.py
**Purpose**: Validates system performance with large datasets.  
**Implementation Details**:
- Generates large test logs (2000+ entries)
- Measures execution time for log parsing and clustering
- Sets performance thresholds (5s for parsing, 3s for clustering)
- Computes processing rates (errors/second)

### timeline_generator_test.py
**Purpose**: Tests the generation of timeline visualizations.  
**Implementation Details**:
- Creates mock step metadata and logs
- Tests simple timeline generation
- Tests cluster timeline generation
- Verifies image file creation

### secure_api_key_test.py
**Purpose**: Tests secure API key handling.  
**Implementation Details**:
- Tests key retrieval from environment variables
- Tests fallback to keyring storage
- Tests graceful handling of missing keys
- Simulates keyring failures

### batch_processor_test.py
**Purpose**: Tests batch processing functionality.  
**Implementation Details**:
- Tests finding test folders
- Tests processing single tests
- Tests batch processing logic
- Mocks pipeline execution

## Test Utilities and Support

### test_utils.py
**Purpose**: Provides common utilities for all tests.  
**Key Features**:
- Dynamic log folder detection with `get_test_folder()`
- Test data generation with `create_test_data()`
- Path management for test resources

### test_config.py
**Purpose**: Provides configuration for tests.  
**Key Features**:
- Defines paths for test data and outputs
- Configures sample files for specialized tests
- Creates required directories for tests

### reports_test_helper.py
**Purpose**: Supports testing of the reports package.  
**Key Features**:
- Compatibility detection for reports modules
- Function imports with fallbacks
- Compatibility reporting

### integration_tester.py / integration_checker.py
**Purpose**: Verifies system integration status.  
**Key Features**:
- Module availability checking
- Function existence verification
- Reports package compatibility testing
- Detailed integration status reporting

## Code Coverage Tools

### coverage_test.bat / coverage_test.sh
**Purpose**: Measures test coverage across the codebase.  
**Features**:
- Runs tests with the coverage module
- Generates console reports with line counts and percentages
- Creates HTML reports for visual coverage analysis
- Highlights untested code sections with color coding

## Test Execution Patterns

### Standard Test Pattern
Most standard tests follow this pattern:
```python
def test_module_name():
    """Test function for the module"""
    # Get test data
    test_dir = get_test_folder()
    
    # Call module function with test data
    result = module_function(test_dir)
    
    # Validate results with explicit success/failure messages
    if not result:
        print("❌ Test failed")
        return False
    print("✅ Test passed")
    return True
```

### UnitTest Pattern
Most unittest-based tests follow this pattern:
```python
class TestModuleName(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        # Create test data
        self.test_data = {...}
        
    def tearDown(self):
        """Clean up test resources"""
        # Remove temporary files
        
    def test_specific_function(self):
        """Test a specific function"""
        # Call function with test data
        result = specific_function(self.test_data)
        
        # Assert expected results
        self.assertEqual(expected, result)
```

### Mock Test Pattern
Mock tests typically follow this pattern:
```python
@mock.patch('module.function_to_mock')
def test_with_mock(self, mock_function):
    # Configure mock
    mock_function.return_value = mock_data
    
    # Call function that uses the mocked component
    result = function_under_test()
    
    # Verify mock was called correctly
    mock_function.assert_called_once_with(expected_args)
    
    # Verify result
    self.assertEqual(expected_result, result)
```

## Test Naming Conventions

- **File Names**: `module_name_test.py`
- **Function-Based Tests**: `test_module_name()`
- **Class-Based Tests**: `TestModuleName`
- **Test Methods**: `test_specific_functionality`
- **Mock Variables**: `mock_function_name`
- **Helper Functions**: `_create_test_data`

## Logging and Reporting

The test runner provides comprehensive reporting:
- Each test is executed with timing information
- Detailed results indicate pass/fail status
- Summary report shows overall pass/fail counts and percentages
- Individual test files provide more granular success/failure details
- Integration status reports indicate missing or misconfigured components
