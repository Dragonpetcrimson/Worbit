# Orbit Analyzer - Development Workflow Guide

## 1. Version Control Workflow

### Branch Naming Conventions

Based on the current project structure and codebase organization, the Orbit project follows these branch naming conventions:

| Branch Type | Naming Pattern | Example | Purpose |
|-------------|----------------|---------|---------|
| Main Branch | `main` or `master` | `main` | Stable, production-ready code |
| Feature Branches | `feature/[feature-name]` | `feature/component-analysis` | New feature development |
| Bug Fix Branches | `fix/[issue-id]` | `fix/component-preservation` | Fixing specific issues |
| Release Branches | `release/v[version]` | `release/v1.2.0` | Preparing for a specific version release |
| Documentation Branches | `docs/[doc-area]` | `docs/path-utils` | Documentation updates |

### Pull Request Process

1. **PR Creation**
   - Create PR with descriptive title referencing feature or issue
   - Fill out PR template with:
     - What the change accomplishes
     - Testing performed
     - Related issues or tickets
     - Component impact analysis

2. **PR Requirements**
   - All tests must pass
   - Code must adhere to style guidelines 
   - Documentation must be updated for API changes
   - Component preservation must be validated for component-related changes
   - No merge conflicts with target branch

3. **Code Review Process**
   - At least one reviewer must approve
   - Address all review comments
   - Update PR with requested changes
   - Re-request review once changes are made
   - For component-related changes, include sample output showing component preservation

### Merge Strategies

1. **Feature Branches**
   - Squash and merge to main
   - Clean commit message summarizing the feature
   - Reference any related tickets or issues

2. **Bug Fix Branches**
   - Merge commit or squash depending on commit history value
   - Reference issue ID in commit message

3. **Release Branches**
   - Create merge commit to preserve release history
   - Tag the merge commit with version number (e.g., v1.2.3)

4. **Branch Cleanup**
   - Delete branches after successful merge
   - Archive release branches

## 2. Feature Development Lifecycle

### From Request to Production

```
Feature Request → Planning → Implementation → Testing → Review → Merge → Release
```

1. **Feature Request/Planning**
   - Document requirements in issue tracker
   - Break down into manageable tasks
   - Estimate complexity and effort
   - Determine component impact

2. **Implementation Phase**
   - Create feature branch
   - Develop code following standards
   - Write unit tests
   - Update documentation
   - Ensure component information is preserved

3. **Testing Phase**
   - Run full test suite
   - Perform integration testing
   - Check test coverage
   - Verify component preservation
   - Manual verification of reports and visualizations

4. **Review and Approval**
   - Submit PR for code review
   - Address feedback
   - Get final approval
   - Demonstrate component integrity

5. **Integration**
   - Merge to main branch
   - Verify integration success
   - Run post-merge checks

6. **Release**
   - Include in version release
   - Update changelog
   - Deploy to production

### Required Artifacts

| Artifact | Purpose | Required For |
|----------|---------|-------------|
| Unit Tests | Verify component functionality | All features |
| Integration Tests | Verify system integration | User-facing features |
| Documentation Update | Update relevant docs | All changes |
| Component Preservation Tests | Verify component data integrity | Component-related changes |
| Path Validation Tests | Verify correct file paths | Path-related changes |
| Visualization Tests | Verify report visualizations | Visualization features |

### Testing Requirements

1. **Unit Testing Requirements**
   - Test all new public functions and methods
   - Mock external dependencies
   - Verify error handling paths
   - Ensure test coverage of new code
   - Test component field preservation

2. **Integration Testing Requirements**
   - Test interactions between components
   - Verify end-to-end functionality
   - Test failure scenarios and recovery
   - Validate file structure and paths

3. **Component Preservation Testing**
   - Verify component information is preserved through processing
   - Validate primary_issue_component consistency
   - Check component_source tracking
   - Validate JSON serialization maintains component data

## 3. Code Style and Standards

### Python Style Guidelines

Based on the current codebase patterns:

1. **General Python Style**
   - Follow PEP 8 guidelines
   - Maximum line length of 100 characters
   - Use 4 spaces for indentation (no tabs)
   - Use snake_case for functions and variables
   - Use CamelCase for classes

2. **Documentation**
   - Use docstrings for modules, classes, and functions
   - Follow Google docstring style with type hints
   - Include Args, Returns, and Raises sections in docstrings

   ```python
   def function_name(param1: Type, param2: Type) -> ReturnType:
       """
       Short description of function.
       
       Args:
           param1: Description of param1
           param2: Description of param2
           
       Returns:
           Description of return value
           
       Raises:
           ExceptionType: Description of when this exception is raised
       """
   ```

3. **Import Ordering**
   - Standard library imports first
   - Third-party library imports second
   - Local application imports third
   - Alphabetical ordering within each group

   ```python
   # Standard library imports
   import os
   import sys
   from datetime import datetime
   
   # Third-party imports
   import numpy as np
   import pandas as pd
   
   # Local application imports
   from config import Config
   from log_analyzer import parse_logs
   from utils.path_utils import get_output_path, OutputType
   ```

### Component-Related Code Standards

1. **Component Identification**
   - Use standardized approach for component identification
   - Prioritize filename-based identification first
   - Fall back to content-based identification for unknown components
   - Always preserve component information during processing

   ```python
   # Good practice for component identification
   def identify_component_from_log_entry(log_entry):
       # First try filename-based identification
       if 'file' in log_entry:
           component, source = identify_component_from_filename(log_entry['file'])
           if component != 'unknown':
               return component, source
               
       # Fall back to content-based identification
       if 'text' in log_entry:
           component, source = identify_component_from_content(log_entry['text'])
           return component, source
           
       return 'unknown', 'default'
   ```

2. **Component Information Preservation**
   - Use consistent field names for component data
   - Preserve component information during serialization
   - Track component_source to identify identification method
   - Maintain primary_issue_component consistency

   ```python
   # Component field preservation example
   COMPONENT_FIELDS = {
       'component', 'component_source', 'source_component',
       'primary_issue_component', 'affected_components'
   }
   ```

### Path Handling Standards

1. **Path Utilities**
   - Use `utils.path_utils` for all path operations
   - Use `OutputType` enum for file type classification
   - Use standardized filenames for consistency
   - Ensure proper directory structure

   ```python
   # Path handling best practice
   from utils.path_utils import (
       get_output_path,
       OutputType,
       normalize_test_id,
       get_standardized_filename
   )
   
   # Get proper path for a visualization
   image_path = get_output_path(
       output_dir, 
       test_id, 
       get_standardized_filename(test_id, "component_distribution", "png"),
       OutputType.VISUALIZATION
   )
   ```

2. **HTML Path References**
   - Validate HTML references using `utils.path_validator`
   - Use relative paths for better portability

### Error Handling Practices

1. **Use Try-Except Blocks**
   - Catch specific exceptions, not generic `Exception` where possible
   - Log exceptions with context
   - Provide meaningful error messages
   - Continue with reduced functionality when possible

   ```python
   try:
       result = process_file(file_path)
   except FileNotFoundError as e:
       logging.error(f"File not found: {file_path}")
       return None
   except PermissionError as e:
       logging.error(f"Permission denied for file: {file_path}")
       return None
   except Exception as e:
       logging.error(f"Unexpected error processing {file_path}: {str(e)}")
       traceback.print_exc()
       return None
   ```

2. **Graceful Degradation**
   - Continue with fallback approach when main functionality fails
   - Provide informative error visualization for failed visualizations
   - Generate placeholder reports when complete data unavailable
   - Log detailed information for debugging

## 4. Common Development Tasks

### Building and Testing Commands

#### Setup and Installation

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install Pydot for enhanced visualization layouts (not required)
# On Linux/macOS:
pip install pydot

# On Windows:
pip install pydot-ng

# Set up OpenAI API key (for GPT features)
# Option 1: Environment variable
export OPENAI_API_KEY=your_api_key

# Option 2: .env file
echo "OPENAI_API_KEY=your_api_key" > .env

# Option 3: System keyring
python -c "import keyring; keyring.set_password('orbit_analyzer', 'openai_api_key', 'your_api_key')"
```

#### Running the Analyzer

```bash
# Interactive mode
python controller.py

# Direct execution with specific test ID
python controller.py SXM-123456

# Batch processing
python batch_processor.py --tests SXM-123456 SXM-789012
python batch_processor.py --all
python batch_processor.py --all --parallel
```

#### Running Visualization Tests

```bash
# Run all visualization tests
python -m unittest tests.visualization_tests

# Run specific visualization test
python -m unittest tests.visualization_tests.TestComponentVisualizer

# Run with memory profiling
python -m memory_profiler tests/visualization_tests.py

# Test visualization with limited memory
python -c "import tests.visualization_tests; tests.visualization_tests.test_memory_constraints()"
```

#### Debug and Verification Tools

```bash
# Validate output directory structure
python -c "from utils.path_validator import print_validation_results; print_validation_results('output/SXM-123456', 'SXM-123456')"

# Check component preservation
python -c "import json; print(json.load(open('output/SXM-123456/json/SXM-123456_component_preservation.json')))"

# Diagnose output structure
python -c "from controller import diagnose_output_structure; diagnose_output_structure('SXM-123456')"

```

#### Building Executable

```bash
python Build_Script.py
```

### Debugging Workflow

1. **Local Debugging**
   - Enable detailed logging with appropriate levels
   - Set `LOG_LEVEL=DEBUG` in environment or `.env` file
   - Check the log file specified in `Config.LOG_FILE`
   - Use the debug directory for additional diagnostics

   ```python
   logging.debug(f"Component distribution: {component_counts}")
   ```

2. **Component Analysis Issues**
   - Check `component_preservation.json` for field integrity
   - Verify component identification with:
     ```
     python -c "from components.direct_component_analyzer import identify_component_from_filename; print(identify_component_from_filename('app_debug.log'))"
     ```
   - Check primary component consistency in logging

3. **Path and Directory Issues**
   - Run path validation using `path_validator.py`
   - Check directory structure with:
     ```
     python -c "import os; print(os.listdir('output/SXM-123456'))"
     ```
   - Verify HTML references in generated reports

4. **Visualization Debugging**
   - Enable visualization feature flags in `config.py`
   - Check debug logs for visualization generation
   - Validate image paths in HTML source
   - Verify thread-local storage is properly cleaned up
   - Check for memory leaks with `memory_profiler`
   - Validate image files using PIL for proper format
   - Test visualization with different matplotlib backends

### Common Troubleshooting Steps

1. **"Module not found" Errors**
   - Verify virtual environment is activated
   - Check requirements are installed
   - Verify import paths are correct
   - Check for circular imports

2. **Component Information Loss**
   - Check JSON serialization is using `ComponentAwareEncoder`
   - Verify processing maintains component fields
   - Check `normalize_data()` preserves component information
   - Validate `preprocess_errors()` handling

3. **Path-Related Issues**
   - Ensure consistent use of `get_output_path()` for all files
   - Verify correct `OutputType` for each file
   - Check HTML references use correct prefix
   - Run path validation to detect misplaced files

4. **Visualization Generation Failures**
   - Check feature flags in `Config` class
   - Verify matplotlib is properly configured with Agg backend
   - Check image directories exist and are writable
   - Verify fallback to placeholder images works
   - Check thread-local storage for thread safety
   - Ensure matplotlib figures are properly closed with `plt.close()`
   - Verify timeout protection works for hung visualizations
   - Check format detection logic is working properly

5. **GPT Integration Issues**
   - Verify OpenAI API key is available
   - Check for rate limiting or quota issues
   - Verify fallback to offline mode works
   - Check prompt formatting and sanitization

## 5. Component Architecture

### Key Components and Relationships

The system's architecture is built around several key components that work together:

1. **Core Analysis Components**
   - `log_analyzer.py`: Parses logs and identifies initial components
   - `error_clusterer.py`: Groups similar errors
   - `direct_component_analyzer.py`: Identifies components and relationships
   - `gpt_summarizer.py`: Generates AI-powered summaries

2. **Enhanced Component Analysis**
   - `components/component_analyzer.py`: Advanced component identification
   - `components/component_integration.py`: Relationship analysis
   - `components/component_visualizer.py`: Visualization generation with layout fallbacks
   - `components/context_aware_clusterer.py`: Enhanced clustering

3. **Report Generation System**
   - `reports/report_manager.py`: Orchestrates report generation
   - `reports/base.py`: Common base classes and utilities
   - `reports/component_report.py`: Component relationship reporting
   - Various specialized report generators (DOCX, Excel, JSON, Markdown)

4. **Path Handling System**
   - `utils/path_utils.py`: Centralized path handling
   - `utils/path_validator.py`: Path validation utilities

### Component Information Flow

```
Controller.py
   |
   |--> parse_logs() -> Initial component identification
   |      |
   |      |--> assign_components_and_relationships() -> Relationship enrichment
   |             |
   |             |--> ComponentIntegration.analyze_logs() -> Graph-based analysis
   |                     |
   |                     |--> Component reports & visualizations
   |
   |--> write_reports() -> Multi-format reporting with component awareness
          |
          |--> ReportManager -> Orchestrated report generation
                 |
                 |--> Multiple specialized report generators
```

### Critical Component Fields

When working with component-related code, these fields must be preserved:

```python
COMPONENT_FIELDS = {
    'component',              # The identified component
    'component_source',       # How the component was identified
    'source_component',       # Original component before processing
    'root_cause_component',   # Component identified as root cause
    'primary_issue_component',# Primary component for reporting
    'affected_components',    # Related components affected
    'expected_component',     # Expected component for validation
    'component_scores',       # Confidence scores for components
    'component_distribution', # Distribution of components
    'parent_component',       # Parent in component hierarchy
    'child_components'        # Children in component hierarchy
}
```

### Component Preservation Mechanisms

The system includes several mechanisms to ensure component information is preserved:

1. **ComponentAwareEncoder**
   - Custom JSON encoder in `reports/base.py`
   - Preserves component fields during serialization
   - Tracks transformations for validation

2. **Component Data Preprocessor**
   - `reports/data_preprocessor.py`
   - Normalizes and validates component fields
   - Ensures consistency between related fields

3. **ReportManager Component Handling**
   - Verifies component preservation before writing reports
   - Tracks component distribution through processing
   - Adds diagnostics for troubleshooting

## 6. Visualization System Architecture

### Visualization Framework

The visualization system has been redesigned to be more reliable and compatible across different environments:

1. **Backend-Agnostic Rendering**
   - Uses `matplotlib.use('Agg')` to ensure compatibility in non-GUI environments
   - Supports multiple output formats (PNG, SVG) with automatic format detection
   - Implements multi-layered layout fallbacks to replace PyGraphviz dependency

2. **Thread Safety Mechanisms**
   - Uses thread-local storage for feature flags and cache
   - Implements timeout protection to prevent hung visualizations
   - Ensures proper resource cleanup to prevent memory leaks

3. **Memory Management**
   - Explicitly closes matplotlib figures with `plt.close()`
   - Uses `_save_figure_with_cleanup()` to ensure proper cleanup in all cases
   - Tracks memory usage with profiling tools

4. **Layout Algorithm Selection**
   - Multi-layered fallback system for graph layouts:
     1. Pydot/Graphviz layout (if available)
     2. Spectral layout (good for tree-like structures)
     3. Shell layout (good for grouped components)
     4. Spring layout (ultimate fallback)

### Key Visualization Components

1. **ComponentVisualizer**
   - Located in `components/component_visualizer.py`
   - Primary visualization engine for component relationships
   - Implements the multi-layered layout system
   - Provides consistent color schemes and formats

2. **VisualizationGenerator**
   - Located in `reports/visualizations.py`
   - Orchestrates visualization generation
   - Implements thread safety with timeout protection
   - Provides fallback to placeholders when needed

3. **Path Handling for Visualizations**
   - Format-aware path generation
   - Consistent paths for backward compatibility
   - Image verification after generation

4. **Feature Flag System**
   - Controlled through `Config` settings
   - Thread-safe feature flag checking
   - Default values for missing flags

### Visualization Types

1. **Component Relationship Diagrams**
   - Show relationships between components
   - Color-code by component type
   - Highlight primary issue component
   - Generated as `component_relationships.png`

2. **Component Error Distribution**
   - Horizontal bar chart of error counts by component
   - Highlight primary issue component
   - Generated as both `component_errors.png` and `component_distribution.png` (for compatibility)

3. **Error Propagation Diagrams**
   - Show how errors propagate between components
   - Color-code by error type (root cause, symptom, etc.)
   - Generated as `error_propagation.png`

4. **Timeline Visualizations**
   - Show errors along a timeline
   - Color-code by severity or cluster
   - Generated as `timeline.png` and `cluster_timeline.png`

### Visualization Tests

The visualization system includes comprehensive tests in `tests/visualization_tests.py`:

1. **Layout Algorithm Tests**
   - Test all layout fallback algorithms
   - Verify layouts work with various graph structures
   - Test empty graphs and single-node graphs

2. **Memory Usage Tests**
   - Verify figures are properly closed
   - Check for memory leaks in repeated generation
   - Test memory constraints

3. **Format Detection Tests**
   - Test PNG and SVG format support
   - Verify format detection logic
   - Test format fallbacks

4. **Thread Safety Tests**
   - Verify thread-local storage works properly
   - Test timeout handling
   - Verify exception handling in threads

## 7. Release Process

### Versioning Strategy

The project follows Semantic Versioning (SemVer):

```
MAJOR.MINOR.PATCH
```

- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality (backwards compatible)
- **PATCH**: Bug fixes (backwards compatible)

Current version is tracked in the `VERSION_FILE` variable in `Build_Script.py`.

### Release Preparation

1. **Pre-Release Checklist**
   - All tests pass on all supported platforms
   - Documentation is updated to reflect changes
   - Component preservation verified on all report types
   - Path validation checks pass
   - Version number updated in:
     - `version.txt`
     - `Build_Script.py`
     - Documentation references

2. **Release Branch Creation**
   - Create `release/v{version}` branch
   - Perform final verification on branch
   - Make any release-specific adjustments

3. **Building Release Artifacts**
   - Run `Build_Script.py` to create executable
   - Verify executable works correctly on target platforms
   - Test with sample logs to verify report generation
   - Create distributable package

### Release Publication

1. **Merge and Tag**
   - Merge release branch to main
   - Tag the release commit with version
   - Push tag to remote repository

2. **Distribution**
   - Upload artifacts to designated location
   - Update download links in documentation
   - Notify stakeholders of new release

3. **Post-Release**
   - Monitor for immediate issues
   - Create patch releases for critical issues
   - Begin planning for next release cycle

### Hotfix Process

For critical issues in production:

1. Create `fix/v{version}-hotfix` branch from release tag
2. Fix the issue with minimal changes
3. Test thoroughly with focus on component preservation
4. Create new patch version tag
5. Merge back to main branch

## 8. Development Tools

### Recommended IDE: VSCode

#### Recommended Extensions

| Extension | Purpose |
|-----------|---------|
| Python | Core Python support |
| Pylance | Python language server |
| Python Test Explorer | Run and debug tests |
| GitLens | Enhanced Git integration |
| markdownlint | Markdown linting |
| Jupyter | Notebook support for data exploration |
| JSON Tools | Help with component schema editing |
| Path Intellisense | Path autocompletion |

#### Workspace Settings

Recommended `.vscode/settings.json`:

```json
{
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length", "100"],
    "editor.formatOnSave": true,
    "editor.rulers": [100],
    "python.testing.pytestEnabled": false,
    "python.testing.unittestEnabled": true,
    "python.testing.nosetestsEnabled": false,
    "python.testing.unittestArgs": ["-v", "-s", "./tests", "-p", "*_test.py"],
    "files.associations": {
        "*.json": "jsonc"
    },
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true
    },
    "search.exclude": {
        "**/node_modules": true,
        "**/bower_components": true,
        "**/*.code-search": true,
        "**/venv": true
    }
}
```

### Environment Variables

Key environment variables for development:

| Variable | Purpose | Default |
|----------|---------|---------|
| `OPENAI_API_KEY` | Authentication key for OpenAI API | None |
| `LOG_BASE_DIR` | Directory containing log files | `./logs` |
| `OUTPUT_BASE_DIR` | Directory for output reports | `./output` |
| `LOG_LEVEL` | Logging verbosity | `INFO` |
| `LOG_FILE` | File to write logs | `orbit_analyzer.log` |
| `DEFAULT_MODEL` | Default GPT model to use | `gpt-3.5-turbo` |
| `ENABLE_OCR` | Whether to enable OCR | `True` |
| `MPLBACKEND` | Matplotlib backend override | `Agg` |

### Feature Flags

Feature flags in `config.py` can be used to control behavior:

| Flag | Purpose | Default |
|------|---------|---------|
| `ENABLE_CLUSTER_TIMELINE` | Enable cluster timeline visualization | `False` |
| `ENABLE_COMPONENT_DISTRIBUTION` | Enable component distribution charts | `True` |
| `ENABLE_ERROR_PROPAGATION` | Enable error propagation visualization | `False` |
| `ENABLE_STEP_REPORT_IMAGES` | Enable step report visualizations | `False` |
| `ENABLE_COMPONENT_REPORT_IMAGES` | Enable component report visualizations | `True` |
| `ENABLE_COMPONENT_RELATIONSHIPS` | Enable component relationship diagrams | `False` |

### Debugging Tools

1. **Path Validation Tools**
   - `utils/path_validator.py` for directory structure validation
   - `check_html_references()` for HTML image reference validation

2. **Component Analysis Debugging**
   - JSON output of component analysis in `json/` subdirectory
   - Component preservation diagnostic in `component_preservation.json`

3. **Log and Debug Information**
   - Main log file specified by `Config.LOG_FILE`
   - Timeline generation debug logs in `debug/` subdirectory
   - Component analysis debug information in JSON files

4. **Visualization Debugging**
   - PIL image verification
   - Memory profiling with `memory_profiler`
   - Thread-local storage inspection
   - Matplotlib backend configuration testing

## 9. Documentation Standards

### Code Documentation

1. **Module-Level Docstrings**
   - Brief description of module purpose
   - List key classes and functions
   - Note any important caveats or requirements

   ```python
   """
   utils/path_utils.py - Centralized path handling utilities for Orbit
   
   This module provides standardized path handling functions to ensure
   consistent file organization across the application.
   
   Key functions:
   - normalize_test_id: Standardize test ID format
   - get_output_path: Get appropriate path for a file based on type
   - setup_output_directories: Create standard directory structure
   """
   ```

2. **Class Docstrings**
   - Class purpose and behavior
   - Inheritance information if relevant
   - Key methods and attributes
   - Usage example if helpful

   ```python
   class ComponentAnalyzer:
       """
       Analyzer for identifying components and their relationships in log entries.
       
       This class processes logs to identify which components generated them,
       using both filename-based and content-based methods. It also analyzes
       relationships between components using the component schema.
       
       Key methods:
       - identify_component_from_log_entry: Determine component for a log entry
       - enrich_log_entries_with_components: Add component info to logs
       - analyze_component_failures: Analyze component errors and relationships
       """
   ```

3. **Function Docstrings**
   - Purpose and behavior
   - Parameter descriptions with types
   - Return value description with type
   - Exceptions that might be raised
   - Usage example for complex functions

   ```python
   def get_output_path(
       base_dir: str, 
       test_id: str, 
       filename: str, 
       output_type: OutputType = OutputType.PRIMARY_REPORT,
       create_dirs: bool = True
   ) -> str:
       """
       Get standardized output path based on file type.
       
       Args:
           base_dir: Base output directory
           test_id: Test ID (will be normalized)
           filename: Filename to use
           output_type: Type of output determining subdirectory
           create_dirs: Whether to create directories if they don't exist
           
       Returns:
           Full path for the output file
       
       Examples:
           >>> get_output_path("output", "SXM-123", "analysis.json", OutputType.JSON_DATA)
           'output/json/analysis.json'
       """
   ```

### Project Documentation

1. **README Files**
   - High-level project overview
   - Quick start instructions
   - Basic usage examples
   - Links to more detailed documentation

2. **Architecture Documentation**
   - Component relationships and data flow
   - Directory structure explanation
   - Key interfaces and extension points

3. **Workflow Documentation**
   - This document (development workflow)
   - Testing procedures
   - Release process

4. **API Documentation**
   - Public interfaces for each module
   - Usage examples
   - Common patterns and best practices

## 10. Test Logs Management

### Sample Logs Organization

Sample test logs should be organized as follows:

```
logs/
├── SXM-123456/                # Test case directory
│   ├── app_debug.log          # Android app logs
│   ├── appium.log             # Appium test logs
│   ├── mimosa.log             # Mimosa component logs
│   ├── phoebe.log             # Phoebe component logs
│   ├── traffic.har            # Network traffic logs
│   └── screenshots/           # Test screenshots
│       ├── step1.png          # Step 1 screenshot
│       └── step2.png          # Step 2 screenshot
├── SXM-789012/                # Another test case
```

### Adding New Test Logs

1. Create directory with normalized test ID
2. Add relevant log files
3. Include diverse component logs for better analysis
4. Add screenshots if available for OCR

### Test Log Privacy

1. Sanitize logs to remove sensitive information
2. Use `sanitize_text()` from `reports/base.py` for text sanitization
3. Avoid including real user data or credentials

## 11. Development Best Practices

### Component-Aware Development

1. **Component Identification Conservation**
   - Preserve component information throughout processing
   - Use consistent field names from `COMPONENT_FIELDS`
   - Track how component was identified with `component_source`

2. **Component Relationship Awareness**
   - Understand the component schema relationships
   - Preserve relationship information in analysis
   - Use the component graph for relationship lookups

3. **Primary Component Consistency**
   - Maintain consistency of `primary_issue_component`
   - Propagate `primary_issue_component` to all data structures
   - Verify `primary_issue_component` before generating reports

### Path Handling Best Practices

1. **Use Path Utilities Consistently**
   - Always use `get_output_path()` for file paths
   - Use correct `OutputType` for each file
   - Use `get_standardized_filename()` for consistent naming

2. **Directory Structure Compliance**
   - Keep main reports in root directory
   - JSON data files in `json/` subdirectory
   - Debug information in `debug/` subdirectory

3. **HTML References**
   - Use relative paths for better portability
   - Validate HTML references with `check_html_references()`

### Visualization Best Practices

1. **Thread Safety**
   - Always use thread-local storage for caching
   - Use daemon threads for visualization generation
   - Implement timeouts to prevent hanging
   - Clean up thread-local storage when done

2. **Memory Management**
   - Always close matplotlib figures with `plt.close()`
   - Use `_save_figure_with_cleanup()` for proper cleanup
   - Avoid storing large data in memory
   - Monitor memory usage with profiling tools

3. **Format Handling**
   - Use `_get_preferred_format()` to select appropriate format
   - Support both PNG and SVG formats where possible
   - Maintain backward compatibility with dual file generation
   - Verify images after generation with PIL

4. **Error Handling**
   - Always create placeholder visualizations on error
   - Log detailed error information for debugging
   - Catch all exceptions in threads
   - Provide informative messages in placeholder images

### Logging Best Practices

1. **Use Appropriate Log Levels**
   - `ERROR`: Serious issues requiring attention
   - `WARNING`: Potential problems or degraded functionality
   - `INFO`: General information about progress
   - `DEBUG`: Detailed information for troubleshooting

2. **Include Context in Log Messages**
   - Log component information in relevant operations
   - Include file paths where appropriate
   - Log operation success/failure clearly

3. **UTF-8 Handling**
   - Use the UTF-8 logging handler from `config.py`
   - Set `PYTHONIOENCODING=utf-8` for consistent encoding
   - Handle potential encoding issues gracefully

### Testing Best Practices

1. **Test Component Preservation**
   - Verify component fields are preserved through processing
   - Check JSON serialization maintains component information
   - Validate `primary_issue_component` consistency

2. **Test Path Handling**
   - Verify files are created in correct locations
   - Check file naming is consistent and standardized
   - Validate HTML references use correct paths

3. **Test Visualization Generation**
   - Verify visualizations are generated correctly with all layout algorithms
   - Test format detection and selection logic
   - Verify thread safety with parallel generation
   - Test memory usage with repeated generation
   - Ensure fallback to placeholders works correctly
   - Verify timeout protection prevents hanging
   - Test visualization generation in non-GUI environments