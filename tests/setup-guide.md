# Orbit Analyzer - Setup & Integration Guide

This guide explains how to set up the Orbit Analyzer system and integrate additional modules like the Gherkin Log Correlation component with your existing test framework.

## Basic Setup

### Prerequisites
- Python 3.8 or higher
- Tesseract OCR (for image processing)
- Git (for version control)

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/orbit-analyzer.git
   cd orbit-analyzer
   ```

2. Create and activate a virtual environment:
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   - Create a `.env` file in the project root with:
     ```
     OPENAI_API_KEY=your_key_here
     LOG_BASE_DIR=./logs
     OUTPUT_BASE_DIR=./output
     LOG_LEVEL=INFO
     DEFAULT_MODEL=gpt-3.5-turbo
     ENABLE_OCR=true
     ```

5. Create required directories:
   ```bash
   mkdir -p logs output
   ```

## Integrating Gherkin Log Correlation

If you want to add the Gherkin Log Correlation capabilities to an existing project:

1. Copy these core files to your project root:
   - `gherkin_log_correlator.py` - Core correlation engine
   - `enhanced_adapters.py` - Specialized adapters for log formats
   - `step_aware_analyzer.py` - Step-aware analysis tools

2. Copy these test files to your `tests` directory:
   - `gherkin_log_correlator_test.py` - Test module for log correlation

3. Update your `tests/test_config.py` file by adding:

```python
# Configuration for Gherkin log correlation tests
'SAMPLE_FEATURE_FILE': r"path/to/your/sample.feature",
'SAMPLE_LOGS_DIR': r"path/to/your/logs",

# Paths for the modules
'GHERKIN_LOG_CORRELATOR_PATH': os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'gherkin_log_correlator.py')),
'ENHANCED_ADAPTERS_PATH': os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'enhanced_adapters.py'))
```

## Component Analysis Integration

To integrate the component analysis capabilities:

1. Copy these files to your project structure:
   - `components/component_analyzer.py`
   - `components/component_integration.py`
   - `components/direct_component_analyzer.py`
   - `components/component_visualizer.py`
   - `components/context_aware_clusterer.py`
   - `components/schemas/component_schema.json`

2. Copy these test files to your `tests` directory:
   - `component_analyzer_test.py`
   - `component_integration_test.py`
   - `component_visualizer_test.py`
   - `context_aware_clusterer_test.py`
   - `direct_component_analyzer_test.py`

3. Ensure your `controller.py` imports and uses the component integration:

```python
# Try to import the component integration module
try:
    from components.component_integration import ComponentIntegration
    COMPONENT_INTEGRATION_AVAILABLE = True
except ImportError as e:
    COMPONENT_INTEGRATION_AVAILABLE = False
    logging.warning(f"Component integration module not available - will use direct component mapping: {str(e)}")

# Later in the pipeline:
if COMPONENT_INTEGRATION_AVAILABLE and logs:
    try:
        # Create component integration instance
        component_integrator = ComponentIntegration(component_schema_path)
        
        # Run component analysis
        component_analysis_results = component_integrator.analyze_logs(
            all_log_entries, errors, output_dir, test_id
        )
        logging.info(f"Component analysis complete: {len(component_analysis_results.get('metrics', {}))} metrics generated")
    except Exception as e:
        logging.error(f"Error in component analysis: {str(e)}")
        component_analysis_results = None
```

## Running Tests

After setup, you can run tests in several ways:

1. Run all tests:
   ```bash
   python run_all_tests.py
   ```

2. Run specific test categories:
   ```bash
   python run_all_tests.py --standard    # Only standard tests
   python run_all_tests.py --unittest    # Only unittest-based tests
   python run_all_tests.py --performance # Only performance tests
   python run_all_tests.py --integration # Only integration tests
   python run_all_tests.py --reports     # Only reports package tests
   ```

3. Run with integration verification:
   ```bash
   python run_all_tests.py --verify-integration
   ```

4. Run test coverage:
   ```bash
   # Windows
   coverage_test.bat
   
   # Unix/Linux
   ./coverage_test.sh
   ```

## Troubleshooting

### Common Issues and Solutions

1. **Missing modules errors**:
   - Check that your Python path includes the project root
   - Verify all dependencies are installed: `pip install -r requirements.txt`

2. **Configuration issues**:
   - Make sure environment variables are correctly set
   - Check that `.env` file exists if you're using it

3. **Test data availability**:
   - Tests will look for SXM-* folders in the logs directory
   - If none are found, minimal test data will be generated automatically

4. **OpenAI API Key issues**:
   - Set the key in your environment: `export OPENAI_API_KEY=your_key`
   - Or use the `.env` file or system keyring
   - You can run in offline mode by setting `DEFAULT_MODEL=none`

5. **Component analysis issues**:
   - Verify `component_schema.json` is in the correct location
   - Check that all component modules are installed

### Checking System Integration

Run the integration check to verify your setup:

```bash
python -m tests.integration_check_test
```

This will check for all required modules and dependencies, and report any issues that need to be addressed.
