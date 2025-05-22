# 🛰️ Orbit Analyzer

A comprehensive tool for analyzing test logs, clustering errors, identifying component relationships, and generating detailed reports to accelerate debugging and root cause analysis of SiriusXM test failures.

---

## 📋 Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Understanding Reports](#understanding-reports)
- [Advanced Features](#advanced-features)
- [Configuration](#configuration)
- [Privacy & Security](#privacy--security)
- [Troubleshooting](#troubleshooting)

---

## ✨ Features

* **Log Analysis**: Extract errors and warnings from various log formats (appium, app_debug, charles, HAR, and more)
* **Component Analysis**: Automatically identify which components (SOA, Android, etc.) are causing issues and map their relationships
* **Error Clustering**: Group similar errors to identify patterns with component-aware clustering
* **Step-Aware Analysis**: Correlate logs with Gherkin test steps for more precise debugging
* **Timeline Visualization**: See when errors occurred during test execution
* **Component Visualizations**: View component relationships, error distribution, and error propagation paths
* **GPT-Powered Summaries**: Get AI-generated analysis of errors and root causes
* **Multi-Format Reports**: Generate comprehensive reports in Excel, DOCX, Markdown, and JSON formats

---

## 🚀 Installation

### Option 1: Download the Executable (Recommended for Windows)

1. Download the latest `orbit_analyzer.exe` file from the release page
2. Place it in a directory of your choice
3. Create a `logs` folder in the same directory (or configure a custom location)

### Option 2: Run from Source

```batch
:: Clone the repository
git clone https://github.com/yourusername/orbit-analyzer.git
cd orbit-analyzer

:: Install dependencies
pip install -r requirements.txt

:: Set up your OpenAI API key (optional, for GPT analysis)
:: In Windows Command Prompt
setx OPENAI_API_KEY your-api-key

:: Or in PowerShell
$env:OPENAI_API_KEY = "your-api-key"
```

---

## 🔧 Usage

### Preparing Your Logs

1. Create a folder for your test using the format: `logs\SXM-xxxxxxx`
2. Copy all log files into this folder (including component-specific logs like app_debug.log, phoebe.log, mimosa.log)
3. Add any screenshots (PNG/JPG) for OCR processing
4. If available, copy the Gherkin feature file (`.feature`) into the folder for step-aware analysis

**Pro Tip:** Include logs from multiple components (SOA, Phoebe, Mimosa, etc.) to get better component analysis and pinpoint the root cause more accurately!

### Running the Analysis

#### Using the Executable

1. Double-click `orbit_analyzer.exe` or run from command line
2. Follow the interactive prompts:
   - Enter the Test ID (e.g., SXM-1234567 or just 1234567)
   - Select the test type (Ymir or Installer)
   - Choose the GPT model (GPT-4, GPT-3.5, or None for offline mode)

#### Using the Command Line

```batch
:: Interactive mode
python controller.py

:: Or with command line arguments
python controller.py SXM-1234567 --model gpt-3.5-turbo --ocr true
```

### Batch Processing

For processing multiple tests at once:

```batch
:: Process specific tests
python batch_processor.py --tests SXM-1234567 SXM-2345678

:: Process all tests in logs directory
python batch_processor.py --all

:: Run in parallel mode (faster)
python batch_processor.py --all --parallel

:: Generate a summary report
python batch_processor.py --all --report batch_results.txt
```

### Output Reports

All reports are generated in the `output\SXM-xxxxxxx` folder with a standardized directory structure:

```
output/
└── SXM-1234567/
    ├── SXM-1234567_log_analysis.xlsx        # Main Excel report
    ├── SXM-1234567_bug_report.docx          # Bug report document
    ├── SXM-1234567_log_analysis.md          # Markdown report
    ├── SXM-1234567_component_report.html    # Component analysis report
    │
    ├── json/                                # Detailed data files
    │   ├── SXM-1234567_log_analysis.json
    │   └── SXM-1234567_component_analysis.json
    │
    └── supporting_images/                   # Visualizations
        ├── SXM-1234567_timeline.png
        ├── SXM-1234567_cluster_timeline.png
        ├── SXM-1234567_component_relationships.png
        └── SXM-1234567_component_distribution.png
```

---

## 📊 Understanding Reports

### Excel Report (SXM-1234567_log_analysis.xlsx)

Comprehensive multi-tab report including:
- **Summary**: AI-generated root cause and impact analysis
- **Scenario**: Background and test scenario details
- **Technical Summary**: Technical error details with component information
- **Component Analysis**: Components involved and their relationships
- **Key Failures**: Individual errors with severity, context, and timestamps
- **Grouped Issues**: Errors clustered by similarity (color-coded)
- **Cluster Summary**: Overview of each error cluster
- **Images extraction**: Text extracted from screenshots

### Component Report (SXM-1234567_component_report.html)

Interactive HTML report showing:
- Primary issue component (root cause)
- Component relationship diagram
- Error distribution across components
- Error propagation paths
- Component hierarchy and dependencies

### Markdown Summary (SXM-1234567_log_analysis.md)

A simple text summary that can be easily copied into:
- Jira tickets
- Slack messages
- Teams chats
- Email

### Bug Report Document (SXM-1234567_bug_report.docx)

A formatted document ready for Jira submission with:
- Affected components and tests
- Root cause details
- Log snippets
- Expected vs. actual behavior

### Visualizations (in supporting_images/)

| Visualization | Shows |
|---------------|-------|
| Timeline | When errors occurred during test execution |
| Cluster Timeline | Error clusters over time with color coding |
| Component Relationships | How different components interact |
| Component Distribution | Which components have the most errors |

---

## 🔍 Advanced Features

### Component Identification and Analysis

Orbit Analyzer automatically identifies these components:
- SOA (SiriusXM application built on Android)
- Android (system components)
- Phoebe (proxy)
- Mimosa (test data provider)
- Charles (HTTP proxy)
- Telesto (coordinator)
- Arecibo (monitor)
- Lapetus (API service)
- Translator

The system also analyzes component relationships to:
- Determine the primary issue component
- Map data flows between components
- Identify which errors are root causes versus symptoms
- Visualize error propagation through the system

### Path Handling and Standardization

The system uses standardized path handling to ensure consistent file organization:
- Primary reports in the root output directory
- JSON data files in the `json/` subdirectory
- Visualizations in the `supporting_images/` subdirectory
- Debug information in the `debug/` subdirectory

### Gherkin Integration

For better test step correlation:
- Place `.feature` files in the test log directory
- The analyzer will automatically find and use them
- This enables step-by-step analysis showing exactly when errors occurred
- Timeline visualizations will show errors by test step

### OCR Capability

The analyzer can extract text from screenshots:
- Place PNG/JPG images in the test log directory
- Text will be extracted and included in the analysis
- Useful for error messages shown on-screen but not logged

### Offline Analysis

If you prefer not to use OpenAI API:
- Choose "None" for the model option
- Analysis will be performed locally without GPT summaries
- All other features (clustering, component analysis, etc.) still work

---

## ⚙️ Configuration

Configure by setting environment variables or using a `.env` file:

| Variable | Purpose | Default |
|----------|---------|---------|
| `OPENAI_API_KEY` | Authentication key for OpenAI API | None |
| `LOG_BASE_DIR` | Directory containing log files | `./logs` |
| `OUTPUT_BASE_DIR` | Directory for output reports | `./output` |
| `LOG_LEVEL` | Logging verbosity | `INFO` |
| `LOG_FILE` | File to write logs | `orbit_analyzer.log` |
| `DEFAULT_MODEL` | Default GPT model to use | `gpt-3.5-turbo` |
| `ENABLE_OCR` | Whether to enable OCR | `True` |

For secure API key storage, you can use the system keyring:

```python
import keyring
keyring.set_password("orbit_analyzer", "openai_api_key", "your-api-key")
```

Feature flags for visualization generation:

```python
# In config.py
ENABLE_CLUSTER_TIMELINE = True
ENABLE_COMPONENT_DISTRIBUTION = True
ENABLE_ERROR_PROPAGATION = True
ENABLE_STEP_REPORT_IMAGES = True
ENABLE_COMPONENT_REPORT_IMAGES = True
ENABLE_STEP_REPORT = True        # Generate step-aware HTML report
ENABLE_COMPONENT_HTML = True    # Generate component analysis HTML report
```

---

## 🔒 Privacy & Security

### Log Data
- All log files are processed locally on your machine
- When using GPT analysis, only error information and limited context are sent to OpenAI's API
- No complete log files are ever transmitted
- The tool automatically sanitizes sensitive information (API keys, passwords, emails, IPs)

### Best Practices
- Use environment variables or keyring for API keys
- Review reports before sharing
- Use offline mode for sensitive data
- Use the `--parallel` option for processing multiple tests efficiently

### OpenAI API Usage
- When using GPT features, data is transmitted following OpenAI's enterprise privacy guidelines
- You can disable GPT features by selecting "None" for the model
- The system uses secure API key handling with multiple fallback options

---

## 📋 Troubleshooting

### Common Issues

| Problem | Solution |
|---------|----------|
| "No test folders found" | Check that your folder starts with `SXM-` and is under `logs/` |
| GPT error / timeout | Check your API key is correct and has available credits |
| Missing screenshots | They're optional — tool works fine without |
| Excel file won't save | Make sure it's not already open |
| "API key not found" | Set your OpenAI API key using one of the methods in Configuration |
| Missing component analysis | Include logs from multiple components (SOA, Phoebe, etc.) |
| Timeline visualization empty | Verify that logs contain timestamp information |

### Detailed Diagnostics

For directory structure and file placement issues:

```bash
python -c "from utils.path_validator import print_validation_results; print_validation_results('output/SXM-1234567', 'SXM-1234567')"
```

For output directory diagnostics:

```bash
python -c "from controller import diagnose_output_structure; diagnose_output_structure('SXM-1234567')"
```

### Support

If you encounter any issues:
1. Check the log file: `orbit_analyzer.log`
2. Run path validation and diagnostics
3. Contact the development team with the error message and diagnostic outputs

---

## 🔍 Getting the Most Out of Orbit

* **Include diverse logs**: Logs from different components provide better analysis of relationships and root causes
* **Organize by test ID**: Use the standard SXM-#### format for your test ID folders
* **Check component reports**: The component analysis can quickly pinpoint which component is causing issues
* **Review visualizations**: The timeline and component visualizations often make patterns obvious at a glance
* **Use batch processing**: When analyzing multiple test failures, use batch processing to save time

---

© 2025 SiriusXM – Internal Use Only