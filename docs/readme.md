# 🛰️ Orbit Analyzer

A comprehensive tool for analyzing test logs, clustering errors, and generating detailed reports to accelerate debugging and root cause analysis of SiriusXM test failures.

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

* **Log Analysis**: Extract errors and warnings from various log formats (appium, app_debug, charles, and more)
* **Error Clustering**: Group similar errors to identify patterns
* **Component Analysis**: Automatically identify which components (SOA, Android, etc.) are causing issues
* **Timeline Visualization**: See when errors occurred during test execution
* **GPT-Powered Summaries**: Get AI-generated analysis of errors and root causes
* **Bug Reports**: Generate ready-to-submit bug report documents for Jira

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
set OPENAI_API_KEY=your-api-key

:: Or in PowerShell
$env:OPENAI_API_KEY = "your-api-key"
```

---

## 🔧 Usage

### Preparing Your Logs

1. Create a folder for your test using the format: `logs\SXM-xxxxxxx`
2. Copy all log files into this folder
3. If available, copy the Gherkin feature file (`.feature`) into the folder for step-aware analysis

### Running the Analysis

#### Using the Executable

1. Double-click `orbit_analyzer.exe`
2. Follow the interactive prompts:
   - Enter the Test ID (e.g., SXM-1234567 or just 1234567)
   - Select the test type (Ymir or Installer)
   - Choose the GPT model (GPT-4, GPT-3.5, or None for offline mode)

#### Using the Command Line

```batch
:: Interactive mode
python controller.py

:: Or with command line arguments
python controller.py --test-id SXM-1234567 --model gpt-3.5-turbo
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
```




### Output Reports

All reports are generated in the `output\SXM-xxxxxxx` folder:
- `log_analysis.html` - Main HTML report
- `SXM-xxxxxxx_bug_report.docx` - Ready-to-submit bug report
- `SXM-xxxxxxx_component_report.html` - Component relationship analysis
- Various visualization files (PNG)

HTML reports are rendered using **Jinja2** templates located in `reports/templates`.

---

## 📊 Understanding Reports

### Main Analysis Report

The main HTML report provides a comprehensive analysis including:
- AI-generated summary of root cause and recommended actions
- Error clusters grouped by similarity
- Component distribution showing which parts of the system are affected
- OCR text extracted from screenshots (if images are present)

### Component Analysis

Orbit analyzes components and generates visualizations showing error distribution and relationships.



### Bug Report Document

The DOCX bug report is formatted for easy submission to Jira and includes:
- Root cause analysis
- Impact assessment
- Recommended actions
- Affected tests and components
- Representative error logs

---

## 🔍 Advanced Features

### Component Identification

Orbit Analyzer automatically identifies these components:
- SOA (SiriusXM app)
- Android (system components)
- Translator
- Mimosa (test data provider)
- Phoebe (proxy)
- Charles (HTTP proxy)
- Telesto (coordinator)
- Arecibo (monitor)
- Lapetus (API service)

### Gherkin Integration

For better test step correlation:
- Place `.feature` files in the test log directory
- The analyzer will automatically find and use them
- This enables step-by-step analysis showing exactly when errors occurred

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

Configure by editing `config.py` or setting environment variables:

* `LOG_BASE_DIR`: Directory containing log folders (default: `./logs`)
* `OUTPUT_BASE_DIR`: Directory for output reports (default: `./output`)
* `OPENAI_API_KEY`: API key for GPT analysis (set as environment variable)
* `DEFAULT_MODEL`: GPT model to use (default: `gpt-3.5-turbo`)
* `ENABLE_OCR`: Whether to extract text from images (default: `True`)
* `ENABLE_STEP_REPORT`: Generate the step-aware HTML report (default: `True`)

For secure API key storage, you can use the Windows Credential Manager:

```python
import keyring
keyring.set_password("orbit_analyzer", "openai_api_key", "your-api-key")
```

---

## 🔒 Privacy & Security

### Log Data
- All log files are processed locally on your machine
- When using GPT analysis, only error information and limited context are sent to OpenAI's API
- No complete log files are ever transmitted
- The tool automatically attempts to sanitize sensitive information

### Best Practices
- Use environment variables or keyring for API keys
- Review reports before sharing
- Use offline mode for sensitive data
- Use the `--parallel` option for processing multiple tests efficiently

### OpenAI API Usage
- When using GPT features, data is transmitted following OpenAI's enterprise privacy guidelines
- You can disable GPT features by selecting "None" for the model

---

## 📋 Troubleshooting

### Common Issues

**Q: The analyzer doesn't find my logs**  
A: Make sure logs are in the correct folder structure: `logs\SXM-xxxxxxx\`

**Q: Step-aware analysis is not working**  
A: Make sure you have a `.feature` file in your test log directory

**Q: The analyzer crashed during processing**  
A: Check permissions and available disk space. For detailed error information, check `orbit_analyzer.log`

**Q: GPT analysis returns an error**  
A: Verify your OpenAI API key is correctly set and has sufficient quota

### Support

If you encounter any issues:
1. Check the log file: `orbit_analyzer.log`
2. Contact the development team with the error message and log file

---

© 2025 SiriusXM – Internal Use Only
