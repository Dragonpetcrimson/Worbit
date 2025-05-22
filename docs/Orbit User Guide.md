# Orbit: User Guide

## Version: May 2025

## Audience: QA Testers, Stakeholders, and Non-Technical Users

## What Is Orbit?

Orbit is your assistant for understanding why tests fail - **without spending hours digging through logs**. When a test fails, it typically generates multiple log files from different components, making it difficult to pinpoint the exact cause. Orbit eliminates this pain point by automatically analyzing all logs and providing a clear, concise explanation of what went wrong.

## What Orbit Does

Orbit automatically:

* ✅ Processes multiple log files and screenshots from failed tests
* ✅ Identifies and extracts error messages
* ✅ Groups similar errors into clusters
* ✅ Identifies which components are causing issues and how they relate
* ✅ Uses AI to determine the root cause
* ✅ Visualizes when errors occurred during test execution
* ✅ Generates comprehensive reports in multiple formats

## Setup Requirements

| Requirement | Description |
|-------------|-------------|
| Python 3.8+ | The programming language Orbit runs on |
| Log Files | Any `.log`, `.txt`, `.chlsj`, or `.har` files from your test |
| Screenshots (Optional) | Any `.png`, `.jpg`, `.jpeg`, `.bmp`, or `.tiff` files captured during test failure |
| OpenAI API Key | For AI-powered analysis (provided by your organization) |

## One-Time Setup

### 1. Install Python

If you don't have Python installed:

* Download from [python.org/downloads](https://python.org/downloads)
* During installation, check "Add Python to PATH"

### 2. Install Required Packages

Open Command Prompt (Windows) or Terminal (Mac/Linux) and run:

```bash
pip install -r requirements.txt
```

If you don't have a requirements file, run:

```bash
pip install openai pandas openpyxl scikit-learn pillow pytesseract matplotlib networkx python-docx
```

### 3. Set Up Your Environment

For OpenAI API Key (required for AI analysis):

For Windows:
```bash
setx OPENAI_API_KEY "your-key-here"
```
(Restart Command Prompt after running this)

For Mac/Linux:
```bash
export OPENAI_API_KEY="your-key-here"
```
Add this line to your `.bashrc` or `.zshrc` file for permanent setup.

Optional environment settings:
```bash
setx LOG_BASE_DIR "path/to/logs"       # Directory for log files (default: ./logs)
setx OUTPUT_BASE_DIR "path/to/output"  # Directory for output files (default: ./output)
setx ENABLE_OCR "True"                 # Enable image text extraction (default: True)
```

## How to Use Orbit

### 1. Prepare Your Log Files

Create a folder for your test ID in the `logs` directory:

```
Orbit/
├── logs/
│   └── SXM-1234567/   <-- Create this folder with your test ID
│       ├── appium.log
│       ├── app_debug.log
│       ├── phoebe.log
│       ├── mimosa.log
│       ├── traffic.har
│       ├── screenshot1.png
│       └── ... other log files
```

**Pro Tip:** Include logs from different components (SOA, Phoebe, Mimosa, etc.) to get better component analysis!

### 2. Run the Analysis

#### Option 1: Interactive Mode
1. Open Command Prompt/Terminal in the Orbit folder
2. Run: `python controller.py`
3. Follow the prompts:
   * Enter your Test ID (e.g., SXM-1234567)
   * Choose the test type (Ymir or Installer)
   * Select GPT model:
     * GPT-4: More accurate but slower
     * GPT-3.5: Faster and less expensive
     * None: For offline use (no AI summary)

#### Option 2: Direct Command
Run with a specific test ID:
```bash
python controller.py SXM-1234567
```

#### Option 3: Batch Processing
Analyze multiple tests at once:
```bash
python batch_processor.py --tests SXM-1234567 SXM-7890123
```

Or analyze all tests in your logs directory:
```bash
python batch_processor.py --all
```

### 3. Review the Results

The analysis will be saved in the `output` folder under your test ID:

```
Orbit/
├── output/
│   └── SXM-1234567/
│       ├── SXM-1234567_log_analysis.xlsx        # Main Excel report
│       ├── SXM-1234567_bug_report.docx          # Bug report document
│       ├── SXM-1234567_log_analysis.md          # Markdown report
│       ├── SXM-1234567_component_report.html    # Component analysis report
│       │
│       ├── json/                                # Detailed data files
│       │   ├── SXM-1234567_log_analysis.json
│       │   └── SXM-1234567_component_analysis.json
│       │
│       └── supporting_images/                   # Visualizations
│           ├── SXM-1234567_timeline.png
│           ├── SXM-1234567_cluster_timeline.png
│           ├── SXM-1234567_component_relationships.png
│           └── SXM-1234567_component_distribution.png
```

## Understanding the Output

### Excel Report (SXM-1234567_log_analysis.xlsx)

The most comprehensive report with multiple tabs:

| Tab | Content |
|-----|---------|
| Summary | AI-generated root cause and impact analysis |
| Scenario | Background and test scenario details |
| Technical Summary | Technical error details with component information |
| Component Analysis | Components involved and their relationships |
| Key Failures | Individual errors with severity, context, and timestamps |
| Grouped Issues | Errors clustered by similarity (color-coded) |
| Cluster Summary | Overview of each error cluster |
| Images extraction | Text extracted from screenshots |

### Component Report (SXM-1234567_component_report.html)

An interactive report showing:
* Which components are experiencing issues
* How components relate to each other
* Error distribution across components
* Possible error propagation paths

### Markdown Summary (SXM-1234567_log_analysis.md)

A simple text summary that can be easily copied into:
* Jira tickets
* Slack messages
* Teams chats
* Email

### Bug Report Document (SXM-1234567_bug_report.docx)

A formatted document ready for Jira submission with:
* Affected components and tests
* Root cause details
* Log snippets
* Expected vs. actual behavior

### Visualizations (in supporting_images/)

| Visualization | Shows |
|---------------|-------|
| Timeline | When errors occurred during test execution |
| Cluster Timeline | Error clusters over time with color coding |
| Component Relationships | How different components interact |
| Component Distribution | Which components have the most errors |

## Common Issues and Solutions

| Problem | Solution |
|---------|----------|
| "API key not set" error | Ensure your OpenAI API key is set properly and not expired |
| Excel file won't save | Close any open instances of the Excel file before running analysis |
| No logs found error | Verify log files are in the correct folder with the proper format |
| GPT analysis failed | Check internet connection; a fallback summary will be provided |
| Feature file not found | For Ymir tests, ensure the feature file contains the test ID |
| Timeline visualization empty | Verify that logs contain timestamp information |
| Missing component analysis | Include logs from multiple components (SOA, Phoebe, etc.) |

## Getting the Most Out of Orbit

* **Include diverse logs**: Logs from different components provide better analysis of relationships and root causes
* **Organize by test ID**: Use the standard SXM-#### format for your test ID folders
* **Check component reports**: The component analysis can quickly pinpoint which component is causing issues
* **Review visualizations**: The timeline and component visualizations often make patterns obvious at a glance
* **Use batch processing**: When analyzing multiple test failures, use batch processing to save time

## Need More Help?

Contact your QA Lead or DevOps team for:
* API key access issues
* Additional log format support
* Custom report requirements
* Integration with your CI/CD pipeline

---

With Orbit, you can spend less time searching logs and more time fixing issues. The tool is designed to make troubleshooting faster and more accessible, even for team members without deep technical expertise.