# 🛰️ Orbit Analyzer – Getting Started Guide

**Last Updated:** May 2025  
**Audience:** QA Testers, Analysts, Technical Leads, and First-Time Users

---

## 🔍 What is Orbit Analyzer?

Orbit is a standalone AI-powered log analysis tool that helps you quickly identify the **root cause of test failures** across multiple logs, screenshots, and HTTP traffic captures.

Just run the `.exe` and Orbit will:

- ✅ Parse your logs and screenshots  
- ✅ Identify which components are experiencing issues  
- ✅ Group similar errors into clusters  
- ✅ Run GPT-based root cause analysis (optional)  
- ✅ Generate Excel, Word, HTML, and visualization reports — all in seconds

---

## 🧰 Requirements

| Requirement         | Details                                                              |
|---------------------|----------------------------------------------------------------------|
| Windows OS          | Windows 10 or later                                                  |
| Logs Folder         | Place your test files under `logs/SXM-<your-test-id>/`               |
| GPT API (Optional)  | Set `OPENAI_API_KEY` environment variable if you want GPT analysis   |

---

## 🛠️ First-Time Setup

### 1. **Unzip the package**

You should have a folder like:

```
orbit_analyzer/
├── orbit_analyzer.exe
├── README.txt
├── Orbit_Getting_Started.md
├── Orbit Technical Guide.pdf
```

---

### 2. **Place your test logs**

Create a folder with your test ID under `logs/`, e.g.:

```
logs/
└── SXM-2302295/
    ├── appium.log
    ├── app_debug.log
    ├── phoebe.log
    ├── mimosa.log
    ├── traffic.har
    ├── screenshot1.png
```

📌 The folder **must start with `SXM-`** for the analyzer to detect it.

**Pro Tip:** Include logs from different components (SOA, Phoebe, Mimosa, etc.) to get better component analysis and pinpoint the root cause more accurately!

---

### 3. **Set up OpenAI API Key (Required for GPT Analysis)**

To use GPT-based features (summary generation, root cause analysis, etc.), you must provide your own OpenAI API key.

#### How to get your API key:
1. Get your API key from https://centerstage.siriusxmpandora.com/esc?id=kb_article&table=kb_knowledge&sysparm_article=KB0013213&searchTerm=chat%20gpt

#### Option 1: Set as environment variable

**On Windows (Command Prompt):**
```cmd
setx OPENAI_API_KEY sk-...
```
(Restart Command Prompt after running this)

**On Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY="sk-..."
```

**On Mac/Linux:**
```bash
export OPENAI_API_KEY=sk-...
```

#### Option 2: Create a .env file

Create a file named `.env` in the same directory as the executable with the following content:
```
OPENAI_API_KEY=your-api-key-here
DEFAULT_MODEL=gpt-3.5-turbo
```

#### Option 3: Additional environment settings (optional)

```
LOG_BASE_DIR=path/to/logs
OUTPUT_BASE_DIR=path/to/output
ENABLE_OCR=True
ENABLE_STEP_REPORT=True
ENABLE_STEP_REPORT_IMAGES=True
```

> If you don't set an API key, Orbit will run in **offline mode** (no AI summary, but everything else still works).

---

### 4. **Run the analyzer**

#### Option 1: Interactive Mode

From a command prompt:

```cmd
cd path\to\orbit_analyzer
orbit_analyzer.exe
```

Follow the prompts:
- Enter your test ID (e.g., `SXM-2302295`)
- Select test type (Ymir or Installer)
- Choose GPT mode (GPT-4, GPT-3.5, or None)

#### Option 2: Direct Command

Run with a specific test ID:
```cmd
orbit_analyzer.exe SXM-2302295
```

#### Option 3: Batch Processing

Analyze multiple tests at once:
```cmd
orbit_analyzer.exe --batch --tests SXM-2302295 SXM-7890123
```

Or analyze all tests in your logs directory:
```cmd
orbit_analyzer.exe --batch --all
```


---

## 📊 Output Location

Reports will be saved in:

```
output/
└── SXM-2302295/
    ├── SXM-2302295_log_analysis.xlsx       # Main Excel report
    ├── SXM-2302295_bug_report.docx         # Bug report document
    ├── SXM-2302295_log_analysis.md         # Markdown report
    ├── SXM-2302295_component_report.html   # Component analysis report
    │
    ├── json/                               # Detailed data files
    │   ├── SXM-2302295_log_analysis.json
    │   └── SXM-2302295_component_analysis.json
    │
    └── supporting_images/                  # Visualizations
        ├── SXM-2302295_timeline.png
        ├── SXM-2302295_cluster_timeline.png
        ├── SXM-2302295_component_relationships.png
        └── SXM-2302295_component_distribution.png
```

---

## 🔍 Understanding the Reports

### Excel Report (SXM-2302295_log_analysis.xlsx)

The most comprehensive report with multiple tabs:
- **Summary**: AI-generated root cause and impact analysis
- **Technical Summary**: Error details with component information
- **Component Analysis**: Components involved and their relationships
- **Grouped Issues**: Errors clustered by similarity (color-coded)
- **Cluster Summary**: Overview of each error cluster
- **Images extraction**: Text extracted from screenshots

### Component Report (SXM-2302295_component_report.html)

An interactive report showing:
- Which components are experiencing issues
- How components relate to each other
- Error distribution across components
- Possible error propagation paths

### Visualizations (in supporting_images/)

- **Timeline**: When errors occurred during test execution
- **Cluster Timeline**: Error clusters over time with color coding
- **Component Relationships**: How different components interact
- **Component Distribution**: Which components have the most errors

---

## 🆘 Troubleshooting

| Problem                  | Solution                                                            |
|--------------------------|---------------------------------------------------------------------|
| "No test folders found"  | Check that your folder starts with `SXM-` and is under `logs/`      |
| GPT error / timeout      | Check your API key is correct and has available credits              |
| Missing screenshots      | They're optional — tool works fine without                           |
| Excel file won't save    | Make sure it's not already open                                     |
| "API key not found"      | Set your OpenAI API key using one of the methods above              |
| Missing component analysis | Include logs from multiple components (SOA, Phoebe, etc.)         |
| Timeline visualization empty | Verify that logs contain timestamp information                   |

---

## 📎 For More Details

See:
- `Orbit User Guide.md` for complete documentation
- `Orbit Technical Guide.pdf` for advanced features

---

## 🧠 Tips for Getting the Most Out of Orbit

- **Include diverse logs**: Logs from different components provide better analysis of relationships and root causes
- **Organize by test ID**: Use the standard SXM-#### format for your test ID folders
- **Check component reports**: The component analysis can quickly pinpoint which component is causing issues
- **Review visualizations**: The timeline and component visualizations often make patterns obvious at a glance
- **Use batch processing**: When analyzing multiple test failures, use batch processing to save time

---

## 🔧 For QA Leads and CI/CD Integration

For CI/CD pipeline integration:

```bash
# Windows path
"C:\path\to\orbit_analyzer.exe" --batch --all --output-report=batch_results.txt

# Linux/Mac (using Python directly)
python batch_processor.py --all --parallel
```

See the `Orbit Technical Guide.pdf` for more advanced integration examples and API usage.

---

© 2025 SiriusXM – Internal Use Only