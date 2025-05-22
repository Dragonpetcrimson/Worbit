# 🛠️ Common Tasks – Orbit Analyzer

This document covers the most frequent tasks developers and QA testers will perform while working with Orbit Analyzer. Use this as a quick reference guide.

---

## 🔍 Run a Single Test

python controller.py
You’ll be prompted for:

Test ID (e.g., SXM-2302295)

Test type (ymir or installer)

Whether to enable GPT and OCR

🧪 Run Batch Tests

python batch_processor.py --all
This processes all logs/SXM-* folders and generates full reports in the output/ directory.

Optional:

python batch_processor.py --ids SXM-2302295 SXM-2302296
🤖 Run with GPT Summary (AI-Powered)

python controller.py --test-id SXM-1234567 --model gpt-4
or disable OCR for faster performance:


python controller.py --test-id SXM-1234567 --model gpt-3.5-turbo --no-ocr
🛠️ Build the Executable
To create a standalone .exe file for Windows:


python Build_Script.py
This uses PyInstaller and places the result in the dist/ or orbit_analyzer/ directory.

✅ Run All Tests

python run_all_tests.py
Or run test categories:

python run_all_tests.py --unittest
python run_all_tests.py --performance
python run_all_tests.py --integration
📊 Generate Coverage Report
On Linux/macOS:

./coverage_test.sh
On Windows:

coverage_test.bat
Outputs HTML report in the htmlcov/ folder.

🔐 Set API Key (GPT)
Windows (Command Prompt)
set OPENAI_API_KEY=your-key

PowerShell

$env:OPENAI_API_KEY="your-key"
macOS/Linux

export OPENAI_API_KEY=your-key
🧰 Common Troubleshooting
Problem	Solution
"No test folders found"	Ensure test folders are in logs/ and start with SXM-
Excel file won't save	Make sure it's not already open
GPT error or timeout	Try --model none to run offline, or check your network/API key
Blank output folder	Check if the pipeline ran successfully and that valid logs were present
OCR fails silently	Ensure ENABLE_OCR is set to True and pytesseract is installed
📂 Output Directory Structure

output/SXM-2302295/
├── log_analysis.xlsx
├── SXM-2302295_bug_report.docx
├── SXM-2302295_step_report.html
├── SXM-2302295_cluster_timeline.png
├── log_analysis.json
├── log_analysis.md
📎 See Also
Orbit_Getting_Started.md

onboarding.md

development-workflow-guide.md