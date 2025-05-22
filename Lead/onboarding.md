# 🚀 Orbit Analyzer – Developer Onboarding

Welcome to the Orbit Analyzer project! This onboarding guide is designed to help you get started quickly by centralizing all the key resources, documentation, and development workflows you'll need.

---

## 📦 Project Overview

Orbit Analyzer is an AI-powered log analysis and reporting tool that:
- Extracts and clusters log errors
- Correlates test steps from Gherkin files
- Supports component-level analysis
- Generates summaries using GPT (optional)
- Outputs Excel, DOCX, HTML, JSON, and PNG reports

---

## 📂 Essential Documentation

### 🧭 Getting Started Guide
Step-by-step instructions for installing and using Orbit Analyzer.  
👉 [`Orbit_Getting_Started.md`](docs/Orbit_Getting_Started.md)

### 🔧 Developer Setup & Environment
How to create your virtual environment, install dependencies, and run the app locally.  
👉 [`development-workflow-guide.md`](docs/development-workflow-guide.md)

### 🧠 Technical Architecture
Detailed design, error clustering logic, GPT integration, and visualizations.  
👉 [`Orbit Deep Dive.md`](docs/Orbit Deep Dive.md)  
👉 [`Orbit Technical Guide.pdf`](docs/Orbit Technical Guide.pdf)

### 📘 User Manual
High-level functional guide aimed at QA users and analysts.  
👉 [`Orbit User Guide.pdf`](docs/Orbit User Guide.pdf)

---

## 🛠️ Common Tasks

- Run single test: `python controller.py`
- Run batch tests: `python batch_processor.py --all`
- Run with GPT: `python controller.py --test-id SXM-1234567 --model gpt-4`
- Build executable: `python Build_Script.py`
- Run tests: `python run_all_tests.py`
- View test coverage: `./coverage_test.sh` or `coverage_test.bat`

---

## 🔐 Environment Variables

Environment variables can be configured using `.env` or directly in your shell:

- `OPENAI_API_KEY` – For GPT integration
- `LOG_BASE_DIR` – Location of input logs (default: `./logs`)
- `OUTPUT_BASE_DIR` – Output directory (default: `./output`)
- `ENABLE_OCR` – Toggle OCR processing (`True` or `False`)

👉 See: `.env.example` for all available variables

---

## 📦 CI/CD & Test Integration

Tests are run on every PR via GitHub Actions. See:
- `.github/workflows/orbit-tests.yml`

Test categories include:
- Unit, Integration, Performance
- Run all: `python run_all_tests.py`

---

## 👥 Development Workflow

- Use `feature/`, `fix/`, and `release/` branches
- Submit PRs to `develop`, not `main`
- Use the [PR Template](.github/PULL_REQUEST_TEMPLATE.md)
- Follow code style and docstring standards from `development-workflow-guide.md`

---

## 🙋 Need Help?

Contact the project maintainer or check the `#orbit-dev` Slack channel for support.

---

© 2025 SiriusXM – Internal Use Only
