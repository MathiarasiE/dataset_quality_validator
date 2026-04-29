# Auto Dataset Quality Validator with CI/CD Integration

This repository provides an end-to-end dataset quality validator that:

- Runs basic and ML-powered data checks (missing values, duplicates, outliers, distribution drift, label consistency).
- Computes a quality score and returns PASS/FAIL.
- Exposes a small Flask UI and API to upload datasets with interactive charts.
- Integrates with GitHub Actions to run dataset validation in CI.
- Includes configurable thresholds via `validator_config.ini`.
- Comes with comprehensive unit tests.

## Quick Start

### 1. Set up environment:

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. Run tests:

```bash
pip install pytest
pytest test_validators.py -v
```

### 3. Run the API and UI:

```bash
python app.py
# then open http://localhost:5000 in your browser
```

### 4. Validate a dataset from CLI (CI/CD style):

```bash
python ci_validate.py --dataset dataset.csv --config validator_config.ini
```

## Supported File Formats

The validator accepts multiple dataset formats:

- **CSV** (Comma-Separated Values)
- **JSON** (Standard JSON array of objects)
- **JSONL** (JSON Lines — one object per line)
- **Parquet** (Apache Parquet columnar format)
- **Excel** (XLSX, XLS)

Edit `validator_config.ini` to adjust:

- `min_quality_score`: Minimum passing score (default: 70)
- Enable/disable specific checks
- Severity thresholds

Example:
```ini
[validation]
min_quality_score = 75

[checks]
enable_knn_outliers = true
```

## Project Structure

```
.
├── app.py                    # Flask app and endpoints
├── ci_validate.py            # CLI validator for CI/CD
├── backend/
│   ├── __init__.py
│   └── validator_pipeline.py # Orchestrates validation
├── basic_validator.py        # Missing values, duplicates, types
├── ml_validator.py           # KNN, K-Means, Decision Tree, Naive Bayes
├── score.py                  # Quality scoring system
├── report_generator.py       # JSON + HTML reports
├── test_validators.py        # Unit tests
├── frontend/
│   ├── index.html           # UI with charts
│   ├── app.js               # Upload + Chart.js integration
│   └── style.css            # Styling
├── .github/workflows/ci.yml # GitHub Actions
├── validator_config.ini     # Configuration
└── dataset.csv              # Sample data

```

## Features

**Basic Validation:**
- Missing values detection
- Duplicate rows detection  
- Data type mismatch detection

**ML-Based Validation:**
- KNN outlier detection in numerical data
- K-Means distribution shift detection
- Decision Tree label consistency checking
- Naive Bayes feature distribution validation

**Reporting:**
- JSON validation report with full details
- HTML dashboard with scores and charts
- CI-friendly exit codes

**Frontend:**
- Drag-and-drop CSV upload
- Real-time validation with progress
- Interactive charts (deduction breakdown, severity distribution)
- Full HTML report viewer

## API Endpoints

- `POST /upload` — Upload CSV, run validation, return report
- `GET /report` — Fetch latest validation report (JSON)
- `GET /score` — Fetch quality score only
- `GET /health` — Health check

## CI/CD Integration

The GitHub Actions pipeline in [.github/workflows/ci.yml](.github/workflows/ci.yml) now:

- Runs `pytest test_validators.py -v` on pull requests and pushes to `main`
- Runs `python ci_validate.py --dataset dataset.csv --config validator_config.ini`
- Uploads `reports/ci/` as a workflow artifact
- Builds the Docker image from [Dockerfile](Dockerfile) and pushes it to GitHub Container Registry on pushes to `main`

The published image will be available as `ghcr.io/<owner>/<repo>:latest` and `ghcr.io/<owner>/<repo>:<sha>`.

## Optional Improvements

- Add test coverage reporting (`pytest --cov`)
- Export reports to S3/cloud storage
- Slack notifications on validation failure
- Email alerts with report summary
- Time-series quality tracking

---

**Built with:** Python, Scikit-learn, Pandas, Flask, Chart.js
