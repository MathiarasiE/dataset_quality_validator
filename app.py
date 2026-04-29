"""
app.py
------
Flask API for Auto Dataset Quality Validator.

Endpoints:
  POST /upload  → Upload CSV dataset, run validation
  GET  /report  → Return latest JSON validation report
  GET  /score   → Return only the quality score
  GET  /health  → Health check
"""

import os
import sys
import json
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# ── Path setup ────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backend.validator_pipeline import run_validation

# ── App setup ─────────────────────────────────────────────────
app = Flask(__name__, static_folder="frontend", static_url_path="")
CORS(app)  # Enable CORS for frontend communication

UPLOAD_FOLDER = "uploads"
REPORT_DIR = "reports"
ALLOWED_EXTENSIONS = {"csv", "json", "jsonl", "parquet", "xlsx", "xls"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# In-memory store for the latest report (production: use a database)
latest_report = {}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_dataset(filepath: str) -> pd.DataFrame:
    """Load dataset from various file formats with robust handling."""
    ext = filepath.rsplit(".", 1)[1].lower()
    
    # Check if file is empty
    if os.path.getsize(filepath) == 0:
        raise ValueError("File is empty. Please upload a non-empty dataset.")
    
    try:
        if ext == "csv":
            return pd.read_csv(filepath)
        elif ext == "json":
            # Try to validate JSON first
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:
                    raise ValueError("JSON file is empty")
                if not (content.startswith('[') or content.startswith('{')):
                    raise ValueError("JSON file must start with '[' or '{'")
            df = pd.read_json(filepath)
            return _flatten_and_clean_df(df)
        elif ext == "jsonl":
            # Validate JSONL format (each line should be valid JSON)
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if not lines:
                    raise ValueError("JSONL file is empty")
                # Try parsing first line to validate format
                import json
                try:
                    json.loads(lines[0])
                except json.JSONDecodeError as e:
                    raise ValueError(f"First line is not valid JSON: {str(e)}")
            df = pd.read_json(filepath, lines=True)
            return _flatten_and_clean_df(df)
        elif ext == "parquet":
            df = pd.read_parquet(filepath)
            return _flatten_and_clean_df(df)
        elif ext in ("xlsx", "xls"):
            df = pd.read_excel(filepath)
            return _flatten_and_clean_df(df)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
    except (ValueError, UnicodeDecodeError) as e:
        raise ValueError(f"Failed to parse {ext.upper()} file: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error reading file: {str(e)}")


def _flatten_and_clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten nested structures and convert list columns to strings."""
    # Convert list/array columns to strings (to avoid 'unhashable type' errors)
    for col in df.columns:
        if df[col].dtype == object:
            # Check if column contains lists or dicts
            try:
                if any(isinstance(x, (list, dict)) for x in df[col].dropna()):
                    # Convert to string representation
                    df[col] = df[col].astype(str)
            except Exception:
                pass
    
    return df


# ── Routes ────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the frontend UI."""
    return send_from_directory("frontend", "index.html")


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint for CI/CD."""
    return jsonify({"status": "ok", "service": "Dataset Quality Validator"}), 200


@app.route("/upload", methods=["POST"])
def upload_dataset():
    """
    POST /upload
    Upload a CSV file, run full validation, return report.
    """
    global latest_report

    if "file" not in request.files:
        return jsonify({"error": "No file part in request."}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"Unsupported file format. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        df = load_dataset(filepath)
    except Exception as e:
        # Provide detailed error message
        error_msg = str(e)
        if "Expected object or value" in error_msg:
            error_msg = "JSON file is malformed or empty. Check that it's valid JSON."
        elif "Expecting value" in error_msg:
            error_msg = "JSON format error. Each line in JSONL must be valid JSON."
        return jsonify({"error": f"Failed to parse file: {error_msg}"}), 422

    if df.empty:
        return jsonify({"error": "Uploaded CSV is empty."}), 422

    # Optionally read label column from form data
    label_col = request.form.get("label_col", None)

    try:
        report = run_validation(
            df,
            label_col=label_col,
            generate_report=True,
            report_dir=REPORT_DIR
        )
        latest_report = report
    except Exception as e:
        return jsonify({"error": f"Validation failed: {str(e)}"}), 500

    # Return a condensed response for the UI
    return jsonify({
        "message": "Validation complete.",
        "filename": filename,
        "rows": report["dataset_info"]["rows"],
        "columns": report["dataset_info"]["columns"],
        "quality_score": report["quality_score"],
        "summary": report["summary"],
        "validation_checks": report["validation_checks"]
    }), 200


@app.route("/report", methods=["GET"])
def get_report():
    """
    GET /report
    Return the latest full validation report as JSON.
    """
    if not latest_report:
        # Try loading from disk
        report_path = os.path.join(REPORT_DIR, "validation_report.json")
        if os.path.exists(report_path):
            with open(report_path) as f:
                return jsonify(json.load(f)), 200
        return jsonify({"error": "No report available. Upload a dataset first."}), 404

    return jsonify(latest_report), 200


@app.route("/score", methods=["GET"])
def get_score():
    """
    GET /score
    Return only the quality score and status.
    """
    if not latest_report:
        report_path = os.path.join(REPORT_DIR, "validation_report.json")
        if os.path.exists(report_path):
            with open(report_path) as f:
                data = json.load(f)
                return jsonify(data.get("quality_score", {})), 200
        return jsonify({"error": "No score available. Upload a dataset first."}), 404

    return jsonify(latest_report.get("quality_score", {})), 200


# ── Entry Point ───────────────────────────────────────────────

if __name__ == "__main__":
    print("🚀 Starting Dataset Quality Validator API...")
    print("   → UI:     http://localhost:5000")
    print("   → Upload: POST http://localhost:5000/upload")
    print("   → Report: GET  http://localhost:5000/report")
    print("   → Score:  GET  http://localhost:5000/score")
    app.run(debug=True, host="0.0.0.0", port=5000)