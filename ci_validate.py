"""
backend/ci_validate.py
-----------------------
Script used by CI/CD pipeline to validate datasets automatically.
Exits with code 1 if quality score is below threshold (fails pipeline).
Exits with code 0 if quality check passes.

Reads configuration from validator_config.ini or accepts CLI args.

Usage:
    python ci_validate.py --dataset path/to/data.csv [--label_col label] [--threshold 70] [--config validator_config.ini]
"""

import argparse
import sys
import os
import json
import configparser
import pandas as pd

# Path setup
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from backend.validator_pipeline import run_validation

CI_REPORT_DIR = "reports/ci"


def load_config(config_file: str = "validator_config.ini") -> dict:
    """Load configuration from INI file."""
    config = configparser.ConfigParser()
    defaults = {
        "min_quality_score": 70.0,
        "enable_missing_values": True,
        "enable_duplicates": True,
        "enable_data_types": True,
        "enable_knn_outliers": True,
        "enable_kmeans_distribution": True,
        "enable_label_consistency": True,
        "enable_distribution_drift": True,
    }
    
    if os.path.exists(config_file):
        config.read(config_file)
        if "validation" in config:
            defaults["min_quality_score"] = config.getfloat("validation", "min_quality_score", fallback=70.0)
    
    return defaults


def main():
    parser = argparse.ArgumentParser(description="Auto Dataset Quality Validator — CI Mode")
    parser.add_argument("--dataset", required=True, help="Path to CSV dataset")
    parser.add_argument("--label_col", default=None, help="Name of the label/target column")
    parser.add_argument("--threshold", type=float, default=None, help="Minimum passing score (overrides config)")
    parser.add_argument("--config", default="validator_config.ini", help="Path to config file")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    threshold = args.threshold if args.threshold is not None else config["min_quality_score"]

    print("=" * 60)
    print("  AUTO DATASET QUALITY VALIDATOR — CI/CD MODE")
    print("=" * 60)
    print(f"  Dataset   : {args.dataset}")
    print(f"  Threshold : {threshold}")
    print("=" * 60)

    # Load dataset
    if not os.path.exists(args.dataset):
        print(f"ERROR: Dataset not found at '{args.dataset}'")
        sys.exit(1)

    try:
        df = pd.read_csv(args.dataset)
    except Exception as e:
        print(f"ERROR: Failed to load CSV — {e}")
        sys.exit(1)

    print(f"\nLoaded dataset: {df.shape[0]} rows x {df.shape[1]} columns\n")

    # Run validation
    try:
        report = run_validation(
            df,
            label_col=args.label_col,
            generate_report=True,
            report_dir=CI_REPORT_DIR
        )
    except Exception as e:
        print(f"\nValidation pipeline error: {e}")
        sys.exit(1)

    score = report["quality_score"]["final_score"]
    status = report["quality_score"]["status"]
    grade = report["quality_score"]["grade"]

    print("\n" + "=" * 60)
    print(f"  QUALITY SCORE : {score}/100 (Grade: {grade})")
    print(f"  THRESHOLD     : {threshold}")
    print(f"  STATUS        : {status}")
    print("=" * 60)

    # Print deduction breakdown
    print("\n  DEDUCTION BREAKDOWN:")
    for item in report["quality_score"]["deduction_breakdown"]:
        if item["deduction"] > 0:
            print(f"    - {item['check']:45s} [{item['severity'].upper():6s}]  -{item['deduction']:.1f} pts")

    # Print top suggestions
    suggestions = report["summary"].get("top_suggestions", [])
    if suggestions:
        print("\n  TOP RECOMMENDATIONS:")
        for i, s in enumerate(suggestions[:5], 1):
            print(f"    {i}. {s}")

    print("\n" + "=" * 60)

    if score >= threshold:
        print(f"  PIPELINE PASSED — Dataset quality score {score} >= {threshold}")
        print("=" * 60)
        sys.exit(0)
    else:
        print(f"  PIPELINE FAILED — Dataset quality score {score} < {threshold}")
        print("     Fix data quality issues before pushing for training.")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
