"""
backend/validator_pipeline.py
------------------------------
Orchestrates all validation steps by calling basic and ML validators,
computing the quality score and generating reports.

Public API:
    run_validation(df, label_col=None, generate_report=True, report_dir='reports')

This file is intentionally lightweight and delegates heavy-lifting
to the existing modules in the repository.
"""
import os
import sys
from typing import Dict, Any, List

# Ensure repository root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd

import basic_validator
import ml_validator
import score as scorer
import report_generator


def _collect_basic_checks(df: pd.DataFrame) -> List[Dict[str, Any]]:
    results = []
    results.append(basic_validator.check_missing_values(df))
    results.append(basic_validator.check_duplicates(df))
    results.append(basic_validator.check_data_types(df))
    return results


def _collect_ml_checks(df: pd.DataFrame, label_col: str = None) -> List[Dict[str, Any]]:
    results = []
    results.append(ml_validator.knn_outlier_detection(df))
    results.append(ml_validator.kmeans_distribution_check(df))
    results.append(ml_validator.decision_tree_label_check(df, label_col=label_col))
    results.append(ml_validator.naive_bayes_distribution_check(df, label_col=label_col))
    return results


def run_validation(
    df: pd.DataFrame,
    label_col: str = None,
    generate_report: bool = True,
    report_dir: str = "reports",
) -> Dict[str, Any]:
    """Run the full validation pipeline and return a structured report.

    The returned report contains:
      - dataset_info
      - validation_checks (list)
      - quality_score (aggregated)
      - summary

    If `generate_report` is True the function will write JSON and HTML
    reports to `report_dir` and include the paths in the returned dict.
    """
    os.makedirs(report_dir, exist_ok=True)

    dataset_info = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "columns_list": [str(c) for c in df.columns.tolist()],
    }

    # Basic checks
    basic_results = _collect_basic_checks(df)

    # ML-based checks
    ml_results = _collect_ml_checks(df, label_col=label_col)

    # All checks combined
    validation_checks = basic_results + ml_results

    # Compute aggregated score
    quality = scorer.compute_quality_score(validation_checks)

    # Build final report
    report = {
        "dataset_info": dataset_info,
        "validation_checks": validation_checks,
        "quality_score": quality,
        "summary": {
            "final_score": quality["final_score"],
            "status": quality["status"],
            "grade": quality["grade"],
            "recommendation": quality.get("recommendation", ""),
        },
    }

    # Persist reports
    if generate_report:
        json_path = os.path.join(report_dir, "validation_report.json")
        html_path = os.path.join(report_dir, "validation_report.html")
        full_report = report_generator.generate_json_report(quality, validation_checks, dataset_info, output_path=json_path)
        report_generator.generate_html_report(full_report, output_path=html_path)
        report["report_paths"] = {"json": json_path, "html": html_path}
        report["full_report"] = full_report

    return report
