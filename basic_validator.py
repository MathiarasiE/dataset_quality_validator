"""
utils/basic_validator.py
------------------------
Handles basic data quality checks:
- Missing values detection
- Duplicate rows detection
- Data type mismatch detection
"""

import pandas as pd
import numpy as np
from typing import Dict, Any


def check_missing_values(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Detect missing values in the dataset.
    Returns count, percentage, and affected columns.
    """
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    total_cells = df.shape[0] * df.shape[1]
    total_missing = int(missing.sum())
    missing_pct = round((total_missing / total_cells) * 100, 2) if total_cells > 0 else 0

    issues = []
    for col, cnt in missing_cols.items():
        col_pct = round((cnt / len(df)) * 100, 2)
        issues.append({
            "column": col,
            "missing_count": int(cnt),
            "missing_percentage": col_pct
        })

    # Severity: >20% missing = high, 5-20% = medium, <5% = low
    severity = "none"
    if missing_pct > 20:
        severity = "high"
    elif missing_pct > 5:
        severity = "medium"
    elif missing_pct > 0:
        severity = "low"

    suggestions = []
    if missing_pct > 0:
        suggestions.append("Impute missing numerical values with mean/median.")
        suggestions.append("Impute missing categorical values with mode or 'Unknown'.")
        if missing_pct > 20:
            suggestions.append("Consider dropping columns with >50% missing values.")

    return {
        "check": "missing_values",
        "total_missing": total_missing,
        "missing_percentage": missing_pct,
        "affected_columns": issues,
        "severity": severity,
        "score_deduction": min(missing_pct * 0.5, 20),  # cap at 20 pts
        "suggestions": suggestions
    }


def check_duplicates(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Detect duplicate rows in the dataset.
    """
    duplicate_count = int(df.duplicated().sum())
    dup_pct = round((duplicate_count / len(df)) * 100, 2) if len(df) > 0 else 0

    severity = "none"
    if dup_pct > 15:
        severity = "high"
    elif dup_pct > 5:
        severity = "medium"
    elif dup_pct > 0:
        severity = "low"

    suggestions = []
    if duplicate_count > 0:
        suggestions.append(f"Remove {duplicate_count} duplicate rows using df.drop_duplicates().")
        suggestions.append("Investigate data collection pipeline for duplication sources.")

    return {
        "check": "duplicates",
        "duplicate_count": duplicate_count,
        "duplicate_percentage": dup_pct,
        "severity": severity,
        "score_deduction": min(dup_pct * 0.3, 10),  # cap at 10 pts
        "suggestions": suggestions
    }


def check_data_types(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Detect data type mismatches — e.g., numeric columns
    that contain string values due to dirty data.
    """
    issues = []
    for col in df.columns:
        # Try to coerce to numeric; if many fail, it's a mismatch
        if df[col].dtype == object:
            coerced = pd.to_numeric(df[col], errors='coerce')
            # How many non-null values failed numeric coercion
            original_non_null = df[col].dropna().shape[0]
            converted_non_null = coerced.dropna().shape[0]
            failed = original_non_null - converted_non_null
            if failed > 0 and converted_non_null > 0:
                # Mixed: some numeric, some strings → mismatch
                issues.append({
                    "column": col,
                    "detected_type": "mixed (numeric + string)",
                    "failed_conversions": int(failed),
                    "suggestion": f"Column '{col}' has mixed types. Standardize to one type."
                })

    severity = "none"
    if len(issues) > 3:
        severity = "high"
    elif len(issues) > 1:
        severity = "medium"
    elif len(issues) > 0:
        severity = "low"

    suggestions = [i["suggestion"] for i in issues]

    return {
        "check": "data_type_mismatch",
        "mismatch_count": len(issues),
        "affected_columns": issues,
        "severity": severity,
        "score_deduction": len(issues) * 2,
        "suggestions": suggestions
    }