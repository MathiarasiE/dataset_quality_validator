"""
utils/scorer.py
---------------
Quality Scoring System:
- Start from 100
- Deduct points based on each validation check
- Return final score + Pass/Fail status
"""

from typing import Dict, Any, List


THRESHOLD = 70  # Pass/Fail threshold (configurable)


def compute_quality_score(validation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate all validation results into a single quality score.

    Args:
        validation_results: List of dicts, each with 'score_deduction', 'check', 'severity'.

    Returns:
        Dict with final_score, status, grade, deduction_breakdown.
    """
    base_score = 100.0
    deduction_breakdown = []

    for result in validation_results:
        deduction = result.get("score_deduction", 0)
        base_score -= deduction
        deduction_breakdown.append({
            "check": result.get("check", "unknown"),
            "severity": result.get("severity", "none"),
            "deduction": round(deduction, 2)
        })

    final_score = round(max(0.0, base_score), 2)

    # Letter grade
    if final_score >= 90:
        grade = "A"
    elif final_score >= 80:
        grade = "B"
    elif final_score >= 70:
        grade = "C"
    elif final_score >= 60:
        grade = "D"
    else:
        grade = "F"

    status = "PASS" if final_score >= THRESHOLD else "FAIL"

    # Count issues by severity
    severity_summary = {"high": 0, "medium": 0, "low": 0, "none": 0}
    for result in validation_results:
        sev = result.get("severity", "none")
        severity_summary[sev] = severity_summary.get(sev, 0) + 1

    return {
        "final_score": final_score,
        "threshold": THRESHOLD,
        "status": status,
        "grade": grade,
        "total_deducted": round(100 - final_score, 2),
        "deduction_breakdown": deduction_breakdown,
        "severity_summary": severity_summary,
        "recommendation": _get_recommendation(final_score, status)
    }


def _get_recommendation(score: float, status: str) -> str:
    if score >= 90:
        return "Dataset is high quality. Safe to use for model training."
    elif score >= 80:
        return "Dataset is good. Minor issues present; review suggestions."
    elif score >= 70:
        return "Dataset passes threshold but has notable issues. Address before production use."
    elif score >= 60:
        return "Dataset quality is poor. Significant cleaning required before training."
    else:
        return "Dataset FAILS quality check. Do NOT use for training. Immediate remediation needed."