"""
utils/report_generator.py
--------------------------
Generates:
- JSON validation report
- HTML dashboard report
- Optional matplotlib charts
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, List


def generate_json_report(
    score_result: Dict[str, Any],
    validation_results: List[Dict[str, Any]],
    dataset_info: Dict[str, Any],
    output_path: str = "reports/validation_report.json"
) -> Dict[str, Any]:
    """
    Build and save a full JSON report.
    """
    report = {
        "report_metadata": {
            "generated_at": datetime.now().isoformat(),
            "tool": "Auto Dataset Quality Validator",
            "version": "1.0.0"
        },
        "dataset_info": dataset_info,
        "quality_score": score_result,
        "validation_checks": validation_results,
        "summary": _build_summary(score_result, validation_results)
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    return report


def _build_summary(score_result, validation_results) -> Dict[str, Any]:
    all_suggestions = []
    high_severity_checks = []
    for r in validation_results:
        if r.get("severity") in ("high", "medium"):
            high_severity_checks.append(r.get("check", "unknown"))
        all_suggestions.extend(r.get("suggestions", []))

    return {
        "final_score": score_result["final_score"],
        "status": score_result["status"],
        "grade": score_result["grade"],
        "high_priority_issues": high_severity_checks,
        "top_suggestions": list(dict.fromkeys(all_suggestions))[:8]  # deduplicated top-8
    }


def generate_html_report(
    report: Dict[str, Any],
    output_path: str = "reports/validation_report.html"
) -> str:
    """
    Generate a styled HTML dashboard from the JSON report.
    """
    score = report["quality_score"]["final_score"]
    status = report["quality_score"]["status"]
    grade = report["quality_score"]["grade"]
    dataset_info = report["dataset_info"]
    checks = report["validation_checks"]
    summary = report["summary"]

    status_color = "#22c55e" if status == "PASS" else "#ef4444"
    score_color = "#22c55e" if score >= 80 else "#f59e0b" if score >= 70 else "#ef4444"

    checks_html = ""
    for c in checks:
        sev = c.get("severity", "none")
        sev_colors = {"high": "#ef4444", "medium": "#f59e0b", "low": "#3b82f6", "none": "#22c55e"}
        color = sev_colors.get(sev, "#6b7280")
        suggestions_html = "".join(f"<li>{s}</li>" for s in c.get("suggestions", []))
        deduction = c.get("score_deduction", 0)

        checks_html += f"""
        <div class="check-card">
          <div class="check-header">
            <span class="check-name">{c.get('check','').replace('_',' ').title()}</span>
            <span class="severity-badge" style="background:{color}">{sev.upper()}</span>
            <span class="deduction">-{deduction:.1f} pts</span>
          </div>
          <ul class="suggestions">{suggestions_html}</ul>
        </div>"""

    suggestions_list = "".join(f"<li>{s}</li>" for s in summary.get("top_suggestions", []))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Dataset Quality Report</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', sans-serif; background: #0f172a; color: #e2e8f0; padding: 2rem; }}
  h1 {{ font-size: 2rem; color: #38bdf8; margin-bottom: 0.5rem; }}
  .meta {{ color: #94a3b8; font-size: 0.85rem; margin-bottom: 2rem; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 2rem; }}
  .card {{ background: #1e293b; border-radius: 12px; padding: 1.5rem; border: 1px solid #334155; }}
  .card-label {{ font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.1em; }}
  .card-value {{ font-size: 2rem; font-weight: 700; margin-top: 0.25rem; }}
  .score-circle {{ text-align:center; background:#1e293b; border-radius:12px; padding:2rem; border:1px solid #334155; }}
  .score-number {{ font-size: 4rem; font-weight: 900; color: {score_color}; }}
  .status-badge {{ display:inline-block; padding:0.4rem 1.2rem; border-radius:999px; font-weight:700; font-size:1rem; background:{status_color}; color:#fff; margin-top:0.5rem; }}
  .section-title {{ font-size: 1.25rem; font-weight: 600; color: #38bdf8; margin: 2rem 0 1rem; }}
  .check-card {{ background:#1e293b; border-radius:10px; padding:1.25rem; border:1px solid #334155; margin-bottom:1rem; }}
  .check-header {{ display:flex; align-items:center; gap:1rem; flex-wrap:wrap; margin-bottom:0.75rem; }}
  .check-name {{ font-weight:600; font-size:1rem; }}
  .severity-badge {{ padding:0.2rem 0.8rem; border-radius:999px; color:#fff; font-size:0.75rem; font-weight:600; }}
  .deduction {{ margin-left:auto; color:#f87171; font-weight:600; }}
  .suggestions {{ padding-left:1.2rem; color:#94a3b8; font-size:0.875rem; line-height:1.6; }}
  .suggestions li {{ margin-bottom:0.3rem; }}
  .top-suggestions {{ background:#1e293b; border-radius:10px; padding:1.25rem; border:1px solid #334155; }}
  .top-suggestions li {{ margin-bottom:0.5rem; color:#cbd5e1; }}
</style>
</head>
<body>
<h1>🔍 Dataset Quality Report</h1>
<div class="meta">Generated: {report['report_metadata']['generated_at']} | Tool: {report['report_metadata']['tool']}</div>

<div class="grid">
  <div class="score-circle">
    <div class="card-label">Quality Score</div>
    <div class="score-number">{score:.0f}</div>
    <div>/ 100</div>
    <div><span class="status-badge">{status}</span></div>
    <div style="margin-top:0.5rem;color:#94a3b8">Grade: {grade}</div>
  </div>
  <div class="card">
    <div class="card-label">Rows</div>
    <div class="card-value" style="color:#38bdf8">{dataset_info.get('rows','-')}</div>
  </div>
  <div class="card">
    <div class="card-label">Columns</div>
    <div class="card-value" style="color:#a78bfa">{dataset_info.get('columns','-')}</div>
  </div>
  <div class="card">
    <div class="card-label">Points Deducted</div>
    <div class="card-value" style="color:#f87171">{report['quality_score']['total_deducted']}</div>
  </div>
</div>

<div class="section-title">📋 Top Recommendations</div>
<div class="top-suggestions"><ul>{suggestions_list}</ul></div>

<div class="section-title">🧪 Validation Checks</div>
{checks_html}

</body>
</html>"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    return output_path