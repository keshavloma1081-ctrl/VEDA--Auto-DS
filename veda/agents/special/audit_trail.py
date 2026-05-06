"""
VEDA — Autonomous Data Science System
agents/special/audit_trail.py — Audit Trail Agent

Creates a complete compliance audit trail:
- All pipeline decisions logged
- Data transformations recorded
- Model decisions documented
- Compliance checks summarized
- Generates audit certificate
"""

import os
import json
import hashlib
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq

from veda.core.base_agent import BaseAgent

load_dotenv()


class AuditTrailAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name="AuditTrailAgent",
            domain="compliance",
            version="1.0.0"
        )
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.3-70b-versatile"

    def _compute_data_hash(self, state: dict) -> str:
        """Compute hash of key pipeline artifacts."""
        key_data = {
            "goal": state.get("goal", ""),
            "run_id": state.get("run_id", ""),
            "model_name": state.get("model_info", {}).get("model_name", ""),
            "auc": state.get("model_info", {}).get("test_metrics", {}).get("auc_roc", 0)
        }
        data_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()[:16].upper()

    def _build_audit_trail(self, state: dict) -> dict:
        """Build complete audit trail from state."""
        model_info = state.get("model_info", {})
        data_profile = state.get("data_profile", {})

        trail = {
            "run_id": state.get("run_id", ""),
            "timestamp": datetime.now().isoformat(),
            "goal": state.get("goal", ""),
            "dataset_path": state.get("dataset_path", ""),

            "data_profile": {
                "rows": data_profile.get("row_count", 0) if data_profile else 0,
                "columns": data_profile.get("col_count", 0) if data_profile else 0,
                "target": data_profile.get("target_column", "") if data_profile else ""
            },

            "data_transformations": state.get("cleaning_diff", []),

            "feature_engineering": {
                "features": state.get("feature_list", [])[:10],
                "total_features": len(state.get("feature_list", []))
            },

            "model": {
                "name": model_info.get("model_name", ""),
                "path": model_info.get("model_path", ""),
                "mlflow_run_id": model_info.get("mlflow_run_id", ""),
                "metrics": model_info.get("test_metrics", {})
            },

            "explainability": {
                "top_features": state.get("explainability", {}).get("top_features", []),
                "method": "SHAP"
            },

            "compliance": {
                "pii_risk": state.get("pii_report", {}).get("risk_level", "NOT_CHECKED"),
                "masking_applied": len(state.get("masking_report", {}).get("changes_applied", [])),
                "gdpr_status": state.get("gdpr_report", {}).get("overall_status", "NOT_CHECKED"),
                "rbi_status": state.get("rbi_report", {}).get("overall_status", "NOT_CHECKED")
            },

            "drift_monitoring": {
                "drift_score": state.get("drift_report", {}).get("drift_score", 0),
                "needs_retraining": state.get("drift_report", {}).get("needs_retraining", False)
            },

            "decision_log": state.get("planner_decision_log", []),
            "pipeline_complete": state.get("pipeline_complete", False),
            "data_hash": self._compute_data_hash(state)
        }

        return trail

    def _generate_audit_certificate(self, trail: dict) -> str:
        """Generate audit certificate using Groq."""
        model = trail["model"]["name"]
        auc = trail["model"]["metrics"].get("auc_roc", 0)
        gdpr = trail["compliance"]["gdpr_status"]
        rbi = trail["compliance"]["rbi_status"]
        pii = trail["compliance"]["pii_risk"]

        prompt = """Generate a formal ML model audit certificate.

Model: """ + str(model) + """
AUC: """ + str(auc) + """
GDPR Status: """ + str(gdpr) + """
RBI Status: """ + str(rbi) + """
PII Risk: """ + str(pii) + """
Run ID: """ + str(trail["run_id"]) + """
Data Hash: """ + str(trail["data_hash"]) + """

Write a formal 4-5 sentence audit certificate that:
1. States what was audited
2. Summarizes compliance findings
3. Notes any conditions or requirements
4. Provides a deployment recommendation

Use formal language appropriate for regulatory documentation."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=400
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return "Audit certificate generation failed: " + str(e)

    def _generate_html_audit_report(self, trail: dict, certificate: str) -> str:
        """Generate HTML audit report."""
        timestamp = trail["timestamp"]
        run_id = trail["run_id"]
        model_name = trail["model"]["name"]
        auc = trail["model"]["metrics"].get("auc_roc", "N/A")
        gdpr = trail["compliance"]["gdpr_status"]
        rbi = trail["compliance"]["rbi_status"]
        pii = trail["compliance"]["pii_risk"]
        data_hash = trail["data_hash"]

        status_color = lambda s: "#2e7d32" if s in ["COMPLIANT", "LOW"] else "#e65100" if s == "NEEDS_ATTENTION" else "#c62828"

        html = """<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>VEDA Compliance Audit Report</title>
<style>
body{font-family:Arial,sans-serif;margin:40px;color:#1a1a2e;}
.header{background:#1F4E79;color:white;padding:30px;border-radius:10px;margin-bottom:25px;}
.header h1{margin:0;font-size:24px;}
.header p{margin:5px 0 0;opacity:.85;font-size:13px;}
.section{margin-bottom:20px;padding:18px;border:1px solid #e0e0e0;border-radius:8px;}
.section h2{color:#1F4E79;border-bottom:2px solid #1F4E79;padding-bottom:6px;margin-top:0;}
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:12px;}
.card{background:#f0f4f8;padding:12px;border-radius:8px;text-align:center;}
.card .val{font-size:20px;font-weight:bold;color:#1F4E79;}
.card .lbl{font-size:11px;color:#666;margin-top:3px;}
.badge{padding:4px 12px;border-radius:20px;font-size:12px;font-weight:bold;display:inline-block;}
.cert{background:#e3f2fd;padding:18px;border-radius:8px;border-left:4px solid #1F4E79;font-style:italic;}
table{width:100%;border-collapse:collapse;margin-top:10px;}
th{background:#1F4E79;color:white;padding:8px;text-align:left;}
td{padding:7px 8px;border-bottom:1px solid #eee;}
.footer{text-align:center;color:#999;font-size:11px;margin-top:30px;padding-top:15px;border-top:1px solid #eee;}
</style>
</head>
<body>
<div class="header">
<h1>VEDA — ML Compliance Audit Report</h1>
<p>Run ID: """ + run_id + """ | Generated: """ + timestamp + """ | Data Hash: """ + data_hash + """</p>
</div>

<div class="section">
<h2>Audit Certificate</h2>
<div class="cert">""" + certificate.replace("\n", "<br>") + """</div>
</div>

<div class="section">
<h2>Model Summary</h2>
<div class="grid">
<div class="card"><div class="val">""" + str(model_name) + """</div><div class="lbl">Model</div></div>
<div class="card"><div class="val">""" + str(auc) + """</div><div class="lbl">AUC-ROC</div></div>
<div class="card"><div class="val">""" + str(trail["data_profile"]["rows"]) + """</div><div class="lbl">Training Rows</div></div>
</div>
</div>

<div class="section">
<h2>Compliance Status</h2>
<table>
<tr><th>Check</th><th>Status</th></tr>
<tr><td>GDPR Compliance</td><td><span class="badge" style="background:#e8f5e9;color:""" + status_color(gdpr) + """">""" + str(gdpr) + """</span></td></tr>
<tr><td>RBI / DPDP Compliance</td><td><span class="badge" style="background:#e8f5e9;color:""" + status_color(rbi) + """">""" + str(rbi) + """</span></td></tr>
<tr><td>PII Risk Level</td><td><span class="badge" style="background:#e8f5e9;color:""" + status_color(pii) + """">""" + str(pii) + """</span></td></tr>
<tr><td>Data Masking</td><td>""" + str(trail["compliance"]["masking_applied"]) + """ operations applied</td></tr>
<tr><td>Drift Score</td><td>""" + str(trail["drift_monitoring"]["drift_score"]) + """</td></tr>
</table>
</div>

<div class="section">
<h2>Top Features</h2>
<p>""" + " | ".join(["<strong>" + f + "</strong>" for f in trail["explainability"]["top_features"][:5]]) + """</p>
</div>

<div class="section">
<h2>Pipeline Decision Log</h2>
<table>
<tr><th>Log Entry</th></tr>
""" + "".join(["<tr><td>" + str(log)[:200] + "</td></tr>" for log in trail["decision_log"][-10:]]) + """
</table>
</div>

<div class="footer">
<p>Generated by VEDA Autonomous Data Science System</p>
<p>Data Hash: """ + data_hash + """ | This report is auto-generated and should be reviewed by a qualified compliance officer.</p>
</div>
</body>
</html>"""
        return html

    def run(self, state: dict) -> dict:
        """
        Audit Trail:
        1. Build complete audit trail
        2. Generate audit certificate
        3. Create HTML audit report
        4. Save all artifacts
        """

        self.log("Building complete audit trail...")
        trail = self._build_audit_trail(state)

        self.log("Generating audit certificate with Groq...")
        certificate = self._generate_audit_certificate(trail)

        self.log("Building HTML audit report...")
        html_report = self._generate_html_audit_report(trail, certificate)

        os.makedirs("outputs", exist_ok=True)
        run_id = state.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))

        trail_path = "outputs/" + run_id + "_audit_trail.json"
        with open(trail_path, "w") as f:
            json.dump(trail, f, indent=2, default=str)

        report_path = "outputs/" + run_id + "_audit_report.html"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_report)

        state["audit_trail"] = trail
        state["audit_certificate"] = certificate
        state.setdefault("planner_decision_log", []).append(
            "[" + datetime.now().isoformat() + "] AuditTrailAgent: audit complete, hash=" +
            trail["data_hash"]
        )

        self.log("=" * 50)
        self.log("AUDIT TRAIL COMPLETE")
        self.log("Data hash   : " + trail["data_hash"])
        self.log("Trail saved : " + trail_path)
        self.log("Report saved: " + report_path)
        self.log("Certificate : " + certificate[:100] + "...")
        self.log("=" * 50)

        return state