"""
VEDA — Autonomous Data Science System
agents/synthetic/synthetic_report.py — Synthetic Data Report Agent

Generates comprehensive synthetic data report:
- Generation quality summary
- Privacy assessment
- Fidelity analysis
- Usage recommendations
"""

import os
import json
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq

from veda.core.base_agent import BaseAgent

load_dotenv()


class SyntheticReportAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name="SyntheticReportAgent",
            domain="synthetic",
            version="1.0.0"
        )
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.3-70b-versatile"

    def _generate_summary(self, state: dict) -> str:
        synthetic = state.get("synthetic_results", {})
        privacy = state.get("privacy_results", {})
        fidelity = state.get("fidelity_results", {})
        augmentation = state.get("augmentation_results", {})

        prompt = """You are a synthetic data expert. Summarize these results.

Generation method: """ + str(synthetic.get("method", "N/A")) + """
Rows generated: """ + str(synthetic.get("synthetic_rows", "N/A")) + """

Fidelity score: """ + str(fidelity.get("fidelity_score", "N/A")) + """%
Grade: """ + str(fidelity.get("grade", "N/A")) + """

Privacy risk: """ + str(privacy.get("overall_privacy_risk", "N/A")) + """
Privacy score: """ + str(privacy.get("privacy_score", "N/A")) + """%

Augmentation: """ + str(augmentation.get("final_train_size", "N/A")) + """ samples

Write 4 sentences:
1. Quality of synthetic data generated
2. Privacy assessment finding
3. Statistical fidelity result
4. Recommendation for using this synthetic data"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=300
            )
            return response.choices[0].message.content.strip()
        except:
            return "Synthetic data generation complete with " + \
                   str(fidelity.get("grade", "N/A")) + " fidelity grade."

    def _build_html_report(self, state: dict, summary: str) -> str:
        synthetic = state.get("synthetic_results", {})
        privacy = state.get("privacy_results", {})
        fidelity = state.get("fidelity_results", {})
        augmentation = state.get("augmentation_results", {})
        run_id = state.get("run_id", "")

        fidelity_score = fidelity.get("fidelity_score", 0)
        privacy_score = privacy.get("privacy_score", 0)
        grade = fidelity.get("grade", "N/A")
        privacy_risk = privacy.get("overall_privacy_risk", "N/A")
        method = synthetic.get("method", "N/A")
        synth_rows = synthetic.get("synthetic_rows", 0)
        real_rows = synthetic.get("real_rows", 0)

        score_color = "#2e7d32" if fidelity_score >= 80 else "#e65100" if fidelity_score >= 60 else "#c62828"
        privacy_color = "#2e7d32" if privacy_risk == "LOW" else "#e65100" if privacy_risk == "MEDIUM" else "#c62828"

        html = """<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>VEDA Synthetic Data Report</title>
<style>
body{font-family:Arial,sans-serif;margin:40px;color:#1a1a2e;}
.header{background:#1F4E79;color:white;padding:25px;border-radius:10px;margin-bottom:20px;}
.header h1{margin:0;font-size:22px;}
.header p{margin:4px 0 0;opacity:.85;font-size:12px;}
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:18px;}
.card{background:#f0f4f8;padding:14px;border-radius:8px;text-align:center;}
.card .val{font-size:22px;font-weight:bold;}
.card .lbl{font-size:11px;color:#666;margin-top:3px;}
.section{margin-bottom:18px;padding:16px;border:1px solid #e0e0e0;border-radius:8px;}
.section h2{color:#1F4E79;border-bottom:2px solid #1F4E79;padding-bottom:6px;margin-top:0;}
.box{background:#e3f2fd;padding:14px;border-radius:8px;border-left:4px solid #1F4E79;}
table{width:100%;border-collapse:collapse;margin-top:8px;}
th{background:#1F4E79;color:white;padding:8px;text-align:left;font-size:12px;}
td{padding:6px 8px;border-bottom:1px solid #eee;font-size:12px;}
.footer{text-align:center;color:#999;font-size:11px;margin-top:30px;padding-top:15px;border-top:1px solid #eee;}
</style>
</head>
<body>
<div class="header">
<h1>VEDA — Synthetic Data Report</h1>
<p>Run ID: """ + run_id + """ | Generated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
</div>

<div class="grid">
<div class="card"><div class="val" style="color:""" + score_color + """">""" + str(fidelity_score) + """%</div><div class="lbl">Fidelity Score</div></div>
<div class="card"><div class="val" style="color:""" + score_color + """">""" + str(grade) + """</div><div class="lbl">Fidelity Grade</div></div>
<div class="card"><div class="val" style="color:""" + privacy_color + """">""" + str(privacy_risk) + """</div><div class="lbl">Privacy Risk</div></div>
<div class="card"><div class="val">""" + str(synth_rows) + """</div><div class="lbl">Synthetic Rows</div></div>
</div>

<div class="section">
<h2>Executive Summary</h2>
<div class="box">""" + summary.replace("\n", "<br>") + """</div>
</div>

<div class="section">
<h2>Generation Results</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Method</td><td>""" + str(method) + """</td></tr>
<tr><td>Real Rows</td><td>""" + str(real_rows) + """</td></tr>
<tr><td>Synthetic Rows</td><td>""" + str(synth_rows) + """</td></tr>
<tr><td>Augmented Samples</td><td>""" + str(augmentation.get("final_train_size", "N/A")) + """</td></tr>
</table>
</div>

<div class="section">
<h2>Privacy Assessment</h2>
<table>
<tr><th>Test</th><th>Risk</th><th>Details</th></tr>
<tr><td>Distance to Closest Record</td><td>""" + str(privacy.get("distance_to_closest_record", {}).get("risk_level", "N/A")) + """</td><td>mean_dcr=""" + str(privacy.get("distance_to_closest_record", {}).get("mean_dcr", "N/A")) + """</td></tr>
<tr><td>Membership Inference</td><td>""" + str(privacy.get("membership_inference", {}).get("risk_level", "N/A")) + """</td><td>attack_auc=""" + str(privacy.get("membership_inference", {}).get("attack_auc", "N/A")) + """</td></tr>
<tr><td>Attribute Disclosure</td><td>""" + str(privacy.get("attribute_disclosure", {}).get("risk_level", "N/A")) + """</td><td>issues=""" + str(privacy.get("attribute_disclosure", {}).get("issues_found", 0)) + """</td></tr>
</table>
</div>

<div class="footer">
<p>Generated by VEDA Autonomous Data Science System</p>
</div>
</body>
</html>"""
        return html

    def run(self, state: dict) -> dict:
        self.log("Generating synthetic data summary...")
        summary = self._generate_summary(state)

        self.log("Building HTML report...")
        html = self._build_html_report(state, summary)

        os.makedirs("outputs", exist_ok=True)
        run_id = state.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
        report_path = "outputs/" + run_id + "_synthetic_report.html"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html)

        fidelity = state.get("fidelity_results", {})
        privacy = state.get("privacy_results", {})

        report = {
            "summary": summary,
            "report_path": report_path,
            "fidelity_score": fidelity.get("fidelity_score", 0),
            "privacy_risk": privacy.get("overall_privacy_risk", "N/A"),
            "grade": fidelity.get("grade", "N/A")
        }

        state["synthetic_report"] = report
        state.setdefault("planner_decision_log", []).append(
            "[" + datetime.now().isoformat() + "] SyntheticReportAgent: " +
            "fidelity=" + str(fidelity.get("fidelity_score", 0)) +
            "% privacy=" + str(privacy.get("overall_privacy_risk", "N/A"))
        )

        self.log("=" * 50)
        self.log("SYNTHETIC REPORT COMPLETE")
        self.log("Fidelity  : " + str(fidelity.get("fidelity_score", 0)) + "%")
        self.log("Privacy   : " + str(privacy.get("overall_privacy_risk", "N/A")))
        self.log("Report    : " + report_path)
        self.log("=" * 50)

        return state