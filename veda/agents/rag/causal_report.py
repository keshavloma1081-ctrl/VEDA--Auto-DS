"""
VEDA — Autonomous Data Science System
agents/rag/causal_report.py — Causal Report Agent

Generates causal analysis report:
- Summarizes all causal findings
- Business recommendations
- Intervention suggestions
- HTML causal report
"""

import os
import json
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq

from veda.core.base_agent import BaseAgent

load_dotenv()


class CausalReportAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name="CausalReportAgent",
            domain="causal",
            version="1.0.0"
        )
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.3-70b-versatile"

    def _generate_causal_summary(self, state: dict) -> str:
        """Generate LLM summary of all causal findings."""
        causal_graph = state.get("causal_graph", {})
        uplift = state.get("uplift_results", {})
        ab_results = state.get("ab_results", {})
        causal_inf = state.get("causal_inference", {})
        goal = state.get("goal", "")

        prompt = """You are a causal inference expert. Summarize these findings.

Goal: """ + goal + """

Causal Graph top causes: """ + str(causal_graph.get("top_causal_features", [])[:3]) + """

Uplift Model:
- S-Learner CATE: """ + str(uplift.get("s_learner", {}).get("avg_cate", "N/A")) + """
- Positive uplift: """ + str(uplift.get("s_learner", {}).get("positive_uplift_pct", "N/A")) + """%

A/B Test:
- Lift: """ + str(ab_results.get("summary", {}).get("relative_lift_pct", "N/A")) + """%
- Recommendation: """ + str(ab_results.get("bayesian", {}).get("recommendation", "N/A")) + """

Causal Inference ATE: """ + str(causal_inf.get("naive_ate", {}).get("ate", "N/A")) + """

Write 4 sentences covering:
1. Main causal drivers identified
2. Treatment effect magnitude
3. A/B test conclusion
4. Recommended business interventions"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=400
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return "Causal analysis complete. Review findings for business interventions."

    def _build_html_report(self, state: dict, summary: str) -> str:
        """Build HTML causal report."""
        causal_graph = state.get("causal_graph", {})
        uplift = state.get("uplift_results", {})
        ab_results = state.get("ab_results", {})
        causal_inf = state.get("causal_inference", {})
        goal = state.get("goal", "")
        run_id = state.get("run_id", "")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        s_cate = uplift.get("s_learner", {}).get("avg_cate", "N/A")
        t_cate = uplift.get("t_learner", {}).get("avg_cate", "N/A")
        lift = ab_results.get("summary", {}).get("relative_lift_pct", "N/A")
        ab_rec = ab_results.get("bayesian", {}).get("recommendation", "N/A")
        naive_ate = causal_inf.get("naive_ate", {}).get("ate", "N/A")
        psm_ate = causal_inf.get("psm_ate", {}).get("ate", "N/A")
        top_causes = causal_graph.get("top_causal_features", [])[:5]
        dag_edges = causal_graph.get("dag", {}).get("edges", [])[:5]

        edges_html = "".join([
            "<tr><td>" + e["from"] + "</td><td>" + e["to"] + "</td><td>" +
            str(round(e["weight"], 4)) + "</td></tr>"
            for e in dag_edges
        ])

        causes_tags = " ".join([
            "<span style='background:#1F4E79;color:white;padding:3px 10px;border-radius:20px;font-size:11px;margin:2px'>" +
            str(c) + "</span>"
            for c in top_causes
        ])

        rec_color = "#2e7d32" if "TREATMENT" in str(ab_rec) else "#e65100"

        html = """<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>VEDA Causal Analysis Report</title>
<style>
body{font-family:Arial,sans-serif;margin:40px;color:#1a1a2e;}
.header{background:#1F4E79;color:white;padding:25px;border-radius:10px;margin-bottom:20px;}
.header h1{margin:0;font-size:22px;}
.header p{margin:4px 0 0;opacity:.85;font-size:12px;}
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:18px;}
.card{background:#f0f4f8;padding:14px;border-radius:8px;text-align:center;}
.card .val{font-size:22px;font-weight:bold;color:#1F4E79;}
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
<h1>VEDA — Causal Analysis Report</h1>
<p>Goal: """ + goal + """ | Run ID: """ + run_id + """ | Generated: """ + timestamp + """</p>
</div>

<div class="grid">
<div class="card"><div class="val">""" + str(s_cate) + """</div><div class="lbl">S-Learner CATE</div></div>
<div class="card"><div class="val">""" + str(lift) + """%</div><div class="lbl">A/B Test Lift</div></div>
<div class="card"><div class="val">""" + str(naive_ate) + """</div><div class="lbl">Naive ATE</div></div>
</div>

<div class="section">
<h2>Executive Summary</h2>
<div class="box">""" + summary.replace("\n", "<br>") + """</div>
</div>

<div class="section">
<h2>Top Causal Drivers</h2>
<p>""" + causes_tags + """</p>
<table>
<tr><th>From</th><th>To</th><th>Strength</th></tr>
""" + edges_html + """
</table>
</div>

<div class="section">
<h2>Uplift Model Results</h2>
<table>
<tr><th>Method</th><th>CATE</th><th>Positive Uplift %</th></tr>
<tr><td>S-Learner</td><td>""" + str(s_cate) + """</td><td>""" + str(uplift.get("s_learner", {}).get("positive_uplift_pct", "N/A")) + """%</td></tr>
<tr><td>T-Learner</td><td>""" + str(t_cate) + """</td><td>""" + str(uplift.get("t_learner", {}).get("positive_uplift_pct", "N/A")) + """%</td></tr>
</table>
</div>

<div class="section">
<h2>A/B Test Results</h2>
<p>Recommendation: <strong style="color:""" + rec_color + """">""" + str(ab_rec) + """</strong></p>
<table>
<tr><th>Metric</th><th>Control</th><th>Treatment</th></tr>
<tr><td>Conversion Rate</td><td>""" + str(ab_results.get("summary", {}).get("control_conversion_rate", "N/A")) + """</td><td>""" + str(ab_results.get("summary", {}).get("treatment_conversion_rate", "N/A")) + """</td></tr>
<tr><td>Relative Lift</td><td colspan="2">""" + str(lift) + """%</td></tr>
<tr><td>Chi-square p-value</td><td colspan="2">""" + str(ab_results.get("chi_square", {}).get("p_value", "N/A")) + """</td></tr>
<tr><td>P(treatment better)</td><td colspan="2">""" + str(ab_results.get("bayesian", {}).get("prob_treatment_better", "N/A")) + """</td></tr>
</table>
</div>

<div class="section">
<h2>Causal Inference</h2>
<table>
<tr><th>Method</th><th>ATE</th><th>Significant</th></tr>
<tr><td>Naive Difference</td><td>""" + str(naive_ate) + """</td><td>""" + str(causal_inf.get("naive_ate", {}).get("significant", "N/A")) + """</td></tr>
<tr><td>Propensity Score Matching</td><td>""" + str(psm_ate) + """</td><td>N/A</td></tr>
</table>
</div>

<div class="footer">
<p>Generated by VEDA Autonomous Data Science System</p>
</div>
</body>
</html>"""
        return html

    def run(self, state: dict) -> dict:
        self.log("Generating causal analysis summary...")
        summary = self._generate_causal_summary(state)
        self.log("Summary: " + summary[:100])

        self.log("Building HTML causal report...")
        html = self._build_html_report(state, summary)

        os.makedirs("outputs", exist_ok=True)
        run_id = state.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
        report_path = "outputs/" + run_id + "_causal_report.html"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html)

        causal_summary = {
            "summary": summary,
            "report_path": report_path,
            "key_findings": {
                "top_causes": state.get("causal_graph", {}).get("top_causal_features", [])[:3],
                "ab_recommendation": state.get("ab_results", {}).get("bayesian", {}).get("recommendation", "N/A"),
                "uplift": state.get("uplift_results", {}).get("s_learner", {}).get("avg_cate", "N/A"),
                "ate": state.get("causal_inference", {}).get("naive_ate", {}).get("ate", "N/A")
            }
        }

        state["causal_summary"] = causal_summary
        state.setdefault("planner_decision_log", []).append(
            "[" + datetime.now().isoformat() + "] CausalReportAgent: report saved to " + report_path
        )

        self.log("=" * 50)
        self.log("CAUSAL REPORT COMPLETE")
        self.log("Report  : " + report_path)
        self.log("Summary : " + summary[:100] + "...")
        self.log("=" * 50)

        return state