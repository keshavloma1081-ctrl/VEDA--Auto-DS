import os
import json
from datetime import datetime
from groq import Groq
from dotenv import load_dotenv
from veda.core.base_agent import BaseAgent

load_dotenv()

class ReportAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="ReportAgent", domain="reports", version="1.0.0")
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.3-70b-versatile"

    def _load_json(self, suffix):
        d = "outputs"
        files = [f for f in os.listdir(d) if f.endswith(suffix)]
        if not files:
            return {}
        with open(os.path.join(d, sorted(files)[-1])) as f:
            return json.load(f)

    def _generate_summary(self, goal, model_name, metrics, explanation, top_features):
        prompt = "You are VEDA. Write 3 bullet points (starting with -) summarizing this ML project.\n"
        prompt += "Goal: " + goal + "\n"
        prompt += "Model: " + model_name + "\n"
        prompt += "AUC=" + str(metrics.get("auc_roc", "N/A")) + " F1=" + str(metrics.get("f1_score", "N/A")) + "\n"
        prompt += "Top features: " + str(top_features[:3]) + "\n"
        prompt += "Be concise and professional."
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()

    def run(self, state):
        goal = state.get("goal", "")
        model_info = state.get("model_info", {})
        model_name = model_info.get("model_name", "LightGBM")
        data_profile = state.get("data_profile", {})
        cleaning_diff = state.get("cleaning_diff", [])
        feature_list = state.get("feature_list", [])

        self.log("Loading pipeline outputs...")
        eval_data = self._load_json("_evaluation.json")
        exp_data = self._load_json("_explainability.json")

        metrics = eval_data
        explanation = exp_data.get("explanation_text", "No explanation.")
        top_features = exp_data.get("top_features", [])
        passed = metrics.get("passed_threshold", False)
        eval_status = "PASSED" if passed else "FAILED"

        auc = str(metrics.get("auc_roc", "N/A"))
        f1 = str(metrics.get("f1_score", "N/A"))
        acc = str(metrics.get("accuracy", "N/A"))
        prec = str(metrics.get("precision", "N/A"))
        rec = str(metrics.get("recall", "N/A"))

        self.log("Generating executive summary...")
        summary = self._generate_summary(goal, model_name, metrics, explanation, top_features)
        summary_html = summary.replace("-", "<br>-").replace("\n", "<br>")

        cleaning_rows = "".join(["<tr><td>" + c + "</td></tr>" for c in cleaning_diff])
        feature_rows = "".join(["<tr><td>" + str(f) + "</td></tr>" for f in feature_list[:10]])
        feature_tags = " ".join(["<span style='background:#1F4E79;color:white;padding:3px 10px;border-radius:20px;font-size:11px;margin:2px'>" + str(f) + "</span>" for f in top_features[:5]])
        explanation_clean = explanation.replace("\n", "<br>")
        run_id = state.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        rows = str(data_profile.get("row_count", "N/A"))
        cols = str(data_profile.get("col_count", "N/A"))
        target = str(data_profile.get("target_column", "N/A"))
        imbalance = str(data_profile.get("has_imbalance", False))
        leakage = str(data_profile.get("has_leakage_risk", False))
        n_changes = str(len(cleaning_diff))
        n_features = str(len(feature_list))
        status_color = "#2e7d32" if passed else "#c62828"
        status_bg = "#e8f5e9" if passed else "#ffebee"

        html = (
            "<!DOCTYPE html><html><head><meta charset=UTF-8>"
            "<title>VEDA AutoDS Report</title><style>"
            "body{font-family:Arial,sans-serif;margin:40px;color:#1a1a2e;}"
            ".header{background:#1F4E79;color:white;padding:30px;border-radius:10px;margin-bottom:30px;}"
            ".header h1{margin:0;font-size:28px;}"
            ".header p{margin:5px 0 0;opacity:.85;font-size:14px;}"
            ".section{margin-bottom:25px;padding:20px;border:1px solid #e0e0e0;border-radius:8px;}"
            ".section h2{color:#1F4E79;border-bottom:2px solid #1F4E79;padding-bottom:8px;}"
            ".grid{display:grid;grid-template-columns:repeat(3,1fr);gap:15px;margin-top:15px;}"
            ".card{background:#f0f4f8;padding:15px;border-radius:8px;text-align:center;}"
            ".card .val{font-size:28px;font-weight:bold;color:#1F4E79;}"
            ".card .lbl{font-size:12px;color:#666;margin-top:5px;}"
            "table{width:100%;border-collapse:collapse;margin-top:10px;}"
            "th{background:#1F4E79;color:white;padding:10px;text-align:left;}"
            "td{padding:8px 10px;border-bottom:1px solid #eee;}"
            "tr:nth-child(even){background:#f9f9f9;}"
            ".box{background:#e3f2fd;padding:20px;border-radius:8px;border-left:4px solid #1F4E79;}"
            ".footer{text-align:center;color:#999;font-size:12px;margin-top:40px;padding-top:20px;border-top:1px solid #eee;}"
            "</style></head><body>"
            "<div class=header>"
            "<h1>VEDA Autonomous Data Science Report</h1>"
            "<p>Goal: " + goal + "</p>"
            "<p>Generated: " + timestamp + " | Run ID: " + run_id + "</p>"
            "</div>"
            "<div class=section><h2>Executive Summary</h2>"
            "<div class=box>" + summary_html + "</div></div>"
            "<div class=section><h2>Model Performance</h2>"
            "<p>Status: <span style='background:" + status_bg + ";color:" + status_color + ";padding:8px 15px;border-radius:20px;font-weight:bold'>" + eval_status + " (AUC threshold 0.70)</span></p>"
            "<div class=grid>"
            "<div class=card><div class=val>" + auc + "</div><div class=lbl>AUC-ROC</div></div>"
            "<div class=card><div class=val>" + f1 + "</div><div class=lbl>F1 Score</div></div>"
            "<div class=card><div class=val>" + acc + "</div><div class=lbl>Accuracy</div></div>"
            "<div class=card><div class=val>" + prec + "</div><div class=lbl>Precision</div></div>"
            "<div class=card><div class=val>" + rec + "</div><div class=lbl>Recall</div></div>"
            "<div class=card><div class=val>" + model_name + "</div><div class=lbl>Model</div></div>"
            "</div></div>"
            "<div class=section><h2>Data Profile</h2>"
            "<table><tr><th>Attribute</th><th>Value</th></tr>"
            "<tr><td>Rows</td><td>" + rows + "</td></tr>"
            "<tr><td>Columns</td><td>" + cols + "</td></tr>"
            "<tr><td>Target Column</td><td>" + target + "</td></tr>"
            "<tr><td>Class Imbalance</td><td>" + imbalance + "</td></tr>"
            "<tr><td>Leakage Risk</td><td>" + leakage + "</td></tr>"
            "</table></div>"
            "<div class=section><h2>Data Cleaning</h2>"
            "<p>" + n_changes + " changes made automatically</p>"
            "<table><tr><th>Change</th></tr>" + cleaning_rows + "</table></div>"
            "<div class=section><h2>Feature Engineering</h2>"
            "<p>" + n_features + " features in final model</p>"
            "<p>Top features: " + feature_tags + "</p>"
            "<table><tr><th>Feature</th></tr>" + feature_rows + "</table></div>"
            "<div class=section><h2>Model Explanation</h2>"
            "<div class=box>" + explanation_clean + "</div></div>"
            "<div class=footer>"
            "<p>Generated by VEDA Autonomous Data Science System</p>"
            "<p>github.com/keshavloma1081-ctrl/VEDA--Auto-DS</p>"
            "</div></body></html>"
        )

        os.makedirs("outputs", exist_ok=True)
        report_path = "outputs/" + run_id + "_report.html"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html)

        state.setdefault("outputs", {})
        state["outputs"]["executive_report_path"] = report_path
        state["pipeline_complete"] = True
        state.setdefault("planner_decision_log", []).append(
            "[" + datetime.now().isoformat() + "] ReportAgent: report saved to " + report_path
        )

        self.log("REPORT COMPLETE — saved to: " + report_path)
        return state
