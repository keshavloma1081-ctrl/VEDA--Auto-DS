"""
VEDA — Autonomous Data Science System
agents/special/rbi_compliance.py — RBI Compliance Agent

Checks ML models for RBI (Reserve Bank of India) compliance:
- Fair lending practices
- Model explainability requirements
- Credit scoring guidelines
- DPDP Act (Digital Personal Data Protection)
- Bias detection
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq

from veda.core.base_agent import BaseAgent

load_dotenv()


class RBIComplianceAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name="RBIComplianceAgent",
            domain="compliance",
            version="1.0.0"
        )
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.3-70b-versatile"

    def _check_model_explainability(self, state: dict) -> dict:
        """Check if model has explainability."""
        explainability = state.get("explainability", {})
        has_shap = bool(explainability.get("feature_importance"))
        has_explanation = bool(explainability.get("explanation_text"))

        issues = []
        if not has_shap:
            issues.append("WARNING: No SHAP values found — RBI requires model explainability for credit decisions")
        if not has_explanation:
            issues.append("WARNING: No plain-English explanation — required for adverse action notices")

        return {
            "has_shap": has_shap,
            "has_explanation": has_explanation,
            "top_features": explainability.get("top_features", []),
            "issues": issues,
            "passed": has_shap and has_explanation
        }

    def _check_fair_lending(self, df: pd.DataFrame) -> dict:
        """Check for protected attributes in model features."""
        protected_attributes = [
            "gender", "sex", "religion", "caste", "race", "ethnicity",
            "nationality", "marital_status", "disability", "age"
        ]

        found_protected = []
        for col in df.columns:
            if any(attr in col.lower() for attr in protected_attributes):
                found_protected.append(col)

        issues = []
        if found_protected:
            issues.append(
                "FAIR LENDING: Protected attributes found in features: " +
                str(found_protected) + " — ensure these don't cause discriminatory outcomes"
            )

        return {
            "protected_attributes_found": found_protected,
            "issues": issues,
            "passed": len(found_protected) == 0,
            "recommendation": "Remove or carefully audit protected attribute usage" if found_protected else "No protected attributes found"
        }

    def _check_model_performance_threshold(self, state: dict) -> dict:
        """Check if model meets minimum performance thresholds."""
        model_info = state.get("model_info", {})
        metrics = model_info.get("test_metrics", {})
        auc = metrics.get("auc_roc", 0)
        f1 = metrics.get("f1_score", 0)

        issues = []
        if auc < 0.7:
            issues.append("CRITICAL: AUC " + str(auc) + " below RBI minimum threshold of 0.70 for credit models")
        if f1 < 0.5:
            issues.append("WARNING: F1 " + str(f1) + " may indicate poor minority class prediction")

        return {
            "auc": auc,
            "f1": f1,
            "issues": issues,
            "passed": auc >= 0.7
        }

    def _check_dpdp_compliance(self, df: pd.DataFrame) -> dict:
        """Check DPDP Act compliance (India's data protection law)."""
        issues = []
        sensitive_cols = []

        dpdp_sensitive = [
            "health", "financial", "biometric", "caste", "religion",
            "political", "sexual", "transgender", "intersex", "disability",
            "aadhaar", "pan", "passport"
        ]

        for col in df.columns:
            if any(kw in col.lower() for kw in dpdp_sensitive):
                sensitive_cols.append(col)

        if sensitive_cols:
            issues.append(
                "DPDP Act: Sensitive personal data found: " + str(sensitive_cols) +
                " — requires explicit consent under DPDP Act 2023"
            )

        data_fiduciary_obligations = [
            "Appoint Data Protection Officer if processing sensitive data at scale",
            "Maintain records of data processing activities",
            "Implement data localisation requirements for sensitive data",
            "Enable data principal rights: access, correction, erasure, grievance"
        ]

        return {
            "sensitive_cols_found": sensitive_cols,
            "issues": issues,
            "dpdp_obligations": data_fiduciary_obligations,
            "passed": len(sensitive_cols) == 0
        }

    def _generate_rbi_assessment(self, checks: dict, goal: str, model_name: str) -> str:
        """Generate RBI compliance assessment."""
        prompt = """You are an RBI (Reserve Bank of India) compliance expert.
Assess this ML model for regulatory compliance.

Model goal: """ + goal + """
Model: """ + model_name + """

Compliance checks:
""" + json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "dpdp_obligations"} for k, v in checks.items()}, indent=2)[:1500] + """

Write a 3-4 sentence RBI compliance assessment covering:
1. Overall compliance status for Indian fintech deployment
2. Critical issues that must be resolved
3. Specific RBI/DPDP Act requirements to address

Be specific to Indian regulatory context."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=300
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return "RBI assessment generation failed: " + str(e)

    def run(self, state: dict) -> dict:
        """
        RBI Compliance Check:
        1. Check model explainability
        2. Check fair lending
        3. Check performance thresholds
        4. Check DPDP compliance
        5. Generate assessment
        """

        self.log("Running RBI compliance checks...")
        d = "outputs"
        files = [f for f in os.listdir(d) if f.endswith("_features.parquet")]
        if not files:
            files = [f for f in os.listdir(d) if f.endswith("_data.parquet")]

        df = pd.DataFrame()
        if files:
            df = pd.read_parquet(os.path.join(d, sorted(files)[-1]))

        goal = state.get("goal", "")
        model_name = state.get("model_info", {}).get("model_name", "Unknown")

        checks = {
            "explainability": self._check_model_explainability(state),
            "fair_lending": self._check_fair_lending(df),
            "performance": self._check_model_performance_threshold(state),
            "dpdp_act": self._check_dpdp_compliance(df)
        }

        total_issues = sum(len(c.get("issues", [])) for c in checks.values())
        critical_issues = [
            issue for c in checks.values()
            for issue in c.get("issues", [])
            if "CRITICAL" in issue
        ]

        self.log("Generating RBI assessment...")
        assessment = self._generate_rbi_assessment(checks, goal, model_name)

        overall_status = "COMPLIANT" if total_issues == 0 else \
                        "CRITICAL" if critical_issues else "NEEDS_ATTENTION"

        rbi_report = {
            "goal": goal,
            "model": model_name,
            "checks": checks,
            "total_issues": total_issues,
            "critical_issues": critical_issues,
            "overall_status": overall_status,
            "assessment": assessment
        }

        os.makedirs("outputs", exist_ok=True)
        run_id = state.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
        report_path = "outputs/" + run_id + "_rbi_report.json"
        with open(report_path, "w") as f:
            json.dump(rbi_report, f, indent=2, default=str)

        state["rbi_report"] = rbi_report
        state.setdefault("planner_decision_log", []).append(
            "[" + datetime.now().isoformat() + "] RBIComplianceAgent: status=" +
            overall_status + " issues=" + str(total_issues)
        )

        self.log("=" * 50)
        self.log("RBI COMPLIANCE CHECK COMPLETE")
        self.log("Status   : " + overall_status)
        self.log("Issues   : " + str(total_issues))
        self.log("Critical : " + str(len(critical_issues)))
        self.log("Assessment: " + assessment[:150] + "...")
        self.log("=" * 50)

        return state