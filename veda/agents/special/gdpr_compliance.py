"""
VEDA — Autonomous Data Science System
agents/special/gdpr_compliance.py — GDPR Compliance Agent

Checks dataset and pipeline for GDPR compliance:
- Right to erasure
- Data minimization
- Purpose limitation
- Consent tracking
- Data retention
"""

import os
import json
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq

from veda.core.base_agent import BaseAgent

load_dotenv()


class GDPRComplianceAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name="GDPRComplianceAgent",
            domain="compliance",
            version="1.0.0"
        )
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.3-70b-versatile"

    def _check_data_minimization(self, df: pd.DataFrame, goal: str) -> dict:
        """Check if only necessary data is collected."""
        total_cols = len(df.columns)
        issues = []

        sensitive_cols = []
        for col in df.columns:
            col_lower = col.lower()
            if any(kw in col_lower for kw in [
                "religion", "race", "ethnicity", "political", "health",
                "biometric", "sexual", "genetic", "criminal"
            ]):
                sensitive_cols.append(col)
                issues.append("SENSITIVE: Column '" + col + "' contains special category data (GDPR Art. 9)")

        return {
            "total_columns": total_cols,
            "sensitive_columns": sensitive_cols,
            "issues": issues,
            "passed": len(sensitive_cols) == 0
        }

    def _check_data_retention(self, df: pd.DataFrame) -> dict:
        """Check for date columns that indicate data age."""
        issues = []
        date_cols = []

        for col in df.columns:
            if df[col].dtype == "datetime64[ns]" or "date" in col.lower():
                date_cols.append(col)

        if not date_cols:
            issues.append("INFO: No date columns found — cannot verify retention period")

        return {
            "date_columns": date_cols,
            "issues": issues,
            "recommendation": "Implement data retention policy — delete records older than defined period"
        }

    def _check_consent(self, df: pd.DataFrame) -> dict:
        """Check for consent tracking columns."""
        consent_cols = [col for col in df.columns if "consent" in col.lower()]
        issues = []

        if not consent_cols:
            issues.append("WARNING: No consent tracking column found — ensure data collection has valid consent basis")

        return {
            "consent_columns": consent_cols,
            "issues": issues,
            "passed": len(consent_cols) > 0
        }

    def _check_right_to_erasure(self, df: pd.DataFrame) -> dict:
        """Check if individual records can be identified and erased."""
        id_cols = [col for col in df.columns if any(kw in col.lower()
                   for kw in ["id", "uuid", "user_id", "customer_id", "email"])]

        return {
            "identifier_columns": id_cols,
            "erasure_possible": len(id_cols) > 0,
            "recommendation": "Ensure deletion API exists for identified records" if id_cols
                             else "WARNING: No identifier columns — erasure requests cannot be fulfilled"
        }

    def _generate_gdpr_assessment(self, checks: dict, goal: str) -> str:
        """Generate GDPR compliance assessment using Groq."""
        prompt = """You are a GDPR compliance expert. Assess this ML project.

Goal: """ + goal + """

Compliance checks:
""" + json.dumps(checks, indent=2)[:1500] + """

Write a 3-4 sentence GDPR compliance assessment covering:
1. Overall compliance status
2. Key risks identified
3. Required actions before deployment

Be specific and actionable."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=300
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return "GDPR assessment generation failed: " + str(e)

    def run(self, state: dict) -> dict:
        """
        GDPR Compliance Check:
        1. Load data
        2. Check data minimization
        3. Check data retention
        4. Check consent tracking
        5. Check right to erasure
        6. Generate compliance report
        """

        self.log("Loading data for GDPR compliance check...")
        d = "outputs"
        files = [f for f in os.listdir(d) if f.endswith("_data.parquet")]
        if not files:
            self.log("No data found", level="WARN")
            return state

        df = pd.read_parquet(os.path.join(d, sorted(files)[-1]))
        goal = state.get("goal", "")

        self.log("Running GDPR compliance checks...")

        checks = {
            "data_minimization": self._check_data_minimization(df, goal),
            "data_retention": self._check_data_retention(df),
            "consent_tracking": self._check_consent(df),
            "right_to_erasure": self._check_right_to_erasure(df)
        }

        # Count issues
        total_issues = sum(len(c.get("issues", [])) for c in checks.values())
        passed_checks = sum(1 for c in checks.values() if c.get("passed", True))

        self.log("Generating GDPR assessment...")
        assessment = self._generate_gdpr_assessment(checks, goal)

        gdpr_report = {
            "goal": goal,
            "dataset_shape": list(df.shape),
            "checks": checks,
            "total_issues": total_issues,
            "passed_checks": passed_checks,
            "overall_status": "COMPLIANT" if total_issues == 0 else "NEEDS_ATTENTION" if total_issues < 3 else "NON_COMPLIANT",
            "assessment": assessment
        }

        os.makedirs("outputs", exist_ok=True)
        run_id = state.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
        report_path = "outputs/" + run_id + "_gdpr_report.json"
        with open(report_path, "w") as f:
            json.dump(gdpr_report, f, indent=2, default=str)

        state["gdpr_report"] = gdpr_report
        state.setdefault("planner_decision_log", []).append(
            "[" + datetime.now().isoformat() + "] GDPRComplianceAgent: status=" +
            gdpr_report["overall_status"] + " issues=" + str(total_issues)
        )

        self.log("=" * 50)
        self.log("GDPR COMPLIANCE CHECK COMPLETE")
        self.log("Status  : " + gdpr_report["overall_status"])
        self.log("Issues  : " + str(total_issues))
        self.log("Assessment: " + assessment[:150] + "...")
        self.log("=" * 50)

        return state