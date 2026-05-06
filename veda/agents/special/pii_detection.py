"""
VEDA — Autonomous Data Science System
agents/special/pii_detection.py — PII Detection Agent

Detects Personal Identifiable Information:
- Names, emails, phone numbers
- Aadhaar, PAN, passport numbers
- Credit card numbers
- IP addresses
- Using Microsoft Presidio + regex patterns
"""

import os
import re
import json
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

from veda.core.base_agent import BaseAgent

load_dotenv()


class PIIDetectionAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name="PIIDetectionAgent",
            domain="compliance",
            version="1.0.0"
        )
        self.pii_patterns = {
            "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "PHONE_IN": r"\b[6-9]\d{9}\b",
            "PHONE_INTL": r"\+\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}",
            "AADHAAR": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
            "PAN": r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
            "CREDIT_CARD": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
            "IP_ADDRESS": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
            "PASSPORT": r"\b[A-Z]{1,2}[0-9]{6,7}\b",
            "GST": r"\b\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z\d]{1}[Z]{1}[A-Z\d]{1}\b",
            "IFSC": r"\b[A-Z]{4}0[A-Z0-9]{6}\b",
            "PINCODE": r"\b[1-9][0-9]{5}\b"
        }

    def _detect_in_text(self, text: str) -> dict:
        """Detect PII in a single text string."""
        if not isinstance(text, str):
            return {}
        findings = {}
        for pii_type, pattern in self.pii_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                findings[pii_type] = list(set(matches))
        return findings

    def _detect_in_column_name(self, col_name: str) -> list:
        """Detect if column name suggests PII."""
        pii_keywords = [
            "name", "email", "phone", "mobile", "address", "dob", "birth",
            "aadhaar", "aadhar", "pan", "passport", "ssn", "nric", "sin",
            "credit", "card", "account", "bank", "salary", "income",
            "gender", "age", "race", "religion", "caste", "ip_address",
            "location", "gps", "lat", "lon", "password", "pin", "cvv"
        ]
        col_lower = col_name.lower().replace("_", " ")
        return [kw for kw in pii_keywords if kw in col_lower]

    def _scan_dataframe(self, df: pd.DataFrame) -> dict:
        """Scan entire dataframe for PII."""
        results = {
            "pii_columns": {},
            "pii_in_values": {},
            "total_pii_found": 0,
            "risk_level": "LOW"
        }

        # Check column names
        for col in df.columns:
            pii_keywords = self._detect_in_column_name(col)
            if pii_keywords:
                results["pii_columns"][col] = {
                    "pii_keywords": pii_keywords,
                    "sample_values": df[col].dropna().head(3).tolist()
                }

        # Check text column values
        text_cols = df.select_dtypes(include=["object"]).columns
        for col in text_cols[:10]:
            col_findings = {}
            sample = df[col].dropna().head(100).astype(str)

            for text in sample:
                findings = self._detect_in_text(text)
                for pii_type, values in findings.items():
                    if pii_type not in col_findings:
                        col_findings[pii_type] = []
                    col_findings[pii_type].extend(values)

            if col_findings:
                results["pii_in_values"][col] = {
                    k: list(set(v))[:3] for k, v in col_findings.items()
                }
                results["total_pii_found"] += len(col_findings)

        # Assess risk level
        total_pii = len(results["pii_columns"]) + results["total_pii_found"]
        if total_pii > 10:
            results["risk_level"] = "HIGH"
        elif total_pii > 3:
            results["risk_level"] = "MEDIUM"
        else:
            results["risk_level"] = "LOW"

        return results

    def _try_presidio(self, texts: list) -> list:
        """Try using Presidio for advanced NER-based PII detection."""
        try:
            from presidio_analyzer import AnalyzerEngine
            analyzer = AnalyzerEngine()
            presidio_results = []
            for text in texts[:10]:
                results = analyzer.analyze(text=str(text), language="en")
                entities = [{"type": r.entity_type, "score": round(r.score, 3)} for r in results]
                if entities:
                    presidio_results.append({"text": str(text)[:100], "entities": entities})
            return presidio_results
        except Exception as e:
            self.log("Presidio not available: " + str(e), level="WARN")
            return []

    def run(self, state: dict) -> dict:
        """
        PII Detection:
        1. Load data
        2. Scan column names for PII keywords
        3. Scan text values with regex patterns
        4. Try Presidio for advanced detection
        5. Generate risk report
        """

        self.log("Loading data for PII scan...")
        d = "outputs"
        files = [f for f in os.listdir(d) if f.endswith("_data.parquet")]
        if not files:
            self.log("No data found for PII scan", level="WARN")
            return state

        df = pd.read_parquet(os.path.join(d, sorted(files)[-1]))
        self.log("Scanning " + str(df.shape[0]) + " rows x " + str(df.shape[1]) + " columns")

        # Scan dataframe
        self.log("Running regex-based PII detection...")
        scan_results = self._scan_dataframe(df)

        # Try Presidio on sample texts
        self.log("Running Presidio NER-based PII detection...")
        text_cols = df.select_dtypes(include=["object"]).columns
        presidio_results = []
        if len(text_cols) > 0:
            sample_texts = df[text_cols[0]].dropna().head(10).tolist()
            presidio_results = self._try_presidio(sample_texts)

        # Final report
        pii_report = {
            "dataset_shape": list(df.shape),
            "pii_columns": scan_results["pii_columns"],
            "pii_in_values": scan_results["pii_in_values"],
            "total_pii_found": scan_results["total_pii_found"],
            "risk_level": scan_results["risk_level"],
            "presidio_findings": presidio_results,
            "recommendations": []
        }

        if scan_results["pii_columns"]:
            pii_report["recommendations"].append(
                "MASK or REMOVE columns: " + str(list(scan_results["pii_columns"].keys()))
            )
        if scan_results["risk_level"] == "HIGH":
            pii_report["recommendations"].append(
                "HIGH RISK: Implement data masking before model training"
            )
        if not pii_report["recommendations"]:
            pii_report["recommendations"].append("No critical PII found — dataset appears safe")

        os.makedirs("outputs", exist_ok=True)
        run_id = state.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
        report_path = "outputs/" + run_id + "_pii_report.json"
        with open(report_path, "w") as f:
            json.dump(pii_report, f, indent=2, default=str)

        state["pii_report"] = pii_report
        state.setdefault("planner_decision_log", []).append(
            "[" + datetime.now().isoformat() + "] PIIDetectionAgent: risk=" +
            scan_results["risk_level"] + " pii_cols=" +
            str(len(scan_results["pii_columns"]))
        )

        self.log("=" * 50)
        self.log("PII DETECTION COMPLETE")
        self.log("Risk level    : " + scan_results["risk_level"])
        self.log("PII columns   : " + str(list(scan_results["pii_columns"].keys())))
        self.log("PII in values : " + str(list(scan_results["pii_in_values"].keys())))
        for rec in pii_report["recommendations"]:
            self.log("REC: " + rec)
        self.log("=" * 50)

        return state