"""
VEDA — Autonomous Data Science System
agents/special/data_masking.py — Data Masking Agent

Masks/anonymizes PII in datasets:
- Email masking
- Phone number masking
- Name pseudonymization
- Numeric generalization
- K-anonymity check
"""

import os
import re
import json
import hashlib
import pandas as pd
import numpy as np
from datetime import datetime

from veda.core.base_agent import BaseAgent


class DataMaskingAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name="DataMaskingAgent",
            domain="compliance",
            version="1.0.0"
        )

    def _mask_email(self, email: str) -> str:
        if not isinstance(email, str):
            return email
        parts = email.split("@")
        if len(parts) == 2:
            local = parts[0]
            masked_local = local[0] + "*" * (len(local) - 2) + local[-1] if len(local) > 2 else "***"
            return masked_local + "@" + parts[1]
        return "***@***.***"

    def _mask_phone(self, phone: str) -> str:
        if not isinstance(phone, str):
            return phone
        digits = re.sub(r"\D", "", phone)
        if len(digits) >= 4:
            return "*" * (len(digits) - 4) + digits[-4:]
        return "****"

    def _pseudonymize(self, value: str, salt: str = "VEDA2024") -> str:
        if not isinstance(value, str):
            return str(value)
        hashed = hashlib.sha256((salt + value).encode()).hexdigest()
        return "USER_" + hashed[:8].upper()

    def _generalize_age(self, age) -> str:
        try:
            age = int(age)
            if age < 18:
                return "<18"
            elif age < 25:
                return "18-24"
            elif age < 35:
                return "25-34"
            elif age < 45:
                return "35-44"
            elif age < 55:
                return "45-54"
            elif age < 65:
                return "55-64"
            else:
                return "65+"
        except:
            return "Unknown"

    def _mask_credit_card(self, cc: str) -> str:
        if not isinstance(cc, str):
            return cc
        digits = re.sub(r"\D", "", cc)
        if len(digits) >= 4:
            return "**** **** **** " + digits[-4:]
        return "**** **** **** ****"

    def _apply_masking(self, df: pd.DataFrame, pii_columns: dict) -> tuple:
        """Apply masking to identified PII columns."""
        masked_df = df.copy()
        changes = []

        for col in df.columns:
            col_lower = col.lower()

            if any(kw in col_lower for kw in ["email", "mail"]):
                if col in masked_df.columns:
                    masked_df[col] = masked_df[col].apply(
                        lambda x: self._mask_email(str(x)) if pd.notna(x) else x
                    )
                    changes.append("MASKED email column: " + col)

            elif any(kw in col_lower for kw in ["phone", "mobile", "tel"]):
                if col in masked_df.columns:
                    masked_df[col] = masked_df[col].apply(
                        lambda x: self._mask_phone(str(x)) if pd.notna(x) else x
                    )
                    changes.append("MASKED phone column: " + col)

            elif any(kw in col_lower for kw in ["name", "fullname", "firstname", "lastname"]):
                if col in masked_df.columns and masked_df[col].dtype == "object":
                    masked_df[col] = masked_df[col].apply(
                        lambda x: self._pseudonymize(str(x)) if pd.notna(x) else x
                    )
                    changes.append("PSEUDONYMIZED name column: " + col)

            elif any(kw in col_lower for kw in ["age", "dob", "birth"]):
                if col in masked_df.columns:
                    try:
                        masked_df[col + "_group"] = masked_df[col].apply(self._generalize_age)
                        masked_df = masked_df.drop(columns=[col])
                        changes.append("GENERALIZED age column: " + col + " -> " + col + "_group")
                    except:
                        pass

            elif any(kw in col_lower for kw in ["card", "credit", "debit"]):
                if col in masked_df.columns:
                    masked_df[col] = masked_df[col].apply(
                        lambda x: self._mask_credit_card(str(x)) if pd.notna(x) else x
                    )
                    changes.append("MASKED card column: " + col)

            elif any(kw in col_lower for kw in ["aadhaar", "aadhar", "ssn", "pan"]):
                if col in masked_df.columns:
                    masked_df[col] = masked_df[col].apply(
                        lambda x: self._pseudonymize(str(x)) if pd.notna(x) else x
                    )
                    changes.append("PSEUDONYMIZED ID column: " + col)

        return masked_df, changes

    def _check_k_anonymity(self, df: pd.DataFrame, quasi_identifiers: list, k: int = 5) -> dict:
        """Check if dataset satisfies k-anonymity."""
        available_qi = [col for col in quasi_identifiers if col in df.columns]
        if not available_qi:
            return {"k_anonymity": "N/A", "min_group_size": 0, "satisfied": True}

        try:
            groups = df.groupby(available_qi).size()
            min_group = int(groups.min())
            satisfied = min_group >= k
            return {
                "k_anonymity": k,
                "min_group_size": min_group,
                "satisfied": satisfied,
                "quasi_identifiers": available_qi
            }
        except Exception as e:
            return {"k_anonymity": k, "error": str(e), "satisfied": False}

    def run(self, state: dict) -> dict:
        """
        Data Masking:
        1. Load data
        2. Apply masking rules
        3. Check k-anonymity
        4. Save masked dataset
        """

        self.log("Loading data for masking...")
        d = "outputs"
        files = [f for f in os.listdir(d) if f.endswith("_data.parquet")]
        if not files:
            self.log("No data found for masking", level="WARN")
            return state

        df = pd.read_parquet(os.path.join(d, sorted(files)[-1]))
        original_shape = df.shape
        self.log("Original shape: " + str(original_shape))

        pii_report = state.get("pii_report", {})
        pii_columns = pii_report.get("pii_columns", {})

        self.log("Applying masking rules...")
        masked_df, changes = self._apply_masking(df, pii_columns)

        if not changes:
            self.log("No PII columns detected for masking — dataset appears clean")
        else:
            for change in changes:
                self.log("  -> " + change)

        # K-anonymity check
        quasi_ids = ["age", "gender", "zip", "location", "city", "state"]
        k_anon = self._check_k_anonymity(masked_df, quasi_ids, k=5)
        self.log("K-anonymity check: " + str(k_anon))

        # Save masked dataset
        os.makedirs("outputs", exist_ok=True)
        run_id = state.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
        masked_path = "outputs/" + run_id + "_masked_data.parquet"
        masked_df.to_parquet(masked_path, index=False)
        self.log("Masked data saved to: " + masked_path)

        masking_report = {
            "original_shape": list(original_shape),
            "masked_shape": list(masked_df.shape),
            "changes_applied": changes,
            "k_anonymity": k_anon,
            "masked_data_path": masked_path
        }

        report_path = "outputs/" + run_id + "_masking_report.json"
        with open(report_path, "w") as f:
            json.dump(masking_report, f, indent=2, default=str)

        state["masking_report"] = masking_report
        state.setdefault("planner_decision_log", []).append(
            "[" + datetime.now().isoformat() + "] DataMaskingAgent: " +
            str(len(changes)) + " masking operations applied"
        )

        self.log("=" * 50)
        self.log("DATA MASKING COMPLETE")
        self.log("Changes applied : " + str(len(changes)))
        self.log("K-anonymity     : " + str(k_anon.get("satisfied", "N/A")))
        self.log("Masked data     : " + masked_path)
        self.log("=" * 50)

        return state