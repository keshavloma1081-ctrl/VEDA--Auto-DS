"""
VEDA — Autonomous Data Science System
agents/rag/ab_testing.py — A/B Testing Agent

Statistical A/B testing:
- Two-sample t-test
- Chi-square test
- Power analysis
- Sample size calculation
- Bayesian A/B test
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats

from veda.core.base_agent import BaseAgent


class ABTestingAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name="ABTestingAgent",
            domain="causal",
            version="1.0.0"
        )

    def _simulate_ab_data(self, n: int = 1000) -> dict:
        """Simulate A/B test data."""
        np.random.seed(42)
        control_rate = 0.10
        treatment_rate = 0.12

        control_conversions = np.random.binomial(1, control_rate, n)
        treatment_conversions = np.random.binomial(1, treatment_rate, n)

        control_revenue = np.random.normal(50, 15, n) * control_conversions
        treatment_revenue = np.random.normal(55, 15, n) * treatment_conversions

        return {
            "control": {
                "n": n,
                "conversions": int(control_conversions.sum()),
                "conversion_rate": round(float(control_conversions.mean()), 4),
                "avg_revenue": round(float(control_revenue.mean()), 4),
                "revenue_std": round(float(control_revenue.std()), 4),
                "raw_conversions": control_conversions,
                "raw_revenue": control_revenue
            },
            "treatment": {
                "n": n,
                "conversions": int(treatment_conversions.sum()),
                "conversion_rate": round(float(treatment_conversions.mean()), 4),
                "avg_revenue": round(float(treatment_revenue.mean()), 4),
                "revenue_std": round(float(treatment_revenue.std()), 4),
                "raw_conversions": treatment_conversions,
                "raw_revenue": treatment_revenue
            }
        }

    def _two_sample_ttest(self, control: np.ndarray,
                           treatment: np.ndarray) -> dict:
        """Two-sample t-test for continuous metrics."""
        stat, p_value = stats.ttest_ind(control, treatment)
        effect_size = (treatment.mean() - control.mean()) / np.sqrt(
            (control.std()**2 + treatment.std()**2) / 2
        )
        return {
            "test": "two_sample_ttest",
            "statistic": round(float(stat), 6),
            "p_value": round(float(p_value), 6),
            "significant": bool(p_value < 0.05),
            "effect_size": round(float(effect_size), 6),
            "lift": round(float((treatment.mean() - control.mean()) / max(abs(control.mean()), 1e-6) * 100), 2)
        }

    def _chi_square_test(self, control_conv: int, control_n: int,
                          treatment_conv: int, treatment_n: int) -> dict:
        """Chi-square test for conversion rates."""
        contingency = np.array([
            [control_conv, control_n - control_conv],
            [treatment_conv, treatment_n - treatment_conv]
        ])
        stat, p_value, dof, expected = stats.chi2_contingency(contingency)
        return {
            "test": "chi_square",
            "statistic": round(float(stat), 6),
            "p_value": round(float(p_value), 6),
            "significant": bool(p_value < 0.05),
            "degrees_of_freedom": int(dof)
        }

    def _power_analysis(self, effect_size: float = 0.2,
                         alpha: float = 0.05,
                         power: float = 0.8) -> dict:
        """Compute required sample size."""
        from scipy.stats import norm
        z_alpha = norm.ppf(1 - alpha/2)
        z_beta = norm.ppf(power)
        n = ((z_alpha + z_beta) / effect_size) ** 2
        return {
            "required_n_per_group": int(np.ceil(n)),
            "total_required_n": int(np.ceil(n) * 2),
            "effect_size": effect_size,
            "alpha": alpha,
            "power": power
        }

    def _bayesian_ab(self, control_conv: int, control_n: int,
                      treatment_conv: int, treatment_n: int,
                      n_samples: int = 10000) -> dict:
        """Bayesian A/B test using Beta distributions."""
        np.random.seed(42)
        alpha_c = 1 + control_conv
        beta_c = 1 + control_n - control_conv
        alpha_t = 1 + treatment_conv
        beta_t = 1 + treatment_n - treatment_conv

        samples_c = np.random.beta(alpha_c, beta_c, n_samples)
        samples_t = np.random.beta(alpha_t, beta_t, n_samples)

        prob_treatment_better = float((samples_t > samples_c).mean())
        expected_lift = float((samples_t - samples_c).mean())

        return {
            "prob_treatment_better": round(prob_treatment_better, 4),
            "expected_lift": round(expected_lift * 100, 4),
            "credible_interval_95": [
                round(float(np.percentile(samples_t - samples_c, 2.5) * 100), 4),
                round(float(np.percentile(samples_t - samples_c, 97.5) * 100), 4)
            ],
            "recommendation": "USE TREATMENT" if prob_treatment_better > 0.95 else
                             "NEEDS MORE DATA" if prob_treatment_better > 0.80 else "USE CONTROL"
        }

    def run(self, state: dict) -> dict:
        self.log("Simulating A/B test data...")
        ab_data = self._simulate_ab_data(n=1000)

        control = ab_data["control"]
        treatment = ab_data["treatment"]

        self.log("Control conversion rate  : " + str(control["conversion_rate"]))
        self.log("Treatment conversion rate: " + str(treatment["conversion_rate"]))

        self.log("Running two-sample t-test on revenue...")
        ttest = self._two_sample_ttest(
            control["raw_revenue"],
            treatment["raw_revenue"]
        )
        self.log("T-test p-value: " + str(ttest["p_value"]) +
                " significant=" + str(ttest["significant"]))

        self.log("Running chi-square test on conversions...")
        chi2 = self._chi_square_test(
            control["conversions"], control["n"],
            treatment["conversions"], treatment["n"]
        )
        self.log("Chi-square p-value: " + str(chi2["p_value"]) +
                " significant=" + str(chi2["significant"]))

        self.log("Running power analysis...")
        power = self._power_analysis(effect_size=0.2)
        self.log("Required n per group: " + str(power["required_n_per_group"]))

        self.log("Running Bayesian A/B test...")
        bayesian = self._bayesian_ab(
            control["conversions"], control["n"],
            treatment["conversions"], treatment["n"]
        )
        self.log("P(treatment better): " + str(bayesian["prob_treatment_better"]))
        self.log("Recommendation: " + bayesian["recommendation"])

        ab_results = {
            "summary": {
                "control_conversion_rate": control["conversion_rate"],
                "treatment_conversion_rate": treatment["conversion_rate"],
                "relative_lift_pct": round(
                    (treatment["conversion_rate"] - control["conversion_rate"]) /
                    max(control["conversion_rate"], 1e-6) * 100, 2
                )
            },
            "t_test": ttest,
            "chi_square": chi2,
            "power_analysis": power,
            "bayesian": bayesian
        }

        os.makedirs("outputs", exist_ok=True)
        run_id = state.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
        path = "outputs/" + run_id + "_ab_results.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(ab_results, f, indent=2)

        state["ab_results"] = ab_results
        state.setdefault("planner_decision_log", []).append(
            "[" + datetime.now().isoformat() + "] ABTestingAgent: " +
            "lift=" + str(ab_results["summary"]["relative_lift_pct"]) + "% " +
            bayesian["recommendation"]
        )

        self.log("=" * 50)
        self.log("A/B TESTING COMPLETE")
        self.log("Lift            : " + str(ab_results["summary"]["relative_lift_pct"]) + "%")
        self.log("Significant     : " + str(chi2["significant"]))
        self.log("Recommendation  : " + bayesian["recommendation"])
        self.log("=" * 50)

        return state