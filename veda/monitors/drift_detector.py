"""
VEDA Model Monitoring + Drift Detection - Fix #6
Detects when model performance degrades in production.
"""
import os, json, logging, hashlib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DriftReport:
    model_id: str
    checked_at: str
    total_predictions: int
    data_drift_detected: bool
    concept_drift_detected: bool
    drift_score: float
    feature_drift: Dict[str, float]
    alert_triggered: bool
    alert_message: str
    recommendation: str

    def to_dict(self):
        return {
            "model_id": self.model_id,
            "checked_at": self.checked_at,
            "total_predictions": self.total_predictions,
            "data_drift_detected": self.data_drift_detected,
            "concept_drift_detected": self.concept_drift_detected,
            "drift_score": round(self.drift_score, 4),
            "feature_drift": {k: round(v, 4) for k, v in self.feature_drift.items()},
            "alert_triggered": self.alert_triggered,
            "alert_message": self.alert_message,
            "recommendation": self.recommendation,
        }


# ─────────────────────────────────────────────────────────────────────────────
# STATISTICAL DRIFT DETECTORS
# ─────────────────────────────────────────────────────────────────────────────

class KSTestDriftDetector:
    """Kolmogorov-Smirnov test for feature drift"""

    def __init__(self, threshold: float = 0.05):
        self.threshold = threshold

    def detect(self, reference: np.ndarray, current: np.ndarray) -> Dict:
        try:
            from scipy import stats
            stat, p_value = stats.ks_2samp(reference, current)
            drift = p_value < self.threshold
            return {
                "ks_statistic": round(float(stat), 4),
                "p_value": round(float(p_value), 4),
                "drift_detected": drift,
                "severity": "high" if stat > 0.3 else "medium" if stat > 0.15 else "low"
            }
        except ImportError:
            # Fallback: simple mean shift detection
            ref_mean, cur_mean = np.mean(reference), np.mean(current)
            ref_std = np.std(reference) + 1e-10
            z_score = abs(ref_mean - cur_mean) / ref_std
            drift = z_score > 2.0
            return {
                "z_score": round(float(z_score), 4),
                "drift_detected": drift,
                "severity": "high" if z_score > 4 else "medium" if z_score > 2 else "low"
            }


class PSIDriftDetector:
    """Population Stability Index for categorical drift"""

    def __init__(self, threshold: float = 0.2):
        self.threshold = threshold

    def detect(self, reference: np.ndarray, current: np.ndarray) -> Dict:
        try:
            ref_counts = pd.Series(reference).value_counts(normalize=True)
            cur_counts = pd.Series(current).value_counts(normalize=True)
            all_cats = set(ref_counts.index) | set(cur_counts.index)

            psi = 0.0
            for cat in all_cats:
                ref_p = ref_counts.get(cat, 0.001)
                cur_p = cur_counts.get(cat, 0.001)
                psi += (cur_p - ref_p) * np.log(cur_p / ref_p)

            drift = psi > self.threshold
            return {
                "psi": round(float(psi), 4),
                "drift_detected": drift,
                "severity": "high" if psi > 0.25 else "medium" if psi > 0.1 else "low"
            }
        except Exception as e:
            return {"error": str(e), "drift_detected": False}


# ─────────────────────────────────────────────────────────────────────────────
# MODEL MONITOR
# ─────────────────────────────────────────────────────────────────────────────

class ModelMonitor:
    """
    Monitors model performance and detects drift.
    Compares current predictions against reference baseline.
    """

    def __init__(
        self,
        model_id: str,
        reference_data: Optional[pd.DataFrame] = None,
        drift_threshold: float = 0.05,
        performance_threshold: float = 0.05
    ):
        self.model_id = model_id
        self.reference_data = reference_data
        self.reference_stats: Dict = {}
        self.drift_threshold = drift_threshold
        self.performance_threshold = performance_threshold
        self.ks_detector = KSTestDriftDetector(threshold=drift_threshold)
        self.psi_detector = PSIDriftDetector()
        self.baseline_accuracy: Optional[float] = None

        if reference_data is not None:
            self._compute_reference_stats(reference_data)

    def _compute_reference_stats(self, data: pd.DataFrame):
        """Compute baseline statistics from reference data"""
        self.reference_stats = {}
        for col in data.select_dtypes(include=[np.number]).columns:
            values = data[col].dropna().values
            if len(values) > 0:
                self.reference_stats[col] = {
                    "values": values,
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "p25": float(np.percentile(values, 25)),
                    "p75": float(np.percentile(values, 75)),
                }
        log.info(f"Reference stats computed for {len(self.reference_stats)} features")

    def set_baseline_accuracy(self, accuracy: float):
        """Set baseline model accuracy for performance monitoring"""
        self.baseline_accuracy = accuracy
        log.info(f"Baseline accuracy set: {accuracy:.4f}")

    def detect_data_drift(self, current_data: pd.DataFrame) -> Dict:
        """Detect feature distribution drift"""
        if not self.reference_stats:
            return {
                "drift_detected": False,
                "message": "No reference data available",
                "feature_results": {}
            }

        feature_results = {}
        drift_detected = False
        total_drift_score = 0.0
        drifted_features = []

        for col, ref_stats in self.reference_stats.items():
            if col not in current_data.columns:
                continue
            current_values = current_data[col].dropna().values
            if len(current_values) < 30:
                continue

            result = self.ks_detector.detect(ref_stats["values"], current_values)
            feature_results[col] = result

            if result.get("drift_detected", False):
                drift_detected = True
                drifted_features.append(col)
                total_drift_score += result.get("ks_statistic", result.get("z_score", 0))

        avg_drift_score = total_drift_score / len(feature_results) if feature_results else 0.0

        return {
            "drift_detected": drift_detected,
            "drift_score": round(avg_drift_score, 4),
            "drifted_features": drifted_features,
            "total_features_checked": len(feature_results),
            "feature_results": feature_results
        }

    def detect_prediction_drift(
        self,
        recent_predictions: List[float],
        window_size: int = 500
    ) -> Dict:
        """Detect concept drift from prediction distribution shift"""
        if not self.baseline_accuracy:
            return {
                "drift_detected": False,
                "message": "No baseline accuracy set"
            }

        if len(recent_predictions) < window_size // 2:
            return {
                "drift_detected": False,
                "message": f"Insufficient predictions ({len(recent_predictions)} < {window_size // 2})"
            }

        # Sliding window comparison
        window1 = np.array(recent_predictions[:window_size // 2])
        window2 = np.array(recent_predictions[-(window_size // 2):])

        result = self.ks_detector.detect(window1, window2)

        # Check accuracy degradation
        recent_accuracy = float(np.mean(np.array(recent_predictions) >= 0.5))
        accuracy_drop = self.baseline_accuracy - recent_accuracy
        accuracy_drift = accuracy_drop > self.performance_threshold

        return {
            "drift_detected": result["drift_detected"] or accuracy_drift,
            "ks_statistic": result.get("ks_statistic", 0),
            "recent_accuracy": round(recent_accuracy, 4),
            "baseline_accuracy": round(self.baseline_accuracy, 4),
            "accuracy_drop": round(accuracy_drop, 4),
            "accuracy_drift_detected": accuracy_drift,
            "distribution_drift_detected": result["drift_detected"]
        }

    def generate_report(
        self,
        current_data: Optional[pd.DataFrame] = None,
        recent_predictions: Optional[List] = None
    ) -> DriftReport:
        """Generate comprehensive drift report"""
        now = datetime.utcnow().isoformat()

        data_drift_result = {}
        pred_drift_result = {}
        feature_drift_scores = {}

        if current_data is not None:
            data_drift_result = self.detect_data_drift(current_data)
            feature_drift_scores = {
                feat: res.get("ks_statistic", res.get("z_score", 0))
                for feat, res in data_drift_result.get("feature_results", {}).items()
            }

        if recent_predictions is not None:
            pred_drift_result = self.detect_prediction_drift(recent_predictions)

        data_drift = data_drift_result.get("drift_detected", False)
        concept_drift = pred_drift_result.get("drift_detected", False)
        drift_score = data_drift_result.get("drift_score", 0.0)

        # Generate alert and recommendation
        alert = data_drift or concept_drift
        alert_msg = ""
        recommendation = "Model performing normally. Continue monitoring."

        if concept_drift and data_drift:
            alert_msg = "CRITICAL: Both data drift and concept drift detected!"
            recommendation = "Immediate retraining required. Review data pipeline."
        elif concept_drift:
            alert_msg = "WARNING: Concept drift detected - model accuracy degrading"
            recommendation = "Schedule model retraining with recent data."
        elif data_drift:
            drifted = data_drift_result.get("drifted_features", [])
            alert_msg = f"WARNING: Data drift in features: {', '.join(drifted[:3])}"
            recommendation = "Monitor closely. Consider retraining if drift persists."

        if alert:
            log.warning(f"Drift alert for model {self.model_id}: {alert_msg}")

        return DriftReport(
            model_id=self.model_id,
            checked_at=now,
            total_predictions=len(recent_predictions) if recent_predictions else 0,
            data_drift_detected=data_drift,
            concept_drift_detected=concept_drift,
            drift_score=drift_score,
            feature_drift=feature_drift_scores,
            alert_triggered=alert,
            alert_message=alert_msg,
            recommendation=recommendation
        )


# ─────────────────────────────────────────────────────────────────────────────
# MONITORING SERVICE
# ─────────────────────────────────────────────────────────────────────────────

class MonitoringService:
    """Central service for managing model monitors"""

    def __init__(self):
        self._monitors: Dict[str, ModelMonitor] = {}
        self._reports: Dict[str, List[DriftReport]] = {}

    def register_model(
        self,
        model_id: str,
        reference_data: Optional[pd.DataFrame] = None,
        baseline_accuracy: Optional[float] = None
    ) -> ModelMonitor:
        monitor = ModelMonitor(
            model_id=model_id,
            reference_data=reference_data
        )
        if baseline_accuracy:
            monitor.set_baseline_accuracy(baseline_accuracy)
        self._monitors[model_id] = monitor
        self._reports[model_id] = []
        log.info(f"Model {model_id} registered for monitoring")
        return monitor

    def check_model(
        self,
        model_id: str,
        current_data: Optional[pd.DataFrame] = None,
        recent_predictions: Optional[List] = None
    ) -> Optional[DriftReport]:
        monitor = self._monitors.get(model_id)
        if not monitor:
            log.warning(f"Model {model_id} not registered")
            return None

        report = monitor.generate_report(current_data, recent_predictions)
        self._reports[model_id].append(report)

        # Save to database if available
        self._save_report(report)

        return report

    def _save_report(self, report: DriftReport):
        """Save drift report to database"""
        try:
            from veda.database.models import SessionLocal, ModelMonitoringLog
            db = SessionLocal()
            try:
                log_entry = ModelMonitoringLog(
                    model_id=report.model_id,
                    period_start=datetime.utcnow() - timedelta(hours=1),
                    period_end=datetime.utcnow(),
                    total_predictions=report.total_predictions,
                    data_drift_detected=report.data_drift_detected,
                    concept_drift_detected=report.concept_drift_detected,
                    prediction_drift_score=report.drift_score,
                    feature_drift_scores=report.feature_drift,
                    alert_triggered=report.alert_triggered,
                    alert_message=report.alert_message
                )
                db.add(log_entry)
                db.commit()
            finally:
                db.close()
        except Exception as e:
            log.warning(f"Could not save drift report: {e}")

    def get_model_health(self, model_id: str) -> dict:
        reports = self._reports.get(model_id, [])
        if not reports:
            return {"model_id": model_id, "status": "no_data"}

        latest = reports[-1]
        recent_alerts = sum(1 for r in reports[-10:] if r.alert_triggered)

        return {
            "model_id": model_id,
            "status": "alert" if latest.alert_triggered else "healthy",
            "latest_check": latest.checked_at,
            "data_drift": latest.data_drift_detected,
            "concept_drift": latest.concept_drift_detected,
            "drift_score": latest.drift_score,
            "recent_alerts": recent_alerts,
            "recommendation": latest.recommendation
        }

    def get_all_health(self) -> List[dict]:
        return [self.get_model_health(mid) for mid in self._monitors]


# Global service
monitoring_service = MonitoringService()


# ─────────────────────────────────────────────────────────────────────────────
# QUICK CHECK FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def quick_drift_check(
    model_id: str,
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    baseline_accuracy: Optional[float] = None
) -> DriftReport:
    """One-shot drift check without registering monitor"""
    monitor = ModelMonitor(
        model_id=model_id,
        reference_data=reference_df,
        drift_threshold=0.05
    )
    if baseline_accuracy:
        monitor.set_baseline_accuracy(baseline_accuracy)
    return monitor.generate_report(current_data=current_df)
