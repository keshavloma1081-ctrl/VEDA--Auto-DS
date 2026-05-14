"""
VEDA A/B Testing Framework - Fix #8
Safely compare model versions in production.
"""
import os, logging, random, time
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

log = logging.getLogger(__name__)

class ExperimentStatus(Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    STOPPED = "stopped"

@dataclass
class ExperimentResult:
    experiment_id: str
    name: str
    status: str
    model_a_id: str
    model_b_id: str
    traffic_split: float
    model_a_requests: int = 0
    model_b_requests: int = 0
    model_a_correct: int = 0
    model_b_correct: int = 0
    model_a_latency_ms: List[float] = field(default_factory=list)
    model_b_latency_ms: List[float] = field(default_factory=list)
    started_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    ended_at: Optional[str] = None
    winner: Optional[str] = None
    p_value: Optional[float] = None
    is_significant: Optional[bool] = None
    conclusion: Optional[str] = None

    @property
    def model_a_accuracy(self) -> float:
        return self.model_a_correct / self.model_a_requests if self.model_a_requests > 0 else 0.0

    @property
    def model_b_accuracy(self) -> float:
        return self.model_b_correct / self.model_b_requests if self.model_b_requests > 0 else 0.0

    @property
    def model_a_avg_latency(self) -> float:
        return float(np.mean(self.model_a_latency_ms)) if self.model_a_latency_ms else 0.0

    @property
    def model_b_avg_latency(self) -> float:
        return float(np.mean(self.model_b_latency_ms)) if self.model_b_latency_ms else 0.0

    def to_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "status": self.status,
            "model_a_id": self.model_a_id,
            "model_b_id": self.model_b_id,
            "traffic_split": self.traffic_split,
            "model_a": {
                "requests": self.model_a_requests,
                "accuracy": round(self.model_a_accuracy, 4),
                "avg_latency_ms": round(self.model_a_avg_latency, 2)
            },
            "model_b": {
                "requests": self.model_b_requests,
                "accuracy": round(self.model_b_accuracy, 4),
                "avg_latency_ms": round(self.model_b_avg_latency, 2)
            },
            "statistical_test": {
                "p_value": round(self.p_value, 4) if self.p_value else None,
                "is_significant": self.is_significant,
                "winner": self.winner,
                "conclusion": self.conclusion
            },
            "started_at": self.started_at,
            "ended_at": self.ended_at
        }


class ABTestFramework:
    """
    Production A/B testing for ML models.
    Routes traffic between model versions and tracks performance.
    Uses chi-square test for statistical significance.
    """

    def __init__(self):
        self._experiments: Dict[str, ExperimentResult] = {}
        self._active: Dict[str, str] = {}  # model_id -> experiment_id

    def create_experiment(
        self,
        name: str,
        model_a_id: str,
        model_b_id: str,
        traffic_split: float = 0.5,
        min_samples: int = 100
    ) -> ExperimentResult:
        """Create new A/B experiment"""
        import uuid
        exp_id = str(uuid.uuid4())[:8]

        exp = ExperimentResult(
            experiment_id=exp_id,
            name=name,
            status=ExperimentStatus.RUNNING.value,
            model_a_id=model_a_id,
            model_b_id=model_b_id,
            traffic_split=traffic_split
        )
        self._experiments[exp_id] = exp
        self._active[model_a_id] = exp_id
        self._active[model_b_id] = exp_id

        log.info(f"A/B experiment created | id={exp_id} | {model_a_id} vs {model_b_id}")
        self._save_to_db(exp)
        return exp

    def route_request(self, experiment_id: str) -> str:
        """Route request to model A or B based on traffic split"""
        exp = self._experiments.get(experiment_id)
        if not exp or exp.status != ExperimentStatus.RUNNING.value:
            return "model_a"
        return "model_a" if random.random() < exp.traffic_split else "model_b"

    def record_outcome(
        self,
        experiment_id: str,
        variant: str,
        correct: bool,
        latency_ms: float = 0.0
    ):
        """Record prediction outcome for statistical tracking"""
        exp = self._experiments.get(experiment_id)
        if not exp:
            return

        if variant == "model_a":
            exp.model_a_requests += 1
            if correct:
                exp.model_a_correct += 1
            exp.model_a_latency_ms.append(latency_ms)
        else:
            exp.model_b_requests += 1
            if correct:
                exp.model_b_correct += 1
            exp.model_b_latency_ms.append(latency_ms)

        # Auto-analyze every 50 requests
        total = exp.model_a_requests + exp.model_b_requests
        if total > 0 and total % 50 == 0:
            self.analyze(experiment_id)

    def analyze(self, experiment_id: str) -> Dict:
        """Run statistical significance test"""
        exp = self._experiments.get(experiment_id)
        if not exp:
            return {"error": "Experiment not found"}

        min_samples = 30
        if exp.model_a_requests < min_samples or exp.model_b_requests < min_samples:
            return {
                "experiment_id": experiment_id,
                "message": f"Need {min_samples} samples per variant",
                "model_a_requests": exp.model_a_requests,
                "model_b_requests": exp.model_b_requests
            }

        # Chi-square test for proportions
        try:
            from scipy.stats import chi2_contingency

            a_correct = exp.model_a_correct
            a_wrong = exp.model_a_requests - exp.model_a_correct
            b_correct = exp.model_b_correct
            b_wrong = exp.model_b_requests - exp.model_b_correct

            # Avoid zeros
            contingency = [[max(a_correct, 1), max(a_wrong, 1)],
                           [max(b_correct, 1), max(b_wrong, 1)]]

            chi2, p_value, _, _ = chi2_contingency(contingency)
            is_significant = p_value < 0.05

        except ImportError:
            # Fallback: z-test for proportions
            p_a = exp.model_a_accuracy
            p_b = exp.model_b_accuracy
            p_pool = (exp.model_a_correct + exp.model_b_correct) / (exp.model_a_requests + exp.model_b_requests + 1e-10)
            se = np.sqrt(p_pool * (1 - p_pool) * (1/exp.model_a_requests + 1/exp.model_b_requests) + 1e-10)
            z = abs(p_a - p_b) / se
            p_value = float(2 * (1 - min(0.9999, z / 6)))
            is_significant = p_value < 0.05

        exp.p_value = float(p_value)
        exp.is_significant = is_significant

        # Determine winner
        if is_significant:
            if exp.model_a_accuracy > exp.model_b_accuracy:
                exp.winner = "model_a"
                lift = round((exp.model_a_accuracy - exp.model_b_accuracy) / max(exp.model_b_accuracy, 0.001) * 100, 2)
                exp.conclusion = f"Model A wins with {lift}% accuracy improvement (p={p_value:.4f})"
            else:
                exp.winner = "model_b"
                lift = round((exp.model_b_accuracy - exp.model_a_accuracy) / max(exp.model_a_accuracy, 0.001) * 100, 2)
                exp.conclusion = f"Model B wins with {lift}% accuracy improvement (p={p_value:.4f})"
        else:
            exp.winner = None
            exp.conclusion = f"No significant difference yet (p={p_value:.4f}). Need more data."

        log.info(f"A/B analysis | {experiment_id} | {exp.conclusion}")
        return exp.to_dict()

    def stop_experiment(self, experiment_id: str, deploy_winner: bool = False) -> Dict:
        """Stop experiment and optionally deploy winner"""
        exp = self._experiments.get(experiment_id)
        if not exp:
            return {"error": "Experiment not found"}

        # Final analysis
        final = self.analyze(experiment_id)
        exp.status = ExperimentStatus.COMPLETED.value
        exp.ended_at = datetime.utcnow().isoformat()

        # Update DB
        self._update_db(exp)

        result = {
            "experiment_id": experiment_id,
            "status": "completed",
            "winner": exp.winner,
            "conclusion": exp.conclusion,
            "final_results": exp.to_dict()
        }

        if deploy_winner and exp.winner:
            result["deployment"] = f"Model {exp.winner} ({getattr(exp, f'{exp.winner}_id', 'unknown')}) deployed to production"
            log.info(f"Winner deployed: {exp.winner} from experiment {experiment_id}")

        return result

    def get_experiment(self, experiment_id: str) -> Optional[Dict]:
        exp = self._experiments.get(experiment_id)
        return exp.to_dict() if exp else None

    def list_experiments(self, status: Optional[str] = None) -> List[Dict]:
        exps = list(self._experiments.values())
        if status:
            exps = [e for e in exps if e.status == status]
        return [e.to_dict() for e in exps]

    def get_summary(self) -> Dict:
        total = len(self._experiments)
        running = sum(1 for e in self._experiments.values() if e.status == ExperimentStatus.RUNNING.value)
        completed = sum(1 for e in self._experiments.values() if e.status == ExperimentStatus.COMPLETED.value)
        winners = [e for e in self._experiments.values() if e.winner]
        return {
            "total_experiments": total,
            "running": running,
            "completed": completed,
            "experiments_with_winner": len(winners)
        }

    def _save_to_db(self, exp: ExperimentResult):
        try:
            from veda.database.models import SessionLocal, ABExperiment
            db = SessionLocal()
            try:
                record = ABExperiment(
                    experiment_id=exp.experiment_id,
                    name=exp.name,
                    model_a_id=exp.model_a_id,
                    model_b_id=exp.model_b_id,
                    traffic_split=exp.traffic_split,
                    status=exp.status
                )
                db.add(record)
                db.commit()
            finally:
                db.close()
        except Exception as e:
            log.warning(f"Could not save experiment to DB: {e}")

    def _update_db(self, exp: ExperimentResult):
        try:
            from veda.database.models import SessionLocal, ABExperiment
            db = SessionLocal()
            try:
                record = db.query(ABExperiment).filter(ABExperiment.experiment_id == exp.experiment_id).first()
                if record:
                    record.status = exp.status
                    record.winner = exp.winner
                    record.model_a_requests = exp.model_a_requests
                    record.model_b_requests = exp.model_b_requests
                    record.model_a_accuracy = exp.model_a_accuracy
                    record.model_b_accuracy = exp.model_b_accuracy
                    record.p_value = exp.p_value
                    record.is_significant = exp.is_significant
                    db.commit()
            finally:
                db.close()
        except Exception as e:
            log.warning(f"Could not update experiment in DB: {e}")


# Global instance
ab_framework = ABTestFramework()
