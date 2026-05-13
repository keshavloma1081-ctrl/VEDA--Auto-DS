"""
Model Versioning and History Tracking
"""
from datetime import datetime
from typing import Optional, List
import json
import hashlib

class ModelVersion:
    """Model version tracking"""
    
    def __init__(
        self,
        model_id: str,
        version: str,
        algorithm: str,
        metrics: dict,
        hyperparameters: dict,
        created_at: datetime = None
    ):
        self.model_id = model_id
        self.version = version
        self.algorithm = algorithm
        self.metrics = metrics
        self.hyperparameters = hyperparameters
        self.created_at = created_at or datetime.utcnow()
        self.checksum = self._compute_checksum()
    
    def _compute_checksum(self) -> str:
        """Compute model checksum"""
        data = {
            "algorithm": self.algorithm,
            "hyperparameters": self.hyperparameters,
            "version": self.version
        }
        return hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()[:16]
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "model_id": self.model_id,
            "version": self.version,
            "algorithm": self.algorithm,
            "metrics": self.metrics,
            "hyperparameters": self.hyperparameters,
            "checksum": self.checksum,
            "created_at": self.created_at.isoformat()
        }


class ModelRegistry:
    """Registry for model versions"""
    
    def __init__(self):
        self.versions: List[ModelVersion] = []
    
    def register(
        self,
        model_id: str,
        algorithm: str,
        metrics: dict,
        hyperparameters: dict
    ) -> ModelVersion:
        """Register a new model version"""
        version = f"v{len(self.versions) + 1}"
        
        model_version = ModelVersion(
            model_id=model_id,
            version=version,
            algorithm=algorithm,
            metrics=metrics,
            hyperparameters=hyperparameters
        )
        
        self.versions.append(model_version)
        return model_version
    
    def get_latest(self, model_id: str) -> Optional[ModelVersion]:
        """Get latest version of a model"""
        versions = [v for v in self.versions if v.model_id == model_id]
        return versions[-1] if versions else None
    
    def get_best(self, model_id: str, metric: str = "accuracy") -> Optional[ModelVersion]:
        """Get best performing version"""
        versions = [v for v in self.versions if v.model_id == model_id]
        
        if not versions:
            return None
        
        return max(
            versions,
            key=lambda v: v.metrics.get(metric, 0)
        )
    
    def list_versions(self, model_id: str) -> List[ModelVersion]:
        """List all versions of a model"""
        return [v for v in self.versions if v.model_id == model_id]
    
    def compare_versions(
        self,
        model_id: str,
        version1: str,
        version2: str
    ) -> dict:
        """Compare two model versions"""
        v1 = next((v for v in self.versions 
                  if v.model_id == model_id and v.version == version1), None)
        v2 = next((v for v in self.versions 
                  if v.model_id == model_id and v.version == version2), None)
        
        if not v1 or not v2:
            return {}
        
        return {
            "version1": v1.to_dict(),
            "version2": v2.to_dict(),
            "metric_diff": {
                metric: v2.metrics.get(metric, 0) - v1.metrics.get(metric, 0)
                for metric in set(v1.metrics.keys()) | set(v2.metrics.keys())
            }
        }


# Global registry
model_registry = ModelRegistry()

# Usage:
"""
# Register a model
version = model_registry.register(
    model_id="customer_churn_pred",
    algorithm="XGBoost",
    metrics={"accuracy": 0.89, "f1": 0.87},
    hyperparameters={"n_estimators": 100, "max_depth": 5}
)

# Get latest version
latest = model_registry.get_latest("customer_churn_pred")

# Get best performing version
best = model_registry.get_best("customer_churn_pred", metric="f1")

# Compare versions
comparison = model_registry.compare_versions(
    "customer_churn_pred", "v1", "v2"
)
"""