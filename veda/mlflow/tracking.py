"""
MLflow Integration
Experiment tracking, model registry, and artifact logging
"""
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import mlflow.pytorch
from mlflow.tracking import MlflowClient
import os
from typing import Dict, Any, Optional
from pathlib import Path

class MLflowTracker:
    """MLflow experiment tracking wrapper"""
    
    def __init__(self):
        # Set tracking URI
        tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'file:./mlruns')
        mlflow.set_tracking_uri(tracking_uri)
        
        self.client = MlflowClient()
        self.experiment_name = "VEDA-AutoML"
        
        # Create experiment if doesn't exist
        try:
            self.experiment_id = mlflow.create_experiment(self.experiment_name)
        except:
            self.experiment_id = mlflow.get_experiment_by_name(self.experiment_name).experiment_id
    
    def start_run(self, run_name: str = None) -> str:
        """Start a new MLflow run"""
        mlflow.set_experiment(self.experiment_name)
        run = mlflow.start_run(run_name=run_name)
        return run.info.run_id
    
    def end_run(self):
        """End current MLflow run"""
        mlflow.end_run()
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters"""
        for key, value in params.items():
            mlflow.log_param(key, value)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics"""
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)
    
    def log_model(self, model, model_type: str, artifact_path: str = "model"):
        """Log model to MLflow"""
        if model_type == "sklearn":
            mlflow.sklearn.log_model(model, artifact_path)
        elif model_type == "xgboost":
            mlflow.xgboost.log_model(model, artifact_path)
        elif model_type == "lightgbm":
            mlflow.lightgbm.log_model(model, artifact_path)
        elif model_type == "pytorch":
            mlflow.pytorch.log_model(model, artifact_path)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def log_artifact(self, local_path: str):
        """Log file artifact"""
        mlflow.log_artifact(local_path)
    
    def log_figure(self, figure, filename: str):
        """Log matplotlib figure"""
        mlflow.log_figure(figure, filename)
    
    def register_model(self, model_uri: str, model_name: str) -> str:
        """Register model to MLflow Model Registry"""
        result = mlflow.register_model(model_uri, model_name)
        return result.version
    
    def transition_model_stage(self, model_name: str, version: str, stage: str):
        """
        Transition model to different stage
        Stages: None, Staging, Production, Archived
        """
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage
        )
    
    def load_model(self, model_uri: str):
        """Load model from MLflow"""
        return mlflow.pyfunc.load_model(model_uri)
    
    def search_runs(self, filter_string: str = "", max_results: int = 100):
        """Search runs in experiment"""
        return self.client.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string=filter_string,
            max_results=max_results
        )
    
    def get_best_run(self, metric: str = "accuracy", ascending: bool = False):
        """Get best run based on metric"""
        runs = self.search_runs()
        
        if not runs:
            return None
        
        sorted_runs = sorted(
            runs,
            key=lambda run: run.data.metrics.get(metric, 0),
            reverse=not ascending
        )
        
        return sorted_runs[0] if sorted_runs else None

# Global instance
mlflow_tracker = MLflowTracker()