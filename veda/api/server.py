"""
VEDA FastAPI Server - Complete Production Version with Report Generation
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
from datetime import datetime, timedelta
import os
import uuid

# Imports
from veda.database.models import Workflow, get_db, init_db
from veda.agents.core_pipeline.planner import PlannerAgent
from veda.core.state import VEDAState

from veda.api.security import (
    validate_dataset_path, validate_goal,
    save_upload_file, SecureWorkflowRequest,
    SecureLoginRequest, SecurityMiddleware,
    ensure_allowed_dirs, get_file_info
)

# Check Celery availability
try:
    from veda.tasks.celery_app import celery_app
    from veda.tasks.workflow_tasks import (
        run_workflow_task,
        run_workflow_priority,
        cancel_workflow_task
    )
    celery_app.control.ping(timeout=1)
    CELERY_AVAILABLE = True
    print("âœ… Celery+Redis connected")
except Exception:
    CELERY_AVAILABLE = False
    print("âš ï¸  Using BackgroundTasks fallback")

# Report generator
try:
    from veda.reports.generator import generate_workflow_report
    REPORTS_AVAILABLE = True
except ImportError:
    REPORTS_AVAILABLE = False

# Auth imports
try:
    from veda.api.auth import authenticate_user, create_access_token, get_current_user
    AUTH_AVAILABLE = True
except ImportError:
    AUTH_AVAILABLE = False

# Initialize FastAPI
app = FastAPI(
    title="VEDA ML API",
    version="2.0.0",
    docs_url="/docs"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup
@app.on_event("startup")
def startup():
    init_db()
    print("âœ… Database initialized")
    print("ðŸš€ VEDA API Started")

# Planner
planner = None
def get_planner():
    global planner
    if planner is None:
        if not os.getenv("GROQ_API_KEY"):
            raise HTTPException(500, "GROQ_API_KEY not set")
        planner = PlannerAgent()
    return planner

# ============================================================================
# SCHEMAS
# ============================================================================

class LoginRequest(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str

class WorkflowRequest(BaseModel):
    dataset_path: str
    goal: str

class WorkflowResponse(BaseModel):
    job_id: str
    status: str
    message: str

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/")
def root():
    return {
        "service": "VEDA ML API",
        "version": "2.0.0",
        "status": "operational",
        "docs": "/docs",
        "authentication": "JWT (POST /auth/login)" if AUTH_AVAILABLE else "disabled",
        "reports": "enabled" if REPORTS_AVAILABLE else "disabled"
    }

@app.get("/health")
def health(db: Session = Depends(get_db)):
    try:
        from sqlalchemy import text
        db.execute(text("SELECT 1")).fetchone()
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            "api": "operational",
            "database": db_status,
            "groq_api_key": "configured" if os.getenv("GROQ_API_KEY") else "missing",
            "auth": "enabled" if AUTH_AVAILABLE else "disabled",
            "reports": "enabled" if REPORTS_AVAILABLE else "disabled"
        }
    }

# ============================================================================
# AUTH ENDPOINTS
# ============================================================================

if AUTH_AVAILABLE:
    @app.post("/auth/login", response_model=TokenResponse)
    def login(req: LoginRequest):
        """Login to get JWT token. Demo: admin/admin123"""
        user = authenticate_user(req.username, req.password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )
        
        token = create_access_token(
            data={"sub": user["username"], "role": user["role"]},
            expires_delta=timedelta(hours=24)
        )
        
        return {"access_token": token, "token_type": "bearer"}

# ============================================================================
# WORKFLOW ENDPOINTS
# ============================================================================

def run_workflow_bg(job_id: str, dataset_path: str, goal: str):
    """Background workflow execution"""
    db = next(get_db())
    try:
        workflow = db.query(Workflow).filter(Workflow.job_id == job_id).first()
        workflow.status = 'running'
        db.commit()
        
        # Execute
        state = VEDAState(dataset_path=dataset_path, user_goal=goal, job_id=job_id)
        planner_inst = get_planner()
        result = planner_inst.execute(state)
        
        # Update
        workflow.status = 'completed'
        workflow.progress = 100.0
        workflow.result = result
        db.commit()
    except Exception as e:
        workflow = db.query(Workflow).filter(Workflow.job_id == job_id).first()
        workflow.status = 'failed'
        workflow.error = str(e)
        db.commit()
    finally:
        db.close()

@app.post("/workflows", response_model=WorkflowResponse)
async def create_workflow(
    req: WorkflowRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user) if AUTH_AVAILABLE else None
):
    """Create ML workflow"""
    
    if not os.path.exists(req.dataset_path):
        raise HTTPException(404, f"Dataset not found: {req.dataset_path}")
    
    job_id = str(uuid.uuid4())
    
    workflow = Workflow(
        job_id=job_id,
        dataset_path=req.dataset_path,
        goal=req.goal,
        status='submitted',
        progress=0.0
    )
    db.add(workflow)
    db.commit()
    
    background_tasks.add_task(run_workflow_bg, job_id, req.dataset_path, req.goal)
    
    return WorkflowResponse(
        job_id=job_id,
        status='submitted',
        message='Workflow submitted'
    )

@app.get("/workflows/{job_id}")
def get_workflow(job_id: str, db: Session = Depends(get_db)):
    """Get workflow status"""
    w = db.query(Workflow).filter(Workflow.job_id == job_id).first()
    if not w:
        raise HTTPException(404, "Job not found")
    return w.to_dict()

@app.get("/workflows")
def list_workflows(limit: int = 10, db: Session = Depends(get_db)):
    """List workflows"""
    workflows = db.query(Workflow).order_by(Workflow.created_at.desc()).limit(limit).all()
    return [w.to_dict() for w in workflows]

@app.get("/stats")
def stats(db: Session = Depends(get_db)):
    """System stats"""
    total = db.query(Workflow).count()
    completed = db.query(Workflow).filter(Workflow.status == 'completed').count()
    failed = db.query(Workflow).filter(Workflow.status == 'failed').count()
    
    return {
        "workflows": {
            "total": total,
            "completed": completed,
            "failed": failed,
            "success_rate": (completed/total*100) if total > 0 else 0
        }
    }

# ============================================================================
# REPORT ENDPOINT
# ============================================================================

if REPORTS_AVAILABLE:
    @app.get("/workflows/{job_id}/report")
    def generate_report(job_id: str, db: Session = Depends(get_db)):
        """
        Generate HTML report for workflow
        
        Returns a professional HTML report with:
        - Workflow summary
        - Performance metrics
        - Model information
        - Visualizations
        - Recommendations
        """
        workflow = db.query(Workflow).filter(Workflow.job_id == job_id).first()
        
        if not workflow:
            raise HTTPException(404, "Workflow not found")
        
        # Generate report
        try:
            report_path = generate_workflow_report(workflow.to_dict())
            
            # Return HTML file
            return FileResponse(
                report_path,
                media_type="text/html",
                filename=f"veda_report_{job_id}.html"
            )
        except Exception as e:
            raise HTTPException(500, f"Failed to generate report: {str(e)}")

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "veda.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

@app.get("/circuits/health")
def circuit_health():
    """Get circuit breaker health for all agents"""
    try:
        from veda.agents.circuit_breaker import get_circuit_health
        return get_circuit_health()
    except ImportError:
        return {"message": "Circuit breaker not configured"}

@app.post("/circuits/reset")
def reset_circuits():
    """Manually reset all circuit breakers to CLOSED"""
    try:
        from veda.agents.circuit_breaker import circuit_registry
        circuit_registry.reset_all()
        return {"message": "All circuits reset to CLOSED"}
    except ImportError:
        return {"message": "Circuit breaker not configured"}


@app.get("/circuits/health")
def circuit_health():
    """Circuit breaker health for all agents"""
    try:
        from veda.agents.circuit_breaker import get_circuit_health
        return get_circuit_health()
    except ImportError:
        return {"message": "Circuit breaker not configured"}

@app.post("/circuits/reset")
def reset_circuits():
    """Reset all circuit breakers to CLOSED"""
    try:
        from veda.agents.circuit_breaker import circuit_registry
        circuit_registry.reset_all()
        return {"message": "All circuits reset to CLOSED"}
    except ImportError:
        return {"message": "Circuit breaker not configured"}


@app.get("/monitoring/health")
def monitoring_health():
    """Get health status of all monitored models"""
    try:
        from veda.monitors.drift_detector import monitoring_service
        return {"models": monitoring_service.get_all_health()}
    except Exception as e:
        return {"error": str(e)}

@app.get("/monitoring/{model_id}")
def model_monitoring(model_id: str):
    """Get drift report for specific model"""
    try:
        from veda.monitors.drift_detector import monitoring_service
        return monitoring_service.get_model_health(model_id)
    except Exception as e:
        return {"error": str(e)}

@app.post("/monitoring/{model_id}/check")
def trigger_drift_check(model_id: str, db: Session = Depends(get_db)):
    """Manually trigger drift check for a model"""
    try:
        from veda.monitors.drift_detector import monitoring_service
        from veda.database.models import Prediction
        preds = db.query(Prediction).filter(
            Prediction.model_id == model_id
        ).order_by(Prediction.created_at.desc()).limit(500).all()
        if not preds:
            return {"message": f"No predictions found for model {model_id}"}
        pred_values = [p.predictions[0] if isinstance(p.predictions, list) else 0 for p in preds]
        report = monitoring_service.check_model(model_id=model_id, recent_predictions=pred_values)
        if report:
            return report.to_dict()
        return {"message": f"Model {model_id} not registered for monitoring"}
    except Exception as e:
        return {"error": str(e)}


@app.get("/optimization/summary/{job_id}")
def optimization_summary(job_id: str, db: Session = Depends(get_db)):
    """Get hyperparameter optimization summary for a workflow"""
    w = db.query(Workflow).filter(Workflow.job_id == job_id).first()
    if not w:
        raise HTTPException(404, "Workflow not found")
    result = w.to_dict()
    opt_summary = result.get("result", {})
    if isinstance(opt_summary, dict):
        return {
            "job_id": job_id,
            "best_model": opt_summary.get("best_model"),
            "best_score": opt_summary.get("best_score"),
            "best_params": opt_summary.get("best_params"),
            "optimization_time": opt_summary.get("total_optimization_time")
        }
    return {"job_id": job_id, "message": "No optimization data available"}


# ── A/B Testing Endpoints (Fix #8) ───────────────────────────────────────────

class ABExperimentRequest(BaseModel):
    name: str
    model_a_id: str
    model_b_id: str
    traffic_split: float = 0.5

class ABOutcomeRequest(BaseModel):
    experiment_id: str
    variant: str
    correct: bool
    latency_ms: float = 0.0

@app.post("/ab-tests")
def create_ab_test(req: ABExperimentRequest):
    """Create new A/B experiment to compare two models"""
    try:
        from veda.agents.ab_testing.framework import ab_framework
        exp = ab_framework.create_experiment(
            name=req.name,
            model_a_id=req.model_a_id,
            model_b_id=req.model_b_id,
            traffic_split=req.traffic_split
        )
        return exp.to_dict()
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/ab-tests")
def list_ab_tests(status: Optional[str] = None):
    """List all A/B experiments"""
    try:
        from veda.agents.ab_testing.framework import ab_framework
        return {"experiments": ab_framework.list_experiments(status), "summary": ab_framework.get_summary()}
    except Exception as e:
        return {"error": str(e)}

@app.get("/ab-tests/{experiment_id}")
def get_ab_test(experiment_id: str):
    """Get A/B experiment details and current results"""
    try:
        from veda.agents.ab_testing.framework import ab_framework
        exp = ab_framework.get_experiment(experiment_id)
        if not exp:
            raise HTTPException(404, "Experiment not found")
        return exp
    except HTTPException:
        raise
    except Exception as e:
        return {"error": str(e)}

@app.post("/ab-tests/{experiment_id}/analyze")
def analyze_ab_test(experiment_id: str):
    """Run statistical significance test on experiment"""
    try:
        from veda.agents.ab_testing.framework import ab_framework
        return ab_framework.analyze(experiment_id)
    except Exception as e:
        return {"error": str(e)}

@app.post("/ab-tests/{experiment_id}/stop")
def stop_ab_test(experiment_id: str, deploy_winner: bool = False):
    """Stop experiment and optionally deploy winner"""
    try:
        from veda.agents.ab_testing.framework import ab_framework
        return ab_framework.stop_experiment(experiment_id, deploy_winner)
    except Exception as e:
        return {"error": str(e)}

@app.post("/ab-tests/outcome")
def record_ab_outcome(req: ABOutcomeRequest):
    """Record prediction outcome for A/B tracking"""
    try:
        from veda.agents.ab_testing.framework import ab_framework
        ab_framework.record_outcome(req.experiment_id, req.variant, req.correct, req.latency_ms)
        return {"message": "Outcome recorded"}
    except Exception as e:
        return {"error": str(e)}


class RegisterDatasetRequest(BaseModel):
    file_path: str
    tags: Optional[List[str]] = []
    notes: Optional[str] = ""
    workflow_id: Optional[str] = None

@app.post("/datasets/register")
def register_dataset(req: RegisterDatasetRequest):
    try:
        from veda.data_versioning.dvc import dvc
        version = dvc.register(file_path=req.file_path, tags=req.tags, notes=req.notes, workflow_id=req.workflow_id)
        return version.to_dict()
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/datasets")
def list_datasets(tag: Optional[str] = None, limit: int = 20):
    try:
        from veda.data_versioning.dvc import dvc
        versions = dvc.list_versions(tag=tag, limit=limit)
        return {"total": len(versions), "stats": dvc.get_stats(), "versions": [v.to_dict() for v in versions]}
    except Exception as e:
        return {"error": str(e)}

@app.get("/datasets/{version_id}")
def get_dataset(version_id: str):
    try:
        from veda.data_versioning.dvc import dvc
        version = dvc.get(version_id)
        if not version:
            raise HTTPException(404, f"Version {version_id} not found")
        return version.to_dict()
    except HTTPException:
        raise
    except Exception as e:
        return {"error": str(e)}

@app.get("/datasets/{version_id}/lineage")
def get_dataset_lineage(version_id: str):
    try:
        from veda.data_versioning.dvc import dvc
        lineage = dvc.get_lineage(version_id)
        return {"version_id": version_id, "lineage": lineage, "depth": len(lineage)}
    except Exception as e:
        return {"error": str(e)}

@app.get("/datasets/compare/{version_id_a}/{version_id_b}")
def compare_datasets(version_id_a: str, version_id_b: str):
    try:
        from veda.data_versioning.dvc import dvc
        return dvc.compare(version_id_a, version_id_b)
    except Exception as e:
        return {"error": str(e)}
