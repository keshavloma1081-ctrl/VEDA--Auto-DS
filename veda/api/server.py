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
