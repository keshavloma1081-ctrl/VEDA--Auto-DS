"""
VEDA FastAPI Server with Database Persistence
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from sqlalchemy.orm import Session
import os
import uuid
from datetime import datetime

# Import VEDA components
from veda.agents.core_pipeline.planner import PlannerAgent
from veda.core.state import VEDAState
from veda.database.models import Workflow, Model, Prediction, get_db, init_db

# Initialize FastAPI
app = FastAPI(
    title="VEDA ML API",
    description="Production REST API for VEDA Autonomous ML System with Database Persistence",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database on startup
@app.on_event("startup")
def startup_event():
    init_db()
    print("🚀 VEDA API Started")

# Lazy planner initialization
planner = None

def get_planner():
    """Get or initialize planner"""
    global planner
    if planner is None:
        if not os.getenv("GROQ_API_KEY"):
            raise HTTPException(
                status_code=500,
                detail="GROQ_API_KEY not set"
            )
        planner = PlannerAgent()
    return planner

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class WorkflowRequest(BaseModel):
    dataset_path: str
    goal: str
    config: Optional[Dict[str, Any]] = {}

class WorkflowResponse(BaseModel):
    job_id: str
    status: str
    message: str
    submitted_at: str

class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: float
    current_step: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: str
    updated_at: str

class PredictionRequest(BaseModel):
    model_id: str
    data: List[List[float]]
    return_probabilities: bool = False

class PredictionResponse(BaseModel):
    predictions: List[Any]
    probabilities: Optional[List[List[float]]] = None
    model_version: str
    inference_time_ms: float

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def run_workflow(job_id: str, dataset_path: str, goal: str, config: Dict):
    """Background task to run VEDA workflow"""
    # Get database session
    db = next(get_db())
    
    try:
        # Update status to running
        workflow = db.query(Workflow).filter(Workflow.job_id == job_id).first()
        workflow.status = 'running'
        workflow.current_step = 'Planning'
        db.commit()
        
        # Initialize state
        state = VEDAState(
            dataset_path=dataset_path,
            user_goal=goal,
            job_id=job_id
        )
        
        # Get planner
        planner_instance = get_planner()
        
        # Execute workflow
        workflow.current_step = 'Executing Pipeline'
        db.commit()
        
        result = planner_instance.execute(state)
        
        # Update with results
        workflow.status = 'completed'
        workflow.progress = 100.0
        workflow.result = result
        workflow.updated_at = datetime.utcnow()
        db.commit()
        
    except Exception as e:
        # Update with error
        workflow = db.query(Workflow).filter(Workflow.job_id == job_id).first()
        workflow.status = 'failed'
        workflow.error = str(e)
        workflow.updated_at = datetime.utcnow()
        db.commit()
    
    finally:
        db.close()

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
def root():
    """Root endpoint"""
    return {
        "service": "VEDA ML API",
        "version": "2.0.0",
        "status": "operational",
        "database": "connected",
        "docs": "/docs"
    }

@app.get("/health")
def health_check(db: Session = Depends(get_db)):
    """Health check with database connectivity"""
    try:
        # Test database connection
        db.execute("SELECT 1")
        db_status = "connected"
    except:
        db_status = "disconnected"
    
    groq_status = "configured" if os.getenv("GROQ_API_KEY") else "missing"
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            "api": "operational",
            "database": db_status,
            "groq_api_key": groq_status,
            "planner": "ready" if planner is not None else "not_initialized"
        }
    }

@app.post("/workflows", response_model=WorkflowResponse)
async def create_workflow(
    request: WorkflowRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Submit a new ML workflow"""
    
    # Check API key
    if not os.getenv("GROQ_API_KEY"):
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY not configured"
        )
    
    # Validate dataset
    if not os.path.exists(request.dataset_path):
        raise HTTPException(
            status_code=404,
            detail=f"Dataset not found: {request.dataset_path}"
        )
    
    # Create workflow in database
    job_id = str(uuid.uuid4())
    
    workflow = Workflow(
        job_id=job_id,
        dataset_path=request.dataset_path,
        goal=request.goal,
        config=request.config,
        status='submitted',
        progress=0.0
    )
    
    db.add(workflow)
    db.commit()
    db.refresh(workflow)
    
    # Start background task
    background_tasks.add_task(
        run_workflow,
        job_id,
        request.dataset_path,
        request.goal,
        request.config
    )
    
    return WorkflowResponse(
        job_id=job_id,
        status='submitted',
        message='Workflow submitted successfully',
        submitted_at=workflow.created_at.isoformat()
    )

@app.get("/workflows/{job_id}", response_model=JobStatus)
def get_workflow_status(job_id: str, db: Session = Depends(get_db)):
    """Get workflow status"""
    workflow = db.query(Workflow).filter(Workflow.job_id == job_id).first()
    
    if not workflow:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JobStatus(**workflow.to_dict())

@app.get("/workflows")
def list_workflows(
    status: Optional[str] = None,
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """List all workflows"""
    query = db.query(Workflow)
    
    if status:
        query = query.filter(Workflow.status == status)
    
    workflows = query.order_by(Workflow.created_at.desc()).limit(limit).all()
    
    return [JobStatus(**w.to_dict()) for w in workflows]

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    db: Session = Depends(get_db)
):
    """Make real-time predictions"""
    import time
    start_time = time.time()
    
    # Mock predictions (TODO: load actual model)
    predictions = [0] * len(request.data)
    probabilities = [[0.3, 0.7]] * len(request.data) if request.return_probabilities else None
    
    inference_time = (time.time() - start_time) * 1000
    
    # Log prediction to database
    prediction = Prediction(
        model_id=request.model_id,
        model_version="1.0.0",
        input_data={"data": request.data},
        predictions=predictions,
        probabilities=probabilities,
        inference_time_ms=inference_time
    )
    db.add(prediction)
    db.commit()
    
    return PredictionResponse(
        predictions=predictions,
        probabilities=probabilities,
        model_version="1.0.0",
        inference_time_ms=inference_time
    )

@app.get("/models")
def list_models(db: Session = Depends(get_db)):
    """List all trained models"""
    models = db.query(Model).order_by(Model.created_at.desc()).all()
    return {
        "models": [m.to_dict() for m in models],
        "count": len(models)
    }

@app.get("/stats")
def get_stats(db: Session = Depends(get_db)):
    """Get system statistics"""
    total_workflows = db.query(Workflow).count()
    completed = db.query(Workflow).filter(Workflow.status == 'completed').count()
    failed = db.query(Workflow).filter(Workflow.status == 'failed').count()
    running = db.query(Workflow).filter(Workflow.status == 'running').count()
    
    total_predictions = db.query(Prediction).count()
    total_models = db.query(Model).count()
    
    return {
        "workflows": {
            "total": total_workflows,
            "completed": completed,
            "failed": failed,
            "running": running,
            "success_rate": (completed / total_workflows * 100) if total_workflows > 0 else 0
        },
        "predictions": {
            "total": total_predictions
        },
        "models": {
            "total": total_models
        }
    }

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