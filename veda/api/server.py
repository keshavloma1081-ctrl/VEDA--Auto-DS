"""
VEDA FastAPI Server
Production REST API for VEDA ML System
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import os
import uuid
from datetime import datetime

# Import VEDA components
from veda.agents.core_pipeline.planner import PlannerAgent
from veda.core.state import VEDAState

# Initialize FastAPI
app = FastAPI(
    title="VEDA ML API",
    description="Production REST API for VEDA Autonomous ML System",
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

# Lazy planner initialization (avoids startup errors)
planner = None

def get_planner():
    """Get or initialize planner (lazy loading)"""
    global planner
    if planner is None:
        if not os.getenv("GROQ_API_KEY"):
            raise HTTPException(
                status_code=500,
                detail="GROQ_API_KEY not set. Cannot initialize planner."
            )
        planner = PlannerAgent()
    return planner

# In-memory job store (replace with database in production)
jobs = {}

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class WorkflowRequest(BaseModel):
    """Request model for ML workflow"""
    dataset_path: str
    goal: str
    config: Optional[Dict[str, Any]] = {}

class WorkflowResponse(BaseModel):
    """Response model for workflow submission"""
    job_id: str
    status: str
    message: str
    submitted_at: str

class JobStatus(BaseModel):
    """Job status response"""
    job_id: str
    status: str
    progress: float
    current_step: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: str
    updated_at: str

class PredictionRequest(BaseModel):
    """Request for real-time predictions"""
    model_id: str
    data: List[List[float]]
    return_probabilities: bool = False

class PredictionResponse(BaseModel):
    """Response for predictions"""
    predictions: List[Any]
    probabilities: Optional[List[List[float]]] = None
    model_version: str
    inference_time_ms: float

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def run_workflow(job_id: str, dataset_path: str, goal: str, config: Dict):
    """Background task to run VEDA workflow"""
    try:
        jobs[job_id]['status'] = 'running'
        jobs[job_id]['current_step'] = 'Planning'
        
        # Initialize state
        state = VEDAState(
            dataset_path=dataset_path,
            user_goal=goal,
            job_id=job_id
        )
        
        # Get planner (lazy initialization)
        planner_instance = get_planner()
        
        # Execute planner
        jobs[job_id]['current_step'] = 'Executing Pipeline'
        result = planner_instance.execute(state)
        
        # Update job
        jobs[job_id]['status'] = 'completed'
        jobs[job_id]['progress'] = 100.0
        jobs[job_id]['result'] = result
        jobs[job_id]['updated_at'] = datetime.utcnow().isoformat()
        
    except Exception as e:
        jobs[job_id]['status'] = 'failed'
        jobs[job_id]['error'] = str(e)
        jobs[job_id]['updated_at'] = datetime.utcnow().isoformat()

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
        "docs": "/docs",
        "endpoints": {
            "health": "/health",
            "workflows": "/workflows",
            "predict": "/predict",
            "models": "/models"
        }
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    groq_status = "configured" if os.getenv("GROQ_API_KEY") else "missing"
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            "api": "operational",
            "groq_api_key": groq_status,
            "planner": "ready" if planner is not None else "not_initialized"
        }
    }

@app.post("/workflows", response_model=WorkflowResponse)
async def create_workflow(
    request: WorkflowRequest,
    background_tasks: BackgroundTasks
):
    """
    Submit a new ML workflow
    
    Creates a background job that processes the dataset through VEDA pipeline
    """
    # Check if API key is set
    if not os.getenv("GROQ_API_KEY"):
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY not configured. Cannot process workflows."
        )
    
    # Validate dataset exists
    if not os.path.exists(request.dataset_path):
        raise HTTPException(
            status_code=404,
            detail=f"Dataset not found: {request.dataset_path}"
        )
    
    # Create job
    job_id = str(uuid.uuid4())
    timestamp = datetime.utcnow().isoformat()
    
    jobs[job_id] = {
        'job_id': job_id,
        'status': 'submitted',
        'progress': 0.0,
        'dataset_path': request.dataset_path,
        'goal': request.goal,
        'created_at': timestamp,
        'updated_at': timestamp
    }
    
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
        submitted_at=timestamp
    )

@app.get("/workflows/{job_id}", response_model=JobStatus)
def get_workflow_status(job_id: str):
    """Get status of a workflow job"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JobStatus(**jobs[job_id])

@app.get("/workflows", response_model=List[JobStatus])
def list_workflows(
    status: Optional[str] = None,
    limit: int = 10
):
    """List all workflows with optional filtering"""
    filtered_jobs = []
    
    for job in jobs.values():
        if status is None or job['status'] == status:
            filtered_jobs.append(JobStatus(**job))
    
    return filtered_jobs[:limit]

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make real-time predictions using a trained model
    """
    import time
    start_time = time.time()
    
    # Mock predictions (TODO: implement actual model loading)
    predictions = [0] * len(request.data)
    probabilities = [[0.3, 0.7]] * len(request.data) if request.return_probabilities else None
    
    inference_time = (time.time() - start_time) * 1000
    
    return PredictionResponse(
        predictions=predictions,
        probabilities=probabilities,
        model_version="1.0.0",
        inference_time_ms=inference_time
    )

@app.get("/models")
def list_models():
    """List all trained models"""
    return {
        "models": [],
        "count": 0,
        "message": "MLflow integration coming soon"
    }

@app.get("/models/{model_id}")
def get_model_info(model_id: str):
    """Get information about a specific model"""
    raise HTTPException(
        status_code=404,
        detail="Model not found. MLflow integration coming soon."
    )

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