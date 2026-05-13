"""
Pydantic Schemas for Request/Response Validation
Enhanced data validation for all API endpoints
"""
from pydantic import BaseModel, Field, validator, ConfigDict
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

# ============================================================================
# ENUMS
# ============================================================================

class WorkflowStatus(str, Enum):
    SUBMITTED = "submitted"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class TaskType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    FORECASTING = "forecasting"

class ModelType(str, Enum):
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    RANDOM_FOREST = "random_forest"
    NEURAL_NETWORK = "neural_network"

# ============================================================================
# REQUEST SCHEMAS
# ============================================================================

class WorkflowRequest(BaseModel):
    """Request to create a new ML workflow"""
    dataset_path: str = Field(..., description="Path to dataset file")
    goal: str = Field(..., min_length=10, max_length=500, description="ML goal in plain English")
    config: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Optional configuration")
    
    @validator('dataset_path')
    def validate_dataset_path(cls, v):
        """Validate dataset path"""
        if not v.strip():
            raise ValueError("Dataset path cannot be empty")
        
        # Check extension
        valid_extensions = ['.csv', '.xlsx', '.xls', '.json', '.parquet']
        if not any(v.lower().endswith(ext) for ext in valid_extensions):
            raise ValueError(f"Dataset must be one of: {', '.join(valid_extensions)}")
        
        return v
    
    @validator('goal')
    def validate_goal(cls, v):
        """Validate goal description"""
        if len(v.strip()) < 10:
            raise ValueError("Goal must be at least 10 characters")
        return v.strip()

class PredictionRequest(BaseModel):
    """Request for real-time predictions"""
    model_id: str = Field(..., description="Model ID to use for predictions")
    data: List[List[float]] = Field(..., min_items=1, description="Input data for predictions")
    return_probabilities: bool = Field(default=False, description="Return class probabilities")
    
    @validator('data')
    def validate_data(cls, v):
        """Validate prediction data"""
        if not v:
            raise ValueError("Data cannot be empty")
        
        # Check all rows have same length
        lengths = [len(row) for row in v]
        if len(set(lengths)) > 1:
            raise ValueError("All data rows must have same length")
        
        return v

class ModelCreateRequest(BaseModel):
    """Request to register a new model"""
    name: str = Field(..., min_length=3, max_length=200)
    version: str = Field(default="1.0.0", pattern=r'^\d+\.\d+\.\d+$')
    model_type: ModelType
    task_type: TaskType
    file_path: str
    metrics: Dict[str, float] = Field(default_factory=dict)
    dataset_path: Optional[str] = None

# ============================================================================
# RESPONSE SCHEMAS
# ============================================================================

class WorkflowResponse(BaseModel):
    """Response after creating workflow"""
    job_id: str
    status: WorkflowStatus
    message: str
    submitted_at: str

class JobStatus(BaseModel):
    """Workflow job status"""
    job_id: str
    status: WorkflowStatus
    progress: float = Field(ge=0.0, le=100.0)
    current_step: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: str
    updated_at: str

class PredictionResponse(BaseModel):
    """Prediction response"""
    predictions: List[Any]
    probabilities: Optional[List[List[float]]] = None
    model_version: str
    inference_time_ms: float
    prediction_id: int

class ModelResponse(BaseModel):
    """Model information"""
    model_id: str
    name: str
    version: str
    model_type: str
    task_type: str
    metrics: Dict[str, float]
    created_at: str

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    components: Dict[str, str]

class StatsResponse(BaseModel):
    """System statistics"""
    workflows: Dict[str, Any]
    predictions: Dict[str, int]
    models: Dict[str, int]

class ErrorResponse(BaseModel):
    """Error response"""
    detail: str
    error_code: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

# ============================================================================
# PAGINATION
# ============================================================================

class PaginationParams(BaseModel):
    """Pagination parameters"""
    page: int = Field(default=1, ge=1, description="Page number")
    page_size: int = Field(default=10, ge=1, le=100, description="Items per page")
    
    @property
    def offset(self) -> int:
        return (self.page - 1) * self.page_size
    
    @property
    def limit(self) -> int:
        return self.page_size

class PaginatedResponse(BaseModel):
    """Paginated response wrapper"""
    items: List[Any]
    total: int
    page: int
    page_size: int
    total_pages: int