"""
VEDA Database Models
SQLAlchemy ORM models for persistent storage
"""
from sqlalchemy import Column, String, Integer, Float, DateTime, JSON, Text, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

Base = declarative_base()

# ============================================================================
# MODELS
# ============================================================================

class Workflow(Base):
    """Workflow job tracking"""
    __tablename__ = 'workflows'
    
    # Primary key
    job_id = Column(String(36), primary_key=True)
    
    # Job metadata
    status = Column(String(20), default='submitted')  # submitted, running, completed, failed
    progress = Column(Float, default=0.0)
    
    # Input
    dataset_path = Column(Text, nullable=False)
    goal = Column(Text, nullable=False)
    config = Column(JSON, default={})
    
    # Execution
    current_step = Column(String(100), nullable=True)
    
    # Results (stored as JSON)
    result = Column(JSON, nullable=True)
    error = Column(Text, nullable=True)
    
    # Artifacts
    model_path = Column(Text, nullable=True)
    dashboard_path = Column(Text, nullable=True)
    report_path = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'job_id': self.job_id,
            'status': self.status,
            'progress': self.progress,
            'dataset_path': self.dataset_path,
            'goal': self.goal,
            'current_step': self.current_step,
            'result': self.result,
            'error': self.error,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class Model(Base):
    """Trained model registry"""
    __tablename__ = 'models'
    
    # Primary key
    model_id = Column(String(36), primary_key=True)
    
    # Metadata
    name = Column(String(200), nullable=False)
    version = Column(String(50), default='1.0.0')
    model_type = Column(String(50))  # xgboost, lightgbm, neural_network
    task_type = Column(String(50))   # classification, regression
    
    # Performance
    metrics = Column(JSON, default={})
    
    # Storage
    file_path = Column(Text, nullable=False)
    mlflow_run_id = Column(String(100), nullable=True)
    
    # Training info
    dataset_path = Column(Text)
    workflow_id = Column(String(36), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'model_id': self.model_id,
            'name': self.name,
            'version': self.version,
            'model_type': self.model_type,
            'task_type': self.task_type,
            'metrics': self.metrics,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class Prediction(Base):
    """Prediction logging for monitoring"""
    __tablename__ = 'predictions'
    
    # Primary key
    prediction_id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Model info
    model_id = Column(String(36), nullable=False)
    model_version = Column(String(50))
    
    # Input/Output
    input_data = Column(JSON, nullable=False)
    predictions = Column(JSON, nullable=False)
    probabilities = Column(JSON, nullable=True)
    
    # Performance
    inference_time_ms = Column(Float)
    
    # Timestamp
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'prediction_id': self.prediction_id,
            'model_id': self.model_id,
            'predictions': self.predictions,
            'inference_time_ms': self.inference_time_ms,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

# ============================================================================
# DATABASE CONNECTION
# ============================================================================

# Database URL (SQLite by default, can switch to PostgreSQL)
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///./veda.db')

# Create engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith('sqlite') else {}
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create tables
def init_db():
    """Initialize database (create tables)"""
    Base.metadata.create_all(bind=engine)
    print("✅ Database initialized")

# Dependency for FastAPI
def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()