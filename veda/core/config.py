"""
Configuration Management
Centralized configuration with environment variable support
"""
from pydantic_settings import BaseSettings
from typing import Optional
from functools import lru_cache

class Settings(BaseSettings):
    """Application settings"""
    
    # Application
    APP_NAME: str = "VEDA"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = False
    
    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 4
    
    # Database
    DATABASE_URL: str = "sqlite:///./veda.db"
    DATABASE_ECHO: bool = False
    
    # Groq API
    GROQ_API_KEY: Optional[str] = None
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    GROQ_MAX_TOKENS: int = 2000
    GROQ_TEMPERATURE: float = 0.1
    
    # MLflow
    MLFLOW_TRACKING_URI: str = "file:./mlruns"
    MLFLOW_EXPERIMENT_NAME: str = "VEDA-AutoML"
    
    # Security
    SECRET_KEY: str = "change-this-in-production-use-openssl-rand-hex-32"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440  # 24 hours
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_DIR: str = "logs"
    
    # File Storage
    DATA_DIR: str = "data"
    OUTPUT_DIR: str = "outputs"
    MODEL_DIR: str = "models"
    
    # Limits
    MAX_UPLOAD_SIZE_MB: int = 100
    MAX_CONCURRENT_JOBS: int = 10
    JOB_TIMEOUT_MINUTES: int = 60
    
    # CORS
    CORS_ORIGINS: list = ["*"]
    
    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()

# Global settings instance
settings = get_settings()