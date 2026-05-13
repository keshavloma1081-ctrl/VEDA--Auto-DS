"""
API Rate Limiting with SlowAPI
"""
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse

# Initialize rate limiter
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["100/minute", "1000/hour"]
)

# Custom rate limit handler
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    """Handle rate limit exceeded"""
    return JSONResponse(
        status_code=429,
        content={
            "error": "Rate limit exceeded",
            "detail": str(exc.detail),
            "retry_after": exc.detail
        }
    )

# Rate limit configurations
RATE_LIMITS = {
    "auth": "5/minute",           # Login attempts
    "workflows": "10/minute",     # Workflow creation
    "api": "100/minute",          # General API calls
    "reports": "20/minute",       # Report generation
}

# Usage in FastAPI endpoints:
"""
from veda.utils.rate_limit import limiter, RATE_LIMITS

@app.post("/auth/login")
@limiter.limit(RATE_LIMITS["auth"])
async def login(request: Request, ...):
    ...

@app.post("/workflows")
@limiter.limit(RATE_LIMITS["workflows"])
async def create_workflow(request: Request, ...):
    ...
"""