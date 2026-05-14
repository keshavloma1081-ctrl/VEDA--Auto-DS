cd C:\Users\kk536\OneDrive\Desktop\VEDA--Auto-DS

# Create security.py directly
@'
import os, re, hashlib, pathlib, logging
from typing import Optional
from fastapi import HTTPException, UploadFile, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, validator, Field
import time

log = logging.getLogger(__name__)

ALLOWED_BASE_DIRS = [
    os.path.abspath("data"),
    os.path.abspath("uploads"),
    os.path.abspath("examples"),
    os.path.abspath("examples/datasets"),
    os.path.abspath("test_data"),
]

ALLOWED_EXTENSIONS = {".csv",".xlsx",".xls",".json",".parquet",".tsv"}
MAX_FILE_SIZE_BYTES = int(os.getenv("MAX_FILE_SIZE_MB","500")) * 1024 * 1024
MIN_FILE_SIZE_BYTES = 10
MIN_GOAL_LENGTH = 10
MAX_GOAL_LENGTH = 1000

DANGEROUS_PATH_PATTERNS = [r"\.\.",r"~",r"\$",r"\|",r";",r"&",r"`",r"etc/passwd",r"proc/",r"/dev/"]
DANGEROUS_GOAL_PATTERNS = [r"ignore previous",r"ignore all",r"system prompt",r"jailbreak",r"<script",r"eval\(",r"exec\(",r"__import__",r"subprocess",r"os\.system"]

def validate_dataset_path(path: str) -> str:
    if not path or not path.strip():
        raise HTTPException(status_code=400, detail="Dataset path cannot be empty")
    path = path.strip()
    path_lower = path.lower()
    for pattern in DANGEROUS_PATH_PATTERNS:
        if re.search(pattern, path_lower):
            log.warning(f"Dangerous path pattern: {pattern} in {path}")
            raise HTTPException(status_code=400, detail="Invalid path: contains forbidden pattern")
    try:
        resolved = pathlib.Path(path).resolve()
    except (OSError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid file path: {str(e)}")
    ext = resolved.suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}")
    path_str = str(resolved)
    allowed = any(path_str.startswith(d) for d in ALLOWED_BASE_DIRS)
    if not allowed:
        log.warning(f"Path traversal attempt: {path} -> {resolved}")
        raise HTTPException(status_code=403, detail="Access denied: path is outside allowed directories. Place dataset in data/ or uploads/ folder.")
    if not resolved.exists():
        raise HTTPException(status_code=404, detail=f"Dataset file not found: {resolved.name}")
    if not resolved.is_file():
        raise HTTPException(status_code=400, detail="Path must point to a file")
    file_size = resolved.stat().st_size
    if file_size < MIN_FILE_SIZE_BYTES:
        raise HTTPException(status_code=400, detail=f"File too small ({file_size} bytes)")
    if file_size > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large ({file_size/(1024*1024):.1f}MB). Max: {MAX_FILE_SIZE_BYTES//(1024*1024)}MB")
    log.info(f"Path validated: {resolved.name}")
    return str(resolved)

def validate_goal(goal: str) -> str:
    if not goal or not goal.strip():
        raise HTTPException(status_code=400, detail="Goal cannot be empty")
    goal = goal.strip()
    if len(goal) < MIN_GOAL_LENGTH:
        raise HTTPException(status_code=400, detail=f"Goal too short. Minimum {MIN_GOAL_LENGTH} characters.")
    if len(goal) > MAX_GOAL_LENGTH:
        raise HTTPException(status_code=400, detail=f"Goal too long. Maximum {MAX_GOAL_LENGTH} characters.")
    goal_lower = goal.lower()
    for pattern in DANGEROUS_GOAL_PATTERNS:
        if re.search(pattern, goal_lower):
            log.warning(f"Suspicious goal pattern: {pattern}")
            raise HTTPException(status_code=400, detail="Goal contains forbidden content")
    goal = re.sub(r"<[^>]+>", "", goal)
    return goal

async def save_upload_file(file: UploadFile, upload_dir: str = "uploads") -> str:
    upload_path = pathlib.Path(upload_dir).resolve()
    upload_path.mkdir(parents=True, exist_ok=True)
    abs_upload = str(upload_path)
    if abs_upload not in ALLOWED_BASE_DIRS:
        ALLOWED_BASE_DIRS.append(abs_upload)
    filename = pathlib.Path(file.filename)
    ext = filename.suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"File type '{ext}' not allowed")
    content = await file.read()
    file_size = len(content)
    if file_size < MIN_FILE_SIZE_BYTES:
        raise HTTPException(status_code=400, detail="File too small")
    if file_size > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large ({file_size/(1024*1024):.1f}MB)")
    file_hash = hashlib.sha256(content).hexdigest()
    safe_name = re.sub(r"[^a-zA-Z0-9._-]", "_", filename.stem)
    unique_filename = f"{file_hash[:8]}_{safe_name}{ext}"
    save_path = upload_path / unique_filename
    with open(save_path, "wb") as f:
        f.write(content)
    log.info(f"File saved: {save_path}")
    return str(save_path)

class SecurityMiddleware(BaseHTTPMiddleware):
    SECURITY_HEADERS = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Cache-Control": "no-store",
    }
    BLOCKED_UAS = ["sqlmap","nikto","nmap","masscan","zgrab","gobuster"]

    async def dispatch(self, request: Request, call_next) -> Response:
        ua = request.headers.get("user-agent","").lower()
        for blocked in self.BLOCKED_UAS:
            if blocked in ua:
                return JSONResponse({"detail":"Forbidden"}, status_code=403)
        start = time.time()
        response = await call_next(request)
        for h, v in self.SECURITY_HEADERS.items():
            response.headers[h] = v
        response.headers["X-Response-Time"] = f"{round((time.time()-start)*1000,2)}ms"
        return response

class SecureWorkflowRequest(BaseModel):
    dataset_path: str = Field(..., description="Path to dataset in data/ or uploads/ directory")
    goal: str = Field(..., min_length=10, max_length=1000, description="ML goal in plain English")
    priority: str = Field(default="normal")
    config: dict = Field(default_factory=dict)

    @validator("dataset_path")
    def check_path(cls, v):
        if not v.strip():
            raise ValueError("Path cannot be empty")
        for p in [r"\.\.",r"etc/passwd",r"proc/"]:
            if re.search(p, v.lower()):
                raise ValueError("Invalid path")
        ext = pathlib.Path(v).suffix.lower()
        if ext not in {".csv",".xlsx",".xls",".json",".parquet",".tsv"}:
            raise ValueError(f"Unsupported file type: {ext}")
        return v.strip()

    @validator("goal")
    def check_goal(cls, v):
        v = v.strip()
        for p in [r"ignore previous",r"system prompt",r"<script",r"eval\("]:
            if re.search(p, v.lower()):
                raise ValueError("Goal contains forbidden content")
        return re.sub(r"<[^>]+>","",v)

    @validator("priority")
    def check_priority(cls, v):
        if v not in {"normal","high"}:
            raise ValueError("Priority must be normal or high")
        return v

class SecureLoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=100)
    password: str = Field(..., min_length=1, max_length=200)

    @validator("username")
    def check_username(cls, v):
        v = v.strip()
        if not re.match(r"^[a-zA-Z0-9_.-]+$", v):
            raise ValueError("Username contains invalid characters")
        return v

def ensure_allowed_dirs():
    for d in ALLOWED_BASE_DIRS:
        pathlib.Path(d).mkdir(parents=True, exist_ok=True)

def get_file_info(path: str) -> dict:
    p = pathlib.Path(path)
    if not p.exists():
        return {}
    s = p.stat()
    return {"filename": p.name, "extension": p.suffix.lower(), "size_bytes": s.st_size, "size_mb": round(s.st_size/(1024*1024),3)}
'@ | Out-File -Encoding utf8 veda\api\security.py

# Verify
Write-Host "security.py created:"
Select-String -Path "veda\api\security.py" -Pattern "validate_dataset_path|SecurityMiddleware|DANGEROUS_PATH" | Select-Object LineNumber, Line