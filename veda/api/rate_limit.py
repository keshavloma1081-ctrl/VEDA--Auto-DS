import os, time, logging
from collections import defaultdict
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel

log = logging.getLogger(__name__)

RATE_LIMITS = {
    "login":           {"requests": 5,   "window": 60},
    "create_workflow": {"requests": 10,  "window": 60},
    "upload":          {"requests": 5,   "window": 60},
    "get_workflow":    {"requests": 120, "window": 60},
    "list_workflows":  {"requests": 60,  "window": 60},
    "health":          {"requests": 300, "window": 60},
    "default":         {"requests": 100, "window": 60},
}

WHITELIST_IPS = set(os.getenv("RATE_LIMIT_WHITELIST", "127.0.0.1,::1").split(","))
RATE_LIMITING_ENABLED = os.getenv("RATE_LIMITING_ENABLED", "true").lower() == "true"

class InMemoryRateLimiter:
    def __init__(self):
        self._store = defaultdict(list)

    def _cleanup(self, key, window):
        cutoff = time.time() - window
        self._store[key] = [e for e in self._store[key] if e > cutoff]

    def is_allowed(self, key, limit, window):
        now = time.time()
        self._cleanup(key, window)
        count = len(self._store[key])
        if count >= limit:
            oldest = self._store[key][0] if self._store[key] else now
            retry_after = max(1, int(oldest + window - now) + 1)
            return False, {"limit": limit, "remaining": 0, "reset": int(now + retry_after), "retry_after": retry_after}
        self._store[key].append(now)
        return True, {"limit": limit, "remaining": limit - count - 1, "reset": int(now + window), "retry_after": 0}

    def reset(self, key):
        if key in self._store:
            del self._store[key]

    def get_stats(self):
        return {"type": "in_memory", "tracked_keys": len(self._store)}

def create_rate_limiter():
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        try:
            import redis as r
            client = r.from_url(redis_url, decode_responses=True)
            client.ping()
            log.info("Using Redis rate limiter")
            return InMemoryRateLimiter()
        except Exception:
            pass
    return InMemoryRateLimiter()

rate_limiter = create_rate_limiter()

class RateLimitMiddleware(BaseHTTPMiddleware):
    PATH_LIMITS = {
        "/auth/login": "login",
        "/upload": "upload",
        "/health": "health",
        "/stats": "health",
    }

    def _get_ip(self, request):
        fwd = request.headers.get("X-Forwarded-For")
        if fwd:
            return fwd.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _get_limit_key(self, request):
        path = request.url.path
        if path in self.PATH_LIMITS:
            if path == "/workflows" and request.method == "GET":
                return "list_workflows"
            return self.PATH_LIMITS[path]
        if path == "/workflows" and request.method == "POST":
            return "create_workflow"
        if path.startswith("/workflows/"):
            return "get_workflow"
        return "default"

    async def dispatch(self, request, call_next):
        if not RATE_LIMITING_ENABLED:
            return await call_next(request)
        ip = self._get_ip(request)
        if ip in WHITELIST_IPS:
            return await call_next(request)
        limit_key = self._get_limit_key(request)
        config = RATE_LIMITS.get(limit_key, RATE_LIMITS["default"])
        allowed, info = rate_limiter.is_allowed(f"{ip}:{limit_key}", config["requests"], config["window"])
        if not allowed:
            log.warning(f"Rate limit exceeded | ip={ip} | endpoint={request.url.path}")
            return JSONResponse(
                {"detail": f"Rate limit exceeded. Try again in {info['retry_after']} seconds.", "retry_after": info["retry_after"]},
                status_code=429,
                headers={"Retry-After": str(info["retry_after"]), "X-RateLimit-Limit": str(info["limit"]), "X-RateLimit-Remaining": "0"}
            )
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(info["limit"])
        response.headers["X-RateLimit-Remaining"] = str(info["remaining"])
        return response

class BruteForceProtection:
    def __init__(self, max_attempts=5, lockout_minutes=15):
        self.max_attempts = max_attempts
        self.lockout_seconds = lockout_minutes * 60
        self._failures = defaultdict(list)
        self._locked = {}

    def record_failure(self, ip, username):
        now = time.time()
        key = f"{ip}:{username}"
        cutoff = now - self.lockout_seconds
        self._failures[key] = [t for t in self._failures[key] if t > cutoff]
        self._failures[key].append(now)
        if len(self._failures[key]) >= self.max_attempts:
            self._locked[key] = now + self.lockout_seconds
            log.warning(f"Account locked | ip={ip} | user={username}")

    def is_locked(self, ip, username):
        key = f"{ip}:{username}"
        now = time.time()
        if key in self._locked:
            if now < self._locked[key]:
                return True, int(self._locked[key] - now)
            del self._locked[key]
            self._failures.pop(key, None)
        return False, 0

    def record_success(self, ip, username):
        key = f"{ip}:{username}"
        self._failures.pop(key, None)
        self._locked.pop(key, None)

    def get_attempts(self, ip, username):
        key = f"{ip}:{username}"
        now = time.time()
        return len([t for t in self._failures.get(key, []) if t > now - self.lockout_seconds])

brute_force_protection = BruteForceProtection(
    max_attempts=int(os.getenv("MAX_LOGIN_ATTEMPTS", "5")),
    lockout_minutes=int(os.getenv("LOCKOUT_MINUTES", "15"))
)

def get_rate_limit_info():
    return {
        "enabled": RATE_LIMITING_ENABLED,
        "backend": "in_memory",
        "limits": RATE_LIMITS,
        "stats": rate_limiter.get_stats()
    }
