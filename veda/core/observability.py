"""
VEDA Observability - Fix #10
Structured logging, correlation IDs, Prometheus metrics, performance tracking.
"""
import os, time, uuid, logging, json
from typing import Optional, Dict, Any
from datetime import datetime
from functools import wraps
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

# ─────────────────────────────────────────────────────────────────────────────
# STRUCTURED LOGGER
# ─────────────────────────────────────────────────────────────────────────────

class StructuredLogger:
    """JSON structured logging with correlation IDs"""

    def __init__(self, name: str, level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(JSONFormatter())
            self.logger.addHandler(handler)
        self._context: Dict[str, Any] = {}

    def bind(self, **kwargs) -> "StructuredLogger":
        """Add context fields to all subsequent logs"""
        new_logger = StructuredLogger.__new__(StructuredLogger)
        new_logger.logger = self.logger
        new_logger._context = {**self._context, **kwargs}
        return new_logger

    def _log(self, level: str, message: str, **kwargs):
        extra = {**self._context, **kwargs}
        getattr(self.logger, level)(message, extra={"structured": extra})

    def info(self, msg: str, **kwargs): self._log("info", msg, **kwargs)
    def warning(self, msg: str, **kwargs): self._log("warning", msg, **kwargs)
    def error(self, msg: str, **kwargs): self._log("error", msg, **kwargs)
    def debug(self, msg: str, **kwargs): self._log("debug", msg, **kwargs)


class JSONFormatter(logging.Formatter):
    """Format logs as JSON for log aggregators (Datadog, CloudWatch, etc.)"""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": "veda_api",
            "version": "2.0.0",
        }
        structured = getattr(record, "structured", {})
        if structured:
            log_data.update(structured)
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_data)


# Global logger
log = StructuredLogger("veda", level=os.getenv("LOG_LEVEL", "INFO"))


# ─────────────────────────────────────────────────────────────────────────────
# CORRELATION ID CONTEXT
# ─────────────────────────────────────────────────────────────────────────────

from contextvars import ContextVar

correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="")

def get_correlation_id() -> str:
    return correlation_id_var.get() or str(uuid.uuid4())[:8]

def set_correlation_id(cid: str):
    correlation_id_var.set(cid)


# ─────────────────────────────────────────────────────────────────────────────
# PROMETHEUS METRICS
# ─────────────────────────────────────────────────────────────────────────────

class MetricsCollector:
    """Lightweight metrics without Prometheus dependency"""

    def __init__(self):
        self._counters: Dict[str, int] = {}
        self._histograms: Dict[str, list] = {}
        self._gauges: Dict[str, float] = {}
        self._started_at = datetime.utcnow().isoformat()

    def counter_inc(self, name: str, labels: Dict = None, value: int = 1):
        key = self._make_key(name, labels)
        self._counters[key] = self._counters.get(key, 0) + value

    def histogram_observe(self, name: str, value: float, labels: Dict = None):
        key = self._make_key(name, labels)
        if key not in self._histograms:
            self._histograms[key] = []
        self._histograms[key].append(value)
        if len(self._histograms[key]) > 10000:
            self._histograms[key] = self._histograms[key][-1000:]

    def gauge_set(self, name: str, value: float, labels: Dict = None):
        key = self._make_key(name, labels)
        self._gauges[key] = value

    def gauge_inc(self, name: str, labels: Dict = None):
        key = self._make_key(name, labels)
        self._gauges[key] = self._gauges.get(key, 0) + 1

    def gauge_dec(self, name: str, labels: Dict = None):
        key = self._make_key(name, labels)
        self._gauges[key] = max(0, self._gauges.get(key, 0) - 1)

    def _make_key(self, name: str, labels: Dict = None) -> str:
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def get_histogram_stats(self, name: str, labels: Dict = None) -> Dict:
        import numpy as np
        key = self._make_key(name, labels)
        values = self._histograms.get(key, [])
        if not values:
            return {}
        arr = np.array(values)
        return {
            "count": len(values),
            "mean": round(float(np.mean(arr)), 4),
            "p50": round(float(np.percentile(arr, 50)), 4),
            "p95": round(float(np.percentile(arr, 95)), 4),
            "p99": round(float(np.percentile(arr, 99)), 4),
            "min": round(float(np.min(arr)), 4),
            "max": round(float(np.max(arr)), 4),
        }

    def get_all_metrics(self) -> Dict:
        metrics = {
            "service": "veda_api",
            "started_at": self._started_at,
            "collected_at": datetime.utcnow().isoformat(),
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "histograms": {}
        }
        for key in self._histograms:
            name = key.split("{")[0]
            metrics["histograms"][key] = self.get_histogram_stats(name)
        return metrics

    def to_prometheus_format(self) -> str:
        """Export in Prometheus text format"""
        lines = []
        lines.append(f"# VEDA Metrics - {datetime.utcnow().isoformat()}")
        for key, value in self._counters.items():
            lines.append(f"veda_{key}_total {value}")
        for key, value in self._gauges.items():
            lines.append(f"veda_{key} {value}")
        for key in self._histograms:
            stats = self.get_histogram_stats(key.split("{")[0])
            if stats:
                lines.append(f"veda_{key}_mean {stats['mean']}")
                lines.append(f"veda_{key}_p95 {stats['p95']}")
                lines.append(f"veda_{key}_p99 {stats['p99']}")
        return "\n".join(lines)


# Global metrics
metrics = MetricsCollector()


# ─────────────────────────────────────────────────────────────────────────────
# OBSERVABILITY MIDDLEWARE
# ─────────────────────────────────────────────────────────────────────────────

class ObservabilityMiddleware(BaseHTTPMiddleware):
    """
    Adds to every request:
    - Correlation ID (X-Correlation-ID header)
    - Request/response logging
    - Latency metrics
    - Error tracking
    """

    SKIP_PATHS = {"/health", "/metrics", "/docs", "/openapi.json", "/redoc"}

    async def dispatch(self, request: Request, call_next) -> Response:
        start_time = time.time()

        # Generate or extract correlation ID
        cid = request.headers.get("X-Correlation-ID") or str(uuid.uuid4())[:12]
        set_correlation_id(cid)

        # Extract request info
        path = request.url.path
        method = request.method
        client_ip = request.headers.get("X-Forwarded-For", "").split(",")[0].strip() or \
                    (request.client.host if request.client else "unknown")

        # Track active requests
        metrics.gauge_inc("active_requests")
        metrics.counter_inc("requests_total", {"method": method, "path": path[:50]})

        # Log request (skip noisy health checks)
        if path not in self.SKIP_PATHS:
            log.info(
                f"Request started",
                correlation_id=cid,
                method=method,
                path=path,
                client_ip=client_ip
            )

        # Process request
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code

        except Exception as e:
            duration_ms = round((time.time() - start_time) * 1000, 2)
            log.error(
                f"Request failed with exception",
                correlation_id=cid,
                method=method,
                path=path,
                error=str(e),
                duration_ms=duration_ms
            )
            metrics.counter_inc("errors_total", {"path": path[:50], "type": type(e).__name__})
            metrics.gauge_dec("active_requests")
            raise

        finally:
            metrics.gauge_dec("active_requests")

        # Calculate duration
        duration_ms = round((time.time() - start_time) * 1000, 2)

        # Record metrics
        metrics.histogram_observe("request_duration_ms", duration_ms, {"path": path[:50]})
        metrics.counter_inc("responses_total", {"status": str(status_code), "path": path[:50]})

        if status_code >= 400:
            metrics.counter_inc("errors_total", {"status": str(status_code), "path": path[:50]})

        if duration_ms > 5000:
            log.warning(
                f"Slow request detected",
                correlation_id=cid,
                path=path,
                duration_ms=duration_ms
            )

        # Log response (skip health checks)
        if path not in self.SKIP_PATHS:
            log.info(
                f"Request completed",
                correlation_id=cid,
                method=method,
                path=path,
                status_code=status_code,
                duration_ms=duration_ms
            )

        # Add headers
        response.headers["X-Correlation-ID"] = cid
        response.headers["X-Response-Time"] = f"{duration_ms}ms"

        return response


# ─────────────────────────────────────────────────────────────────────────────
# PERFORMANCE DECORATOR
# ─────────────────────────────────────────────────────────────────────────────

def track_performance(operation_name: str):
    """Decorator to track function performance"""
    def decorator(func):
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.time()
            cid = get_correlation_id()
            try:
                result = func(*args, **kwargs)
                duration_ms = round((time.time() - start) * 1000, 2)
                metrics.histogram_observe(f"operation_duration_ms", duration_ms, {"op": operation_name})
                log.debug(f"Operation completed", operation=operation_name, duration_ms=duration_ms, correlation_id=cid)
                return result
            except Exception as e:
                duration_ms = round((time.time() - start) * 1000, 2)
                metrics.counter_inc("operation_errors_total", {"op": operation_name})
                log.error(f"Operation failed", operation=operation_name, duration_ms=duration_ms, error=str(e), correlation_id=cid)
                raise

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.time()
            cid = get_correlation_id()
            try:
                result = await func(*args, **kwargs)
                duration_ms = round((time.time() - start) * 1000, 2)
                metrics.histogram_observe(f"operation_duration_ms", duration_ms, {"op": operation_name})
                return result
            except Exception as e:
                duration_ms = round((time.time() - start) * 1000, 2)
                metrics.counter_inc("operation_errors_total", {"op": operation_name})
                log.error(f"Operation failed", operation=operation_name, error=str(e), correlation_id=cid)
                raise

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    return decorator


# ─────────────────────────────────────────────────────────────────────────────
# WORKFLOW LOGGER
# ─────────────────────────────────────────────────────────────────────────────

class WorkflowLogger:
    """Structured logging for ML workflow execution"""

    def __init__(self, job_id: str, goal: str):
        self.job_id = job_id
        self.goal = goal
        self.steps: list = []
        self.start_time = time.time()
        self.logger = log.bind(job_id=job_id)

    def step_start(self, step_name: str, step_num: int):
        self.logger.info(f"Step started", step=step_name, step_num=step_num)
        self.steps.append({"step": step_name, "num": step_num, "started_at": time.time(), "status": "running"})
        metrics.counter_inc("pipeline_steps_total", {"step": step_name})

    def step_complete(self, step_name: str, step_num: int, result_summary: Dict = None):
        duration = time.time() - (self.steps[-1]["started_at"] if self.steps else self.start_time)
        self.logger.info(f"Step completed", step=step_name, step_num=step_num, duration_ms=round(duration*1000, 2))
        if self.steps:
            self.steps[-1].update({"status": "completed", "duration_seconds": round(duration, 3)})
        metrics.histogram_observe("step_duration_ms", duration*1000, {"step": step_name})

    def step_failed(self, step_name: str, error: str):
        self.logger.error(f"Step failed", step=step_name, error=error)
        if self.steps:
            self.steps[-1].update({"status": "failed", "error": error})
        metrics.counter_inc("pipeline_step_errors_total", {"step": step_name})

    def workflow_complete(self, metrics_dict: Dict = None):
        total_duration = round(time.time() - self.start_time, 2)
        self.logger.info(
            f"Workflow completed",
            total_duration_seconds=total_duration,
            steps_completed=len([s for s in self.steps if s.get("status") == "completed"]),
            metrics=metrics_dict or {}
        )
        metrics.counter_inc("workflows_completed_total")
        metrics.histogram_observe("workflow_duration_seconds", total_duration)
        metrics.gauge_inc("total_workflows_processed")

    def workflow_failed(self, error: str):
        total_duration = round(time.time() - self.start_time, 2)
        self.logger.error(
            f"Workflow failed",
            total_duration_seconds=total_duration,
            error=error,
            steps_attempted=len(self.steps)
        )
        metrics.counter_inc("workflows_failed_total")

    def get_summary(self) -> Dict:
        return {
            "job_id": self.job_id,
            "total_duration_seconds": round(time.time() - self.start_time, 2),
            "steps": self.steps
        }


def get_observability_status() -> Dict:
    """Get full observability status"""
    return {
        "metrics": metrics.get_all_metrics(),
        "logging": {"format": "json", "level": os.getenv("LOG_LEVEL", "INFO")},
        "correlation_ids": "enabled",
        "performance_tracking": "enabled"
    }
