"""
Prometheus Metrics for Monitoring
"""
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import time

# Metrics
workflow_counter = Counter(
    'veda_workflows_total',
    'Total number of workflows created',
    ['status']
)

workflow_duration = Histogram(
    'veda_workflow_duration_seconds',
    'Workflow execution time in seconds'
)

api_requests = Counter(
    'veda_api_requests_total',
    'Total API requests',
    ['endpoint', 'method', 'status']
)

api_latency = Histogram(
    'veda_api_latency_seconds',
    'API request latency',
    ['endpoint']
)

active_workflows = Gauge(
    'veda_active_workflows',
    'Currently running workflows'
)


class MetricsMiddleware:
    """FastAPI middleware to track metrics"""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        
        start_time = time.time()
        
        # Track request
        endpoint = scope["path"]
        method = scope["method"]
        
        # Execute request
        status_code = 200
        
        try:
            await self.app(scope, receive, send)
        except Exception as e:
            status_code = 500
            raise
        finally:
            # Record metrics
            duration = time.time() - start_time
            api_requests.labels(
                endpoint=endpoint,
                method=method,
                status=status_code
            ).inc()
            
            api_latency.labels(endpoint=endpoint).observe(duration)


def get_metrics():
    """Get current metrics in Prometheus format"""
    return generate_latest()


# Usage in workflow execution:
"""
# Start workflow
active_workflows.inc()
workflow_counter.labels(status='started').inc()

start_time = time.time()

try:
    # Execute workflow
    ...
    workflow_counter.labels(status='completed').inc()
finally:
    duration = time.time() - start_time
    workflow_duration.observe(duration)
    active_workflows.dec()
"""