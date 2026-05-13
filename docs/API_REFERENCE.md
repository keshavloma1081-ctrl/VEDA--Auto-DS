# 📡 VEDA API Reference

Complete API documentation for VEDA v2.0

Base URL: `http://localhost:8000`

## Authentication

VEDA uses JWT Bearer token authentication.

### POST /auth/login

Login to get access token.

**Request:**
```json
{
  "username": "admin",
  "password": "admin123"
}
```

**Response:**
```json
{
  "access_token": "eyJhbGc...",
  "token_type": "bearer"
}
```

**Demo Credentials:**
- Admin: `admin` / `admin123`
- User: `user` / `user123`

---

## System Endpoints

### GET /

Get API information.

**Response:**
```json
{
  "service": "VEDA ML API",
  "version": "2.0.0",
  "status": "operational",
  "docs": "/docs",
  "authentication": "JWT (POST /auth/login)",
  "reports": "enabled"
}
```

### GET /health

Health check with component status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2026-05-13T10:00:00",
  "components": {
    "api": "operational",
    "database": "connected",
    "groq_api_key": "configured",
    "auth": "enabled",
    "reports": "enabled"
  }
}
```

### GET /stats

System statistics.

**Response:**
```json
{
  "workflows": {
    "total": 10,
    "completed": 7,
    "failed": 1,
    "running": 2,
    "success_rate": 70.0
  }
}
```

---

## Workflow Endpoints

### POST /workflows

Create new ML workflow.

**Authentication:** Required

**Request:**
```json
{
  "dataset_path": "data/my_data.csv",
  "goal": "predict customer churn based on usage patterns"
}
```

**Response:**
```json
{
  "job_id": "a1b2c3d4-5678-90ef-ghij-klmnopqrstuv",
  "status": "submitted",
  "message": "Workflow submitted"
}
```

**Status Codes:**
- `200`: Success
- `401`: Unauthorized (missing/invalid token)
- `404`: Dataset not found
- `422`: Validation error

### GET /workflows/{job_id}

Get workflow status and results.

**Response:**
```json
{
  "job_id": "a1b2c3d4-5678-90ef-ghij-klmnopqrstuv",
  "status": "completed",
  "progress": 100.0,
  "dataset_path": "data/my_data.csv",
  "goal": "predict customer churn",
  "current_step": "Evaluation Complete",
  "result": {
    "model_type": "classification",
    "best_model": "xgboost",
    "metrics": {
      "accuracy": 0.8750,
      "precision": 0.8621,
      "recall": 0.8432,
      "f1_score": 0.8525
    }
  },
  "error": null,
  "created_at": "2026-05-13T10:00:00",
  "updated_at": "2026-05-13T10:02:15"
}
```

**Status Values:**
- `submitted`: Workflow queued
- `running`: Currently processing
- `completed`: Successfully finished
- `failed`: Error occurred

### GET /workflows

List all workflows.

**Query Parameters:**
- `limit` (optional): Max results (default: 10)

**Response:**
```json
[
  {
    "job_id": "...",
    "status": "completed",
    "goal": "predict sales",
    "created_at": "2026-05-13T10:00:00"
  }
]
```

### GET /workflows/{job_id}/report

Generate HTML report for workflow.

**Response:** HTML file download

**Report includes:**
- Workflow summary
- Performance metrics
- Model information
- Feature importance
- Execution timeline
- Recommendations

---

## Error Responses

All endpoints return errors in this format:

```json
{
  "detail": "Error message here"
}
```

**Common Status Codes:**
- `400`: Bad Request
- `401`: Unauthorized
- `403`: Forbidden
- `404`: Not Found
- `422`: Validation Error
- `500`: Internal Server Error

---

## Rate Limits

- **Login**: 10 requests/minute
- **Workflow Creation**: 5 requests/minute
- **Other endpoints**: 100 requests/minute

---

## Examples

### Complete Workflow (Python)

```python
import requests
import time

BASE_URL = "http://localhost:8000"

# 1. Login
resp = requests.post(f"{BASE_URL}/auth/login", json={
    "username": "admin",
    "password": "admin123"
})
token = resp.json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}

# 2. Create workflow
resp = requests.post(f"{BASE_URL}/workflows", headers=headers, json={
    "dataset_path": "data/customers.csv",
    "goal": "predict customer lifetime value"
})
job_id = resp.json()["job_id"]
print(f"Started: {job_id}")

# 3. Monitor progress
while True:
    resp = requests.get(f"{BASE_URL}/workflows/{job_id}")
    status = resp.json()
    
    print(f"Status: {status['status']} - {status['progress']}%")
    
    if status['status'] in ['completed', 'failed']:
        break
    
    time.sleep(5)

# 4. Get report
report_url = f"{BASE_URL}/workflows/{job_id}/report"
print(f"Report: {report_url}")
```

### Complete Workflow (cURL)

```bash
# Login
TOKEN=$(curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin123"}' \
  | jq -r '.access_token')

# Create workflow
JOB_ID=$(curl -X POST http://localhost:8000/workflows \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"dataset_path":"data/sales.csv","goal":"forecast revenue"}' \
  | jq -r '.job_id')

# Check status
curl http://localhost:8000/workflows/$JOB_ID | jq

# Download report
curl http://localhost:8000/workflows/$JOB_ID/report -o report.html
```

---

## Interactive Documentation

Visit http://localhost:8000/docs for interactive Swagger UI with:
- Live API testing
- Request/response examples
- Schema validation
- Authentication testing