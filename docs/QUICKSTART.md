# 🚀 VEDA Quick Start Guide

Get VEDA running in 5 minutes!

## Prerequisites

- Python 3.10+
- GROQ API Key ([Get one free](https://console.groq.com))
- 4GB RAM minimum

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/VEDA--Auto-DS.git
cd VEDA--Auto-DS
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### 3. Configure Environment

```bash
# Create .env file
echo "GROQ_API_KEY=your_groq_api_key_here" > .env
```

### 4. Initialize Database

```bash
python -c "from veda.database.models import init_db; init_db()"
```

## Running VEDA

### Start API Server

```bash
# Terminal 1: Start backend
uvicorn veda.api.server:app --host 0.0.0.0 --port 8000 --reload
```

### Start Dashboard (Optional)

```bash
# Terminal 2: Start frontend
streamlit run veda/ui/dashboard.py
```

## Your First Workflow

### Option 1: Using Dashboard UI

1. Open http://localhost:8501
2. Login with `admin` / `admin123`
3. Upload CSV file
4. Enter ML goal (e.g., "predict customer churn")
5. Click "Start Workflow"
6. Monitor progress in Job Monitor tab

### Option 2: Using API

```bash
# Login
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin123"}'

# Copy the access_token

# Create workflow
curl -X POST http://localhost:8000/workflows \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_path": "test_data/sample.csv",
    "goal": "predict customer churn"
  }'

# Check status
curl http://localhost:8000/workflows/JOB_ID
```

### Option 3: Using Python SDK

```python
import requests

# Login
response = requests.post(
    "http://localhost:8000/auth/login",
    json={"username": "admin", "password": "admin123"}
)
token = response.json()["access_token"]

# Create workflow
response = requests.post(
    "http://localhost:8000/workflows",
    headers={"Authorization": f"Bearer {token}"},
    json={
        "dataset_path": "data/my_dataset.csv",
        "goal": "predict sales for next quarter"
    }
)

job_id = response.json()["job_id"]
print(f"Workflow started: {job_id}")
```

## What Happens Next?

VEDA will automatically:

1. ✅ **Analyze** your dataset
2. ✅ **Clean** and preprocess data
3. ✅ **Engineer** relevant features
4. ✅ **Train** multiple ML models
5. ✅ **Evaluate** and select best model
6. ✅ **Generate** performance report

Typical completion time: 2-10 minutes depending on dataset size.

## Next Steps

- 📖 Read [API Reference](API_REFERENCE.md)
- 🚀 Deploy to production: [Deployment Guide](DEPLOYMENT.md)
- 🏗️ Understand architecture: [Architecture Overview](ARCHITECTURE.md)
- 🐛 Report issues: [GitHub Issues](https://github.com/YOUR_USERNAME/VEDA--Auto-DS/issues)

## Troubleshooting

### "GROQ_API_KEY not set"
- Make sure `.env` file exists in project root
- Verify API key is valid at https://console.groq.com

### "Dataset not found"
- Use absolute path or path relative to project root
- Ensure CSV file is readable

### Port already in use
- Change port: `uvicorn veda.api.server:app --port 8001`
- Or kill process using port 8000

### Database errors
- Delete `veda.db` and reinitialize: `rm veda.db && python -c "from veda.database.models import init_db; init_db()"`

## Need Help?

- 📧 Email: your.email@example.com
- 💬 Discord: [Join our community]
- 📝 Documentation: [Full Docs](../README.md)