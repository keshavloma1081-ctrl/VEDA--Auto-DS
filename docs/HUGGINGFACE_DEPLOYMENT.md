# 🤗 Hugging Face Spaces Deployment Guide

Deploy VEDA to Hugging Face Spaces for free with GPU support!

## Why Hugging Face Spaces?

- ✅ **FREE** GPU/CPU hosting
- ✅ Auto-deploys from GitHub
- ✅ Built-in authentication
- ✅ Community visibility
- ✅ Perfect for ML demos
- ✅ No credit card required

---

## Prerequisites

- Hugging Face account (sign up at [huggingface.co](https://huggingface.co))
- VEDA repository on GitHub
- GROQ API key

---

## Method 1: Deploy via Hugging Face Web Interface (Easiest)

### Step 1: Create New Space

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Click **"Create new Space"**
3. Configure:
   - **Owner:** Your username
   - **Space name:** `veda-ml-platform`
   - **License:** MIT
   - **Space SDK:** **Docker**
   - **Visibility:** Public (or Private)

### Step 2: Link GitHub Repository

1. After creating Space, go to **Settings**
2. Scroll to **"Repository"**
3. Click **"Link a GitHub repository"**
4. Select `VEDA--Auto-DS` repository
5. Enable **"Auto-sync"** ✅

### Step 3: Add Dockerfile for Hugging Face

Create `Dockerfile.hf` in your repo:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Install VEDA package
RUN pip install -e .

# Create directories
RUN mkdir -p data outputs logs models

# Expose port 7860 (HF Spaces default)
EXPOSE 7860

ENV PYTHONUNBUFFERED=1

# Run both API and Streamlit
CMD uvicorn veda.api.server:app --host 0.0.0.0 --port 7860 & \
    streamlit run veda/ui/dashboard.py --server.port 8501 --server.address 0.0.0.0
```

### Step 4: Add Hugging Face Configuration

Create `README.md` in root (if not exists) with HF metadata:

```markdown
---
title: VEDA - Autonomous ML Platform
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# 🤖 VEDA - Autonomous Machine Learning Platform

Production-ready autonomous ML system powered by multi-agent AI.

## Features

- 🤖 11 specialized AI agents
- 🚀 FastAPI REST API
- 🎨 Streamlit Dashboard
- 📊 Real-time monitoring
- 🔒 JWT Authentication

## Usage

1. Navigate to the Space URL
2. Login with: `admin` / `admin123`
3. Upload your dataset
4. Enter your ML goal
5. Watch the agents work!

## Documentation

See [GitHub Repository](https://github.com/keshavloma1081-ctrl/VEDA--Auto-DS) for full docs.
```

### Step 5: Configure Secrets

In Hugging Face Space settings:

1. Go to **Settings** → **Repository secrets**
2. Add secrets:

GROQ_API_KEY=your_groq_api_key_here
SECRET_KEY=your_32_char_secret_key_here

### Step 6: Push and Deploy

```bash
# Commit HF configuration
git add Dockerfile.hf README.md
git commit -m "🤗 Added Hugging Face Spaces configuration"
git push origin main
```

Hugging Face will automatically:
1. Detect the changes
2. Build the Docker image
3. Deploy your app
4. Give you a URL: `https://huggingface.co/spaces/YOUR_USERNAME/veda-ml-platform`

---

## Method 2: Deploy via Hugging Face CLI

### Install Hugging Face CLI

```bash
pip install huggingface_hub
```

### Login

```bash
huggingface-cli login
# Enter your HF token
```

### Create Space

```bash
# Create new Space
huggingface-cli repo create veda-ml-platform --type space --space_sdk docker

# Clone the Space repository
git clone https://huggingface.co/spaces/YOUR_USERNAME/veda-ml-platform
cd veda-ml-platform

# Copy your VEDA files
cp -r /path/to/VEDA--Auto-DS/* .

# Add HF-specific files
# (Dockerfile.hf and README.md from above)

# Commit and push
git add .
git commit -m "Initial VEDA deployment"
git push
```

---

## Method 3: Streamlit-Only Deployment (Simpler)

For Streamlit-only (no API), use:

**Create `app.py` in root:**

```python
"""
VEDA Streamlit App for Hugging Face Spaces
"""
import sys
import os

# Add veda to path
sys.path.insert(0, os.path.dirname(__file__))

# Run Streamlit dashboard
if __name__ == "__main__":
    import subprocess
    subprocess.run([
        "streamlit", "run", 
        "veda/ui/dashboard.py",
        "--server.port", "7860",
        "--server.address", "0.0.0.0"
    ])
```

**Update README.md:**

```markdown
---
title: VEDA ML Platform
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: false
---
```

**Requirements:**

Create `requirements.txt` with all dependencies.

---

## Configuration Files

### For Full Stack (API + Dashboard)

**Dockerfile.hf:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc g++ git && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install -e .
RUN mkdir -p data outputs logs models

EXPOSE 7860

ENV PYTHONUNBUFFERED=1

CMD uvicorn veda.api.server:app --host 0.0.0.0 --port 7860
```

### For Streamlit Only

**README.md header:**
```yaml
---
title: VEDA
sdk: streamlit
app_file: app.py
---
```

---

## Environment Variables

Set in HF Space Settings → Repository secrets:

```env
GROQ_API_KEY=your_groq_api_key
SECRET_KEY=your_secret_key_32_chars
DATABASE_URL=sqlite:///./veda.db
LOG_LEVEL=INFO
```

---

## Post-Deployment

### 1. Verify Deployment

Visit: `https://huggingface.co/spaces/YOUR_USERNAME/veda-ml-platform`

### 2. Test Features

- ✅ Dashboard loads
- ✅ Login works
- ✅ File upload works
- ✅ Workflow creation works

### 3. Monitor Logs

In HF Space:
- Click **"Logs"** tab
- Real-time application logs
- Error tracking

### 4. Share Your Space

Your Space URL: