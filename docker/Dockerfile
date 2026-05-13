# VEDA Production Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Install VEDA
RUN pip install -e .

# Create directories
RUN mkdir -p data outputs logs models

EXPOSE 8000

ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Run application
CMD uvicorn veda.api.server:app --host 0.0.0.0 --port $PORT
