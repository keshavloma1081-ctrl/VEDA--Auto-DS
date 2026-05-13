# 🚀 VEDA Deployment Guide

Deploy VEDA to production on Railway, AWS, or Docker.

## Prerequisites

- Git repository
- GROQ API Key
- Platform account (Railway/AWS/etc.)

---

## Option 1: Railway Deployment (Recommended)

Easiest deployment - 5 minutes to production!

### Step 1: Prepare Files

Ensure these files exist:

**Procfile:**

web: uvicorn veda.api.server:app --host 0.0.0.0 --port $PORT

**railway.json:**
```json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "uvicorn veda.api.server:app --host 0.0.0.0 --port $PORT",
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}
```

### Step 2: Deploy to Railway

1. **Sign up**: https://railway.app
2. **New Project** → **Deploy from GitHub repo**
3. **Select** `VEDA--Auto-DS`
4. **Add Variables**:

GROQ_API_KEY=your_key_here
PORT=8000

5. **Deploy** (automatic)

### Step 3: Get Your URL

Railway provides: `https://veda-production.up.railway.app`

### Step 4: Test

```bash
curl https://your-app.railway.app/health
```

---

## Option 2: Docker Deployment

### Build Image

```bash
docker build -t veda:latest -f docker/Dockerfile .
```

### Run Container

```bash
docker run -d \
  -p 8000:8000 \
  -e GROQ_API_KEY=your_key \
  --name veda-api \
  veda:latest
```

### Docker Compose

```yaml
version: '3.8'

services:
  api:
    build:
      context: .
      dockerfile: docker/Dockerfile
    ports:
      - "8000:8000"
    environment:
      - GROQ_API_KEY=${GROQ_API_KEY}
      - DATABASE_URL=postgresql://user:pass@db:5432/veda
    depends_on:
      - db
  
  db:
    image: postgres:15
    environment:
      - POSTGRES_PASSWORD=vedapass
      - POSTGRES_DB=veda
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

Run:
```bash
docker-compose up -d
```

---

## Option 3: AWS EC2

### Launch EC2 Instance

1. **AMI**: Ubuntu 22.04 LTS
2. **Instance Type**: t3.medium (recommended)
3. **Security Group**: Allow ports 22, 80, 8000

### Install Dependencies

```bash
# SSH into instance
ssh ubuntu@your-ec2-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python
sudo apt install python3.11 python3-pip -y

# Clone repo
git clone https://github.com/YOUR_USERNAME/VEDA--Auto-DS.git
cd VEDA--Auto-DS

# Install dependencies
pip3 install -r requirements.txt
pip3 install -e .
```

### Configure Environment

```bash
echo "GROQ_API_KEY=your_key" > .env
```

### Run with Systemd

Create `/etc/systemd/system/veda.service`:

```ini
[Unit]
Description=VEDA ML API
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/VEDA--Auto-DS
Environment="PATH=/home/ubuntu/.local/bin"
ExecStart=/usr/bin/python3 -m uvicorn veda.api.server:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable veda
sudo systemctl start veda
sudo systemctl status veda
```

### Setup Nginx (Optional)

```bash
sudo apt install nginx -y
```

Create `/etc/nginx/sites-available/veda`:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

Enable:
```bash
sudo ln -s /etc/nginx/sites-available/veda /etc/nginx/sites-enabled/
sudo systemctl restart nginx
```

---

## Environment Variables

Required: GROQ_API_KEY=your_groq_api_key

DATABASE_URL=sqlite:///./veda.db
API_PORT=8000
API_WORKERS=4
LOG_LEVEL=INFO
MAX_UPLOAD_SIZE_MB=100

---

## Monitoring

### Health Checks

```bash
# API health
curl https://your-domain.com/health

# System stats
curl https://your-domain.com/stats
```

### Logs

**Docker:**
```bash
docker logs -f veda-api
```

**Systemd:**
```bash
sudo journalctl -u veda -f
```

**Railway:**
- View in Railway dashboard

---

## Security Checklist

- [ ] Change default passwords
- [ ] Use HTTPS (SSL certificate)
- [ ] Set strong JWT SECRET_KEY
- [ ] Enable rate limiting
- [ ] Configure CORS properly
- [ ] Use environment variables for secrets
- [ ] Enable firewall (UFW/Security Groups)
- [ ] Regular security updates
- [ ] Backup database regularly

---

## Scaling

### Horizontal Scaling

Add more workers:
```bash
uvicorn veda.api.server:app --workers 4
```

### Database Scaling

Switch to PostgreSQL for production:

```python
# .env
DATABASE_URL=postgresql://user:pass@host:5432/veda
```

### Caching

Add Redis for caching:
```bash
pip install redis
```

---

## Troubleshooting

### Port binding failed
- Check if port 8000 is available
- Use different port: `--port 8001`

### Database connection error
- Verify DATABASE_URL
- Check database service is running

### Out of memory
- Increase instance size
- Reduce API_WORKERS
- Enable swap

### Slow response
- Add more workers
- Scale horizontally
- Optimize database queries

---

## Production Checklist

- [ ] All tests passing
- [ ] Environment variables set
- [ ] Database backed up
- [ ] SSL certificate installed
- [ ] Monitoring configured
- [ ] Error tracking enabled
- [ ] Documentation updated
- [ ] Load testing completed
- [ ] Security audit passed