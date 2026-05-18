# Deployment Guide — AI Agent Playground

## Quick Deploy (3 minutes)

### Prerequisites
- Linux server (Ubuntu 22.04 recommended, 2C2G minimum)
- Domain name (optional)
- Docker + Docker Compose installed

### Step 1: Clone and configure

```bash
git clone https://github.com/aidless/ai-agent-playground.git
cd ai-agent-playground
cp .env.example .env
```

Edit `.env` — these are MANDATORY:

```bash
# Required
DEEPSEEK_API_KEY=sk-your-deepseek-key-here
GATEWAY_API_KEY=$(openssl rand -hex 32)
IDENTITY_SIGNING_KEY=$(openssl rand -hex 32)
SESSION_SECRET=$(openssl rand -hex 32)

# Production settings
APP_ENV=production
CORS_ORIGINS=https://your-domain.com
```

### Step 2: Start

```bash
./deploy.sh start
```

### Step 3: Verify

```bash
curl -H "Authorization: Bearer $GATEWAY_API_KEY" http://localhost:8000/health
curl http://localhost:8000/autopilot/status
```

---

## Manual Deployment

### Option A: Docker Compose (Recommended)

```bash
# Agent only (no Ollama — uses DeepSeek API exclusively)
docker-compose up -d

# Full stack (agent + Ollama + ChromaDB)
docker-compose --profile full up -d

# View logs
docker-compose logs -f agent

# Stop
docker-compose down
```

### Option B: Bare Metal

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Start server
APP_ENV=production uv run uvicorn agent.server:app \
  --host 0.0.0.0 --port 8000 --workers 4
```

### Option C: Nginx Reverse Proxy

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout 300s;
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/agent /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx

# Add HTTPS
sudo certbot --nginx -d your-domain.com
```

---

## Health Monitoring

### Uptime check (cron every minute)
```bash
*/1 * * * * curl -sf http://localhost:8000/health || \
  echo "Agent down at $(date)" >> /var/log/agent-alerts.log
```

### Resource limits (docker-compose.yml already configured)
```yaml
deploy:
  resources:
    limits:
      cpus: '2'
      memory: 2G
```

### Auto-restart
Already configured in docker-compose.yml: `restart: unless-stopped`

---

## Maintenance Commands

```bash
# View all subsystem status
curl http://localhost:8000/super/status | jq

# View security intrusion events
curl http://localhost:8000/security/intrusion | jq

# Trigger autonomous self-improvement
curl -X POST http://localhost:8000/autopilot/solve \
  -H "Content-Type: application/json" \
  -d '{"task": "Review system health and suggest improvements"}'

# Run penetration test
uv run python scripts/pentest.py

# Run benchmarks
uv run python scripts/benchmark_engines.py
uv run python scripts/hard_benchmark.py

# View logs
docker-compose logs -f --tail=100 agent
```

---

## Security Checklist

- [ ] `GATEWAY_API_KEY` set to strong random value
- [ ] `IDENTITY_SIGNING_KEY` set to strong random value
- [ ] `APP_ENV=production`
- [ ] `CORS_ORIGINS` restricted to your domain (not `*`)
- [ ] `.env` file permissions: `chmod 600 .env`
- [ ] Firewall: only ports 80/443 open, 8000 internal only
- [ ] HTTPS enabled via certbot
- [ ] Regular `docker-compose pull` for updates
- [ ] Audit logs rotated (auto, 90-day retention)
- [ ] Run `uv run python scripts/pentest.py` monthly

---

## Scaling

```bash
# Horizontal scaling (Docker Swarm)
docker swarm init
docker stack deploy -c docker-compose.yml agent-stack

# Vertical scaling
# docker-compose.yml → increase CPU/memory limits
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| 502 Bad Gateway | Agent container crashed — `docker-compose restart agent` |
| High latency | Check DeepSeek API status at https://status.deepseek.com |
| Quota exceeded | Increase budget in `.env`: `BUDGET_DAILY=10.0` |
| Ollama not found | Start with profile: `docker-compose --profile full up -d` |
| Port already in use | `sudo lsof -i :8000` and kill the process |
