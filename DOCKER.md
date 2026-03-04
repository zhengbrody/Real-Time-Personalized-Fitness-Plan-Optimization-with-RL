# Docker Deployment Guide

This guide explains how to run ProFit AI using Docker and Docker Compose.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Service Architecture](#service-architecture)
- [Configuration](#configuration)
- [Common Tasks](#common-tasks)
- [Troubleshooting](#troubleshooting)
- [Production Considerations](#production-considerations)

## Prerequisites

### Required Software

- **Docker**: Version 20.10 or higher
- **Docker Compose**: Version 2.0 or higher

### Installation

#### macOS
```bash
# Install Docker Desktop (includes Docker Compose)
brew install --cask docker
```

#### Linux
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

#### Windows
Download and install [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop).

### Verify Installation

```bash
docker --version
docker-compose --version
```

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/your-username/RL.git
cd RL
```

### 2. Configure Environment

```bash
# Create .env file from example
cp .env.example .env

# Edit .env with your API key
nano .env
```

Add your OpenAI API key:
```
OPENAI_API_KEY=sk-your-key-here
AGENT_MODEL=gpt-4
DATABASE_URL=sqlite:///data/fitness.db
```

### 3. Start Services

```bash
# Start all services in detached mode
docker-compose up -d

# View logs
docker-compose logs -f
```

### 4. Access Application

- **Web Interface**: http://localhost:8501
- **API Server**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Kafka**: localhost:9092
- **Redis**: localhost:6379

### 5. Stop Services

```bash
# Stop all services
docker-compose down

# Stop and remove volumes (clears all data)
docker-compose down -v
```

## Service Architecture

### Services Included

| Service | Container Name | Port | Purpose |
|---------|---------------|------|---------|
| `web` | profit-web | 8501 | Streamlit web interface |
| `api` | profit-api | 8000 | FastAPI backend server |
| `kafka` | profit-kafka | 9092 | Event streaming for online learning |
| `zookeeper` | profit-zookeeper | 2181 | Kafka coordination |
| `redis` | profit-redis | 6379 | Feature caching |
| `kafka-consumer` | profit-kafka-consumer | - | Online learning worker |

### Network

All services run on the `profit-network` bridge network, allowing inter-service communication.

### Volumes

- `redis-data`: Persistent Redis data
- `./data`: Mounted to `/app/data` for SQLite database
- `./models`: Mounted to `/app/models` for ML models

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# OpenAI Configuration
OPENAI_API_KEY=sk-your-key-here
AGENT_MODEL=gpt-4

# Database
DATABASE_URL=sqlite:///data/fitness.db

# Kafka (used by services)
KAFKA_BOOTSTRAP_SERVERS=kafka:29092

# Redis (used by services)
REDIS_HOST=redis
REDIS_PORT=6379
```

### Customizing Services

Edit `docker-compose.yml` to customize:

**Change Ports:**
```yaml
services:
  web:
    ports:
      - "8080:8501"  # Map host port 8080 to container 8501
```

**Add Environment Variables:**
```yaml
services:
  api:
    environment:
      - MY_CUSTOM_VAR=value
```

**Resource Limits:**
```yaml
services:
  api:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
```

## Common Tasks

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f web
docker-compose logs -f api

# Last 100 lines
docker-compose logs --tail=100 web
```

### Restart Services

```bash
# Restart all
docker-compose restart

# Restart specific service
docker-compose restart web
```

### Rebuild After Code Changes

```bash
# Rebuild and restart
docker-compose up -d --build

# Rebuild specific service
docker-compose build api
docker-compose up -d api
```

### Execute Commands in Container

```bash
# Open shell in API container
docker-compose exec api /bin/bash

# Run Python script
docker-compose exec api python src/recommendation/train.py

# Run pytest
docker-compose exec api pytest tests/
```

### Check Service Health

```bash
# Check status
docker-compose ps

# Detailed inspect
docker inspect profit-api

# Check health
curl http://localhost:8000/health
```

### Database Access

```bash
# Access SQLite database
docker-compose exec api sqlite3 /app/data/fitness.db

# Export data
docker-compose exec api sqlite3 /app/data/fitness.db .dump > backup.sql

# Import data
cat backup.sql | docker-compose exec -T api sqlite3 /app/data/fitness.db
```

### Scale Services

```bash
# Run multiple API instances
docker-compose up -d --scale api=3
```

## Troubleshooting

### Port Already in Use

**Error:**
```
ERROR: for profit-web  Cannot start service web: driver failed programming external connectivity on endpoint profit-web: Bind for 0.0.0.0:8501 failed: port is already allocated
```

**Solution:**
```bash
# Find process using port
lsof -i :8501

# Kill process
kill -9 <PID>

# Or change port in docker-compose.yml
```

### Service Won't Start

**Check logs:**
```bash
docker-compose logs api
```

**Common issues:**
- Missing `.env` file → Create it with required variables
- Invalid API key → Check `OPENAI_API_KEY` in `.env`
- Kafka not ready → Wait for Kafka to start (takes 30-60 seconds)

### Data Loss After Restart

**Cause:** Using `docker-compose down -v` removes volumes.

**Solution:**
```bash
# Stop without removing volumes
docker-compose down

# Or backup data first
cp -r data/ data_backup/
```

### Kafka Connection Errors

**Wait for Kafka to be ready:**
```bash
# Check Kafka health
docker-compose ps kafka

# View Kafka logs
docker-compose logs kafka

# Restart dependent services after Kafka is healthy
docker-compose restart api kafka-consumer
```

### Out of Memory

**Increase Docker memory limit:**
- Docker Desktop → Settings → Resources → Memory
- Increase to 4GB+ for full stack

### Permission Issues

**Fix file permissions:**
```bash
# Ensure data directories are writable
chmod -R 755 data/
chmod -R 755 models/

# Or run as root (development only)
docker-compose exec -u root api chown -R profit:profit /app/data
```

## Production Considerations

### Security

1. **Don't commit `.env` file**
   ```bash
   # Add to .gitignore
   echo ".env" >> .gitignore
   ```

2. **Use secrets management**
   - Docker Secrets
   - AWS Secrets Manager
   - HashiCorp Vault

3. **Non-root user**
   - Production Dockerfile uses `profit` user
   - Never run containers as root in production

4. **Network isolation**
   - Use separate networks for different service tiers
   - Limit exposed ports

### Performance

1. **Use production image**
   ```yaml
   services:
     api:
       build:
         target: production
   ```

2. **Enable caching**
   - Redis for feature store
   - CDN for static assets

3. **Resource limits**
   ```yaml
   deploy:
     resources:
       limits:
         cpus: '2'
         memory: 4G
       reservations:
         cpus: '1'
         memory: 2G
   ```

### High Availability

1. **Scale services**
   ```bash
   docker-compose up -d --scale api=3 --scale web=2
   ```

2. **Load balancing**
   - Use nginx or Traefik as reverse proxy
   - Health checks for automatic failover

3. **Database**
   - Switch from SQLite to PostgreSQL
   - Use managed database service (RDS, Cloud SQL)

### Monitoring

1. **Add logging**
   ```yaml
   services:
     api:
       logging:
         driver: "json-file"
         options:
           max-size: "10m"
           max-file: "3"
   ```

2. **Health checks**
   - Already configured in Dockerfile
   - Monitor with Prometheus/Grafana

3. **Alerts**
   - Set up alerts for service failures
   - Monitor resource usage

### Backup

```bash
# Backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)

# Backup database
docker-compose exec -T api sqlite3 /app/data/fitness.db .dump > backup_$DATE.sql

# Backup volumes
docker run --rm -v profit-redis-data:/data -v $(pwd):/backup alpine tar czf /backup/redis_$DATE.tar.gz /data

# Upload to S3
aws s3 cp backup_$DATE.sql s3://my-bucket/backups/
```

## Next Steps

- Read [QUICK_START.md](QUICK_START.md) for usage guide
- See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup
- Check [README.md](README.md) for architecture details

## Need Help?

- GitHub Issues: Report bugs or request features
- Documentation: Check existing docs
- Community: Join discussions
