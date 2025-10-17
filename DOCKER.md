# Docker Setup for Slide Review Agent

This document provides instructions for running the Slide Review Agent using Docker.

## Prerequisites

- Docker installed (version 20.10 or higher)
- Docker Compose installed (version 2.0 or higher)
- LLM API key (HuggingFace, Groq, or other supported provider)

## Quick Start

### 1. Set up environment variables

Create a `.env` file in the project root with your LLM configuration:

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your LLM credentials
# Required variables:
LLM_PROVIDER=huggingface
LLM_API_KEY=your_api_key_here
LLM_MODEL=google/gemma-2-2b-it
LLM_API_ENDPOINT=https://router.huggingface.co/v1/chat/completions
```

### 2. Build and run with Docker Compose

```bash
# Build and start the container
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the container
docker-compose down
```

### 3. Access the application

- Frontend UI: http://localhost:8000/app
- API root: http://localhost:8000
- Health check: http://localhost:8000/health

## Building the Docker Image

### Using Docker Compose (Recommended)

```bash
docker-compose build
docker-compose up -d
```

### Using Docker directly

```bash
# Build the image
docker build -t slide-review-agent:latest .

# Run the container
docker run -d \
  --name slide-review-agent \
  -p 8000:8000 \
  -e LLM_PROVIDER=huggingface \
  -e LLM_API_KEY=your_api_key_here \
  -e LLM_MODEL=google/gemma-2-2b-it \
  -e LLM_API_ENDPOINT=https://router.huggingface.co/v1/chat/completions \
  -v $(pwd)/data/uploads:/app/data/uploads \
  -v $(pwd)/data/outputs:/app/data/outputs \
  -v $(pwd)/data/logs:/app/data/logs \
  slide-review-agent:latest
```

## Environment Variables

### Required Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `LLM_PROVIDER` | LLM provider name | `huggingface`, `groq` |
| `LLM_API_KEY` | API key for LLM provider | `hf_xxxxx...` |
| `LLM_MODEL` | Model identifier | `google/gemma-2-2b-it` |
| `LLM_API_ENDPOINT` | API endpoint URL | `https://router.huggingface.co/v1/chat/completions` |

### Optional Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_NAME` | `slide-review-agent` | Application name |
| `ENVIRONMENT` | `production` | Environment setting |
| `DEBUG` | `false` | Enable debug mode |
| `MAX_FILE_SIZE_MB` | `50` | Maximum upload file size in MB |
| `ENABLE_REWRITE` | `true` | Enable LLM-based rewrite suggestions |
| `REWRITE_BUDGET_MAX_CALLS` | `5` | Max LLM calls per window |
| `REWRITE_BUDGET_WINDOW_HOURS` | `24` | Time window for rate limiting |
| `VADER_NEG_THRESHOLD` | `-0.05` | Negativity threshold for sentiment |
| `TEMPERATURE` | `0.3` | LLM temperature setting |

## Data Persistence

The application uses three main data directories that are mounted as volumes:

- `./data/uploads` - Uploaded presentation files
- `./data/outputs` - Analysis results and normalized documents
- `./data/logs` - Application logs

These directories are automatically created and persisted across container restarts.

## Health Check

The container includes a built-in health check that runs every 30 seconds:

```bash
# Check container health status
docker ps

# View health check logs
docker inspect --format='{{json .State.Health}}' slide-review-agent
```

## Resource Limits

The default Docker Compose configuration sets these resource limits:

- **CPU Limit**: 2 cores
- **Memory Limit**: 4GB
- **CPU Reservation**: 1 core
- **Memory Reservation**: 2GB

Adjust these in `docker-compose.yml` based on your system resources and workload.

## Troubleshooting

### Container fails to start

Check the logs for errors:
```bash
docker-compose logs -f
```

### LLM connection issues

Verify your LLM configuration:
```bash
# Check environment variables
docker exec slide-review-agent env | grep LLM

# Test LLM health
docker exec slide-review-agent python -m backend.services.llm_health
```

### Permission issues with volumes

Ensure the data directories have proper permissions:
```bash
mkdir -p data/uploads data/outputs data/logs
chmod -R 755 data/
```

### Memory issues

If the container runs out of memory, increase the memory limit in `docker-compose.yml`:
```yaml
deploy:
  resources:
    limits:
      memory: 8G  # Increase from 4G
```

## Development Setup

For development with live code reloading:

```bash
# Use a bind mount for the code directory
docker run -d \
  --name slide-review-agent-dev \
  -p 8000:8000 \
  -e DEBUG=true \
  -e LLM_PROVIDER=huggingface \
  -e LLM_API_KEY=your_api_key_here \
  -v $(pwd):/app \
  -v $(pwd)/data/uploads:/app/data/uploads \
  -v $(pwd)/data/outputs:/app/data/outputs \
  -v $(pwd)/data/logs:/app/data/logs \
  slide-review-agent:latest \
  uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

## Stopping and Cleaning Up

```bash
# Stop the container
docker-compose down

# Stop and remove volumes (WARNING: This deletes all data)
docker-compose down -v

# Remove the image
docker rmi slide-review-agent:latest

# Clean up unused Docker resources
docker system prune -a
```

## Security Considerations

1. **API Keys**: Never commit `.env` files with real API keys to version control
2. **Network**: In production, configure proper CORS settings and restrict access
3. **Volumes**: Ensure proper file permissions on mounted volumes
4. **Updates**: Regularly update the base image and dependencies for security patches

## Production Deployment

For production deployment, consider:

1. Using a reverse proxy (nginx, traefik) with SSL/TLS
2. Setting up proper logging and monitoring
3. Implementing backup strategies for data volumes
4. Using Docker secrets for sensitive environment variables
5. Setting up container orchestration (Kubernetes, Docker Swarm)

Example with nginx reverse proxy:

```yaml
services:
  slide-review-agent:
    # ... existing config ...

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./certs:/etc/nginx/certs:ro
    depends_on:
      - slide-review-agent
```
