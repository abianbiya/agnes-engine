# 🐳 Docker Deployment Guide - RAG Chatbot

Complete guide to run, test, and manage your RAG Chatbot using Docker.

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Configuration](#configuration)
4. [Running with Docker](#running-with-docker)
5. [Testing the Deployment](#testing-the-deployment)
6. [Service Management](#service-management)
7. [Troubleshooting](#troubleshooting)
8. [Production Deployment](#production-deployment)

---

## Prerequisites

### System Requirements

- **Docker:** Version 20.10+ ✅ (You have v28.0.4)
- **Docker Compose:** Version 2.0+ ✅ (You have v2.34.0)
- **RAM:** Minimum 4GB, Recommended 8GB
- **Disk Space:** Minimum 5GB free space
- **macOS:** Version 10.15+ (Catalina or later)

### API Keys Required

- **OpenAI API Key** (required for OpenAI provider)
  - Get from: https://platform.openai.com/api-keys
  - Alternative: Use Ollama for local LLM (no API key needed)

---

## Quick Start

### 1. Create Environment File

```bash
cd /Applications/XAMPP/xamppfiles/htdocs/airesearch/langgraph/rag-chatbot

# Copy example environment file
cp .env.example .env

# Edit with your API key
nano .env  # or use your preferred editor
```

**Minimum required configuration in `.env`:**

```bash
# Replace with your actual OpenAI API key
OPENAI_API_KEY=sk-your-actual-api-key-here

# Basic configuration
LLM_MODEL=gpt-4
EMBEDDING_MODEL=text-embedding-ada-002
ENVIRONMENT=production
```

### 2. Start All Services

```bash
# Make scripts executable
chmod +x scripts/start.sh scripts/stop.sh

# Start services (builds on first run)
./scripts/start.sh --build
```

### 3. Verify Services Are Running

```bash
# Check status
docker compose -f docker/docker-compose.yml ps

# You should see 4 services running:
# - rag-api (port 8080)
# - chromadb (port 8000)
# - mcp-server (port 3000)
# - redis (port 6379)
```

---

## Configuration

### Environment Variables

Edit `.env` file to customize configuration:

#### LLM Configuration

```bash
# OpenAI Configuration (Default)
OPENAI_API_KEY=sk-your-key-here
LLM_PROVIDER=openai
LLM_MODEL=gpt-4
LLM_TEMPERATURE=0.7

# Or use Ollama (Local LLM - No API Key Required)
# LLM_PROVIDER=ollama
# OLLAMA_MODEL=llama2
# OLLAMA_BASE_URL=http://host.docker.internal:11434
```

#### Embedding Configuration

```bash
# OpenAI Embeddings (Default)
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-ada-002

# Or HuggingFace Embeddings (Local - Free)
# EMBEDDING_PROVIDER=huggingface
# HUGGINGFACE_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

#### ChromaDB Configuration

```bash
CHROMA_HOST=chromadb          # Service name in Docker
CHROMA_PORT=8000
CHROMA_COLLECTION=documents
CHROMA_IN_MEMORY=false        # Persist data
```

#### Retrieval Configuration

```bash
RETRIEVAL_K=4                 # Number of documents to retrieve
USE_MMR=true                  # Use Maximum Marginal Relevance
MMR_DIVERSITY=0.5             # 0.0=relevance, 1.0=diversity
USE_RERANKER=false            # Enable reranking (slower but better)
```

#### Logging Configuration

```bash
LOG_LEVEL=INFO                # DEBUG, INFO, WARNING, ERROR
LOG_FORMAT=json               # json or console
DEBUG=false
```

---

## Running with Docker

### Using the Start Script (Recommended)

```bash
# Basic start (uses cached images)
./scripts/start.sh

# Start with fresh build
./scripts/start.sh --build

# Start in foreground (see logs in terminal)
./scripts/start.sh --build --foreground

# Start with latest base images
./scripts/start.sh --build --pull

# Fresh start (removes all data - CAUTION!)
./scripts/start.sh --build --remove-volumes
```

### Using Docker Compose Directly

```bash
cd /Applications/XAMPP/xamppfiles/htdocs/airesearch/langgraph/rag-chatbot

# Build images
docker compose -f docker/docker-compose.yml build

# Start services in background
docker compose -f docker/docker-compose.yml up -d

# Start services in foreground (see logs)
docker compose -f docker/docker-compose.yml up

# Stop services
docker compose -f docker/docker-compose.yml down

# Stop and remove volumes (deletes data)
docker compose -f docker/docker-compose.yml down -v
```

### Service URLs

Once running, access services at:

- **RAG API:** http://localhost:8080
- **API Documentation (Swagger):** http://localhost:8080/docs
- **API Documentation (ReDoc):** http://localhost:8080/redoc
- **Health Check:** http://localhost:8080/api/v1/health
- **ChromaDB:** http://localhost:8000
- **MCP Server:** http://localhost:3000 (if using SSE transport)
- **Redis:** localhost:6379

---

## Testing the Deployment

### 1. Health Check

```bash
# Check API health
curl http://localhost:8080/api/v1/health | jq

# Expected output:
# {
#   "status": "healthy",
#   "timestamp": "2025-12-24T05:20:00Z",
#   "version": "1.0.0",
#   "services": {
#     "vectorstore": true,
#     "llm": true,
#     "embeddings": true
#   }
# }
```

### 2. Test Document Ingestion

```bash
# Ingest a sample document
curl -X POST http://localhost:8080/api/v1/ingest/text \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Python is a high-level programming language. It is widely used for web development, data analysis, and artificial intelligence.",
    "metadata": {
      "source": "test_document",
      "category": "programming"
    }
  }' | jq

# Expected output:
# {
#   "success": true,
#   "message": "Text ingested successfully",
#   "document_ids": ["uuid-here"],
#   "chunks_created": 1
# }
```

### 3. Test Chat Functionality

```bash
# Create a chat session
SESSION_ID=$(curl -X POST http://localhost:8080/api/v1/chat/sessions \
  -H "Content-Type: application/json" | jq -r '.session_id')

echo "Session ID: $SESSION_ID"

# Send a question
curl -X POST "http://localhost:8080/api/v1/chat/$SESSION_ID" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is Python used for?"
  }' | jq

# Expected output:
# {
#   "answer": "Python is widely used for web development, data analysis, and artificial intelligence...",
#   "sources": [...],
#   "session_id": "session-uuid",
#   "metadata": {...}
# }
```

### 4. Test Question Answering (Stateless)

```bash
# Ask a question without session
curl -X POST http://localhost:8080/api/v1/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is Python?"
  }' | jq

# Expected output:
# {
#   "answer": "Python is a high-level programming language...",
#   "sources": [...],
#   "metadata": {...}
# }
```

### 5. View Service Logs

```bash
# View all logs
docker compose -f docker/docker-compose.yml logs

# View specific service logs
docker compose -f docker/docker-compose.yml logs rag-api
docker compose -f docker/docker-compose.yml logs chromadb
docker compose -f docker/docker-compose.yml logs mcp-server

# Follow logs in real-time
docker compose -f docker/docker-compose.yml logs -f rag-api

# View last 100 lines
docker compose -f docker/docker-compose.yml logs --tail=100 rag-api
```

### 6. Interactive Testing via Swagger UI

1. Open browser: http://localhost:8080/docs
2. Test endpoints interactively
3. View request/response schemas
4. Try different parameters

---

## Service Management

### Check Service Status

```bash
# List running services
docker compose -f docker/docker-compose.yml ps

# Check health of services
docker compose -f docker/docker-compose.yml ps --format json | jq

# View resource usage
docker stats
```

### Restart Services

```bash
# Restart all services
docker compose -f docker/docker-compose.yml restart

# Restart specific service
docker compose -f docker/docker-compose.yml restart rag-api

# Restart with rebuild
docker compose -f docker/docker-compose.yml up -d --build rag-api
```

### Scale Services (If Needed)

```bash
# Scale rag-api to 3 instances (requires load balancer)
docker compose -f docker/docker-compose.yml up -d --scale rag-api=3
```

### Access Container Shell

```bash
# Access RAG API container
docker exec -it rag-api /bin/bash

# Access ChromaDB container
docker exec -it chromadb /bin/bash

# Run Python in RAG API container
docker exec -it rag-api python
```

### View Container Details

```bash
# Inspect RAG API container
docker inspect rag-api

# View environment variables
docker exec rag-api env

# Check port mappings
docker port rag-api
```

---

## Troubleshooting

### Service Won't Start

**Problem:** Service fails to start or exits immediately

**Solutions:**

1. **Check logs:**
   ```bash
   docker compose -f docker/docker-compose.yml logs rag-api
   ```

2. **Verify environment variables:**
   ```bash
   docker compose -f docker/docker-compose.yml config
   ```

3. **Check if ports are in use:**
   ```bash
   lsof -i :8080  # Check RAG API port
   lsof -i :8000  # Check ChromaDB port
   ```

4. **Remove and recreate:**
   ```bash
   docker compose -f docker/docker-compose.yml down -v
   ./scripts/start.sh --build
   ```

### Health Check Fails

**Problem:** Health check returns unhealthy status

**Solutions:**

1. **Check service dependencies:**
   ```bash
   # ChromaDB must be healthy first
   curl http://localhost:8000/api/v1/heartbeat
   ```

2. **Verify API key:**
   ```bash
   # Check if OPENAI_API_KEY is set
   docker exec rag-api env | grep OPENAI
   ```

3. **Check network connectivity:**
   ```bash
   docker network inspect rag-network
   ```

### ChromaDB Connection Error

**Problem:** RAG API can't connect to ChromaDB

**Solutions:**

1. **Verify ChromaDB is running:**
   ```bash
   docker compose -f docker/docker-compose.yml ps chromadb
   ```

2. **Check ChromaDB logs:**
   ```bash
   docker compose -f docker/docker-compose.yml logs chromadb
   ```

3. **Test direct connection:**
   ```bash
   curl http://localhost:8000/api/v1/heartbeat
   ```

4. **Check network:**
   ```bash
   docker exec rag-api ping chromadb
   ```

### Out of Memory

**Problem:** Services crash due to memory issues

**Solutions:**

1. **Increase Docker memory:**
   - Docker Desktop → Settings → Resources → Memory
   - Increase to at least 4GB, recommend 8GB

2. **Reduce model size:**
   ```bash
   # In .env, use smaller model
   LLM_MODEL=gpt-3.5-turbo
   ```

3. **Use Ollama with smaller model:**
   ```bash
   # In .env
   LLM_PROVIDER=ollama
   OLLAMA_MODEL=llama2:7b
   ```

### Port Already in Use

**Problem:** "Port 8080 is already allocated"

**Solutions:**

1. **Find and kill process:**
   ```bash
   lsof -i :8080
   kill -9 <PID>
   ```

2. **Change port in docker-compose.yml:**
   ```yaml
   services:
     rag-api:
       ports:
         - "8081:8080"  # Change host port
   ```

### Logs Not Showing

**Problem:** No logs visible

**Solutions:**

1. **Check log driver:**
   ```bash
   docker info | grep "Logging Driver"
   ```

2. **View logs with different methods:**
   ```bash
   # Docker logs
   docker logs rag-api
   
   # Compose logs
   docker compose -f docker/docker-compose.yml logs rag-api
   ```

---

## Production Deployment

### Security Best Practices

1. **Use secrets for sensitive data:**
   ```bash
   # Don't commit .env to git
   echo ".env" >> .gitignore
   ```

2. **Use non-root user:** ✅ Already configured in Dockerfile

3. **Limit resource usage:**
   ```yaml
   # Add to docker-compose.yml
   services:
     rag-api:
       deploy:
         resources:
           limits:
             cpus: '2.0'
             memory: 2G
           reservations:
             cpus: '1.0'
             memory: 1G
   ```

4. **Enable HTTPS:** Use reverse proxy (nginx, Traefik, Caddy)

5. **Configure firewall:** Only expose necessary ports

### Monitoring

1. **Health checks:**
   ```bash
   # Automated health monitoring
   watch -n 30 'curl -s http://localhost:8080/api/v1/health | jq'
   ```

2. **Resource monitoring:**
   ```bash
   # Monitor resource usage
   docker stats --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}"
   ```

3. **Log aggregation:**
   - Use ELK Stack (Elasticsearch, Logstash, Kibana)
   - Or use cloud logging (AWS CloudWatch, Google Cloud Logging)

### Backup Strategy

1. **Backup ChromaDB data:**
   ```bash
   # Create backup
   docker run --rm \
     -v rag-chatbot_chroma-data:/data \
     -v $(pwd)/backups:/backup \
     alpine tar czf /backup/chroma-backup-$(date +%Y%m%d-%H%M%S).tar.gz -C /data .
   ```

2. **Restore from backup:**
   ```bash
   # Stop services
   docker compose -f docker/docker-compose.yml down
   
   # Restore data
   docker run --rm \
     -v rag-chatbot_chroma-data:/data \
     -v $(pwd)/backups:/backup \
     alpine sh -c "cd /data && tar xzf /backup/chroma-backup-YYYYMMDD-HHMMSS.tar.gz"
   
   # Restart services
   docker compose -f docker/docker-compose.yml up -d
   ```

3. **Automated backups:**
   ```bash
   # Add to crontab
   0 2 * * * /path/to/backup-script.sh
   ```

### Update Strategy

1. **Pull latest code:**
   ```bash
   cd /Applications/XAMPP/xamppfiles/htdocs/airesearch/langgraph/rag-chatbot
   git pull
   ```

2. **Rebuild and restart:**
   ```bash
   docker compose -f docker/docker-compose.yml build
   docker compose -f docker/docker-compose.yml up -d
   ```

3. **Zero-downtime deployment:**
   ```bash
   # Use blue-green deployment or rolling updates
   docker compose -f docker/docker-compose.yml up -d --no-deps --build rag-api
   ```

---

## Performance Tuning

### Optimize ChromaDB

```bash
# In .env, increase batch size for ingestion
CHROMA_BATCH_SIZE=100

# Enable persistence
CHROMA_IN_MEMORY=false
```

### Optimize LLM Calls

```bash
# Use streaming for faster perceived response
USE_STREAMING=true

# Reduce token count
LLM_MAX_TOKENS=1024

# Lower temperature for faster responses
LLM_TEMPERATURE=0.3
```

### Enable Redis Caching

```bash
# In .env
REDIS_ENABLED=true
SESSION_TTL=3600  # 1 hour cache
```

---

## Additional Commands

### Clean Up Resources

```bash
# Remove stopped containers
docker container prune

# Remove unused images
docker image prune -a

# Remove unused volumes
docker volume prune

# Remove everything (CAUTION!)
docker system prune -a --volumes
```

### Export/Import Images

```bash
# Export image
docker save rag-chatbot-api:latest | gzip > rag-api-image.tar.gz

# Import image
docker load < rag-api-image.tar.gz
```

### Database Management

```bash
# View ChromaDB collections
curl http://localhost:8000/api/v1/collections | jq

# Get collection stats
curl http://localhost:8000/api/v1/collections/documents | jq

# Delete collection (CAUTION!)
curl -X DELETE http://localhost:8000/api/v1/collections/documents
```

---

## Quick Reference

### Essential Commands

| Action | Command |
|--------|---------|
| Start services | `./scripts/start.sh --build` |
| Stop services | `docker compose -f docker/docker-compose.yml down` |
| View logs | `docker compose -f docker/docker-compose.yml logs -f` |
| Restart service | `docker compose -f docker/docker-compose.yml restart rag-api` |
| Check health | `curl http://localhost:8080/api/v1/health` |
| Access docs | Open http://localhost:8080/docs |
| Shell access | `docker exec -it rag-api /bin/bash` |

### Service Ports

| Service | Port | URL |
|---------|------|-----|
| RAG API | 8080 | http://localhost:8080 |
| ChromaDB | 8000 | http://localhost:8000 |
| MCP Server | 3000 | http://localhost:3000 |
| Redis | 6379 | redis://localhost:6379 |

---

## Support

### Getting Help

1. **Check logs:** Always start with logs
   ```bash
   docker compose -f docker/docker-compose.yml logs
   ```

2. **Verify configuration:**
   ```bash
   docker compose -f docker/docker-compose.yml config
   ```

3. **Check system resources:**
   ```bash
   docker stats
   docker system df
   ```

### Known Issues

- **ChromaDB initialization delay:** Wait 30-60 seconds after first start
- **OpenAI rate limits:** Use smaller models or add delays between requests
- **Memory usage:** Embedding models can use significant RAM

---

## Next Steps

1. ✅ Start the services
2. ✅ Test the health endpoint
3. ✅ Ingest sample documents
4. ✅ Test chat functionality
5. ✅ Explore API documentation
6. 🚀 Build your application!

---

**Docker Environment Status:** ✅ Ready to Deploy

All Docker files are properly configured. Follow the Quick Start section above to launch your RAG Chatbot!
