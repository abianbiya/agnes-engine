#!/bin/bash
# =============================================================================
# Docker Compose Start Script
# =============================================================================
# Starts all RAG chatbot services using Docker Compose
# Usage: ./scripts/start.sh [OPTIONS]
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DOCKER_DIR="$PROJECT_ROOT/docker"

# Default options
BUILD=false
DETACH=true
PULL=false
REMOVE_VOLUMES=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -b|--build)
            BUILD=true
            shift
            ;;
        -f|--foreground)
            DETACH=false
            shift
            ;;
        -p|--pull)
            PULL=true
            shift
            ;;
        -v|--remove-volumes)
            REMOVE_VOLUMES=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -b, --build           Build images before starting"
            echo "  -f, --foreground      Run in foreground (don't detach)"
            echo "  -p, --pull            Pull latest base images before building"
            echo "  -v, --remove-volumes  Remove volumes before starting (CAUTION: deletes data)"
            echo "  -h, --help            Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Check if .env file exists
if [ ! -f "$PROJECT_ROOT/.env" ]; then
    echo -e "${YELLOW}Warning: .env file not found${NC}"
    echo "Creating .env from .env.example..."
    if [ -f "$PROJECT_ROOT/.env.example" ]; then
        cp "$PROJECT_ROOT/.env.example" "$PROJECT_ROOT/.env"
        echo -e "${GREEN}.env file created${NC}"
        echo -e "${YELLOW}Please edit .env and add your API keys before continuing${NC}"
        echo ""
        echo "Required variables:"
        echo "  - OPENAI_API_KEY (for OpenAI provider)"
        echo ""
        read -p "Press Enter to continue after editing .env..."
    else
        echo -e "${RED}Error: .env.example not found${NC}"
        exit 1
    fi
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}Error: Docker is not running${NC}"
    echo "Please start Docker and try again"
    exit 1
fi

# Check if docker-compose.yml exists
if [ ! -f "$DOCKER_DIR/docker-compose.yml" ]; then
    echo -e "${RED}Error: docker-compose.yml not found in $DOCKER_DIR${NC}"
    exit 1
fi

echo -e "${GREEN}Starting RAG Chatbot Services...${NC}"
echo ""

# Change to docker directory
cd "$DOCKER_DIR"

# Remove volumes if requested
if [ "$REMOVE_VOLUMES" = true ]; then
    echo -e "${YELLOW}Removing existing volumes...${NC}"
    docker compose down -v
    echo -e "${GREEN}Volumes removed${NC}"
    echo ""
fi

# Pull images if requested
if [ "$PULL" = true ]; then
    echo -e "${GREEN}Pulling latest base images...${NC}"
    docker compose pull
    echo ""
fi

# Build images if requested
if [ "$BUILD" = true ]; then
    echo -e "${GREEN}Building Docker images...${NC}"
    docker compose build
    echo ""
fi

# Start services
echo -e "${GREEN}Starting services...${NC}"
if [ "$DETACH" = true ]; then
    docker compose up -d
    
    echo ""
    echo -e "${GREEN}Services started successfully!${NC}"
    echo ""
    echo "Service URLs:"
    echo "  - RAG API:    http://localhost:8080"
    echo "  - API Docs:   http://localhost:8080/docs"
    echo "  - ChromaDB:   http://localhost:8000"
    echo "  - MCP Server: http://localhost:3000 (if using SSE transport)"
    echo "  - Redis:      localhost:6379"
    echo ""
    echo "View logs:"
    echo "  docker compose -f $DOCKER_DIR/docker-compose.yml logs -f [service-name]"
    echo ""
    echo "Stop services:"
    echo "  docker compose -f $DOCKER_DIR/docker-compose.yml down"
    echo ""
else
    docker compose up
fi
