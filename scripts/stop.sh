#!/bin/bash
# =============================================================================
# Docker Compose Stop Script
# =============================================================================
# Stops all RAG chatbot services
# Usage: ./scripts/stop.sh [OPTIONS]
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
REMOVE_VOLUMES=false
REMOVE_IMAGES=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--remove-volumes)
            REMOVE_VOLUMES=true
            shift
            ;;
        -i|--remove-images)
            REMOVE_IMAGES=true
            shift
            ;;
        -a|--all)
            REMOVE_VOLUMES=true
            REMOVE_IMAGES=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -v, --remove-volumes  Remove volumes (CAUTION: deletes data)"
            echo "  -i, --remove-images   Remove built images"
            echo "  -a, --all             Remove volumes and images"
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

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}Error: Docker is not running${NC}"
    exit 1
fi

# Check if docker-compose.yml exists
if [ ! -f "$DOCKER_DIR/docker-compose.yml" ]; then
    echo -e "${RED}Error: docker-compose.yml not found in $DOCKER_DIR${NC}"
    exit 1
fi

echo -e "${GREEN}Stopping RAG Chatbot Services...${NC}"
echo ""

# Change to docker directory
cd "$DOCKER_DIR"

# Build docker-compose command
COMPOSE_CMD="docker compose down"

if [ "$REMOVE_VOLUMES" = true ]; then
    echo -e "${YELLOW}Warning: This will remove all volumes and delete data${NC}"
    read -p "Are you sure? (y/N) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled"
        exit 0
    fi
    COMPOSE_CMD="$COMPOSE_CMD -v"
fi

if [ "$REMOVE_IMAGES" = true ]; then
    COMPOSE_CMD="$COMPOSE_CMD --rmi local"
fi

# Stop services
eval "$COMPOSE_CMD"

echo ""
echo -e "${GREEN}Services stopped successfully!${NC}"

if [ "$REMOVE_VOLUMES" = true ]; then
    echo -e "${YELLOW}Volumes removed${NC}"
fi

if [ "$REMOVE_IMAGES" = true ]; then
    echo -e "${YELLOW}Images removed${NC}"
fi
