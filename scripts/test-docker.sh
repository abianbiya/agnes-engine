#!/bin/bash
# =============================================================================
# Docker Deployment Testing Script
# =============================================================================
# Tests all major endpoints of the RAG Chatbot after Docker deployment
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# API Base URL
API_URL="${API_URL:-http://localhost:8080}"
TEST_TIMEOUT=10

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0

# Print header
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}RAG Chatbot Docker Deployment Tests${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "API URL: ${YELLOW}$API_URL${NC}"
echo ""

# Helper function to print test results
print_result() {
    local test_name=$1
    local result=$2
    local message=$3
    
    if [ "$result" = "PASS" ]; then
        echo -e "${GREEN}✓ PASS${NC} - $test_name"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC} - $test_name: $message"
        ((TESTS_FAILED++))
    fi
}

# Helper function to make API calls
api_call() {
    local method=$1
    local endpoint=$2
    local data=$3
    
    if [ -n "$data" ]; then
        curl -s -X "$method" "$API_URL$endpoint" \
            -H "Content-Type: application/json" \
            -d "$data" \
            --connect-timeout $TEST_TIMEOUT \
            --max-time $TEST_TIMEOUT
    else
        curl -s -X "$method" "$API_URL$endpoint" \
            --connect-timeout $TEST_TIMEOUT \
            --max-time $TEST_TIMEOUT
    fi
}

# Test 1: Root endpoint
echo -e "${BLUE}Test 1: Root Endpoint${NC}"
RESPONSE=$(api_call GET "/" 2>&1)
if [ $? -eq 0 ] && echo "$RESPONSE" | grep -q "RAG Chatbot API"; then
    print_result "Root endpoint accessible" "PASS"
else
    print_result "Root endpoint accessible" "FAIL" "Could not reach API"
fi
echo ""

# Test 2: Health check
echo -e "${BLUE}Test 2: Health Check${NC}"
RESPONSE=$(api_call GET "/api/v1/health" 2>&1)
if [ $? -eq 0 ]; then
    STATUS=$(echo "$RESPONSE" | jq -r '.status' 2>/dev/null)
    if [ "$STATUS" = "healthy" ] || [ "$STATUS" = "degraded" ]; then
        print_result "Health check" "PASS"
        echo "$RESPONSE" | jq '.' 2>/dev/null || echo "$RESPONSE"
    else
        print_result "Health check" "FAIL" "Status: $STATUS"
        echo "$RESPONSE"
    fi
else
    print_result "Health check" "FAIL" "Request failed"
fi
echo ""

# Test 3: ChromaDB connectivity
echo -e "${BLUE}Test 3: ChromaDB Connectivity${NC}"
CHROMA_RESPONSE=$(curl -s http://localhost:8000/api/v1/heartbeat 2>&1)
if [ $? -eq 0 ]; then
    HEARTBEAT=$(echo "$CHROMA_RESPONSE" | jq -r '.["nanosecond heartbeat"]' 2>/dev/null)
    if [ -n "$HEARTBEAT" ]; then
        print_result "ChromaDB accessible" "PASS"
    else
        print_result "ChromaDB accessible" "FAIL" "Invalid response"
    fi
else
    print_result "ChromaDB accessible" "FAIL" "Cannot connect to ChromaDB"
fi
echo ""

# Test 4: Document ingestion
echo -e "${BLUE}Test 4: Document Ingestion${NC}"
INGEST_DATA='{
  "text": "Docker is a platform for developing, shipping, and running applications in containers. Containers allow developers to package applications with all dependencies.",
  "metadata": {
    "source": "test_docker",
    "timestamp": "'"$(date -u +%Y-%m-%dT%H:%M:%SZ)"'"
  }
}'

INGEST_RESPONSE=$(api_call POST "/api/v1/ingest/text" "$INGEST_DATA" 2>&1)
if [ $? -eq 0 ]; then
    SUCCESS=$(echo "$INGEST_RESPONSE" | jq -r '.success' 2>/dev/null)
    if [ "$SUCCESS" = "true" ]; then
        print_result "Document ingestion" "PASS"
        DOC_IDS=$(echo "$INGEST_RESPONSE" | jq -r '.document_ids[]' 2>/dev/null)
        echo -e "  Document IDs: ${YELLOW}$DOC_IDS${NC}"
    else
        print_result "Document ingestion" "FAIL" "Success flag is false"
        echo "$INGEST_RESPONSE" | jq '.' 2>/dev/null || echo "$INGEST_RESPONSE"
    fi
else
    print_result "Document ingestion" "FAIL" "Request failed"
fi
echo ""

# Wait a moment for ingestion to complete
echo -e "${YELLOW}Waiting 3 seconds for ingestion to complete...${NC}"
sleep 3
echo ""

# Test 5: Question answering (stateless)
echo -e "${BLUE}Test 5: Question Answering (Stateless)${NC}"
ASK_DATA='{
  "question": "What is Docker used for?"
}'

ASK_RESPONSE=$(api_call POST "/api/v1/ask" "$ASK_DATA" 2>&1)
if [ $? -eq 0 ]; then
    ANSWER=$(echo "$ASK_RESPONSE" | jq -r '.answer' 2>/dev/null)
    if [ -n "$ANSWER" ] && [ "$ANSWER" != "null" ]; then
        print_result "Question answering" "PASS"
        echo -e "  Answer preview: ${YELLOW}$(echo "$ANSWER" | cut -c1-100)...${NC}"
        SOURCES=$(echo "$ASK_RESPONSE" | jq -r '.sources | length' 2>/dev/null)
        echo -e "  Sources found: ${YELLOW}$SOURCES${NC}"
    else
        print_result "Question answering" "FAIL" "No answer received"
        echo "$ASK_RESPONSE" | jq '.' 2>/dev/null || echo "$ASK_RESPONSE"
    fi
else
    print_result "Question answering" "FAIL" "Request failed"
fi
echo ""

# Test 6: Chat session creation
echo -e "${BLUE}Test 6: Chat Session Creation${NC}"
SESSION_RESPONSE=$(api_call POST "/api/v1/chat/sessions" '{}' 2>&1)
if [ $? -eq 0 ]; then
    SESSION_ID=$(echo "$SESSION_RESPONSE" | jq -r '.session_id' 2>/dev/null)
    if [ -n "$SESSION_ID" ] && [ "$SESSION_ID" != "null" ]; then
        print_result "Session creation" "PASS"
        echo -e "  Session ID: ${YELLOW}$SESSION_ID${NC}"
    else
        print_result "Session creation" "FAIL" "No session ID received"
        echo "$SESSION_RESPONSE"
    fi
else
    print_result "Session creation" "FAIL" "Request failed"
fi
echo ""

# Test 7: Chat with session
if [ -n "$SESSION_ID" ] && [ "$SESSION_ID" != "null" ]; then
    echo -e "${BLUE}Test 7: Chat with Session${NC}"
    CHAT_DATA='{
      "question": "Can you explain Docker containers?"
    }'
    
    CHAT_RESPONSE=$(api_call POST "/api/v1/chat/$SESSION_ID" "$CHAT_DATA" 2>&1)
    if [ $? -eq 0 ]; then
        CHAT_ANSWER=$(echo "$CHAT_RESPONSE" | jq -r '.answer' 2>/dev/null)
        if [ -n "$CHAT_ANSWER" ] && [ "$CHAT_ANSWER" != "null" ]; then
            print_result "Chat with session" "PASS"
            echo -e "  Answer preview: ${YELLOW}$(echo "$CHAT_ANSWER" | cut -c1-100)...${NC}"
        else
            print_result "Chat with session" "FAIL" "No answer received"
            echo "$CHAT_RESPONSE" | jq '.' 2>/dev/null || echo "$CHAT_RESPONSE"
        fi
    else
        print_result "Chat with session" "FAIL" "Request failed"
    fi
    echo ""
fi

# Test 8: API Documentation
echo -e "${BLUE}Test 8: API Documentation${NC}"
DOCS_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" "$API_URL/docs" --connect-timeout $TEST_TIMEOUT)
if [ "$DOCS_RESPONSE" = "200" ]; then
    print_result "API docs accessible" "PASS"
    echo -e "  Visit: ${YELLOW}$API_URL/docs${NC}"
else
    print_result "API docs accessible" "FAIL" "HTTP $DOCS_RESPONSE"
fi
echo ""

# Test 9: List collections
echo -e "${BLUE}Test 9: List Vector Collections${NC}"
COLLECTIONS_RESPONSE=$(api_call GET "/api/v1/collections" 2>&1)
if [ $? -eq 0 ]; then
    COLLECTION_COUNT=$(echo "$COLLECTIONS_RESPONSE" | jq -r '.collections | length' 2>/dev/null)
    if [ -n "$COLLECTION_COUNT" ]; then
        print_result "List collections" "PASS"
        echo -e "  Collections found: ${YELLOW}$COLLECTION_COUNT${NC}"
    else
        print_result "List collections" "FAIL" "Invalid response"
    fi
else
    print_result "List collections" "FAIL" "Request failed"
fi
echo ""

# Print summary
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Test Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Passed:${NC} $TESTS_PASSED"
echo -e "${RED}Failed:${NC} $TESTS_FAILED"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
    echo ""
    echo -e "Your RAG Chatbot is ${GREEN}fully operational${NC}!"
    echo ""
    echo "Next steps:"
    echo "  1. Visit the API docs: $API_URL/docs"
    echo "  2. Start building your application"
    echo "  3. Ingest your own documents"
    exit 0
else
    echo -e "${RED}✗ Some tests failed${NC}"
    echo ""
    echo "Troubleshooting:"
    echo "  1. Check logs: docker compose -f docker/docker-compose.yml logs"
    echo "  2. Verify .env configuration"
    echo "  3. Ensure all services are running: docker compose ps"
    exit 1
fi
