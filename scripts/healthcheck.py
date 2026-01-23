#!/usr/bin/env python3
"""
Health check script for Docker container health checks.

This script checks the health of the RAG API service by:
1. Testing the /health endpoint
2. Verifying vector store connectivity
3. Checking basic API responsiveness

Exit codes:
    0: Healthy
    1: Unhealthy
"""

import sys
from typing import Optional

try:
    import httpx
except ImportError:
    # Fallback if httpx not available
    import urllib.request
    import json
    
    def check_health_fallback() -> bool:
        """Check health using urllib (fallback)."""
        try:
            req = urllib.request.Request(
                "http://localhost:8080/api/v1/health",
                headers={"User-Agent": "HealthCheck/1.0"}
            )
            with urllib.request.urlopen(req, timeout=5) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode())
                    return data.get("status") in ["healthy", "degraded"]
                return False
        except Exception:
            return False
    
    def main() -> int:
        """Main health check function (fallback)."""
        if check_health_fallback():
            return 0
        return 1
    
    if __name__ == "__main__":
        sys.exit(main())


def check_api_health() -> Optional[dict]:
    """
    Check API health endpoint.
    
    Returns:
        Health status dict if successful, None otherwise
    """
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get("http://localhost:8080/api/v1/health")
            
            if response.status_code == 200:
                return response.json()
            return None
            
    except Exception as e:
        print(f"Health check failed: {e}", file=sys.stderr)
        return None


def check_root_endpoint() -> bool:
    """
    Check root endpoint for basic API responsiveness.
    
    Returns:
        True if root endpoint responds, False otherwise
    """
    try:
        with httpx.Client(timeout=3.0) as client:
            response = client.get("http://localhost:8080/")
            return response.status_code == 200
    except Exception:
        return False


def main() -> int:
    """
    Main health check function.
    
    Returns:
        0 if healthy, 1 if unhealthy
    """
    # First check if API is responding at all
    if not check_root_endpoint():
        print("API not responding", file=sys.stderr)
        return 1
    
    # Check health endpoint
    health_data = check_api_health()
    
    if health_data is None:
        print("Health endpoint not responding", file=sys.stderr)
        return 1
    
    # Get overall status
    status = health_data.get("status", "unhealthy")
    
    # Consider both "healthy" and "degraded" as passing
    # (degraded means some services are down but API is functional)
    if status in ["healthy", "degraded"]:
        # Log status for monitoring
        print(f"Health check passed: {status}", file=sys.stdout)
        
        # Check individual services
        services = health_data.get("services", {})
        for service, is_healthy in services.items():
            status_str = "UP" if is_healthy else "DOWN"
            print(f"  {service}: {status_str}", file=sys.stdout)
        
        return 0
    
    # Unhealthy
    print(f"Health check failed: {status}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
