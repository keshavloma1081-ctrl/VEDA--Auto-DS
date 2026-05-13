"""
Test VEDA API Client Integration
Tests using httpx client with proper error handling
"""
import pytest
import httpx
from pathlib import Path

BASE_URL = "http://localhost:8000"

class VEDAClient:
    """Simple VEDA API client for testing"""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.token = None
    
    def login(self, username: str, password: str):
        """Login and store token"""
        response = httpx.post(
            f"{self.base_url}/auth/login",
            json={"username": username, "password": password}
        )
        response.raise_for_status()
        self.token = response.json()["access_token"]
        return self.token
    
    def _headers(self):
        """Get auth headers"""
        if not self.token:
            raise ValueError("Not authenticated. Call login() first.")
        return {"Authorization": f"Bearer {self.token}"}
    
    def create_workflow(self, dataset_path: str, goal: str):
        """Create a new workflow"""
        response = httpx.post(
            f"{self.base_url}/workflows",
            headers=self._headers(),
            json={"dataset_path": dataset_path, "goal": goal},
            timeout=30.0
        )
        response.raise_for_status()
        return response.json()
    
    def get_workflow(self, job_id: str):
        """Get workflow status"""
        response = httpx.get(
            f"{self.base_url}/workflows/{job_id}",
            headers=self._headers()
        )
        response.raise_for_status()
        return response.json()
    
    def list_workflows(self):
        """List all workflows"""
        response = httpx.get(
            f"{self.base_url}/workflows",
            headers=self._headers()
        )
        response.raise_for_status()
        return response.json()
    
    def get_report(self, job_id: str):
        """Get HTML report"""
        response = httpx.get(
            f"{self.base_url}/workflows/{job_id}/report",
            headers=self._headers(),
            timeout=30.0
        )
        response.raise_for_status()
        return response.text


@pytest.fixture
def client():
    """Create authenticated client"""
    c = VEDAClient()
    c.login("admin", "admin123")
    return c


def test_client_workflow(client):
    """Test complete workflow using client"""
    # Create workflow
    result = client.create_workflow(
        dataset_path="examples/datasets/customer_churn.csv",
        goal="predict churn"
    )
    
    assert "job_id" in result
    job_id = result["job_id"]
    
    # Check status
    status = client.get_workflow(job_id)
    assert status["id"] == job_id
    
    # List workflows
    workflows = client.list_workflows()
    assert any(w["id"] == job_id for w in workflows)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])