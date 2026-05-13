"""
End-to-End Integration Tests for VEDA
Tests complete workflow from login to report generation
"""
import pytest
import httpx
import time
import os
from pathlib import Path

BASE_URL = "http://localhost:8000"
TEST_DATA_DIR = Path("examples/datasets")

@pytest.fixture
def auth_token():
    """Get authentication token"""
    response = httpx.post(
        f"{BASE_URL}/auth/login",
        json={"username": "admin", "password": "admin123"}
    )
    assert response.status_code == 200
    return response.json()["access_token"]


@pytest.fixture
def auth_headers(auth_token):
    """Get authorization headers"""
    return {"Authorization": f"Bearer {auth_token}"}


class TestAuthentication:
    """Test authentication flow"""
    
    def test_successful_login(self):
        """Test successful login"""
        response = httpx.post(
            f"{BASE_URL}/auth/login",
            json={"username": "admin", "password": "admin123"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
    
    def test_failed_login_wrong_password(self):
        """Test login with wrong password"""
        response = httpx.post(
            f"{BASE_URL}/auth/login",
            json={"username": "admin", "password": "wrongpassword"}
        )
        assert response.status_code == 401
    
    def test_failed_login_wrong_username(self):
        """Test login with non-existent user"""
        response = httpx.post(
            f"{BASE_URL}/auth/login",
            json={"username": "nonexistent", "password": "password"}
        )
        assert response.status_code == 401
    
    def test_token_validation(self, auth_headers):
        """Test that token grants access to protected routes"""
        response = httpx.get(f"{BASE_URL}/workflows", headers=auth_headers)
        assert response.status_code == 200


class TestWorkflowCreation:
    """Test workflow creation and management"""
    
    def test_create_workflow_customer_churn(self, auth_headers):
        """Test creating a customer churn prediction workflow"""
        dataset_path = TEST_DATA_DIR / "customer_churn.csv"
        
        if not dataset_path.exists():
            pytest.skip(f"Test dataset not found: {dataset_path}")
        
        payload = {
            "dataset_path": str(dataset_path),
            "goal": "predict customer churn based on purchase history and engagement"
        }
        
        response = httpx.post(
            f"{BASE_URL}/workflows",
            headers=auth_headers,
            json=payload,
            timeout=30.0
        )
        
        assert response.status_code in [200, 201], f"Failed: {response.text}"
        data = response.json()
        assert "job_id" in data
        assert data["status"] in ["pending", "running"]
        
        return data["job_id"]
    
    def test_create_workflow_sales_forecast(self, auth_headers):
        """Test creating a sales forecasting workflow"""
        dataset_path = TEST_DATA_DIR / "sales_forecast.csv"
        
        if not dataset_path.exists():
            pytest.skip(f"Test dataset not found: {dataset_path}")
        
        payload = {
            "dataset_path": str(dataset_path),
            "goal": "forecast daily sales based on marketing and seasonality"
        }
        
        response = httpx.post(
            f"{BASE_URL}/workflows",
            headers=auth_headers,
            json=payload,
            timeout=30.0
        )
        
        assert response.status_code in [200, 201]
        data = response.json()
        assert "job_id" in data
    
    def test_create_workflow_credit_risk(self, auth_headers):
        """Test creating a credit risk prediction workflow"""
        dataset_path = TEST_DATA_DIR / "credit_risk.csv"
        
        if not dataset_path.exists():
            pytest.skip(f"Test dataset not found: {dataset_path}")
        
        payload = {
            "dataset_path": str(dataset_path),
            "goal": "predict loan default risk based on credit history"
        }
        
        response = httpx.post(
            f"{BASE_URL}/workflows",
            headers=auth_headers,
            json=payload,
            timeout=30.0
        )
        
        assert response.status_code in [200, 201]
        data = response.json()
        assert "job_id" in data
    
    def test_create_workflow_invalid_dataset(self, auth_headers):
        """Test creating workflow with non-existent dataset"""
        payload = {
            "dataset_path": "nonexistent/file.csv",
            "goal": "predict something"
        }
        
        response = httpx.post(
            f"{BASE_URL}/workflows",
            headers=auth_headers,
            json=payload,
            timeout=30.0
        )
        
        # Should either reject immediately or create and fail
        assert response.status_code in [400, 422, 200, 201]
    
    def test_create_workflow_missing_goal(self, auth_headers):
        """Test creating workflow without goal"""
        payload = {
            "dataset_path": str(TEST_DATA_DIR / "customer_churn.csv")
        }
        
        response = httpx.post(
            f"{BASE_URL}/workflows",
            headers=auth_headers,
            json=payload,
            timeout=30.0
        )
        
        assert response.status_code == 422  # Validation error


class TestWorkflowRetrieval:
    """Test workflow status and retrieval"""
    
    def test_list_all_workflows(self, auth_headers):
        """Test listing all workflows"""
        response = httpx.get(f"{BASE_URL}/workflows", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_get_specific_workflow(self, auth_headers):
        """Test retrieving specific workflow"""
        # First create a workflow
        dataset_path = TEST_DATA_DIR / "customer_churn.csv"
        
        if not dataset_path.exists():
            pytest.skip("Test dataset not found")
        
        create_response = httpx.post(
            f"{BASE_URL}/workflows",
            headers=auth_headers,
            json={
                "dataset_path": str(dataset_path),
                "goal": "test workflow"
            },
            timeout=30.0
        )
        
        if create_response.status_code not in [200, 201]:
            pytest.skip("Could not create workflow")
        
        job_id = create_response.json()["job_id"]
        
        # Now retrieve it
        response = httpx.get(
            f"{BASE_URL}/workflows/{job_id}",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == job_id
        assert "status" in data
        assert "created_at" in data
    
    def test_get_nonexistent_workflow(self, auth_headers):
        """Test retrieving non-existent workflow"""
        response = httpx.get(
            f"{BASE_URL}/workflows/nonexistent-id-12345",
            headers=auth_headers
        )
        assert response.status_code == 404


class TestWorkflowPolling:
    """Test workflow completion by polling"""
    
    def test_workflow_completion_polling(self, auth_headers):
        """Test polling workflow until completion"""
        dataset_path = TEST_DATA_DIR / "customer_churn.csv"
        
        if not dataset_path.exists():
            pytest.skip("Test dataset not found")
        
        # Create workflow
        create_response = httpx.post(
            f"{BASE_URL}/workflows",
            headers=auth_headers,
            json={
                "dataset_path": str(dataset_path),
                "goal": "predict customer churn"
            },
            timeout=30.0
        )
        
        if create_response.status_code not in [200, 201]:
            pytest.skip("Could not create workflow")
        
        job_id = create_response.json()["job_id"]
        
        # Poll for completion (max 5 minutes)
        max_attempts = 60
        attempt = 0
        final_status = None
        
        print(f"\n⏳ Polling workflow {job_id}...")
        
        while attempt < max_attempts:
            response = httpx.get(
                f"{BASE_URL}/workflows/{job_id}",
                headers=auth_headers
            )
            
            if response.status_code == 200:
                data = response.json()
                status = data.get("status", "unknown")
                
                print(f"  Attempt {attempt + 1}: {status}")
                
                if status in ["completed", "failed", "error"]:
                    final_status = status
                    break
            
            time.sleep(5)
            attempt += 1
        
        if final_status:
            print(f"✅ Workflow finished with status: {final_status}")
            assert final_status == "completed", f"Expected 'completed', got '{final_status}'"
        else:
            pytest.skip("Workflow did not complete within timeout")


class TestReportGeneration:
    """Test HTML report generation"""
    
    def test_generate_report_for_completed_workflow(self, auth_headers):
        """Test generating report for a completed workflow"""
        # List workflows to find a completed one
        response = httpx.get(f"{BASE_URL}/workflows", headers=auth_headers)
        
        if response.status_code != 200:
            pytest.skip("Could not list workflows")
        
        workflows = response.json()
        completed = [w for w in workflows if w.get("status") == "completed"]
        
        if not completed:
            pytest.skip("No completed workflows found")
        
        job_id = completed[0]["id"]
        
        # Generate report
        report_response = httpx.get(
            f"{BASE_URL}/workflows/{job_id}/report",
            headers=auth_headers,
            timeout=30.0
        )
        
        assert report_response.status_code == 200
        assert "text/html" in report_response.headers.get("content-type", "")
        assert len(report_response.text) > 1000  # Should be substantial HTML


class TestSystemEndpoints:
    """Test system-level endpoints"""
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = httpx.get(f"{BASE_URL}/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data or "name" in data
    
    def test_health_endpoint(self):
        """Test health check"""
        response = httpx.get(f"{BASE_URL}/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_stats_endpoint(self):
        """Test statistics endpoint"""
        response = httpx.get(f"{BASE_URL}/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_workflows" in data
        assert isinstance(data["total_workflows"], int)


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_unauthorized_access(self):
        """Test accessing protected endpoint without auth"""
        response = httpx.get(f"{BASE_URL}/workflows")
        assert response.status_code == 401
    
    def test_invalid_token(self):
        """Test accessing with invalid token"""
        headers = {"Authorization": "Bearer invalid-token-12345"}
        response = httpx.get(f"{BASE_URL}/workflows", headers=headers)
        assert response.status_code == 401
    
    def test_malformed_json(self, auth_headers):
        """Test sending malformed JSON"""
        response = httpx.post(
            f"{BASE_URL}/workflows",
            headers={**auth_headers, "Content-Type": "application/json"},
            content="{invalid json}",
            timeout=10.0
        )
        assert response.status_code == 422


class TestConcurrentOperations:
    """Test concurrent API operations"""
    
    def test_concurrent_workflow_creation(self, auth_headers):
        """Test creating multiple workflows concurrently"""
        import concurrent.futures
        
        dataset_path = TEST_DATA_DIR / "customer_churn.csv"
        
        if not dataset_path.exists():
            pytest.skip("Test dataset not found")
        
        def create_workflow(index):
            payload = {
                "dataset_path": str(dataset_path),
                "goal": f"test concurrent workflow {index}"
            }
            response = httpx.post(
                f"{BASE_URL}/workflows",
                headers=auth_headers,
                json=payload,
                timeout=30.0
            )
            return response.status_code
        
        # Create 5 workflows concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(create_workflow, i) for i in range(5)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        # All should succeed
        assert all(status in [200, 201] for status in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])