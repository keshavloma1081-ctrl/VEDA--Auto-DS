"""
Performance and Load Testing for VEDA API
"""
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import pytest

BASE_URL = "http://localhost:8000"

class PerformanceMetrics:
    """Track and report performance metrics"""
    
    def __init__(self):
        self.response_times = []
        self.errors = []
        self.success_count = 0
        self.failure_count = 0
    
    def add_result(self, duration: float, success: bool, error: str = None):
        """Record a request result"""
        self.response_times.append(duration)
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
            if error:
                self.errors.append(error)
    
    def get_stats(self) -> dict:
        """Calculate statistics"""
        if not self.response_times:
            return {}
        
        return {
            "total_requests": len(self.response_times),
            "successful": self.success_count,
            "failed": self.failure_count,
            "success_rate": (self.success_count / len(self.response_times)) * 100,
            "avg_response_time": statistics.mean(self.response_times),
            "median_response_time": statistics.median(self.response_times),
            "min_response_time": min(self.response_times),
            "max_response_time": max(self.response_times),
            "p95_response_time": sorted(self.response_times)[int(len(self.response_times) * 0.95)],
            "p99_response_time": sorted(self.response_times)[int(len(self.response_times) * 0.99)],
        }


def make_request(url: str, method: str = "GET", **kwargs):
    """Make a single HTTP request and measure time"""
    start_time = time.time()
    try:
        if method == "GET":
            response = requests.get(url, **kwargs)
        elif method == "POST":
            response = requests.post(url, **kwargs)
        
        duration = time.time() - start_time
        success = response.status_code < 400
        return duration, success, None
    except Exception as e:
        duration = time.time() - start_time
        return duration, False, str(e)


def load_test_endpoint(url: str, num_requests: int = 100, 
                       concurrent: int = 10, method: str = "GET", **kwargs):
    """Load test an endpoint with concurrent requests"""
    metrics = PerformanceMetrics()
    
    with ThreadPoolExecutor(max_workers=concurrent) as executor:
        futures = [
            executor.submit(make_request, url, method, **kwargs)
            for _ in range(num_requests)
        ]
        
        for future in as_completed(futures):
            duration, success, error = future.result()
            metrics.add_result(duration, success, error)
    
    return metrics.get_stats()


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

def test_root_endpoint_performance():
    """Test root endpoint performance"""
    print("\n🔥 Testing Root Endpoint Performance...")
    
    stats = load_test_endpoint(f"{BASE_URL}/", num_requests=100, concurrent=10)
    
    print(f"\n📊 Results:")
    print(f"  Total Requests: {stats['total_requests']}")
    print(f"  Success Rate: {stats['success_rate']:.2f}%")
    print(f"  Avg Response Time: {stats['avg_response_time']*1000:.2f}ms")
    print(f"  Median: {stats['median_response_time']*1000:.2f}ms")
    print(f"  P95: {stats['p95_response_time']*1000:.2f}ms")
    print(f"  P99: {stats['p99_response_time']*1000:.2f}ms")
    print(f"  Min: {stats['min_response_time']*1000:.2f}ms")
    print(f"  Max: {stats['max_response_time']*1000:.2f}ms")
    
    # Assertions
    assert stats['success_rate'] >= 99, "Success rate should be at least 99%"
    assert stats['avg_response_time'] < 0.1, "Avg response time should be under 100ms"
    assert stats['p95_response_time'] < 0.2, "P95 should be under 200ms"


def test_health_endpoint_performance():
    """Test health check performance"""
    print("\n🔥 Testing Health Endpoint Performance...")
    
    stats = load_test_endpoint(f"{BASE_URL}/health", num_requests=100, concurrent=10)
    
    print(f"\n📊 Results:")
    print(f"  Success Rate: {stats['success_rate']:.2f}%")
    print(f"  Avg Response Time: {stats['avg_response_time']*1000:.2f}ms")
    print(f"  P95: {stats['p95_response_time']*1000:.2f}ms")
    
    assert stats['success_rate'] >= 99
    assert stats['avg_response_time'] < 0.15


def test_stats_endpoint_performance():
    """Test stats endpoint performance"""
    print("\n🔥 Testing Stats Endpoint Performance...")
    
    stats = load_test_endpoint(f"{BASE_URL}/stats", num_requests=50, concurrent=5)
    
    print(f"\n📊 Results:")
    print(f"  Success Rate: {stats['success_rate']:.2f}%")
    print(f"  Avg Response Time: {stats['avg_response_time']*1000:.2f}ms")
    
    assert stats['success_rate'] >= 95
    assert stats['avg_response_time'] < 0.2


def test_authentication_performance():
    """Test login endpoint performance"""
    print("\n🔥 Testing Authentication Performance...")
    
    payload = {"username": "admin", "password": "admin123"}
    
    stats = load_test_endpoint(
        f"{BASE_URL}/auth/login",
        num_requests=50,
        concurrent=5,
        method="POST",
        json=payload
    )
    
    print(f"\n📊 Results:")
    print(f"  Success Rate: {stats['success_rate']:.2f}%")
    print(f"  Avg Response Time: {stats['avg_response_time']*1000:.2f}ms")
    
    assert stats['success_rate'] >= 95
    assert stats['avg_response_time'] < 0.3  # Auth is slower due to bcrypt


def test_concurrent_workflow_list():
    """Test concurrent workflow listing"""
    print("\n🔥 Testing Concurrent Workflow Listing...")
    
    stats = load_test_endpoint(
        f"{BASE_URL}/workflows",
        num_requests=50,
        concurrent=10
    )
    
    print(f"\n📊 Results:")
    print(f"  Success Rate: {stats['success_rate']:.2f}%")
    print(f"  Avg Response Time: {stats['avg_response_time']*1000:.2f}ms")
    
    assert stats['success_rate'] >= 95


# ============================================================================
# STRESS TESTS
# ============================================================================

def test_stress_test_100_concurrent():
    """Stress test with 100 concurrent requests"""
    print("\n🔥 STRESS TEST: 100 Concurrent Requests...")
    
    stats = load_test_endpoint(
        f"{BASE_URL}/health",
        num_requests=200,
        concurrent=100
    )
    
    print(f"\n📊 Stress Test Results:")
    print(f"  Total Requests: {stats['total_requests']}")
    print(f"  Success Rate: {stats['success_rate']:.2f}%")
    print(f"  Avg Response Time: {stats['avg_response_time']*1000:.2f}ms")
    print(f"  P95: {stats['p95_response_time']*1000:.2f}ms")
    print(f"  Max: {stats['max_response_time']*1000:.2f}ms")
    
    # Under stress, allow for more failures
    assert stats['success_rate'] >= 90


# ============================================================================
# BENCHMARK SUITE
# ============================================================================

def test_full_benchmark_suite():
    """Run complete performance benchmark"""
    print("\n" + "="*70)
    print("🚀 VEDA PERFORMANCE BENCHMARK SUITE")
    print("="*70)
    
    endpoints = [
        ("/", "Root", 100, 10),
        ("/health", "Health", 100, 10),
        ("/stats", "Stats", 50, 5),
    ]
    
    results = {}
    
    for path, name, requests, concurrent in endpoints:
        print(f"\n📊 Testing {name} Endpoint...")
        stats = load_test_endpoint(
            f"{BASE_URL}{path}",
            num_requests=requests,
            concurrent=concurrent
        )
        results[name] = stats
        
        print(f"  ✅ {name}: {stats['avg_response_time']*1000:.2f}ms avg")
    
    # Summary
    print("\n" + "="*70)
    print("📈 BENCHMARK SUMMARY")
    print("="*70)
    
    for name, stats in results.items():
        print(f"\n{name} Endpoint:")
        print(f"  Success Rate: {stats['success_rate']:.1f}%")
        print(f"  Avg: {stats['avg_response_time']*1000:.1f}ms")
        print(f"  P95: {stats['p95_response_time']*1000:.1f}ms")
        print(f"  P99: {stats['p99_response_time']*1000:.1f}ms")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    # Run benchmarks
    pytest.main([__file__, "-v", "-s"])