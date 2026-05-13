"""
Redis Caching Layer
"""
import redis
import json
import os
from typing import Optional, Any
from functools import wraps
import hashlib

# Redis configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "false").lower() == "true"

# Initialize Redis client
if CACHE_ENABLED:
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        redis_client.ping()
        print("✅ Redis cache connected")
    except:
        print("⚠️ Redis not available, caching disabled")
        CACHE_ENABLED = False
        redis_client = None
else:
    redis_client = None
    print("💾 Redis caching disabled (using in-memory)")


class Cache:
    """Simple cache wrapper"""
    
    @staticmethod
    def get(key: str) -> Optional[Any]:
        """Get value from cache"""
        if not CACHE_ENABLED or not redis_client:
            return None
        
        try:
            value = redis_client.get(key)
            if value:
                return json.loads(value)
        except Exception as e:
            print(f"Cache get error: {e}")
        return None
    
    @staticmethod
    def set(key: str, value: Any, ttl: int = 300):
        """Set value in cache with TTL (seconds)"""
        if not CACHE_ENABLED or not redis_client:
            return False
        
        try:
            redis_client.setex(
                key,
                ttl,
                json.dumps(value)
            )
            return True
        except Exception as e:
            print(f"Cache set error: {e}")
            return False
    
    @staticmethod
    def delete(key: str):
        """Delete key from cache"""
        if not CACHE_ENABLED or not redis_client:
            return
        
        try:
            redis_client.delete(key)
        except Exception as e:
            print(f"Cache delete error: {e}")
    
    @staticmethod
    def clear_pattern(pattern: str):
        """Clear all keys matching pattern"""
        if not CACHE_ENABLED or not redis_client:
            return
        
        try:
            for key in redis_client.scan_iter(pattern):
                redis_client.delete(key)
        except Exception as e:
            print(f"Cache clear error: {e}")


def cache_result(ttl: int = 300, key_prefix: str = "veda"):
    """Decorator to cache function results"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            key_data = f"{func.__name__}:{args}:{kwargs}"
            cache_key = f"{key_prefix}:{hashlib.md5(key_data.encode()).hexdigest()}"
            
            # Try to get from cache
            cached = Cache.get(cache_key)
            if cached is not None:
                return cached
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Store in cache
            Cache.set(cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator


# Example usage
"""
@cache_result(ttl=600, key_prefix="workflows")
async def get_workflow_stats():
    # Expensive computation
    return {"total": 100, "completed": 80}
"""