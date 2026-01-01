import json
import logging
import os
from typing import Optional, List
from uuid import UUID

logger = logging.getLogger(__name__)

# Redis connection (supports both standard Redis and Upstash REST API)
redis_conn = None

try:
    # Check for Upstash Redis (REST API)
    upstash_url = os.getenv("UPSTASH_REDIS_REST_URL")
    upstash_token = os.getenv("UPSTASH_REDIS_REST_TOKEN")
    
    if upstash_url and upstash_token:
        try:
            from upstash_redis import Redis as UpstashRedis
            redis_conn = UpstashRedis(url=upstash_url, token=upstash_token)
            # Test connection
            redis_conn.ping()
            logger.info("Upstash Redis connection established for transcript caching")
        except ImportError:
            logger.warning("upstash-redis package not installed. Install with: pip install upstash-redis")
            redis_conn = None
        except Exception as conn_error:
            logger.warning(f"Upstash Redis connection failed (will continue without caching): {str(conn_error)}")
            redis_conn = None
    else:
        # Fallback to standard Redis
        import redis
        
        # Get Redis URL from environment
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        
        # Validate Redis URL format
        if redis_url and (redis_url.startswith("redis://") or redis_url.startswith("rediss://")):
            try:
                # Parse Redis URL and create connection
                if redis_url.startswith("rediss://"):
                    # SSL connection
                    redis_conn = redis.from_url(
                        redis_url,
                        ssl_cert_reqs=None,
                        decode_responses=True,  # Use text mode for JSON
                        socket_connect_timeout=2,  # 2 second timeout
                        socket_timeout=2,
                        retry_on_timeout=False,
                        health_check_interval=30
                    )
                else:
                    # Non-SSL connection
                    redis_conn = redis.from_url(
                        redis_url,
                        decode_responses=True,
                        socket_connect_timeout=2,  # 2 second timeout
                        socket_timeout=2,
                        retry_on_timeout=False,
                        health_check_interval=30
                    )
                
                # Test connection immediately
                redis_conn.ping()
                logger.info("Standard Redis connection established for transcript caching")
            except Exception as conn_error:
                logger.warning(f"Redis connection failed (will continue without caching): {str(conn_error)}")
                redis_conn = None
        else:
            logger.warning(f"Invalid REDIS_URL: {redis_url}. Transcript caching disabled.")
            redis_conn = None

except (ImportError, Exception) as e:
    logger.warning(f"Redis not available for transcript caching: {str(e)}")
    redis_conn = None


def _get_transcript_key(video_id: UUID) -> str:
    """Get Redis key for transcript."""
    return f"transcript:{video_id}"


def save_transcript_to_redis(video_id: UUID, segments: List[dict], ttl: int = 86400 * 7) -> bool:
    """
    Save transcript segments to Redis cache.
    
    Args:
        video_id: Video UUID
        segments: List of transcript segment dictionaries
        ttl: Time to live in seconds (default: 7 days)
        
    Returns:
        True if saved successfully, False otherwise
    """
    if not redis_conn:
        return False
    
    try:
        key = _get_transcript_key(video_id)
        value = json.dumps(segments)
        redis_conn.setex(key, ttl, value)
        logger.info(f"Saved transcript to Redis for video {video_id} ({len(segments)} segments, TTL: {ttl}s)")
        return True
    except Exception as e:
        # Log as warning instead of error - Redis is optional
        logger.warning(f"Failed to save transcript to Redis for video {video_id} (non-blocking): {str(e)}")
        return False


def get_transcript_from_redis(video_id: UUID) -> Optional[List[dict]]:
    """
    Get transcript segments from Redis cache.
    
    Args:
        video_id: Video UUID
        
    Returns:
        List of transcript segment dictionaries if found, None otherwise
    """
    if not redis_conn:
        return None
    
    try:
        key = _get_transcript_key(video_id)
        value = redis_conn.get(key)
        if value:
            segments = json.loads(value)
            logger.info(f"Retrieved transcript from Redis for video {video_id} ({len(segments)} segments)")
            return segments
        return None
    except Exception as e:
        # Log as warning instead of error - Redis is optional
        logger.warning(f"Failed to get transcript from Redis for video {video_id} (non-blocking): {str(e)}")
        return None


def delete_transcript_from_redis(video_id: UUID) -> bool:
    """
    Delete transcript from Redis cache.
    
    Args:
        video_id: Video UUID
        
    Returns:
        True if deleted successfully, False otherwise
    """
    if not redis_conn:
        return False
    
    try:
        key = _get_transcript_key(video_id)
        deleted = redis_conn.delete(key)
        if deleted:
            logger.info(f"Deleted transcript from Redis for video {video_id}")
        return bool(deleted)
    except Exception as e:
        # Log as warning instead of error - Redis is optional
        logger.warning(f"Failed to delete transcript from Redis for video {video_id} (non-blocking): {str(e)}")
        return False


def test_redis_connection() -> dict:
    """
    Test Redis connection and return status.
    
    Returns:
        Dictionary with connection status and test results
    """
    result = {
        "connected": False,
        "error": None,
        "test_write": False,
        "test_read": False,
        "test_delete": False,
    }
    
    if not redis_conn:
        result["error"] = "Redis connection not available (redis_conn is None)"
        return result
    
    try:
        # Test 1: Ping Redis
        redis_conn.ping()
        result["connected"] = True
        logger.info("✅ Redis ping successful")
        
        # Test 2: Write test
        test_key = "test:transcript:connection"
        test_value = json.dumps({"test": "data", "timestamp": "2024-01-01"})
        redis_conn.setex(test_key, 10, test_value)  # 10 second TTL
        result["test_write"] = True
        logger.info("✅ Redis write test successful")
        
        # Test 3: Read test
        retrieved = redis_conn.get(test_key)
        if retrieved and retrieved == test_value:
            result["test_read"] = True
            logger.info("✅ Redis read test successful")
        else:
            result["error"] = f"Read test failed: expected {test_value}, got {retrieved}"
        
        # Test 4: Delete test
        deleted = redis_conn.delete(test_key)
        if deleted:
            result["test_delete"] = True
            logger.info("✅ Redis delete test successful")
        else:
            result["error"] = "Delete test failed: key not deleted"
        
        # Verify deletion
        verify = redis_conn.get(test_key)
        if verify is None:
            logger.info("✅ Redis delete verification successful")
        else:
            result["error"] = "Delete verification failed: key still exists"
        
    except Exception as e:
        result["error"] = str(e)
        logger.error(f"❌ Redis connection test failed: {str(e)}", exc_info=True)
    
    return result

