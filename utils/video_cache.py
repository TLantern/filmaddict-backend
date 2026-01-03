import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, Optional
from uuid import UUID

logger = logging.getLogger(__name__)

# Global cache to store local video paths by video_id
_video_cache: Dict[str, str] = {}

# Directory for storing local video cache files
VIDEO_CACHE_DIR = os.path.join(tempfile.gettempdir(), "filmaddict_video_cache")
os.makedirs(VIDEO_CACHE_DIR, exist_ok=True)


def get_cached_video_path(video_id: UUID) -> Optional[str]:
    """
    Get the local cached video path if it exists and is valid.
    
    Args:
        video_id: UUID of the video
        
    Returns:
        Local file path to cached video, or None if not found or invalid
    """
    video_id_str = str(video_id)
    cached_path = _video_cache.get(video_id_str)
    
    if cached_path and os.path.exists(cached_path):
        # Validate file size (should be at least 10KB for a valid video)
        try:
            file_size = os.path.getsize(cached_path)
            if file_size >= 10000:
                return cached_path
            else:
                logger.warning(f"Cached video for {video_id} is too small ({file_size} bytes), invalidating cache")
                _video_cache.pop(video_id_str, None)
                try:
                    os.remove(cached_path)
                except:
                    pass
                return None
        except OSError:
            pass
    
    # Clean up stale entry if file doesn't exist
    if cached_path:
        _video_cache.pop(video_id_str, None)
    
    return None


def set_cached_video_path(video_id: UUID, video_path: str) -> None:
    """
    Store the local cached video path.
    Also cleans up any previous cache for this video.
    
    Args:
        video_id: UUID of the video
        video_path: Path to the local cached video file
    """
    video_id_str = str(video_id)
    
    # Clean up old cache if it exists
    old_cache = _video_cache.get(video_id_str)
    if old_cache and old_cache != video_path and os.path.exists(old_cache):
        try:
            os.remove(old_cache)
            logger.info(f"Cleaned up old video cache for video {video_id}: {old_cache}")
        except Exception as e:
            logger.warning(f"Failed to clean up old video cache {old_cache}: {str(e)}")
    
    _video_cache[video_id_str] = video_path
    logger.info(f"Stored cached video path for video {video_id}: {video_path}")


def clear_cached_video(video_id: UUID) -> None:
    """
    Clear and delete the cached video for a video.
    
    Args:
        video_id: UUID of the video
    """
    video_id_str = str(video_id)
    cached_path = _video_cache.pop(video_id_str, None)
    
    if cached_path and os.path.exists(cached_path):
        try:
            os.remove(cached_path)
            logger.info(f"Cleared cached video for video {video_id}: {cached_path}")
        except Exception as e:
            logger.warning(f"Failed to delete cached video {cached_path}: {str(e)}")


def get_cached_video_file_path(video_id: UUID) -> str:
    """
    Generate a file path for a new cached video file.
    
    Args:
        video_id: UUID of the video
        
    Returns:
        Full path to where cached video file should be saved
    """
    filename = f"video_{video_id}.mp4"
    return os.path.join(VIDEO_CACHE_DIR, filename)

