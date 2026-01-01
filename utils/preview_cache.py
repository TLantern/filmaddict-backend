import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, Optional
from uuid import UUID

logger = logging.getLogger(__name__)

# Global cache to store local preview paths by video_id
_preview_cache: Dict[str, str] = {}

# Directory for storing local preview files
PREVIEW_DIR = os.path.join(tempfile.gettempdir(), "filmaddict_previews")
os.makedirs(PREVIEW_DIR, exist_ok=True)


def get_preview_path(video_id: UUID) -> Optional[str]:
    """
    Get the local preview path for a video if it exists.
    
    Args:
        video_id: UUID of the video
        
    Returns:
        Local file path to preview, or None if not found
    """
    video_id_str = str(video_id)
    preview_path = _preview_cache.get(video_id_str)
    
    if preview_path and os.path.exists(preview_path):
        return preview_path
    
    # Clean up stale entry if file doesn't exist
    if preview_path:
        _preview_cache.pop(video_id_str, None)
    
    return None


def set_preview_path(video_id: UUID, preview_path: str) -> None:
    """
    Store the local preview path for a video.
    Also cleans up any previous preview for this video.
    
    Args:
        video_id: UUID of the video
        preview_path: Path to the local preview file
    """
    video_id_str = str(video_id)
    
    # Clean up old preview if it exists
    old_preview = _preview_cache.get(video_id_str)
    if old_preview and old_preview != preview_path and os.path.exists(old_preview):
        try:
            os.remove(old_preview)
            logger.info(f"Cleaned up old preview for video {video_id}: {old_preview}")
        except Exception as e:
            logger.warning(f"Failed to clean up old preview {old_preview}: {str(e)}")
    
    _preview_cache[video_id_str] = preview_path
    logger.info(f"Stored preview path for video {video_id}: {preview_path}")


def clear_preview(video_id: UUID) -> None:
    """
    Clear and delete the local preview for a video.
    
    Args:
        video_id: UUID of the video
    """
    video_id_str = str(video_id)
    preview_path = _preview_cache.pop(video_id_str, None)
    
    if preview_path and os.path.exists(preview_path):
        try:
            os.remove(preview_path)
            logger.info(f"Cleared preview for video {video_id}: {preview_path}")
        except Exception as e:
            logger.warning(f"Failed to delete preview {preview_path}: {str(e)}")


def get_preview_file_path(video_id: UUID) -> str:
    """
    Generate a file path for a new preview file.
    
    Args:
        video_id: UUID of the video
        
    Returns:
        Full path to where preview file should be saved
    """
    filename = f"preview_{video_id}.mp4"
    return os.path.join(PREVIEW_DIR, filename)

