import asyncio
import logging
import os
import re
from pathlib import Path
from typing import Optional, Tuple

import yt_dlp

logger = logging.getLogger(__name__)

YOUTUBE_URL_PATTERN = re.compile(
    r"(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]{11})"
)


def validate_youtube_url(url: str) -> bool:
    """
    Validate YouTube URL format.
    
    Args:
        url: YouTube URL to validate
        
    Returns:
        True if URL is valid, False otherwise
    """
    return bool(YOUTUBE_URL_PATTERN.match(url))


def get_format_preference() -> str:
    """
    Get YouTube format preference from environment variable.
    
    Returns:
        Format preference string (default: "best")
    """
    return os.getenv("YOUTUBE_FORMAT", "best")


async def download_youtube_video(
    url: str, output_dir: str, format_preference: Optional[str] = None
) -> Tuple[str, float]:
    """
    Download YouTube video using yt-dlp.
    
    Args:
        url: YouTube video URL
        output_dir: Directory to save the downloaded video
        format_preference: Format preference (defaults to YOUTUBE_FORMAT env var or "best")
        
    Returns:
        Tuple of (local_file_path, duration_in_seconds)
        
    Raises:
        ValueError: If URL is invalid
        Exception: If download fails
    """
    if not validate_youtube_url(url):
        raise ValueError(f"Invalid YouTube URL: {url}")
    
    format_pref = format_preference or get_format_preference()
    
    os.makedirs(output_dir, exist_ok=True)
    
    ydl_opts = {
        "format": format_pref,
        "outtmpl": os.path.join(output_dir, "%(id)s.%(ext)s"),
        "quiet": False,
        "no_warnings": False,
        "extract_flat": False,
    }
    
    duration = 0.0
    downloaded_file_path = None
    
    def _download():
        """Synchronous download function to run in executor."""
        nonlocal duration, downloaded_file_path
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            logger.info(f"Downloading YouTube video: {url}")
            
            info = ydl.extract_info(url, download=True)
            
            if not info:
                raise Exception("Failed to extract video information")
            
            duration = float(info.get("duration", 0))
            
            video_id = info.get("id")
            ext = info.get("ext", "mp4")
            downloaded_file_path = os.path.join(output_dir, f"{video_id}.{ext}")
            
            if not os.path.exists(downloaded_file_path):
                raise Exception(f"Downloaded file not found: {downloaded_file_path}")
            
            logger.info(f"Successfully downloaded YouTube video: {downloaded_file_path} (duration: {duration}s)")
            
            return downloaded_file_path, duration
    
    try:
        loop = asyncio.get_event_loop()
        downloaded_file_path, duration = await loop.run_in_executor(None, _download)
        return downloaded_file_path, duration
            
    except yt_dlp.utils.DownloadError as e:
        logger.error(f"yt-dlp download error: {str(e)}")
        if downloaded_file_path and os.path.exists(downloaded_file_path):
            try:
                os.remove(downloaded_file_path)
            except Exception:
                pass
        raise Exception(f"Failed to download YouTube video: {str(e)}")
    except Exception as e:
        logger.error(f"Error downloading YouTube video: {str(e)}", exc_info=True)
        if downloaded_file_path and os.path.exists(downloaded_file_path):
            try:
                os.remove(downloaded_file_path)
            except Exception:
                pass
        raise

