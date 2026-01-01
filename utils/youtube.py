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


def get_cookie_config() -> dict:
    """
    Get YouTube cookie configuration from environment variables.
    
    Supports two methods:
    1. YOUTUBE_COOKIES: Path to a cookies file (Netscape format)
    2. YOUTUBE_COOKIES_FROM_BROWSER: Browser name (e.g., "chrome", "firefox", "safari", "edge")
    
    Returns:
        Dictionary with cookie configuration for yt-dlp, or empty dict if not configured
    """
    cookie_config = {}
    
    # Method 1: Use cookies file
    cookies_file = os.getenv("YOUTUBE_COOKIES")
    if cookies_file:
        cookies_path = Path(cookies_file).expanduser()
        if cookies_path.exists():
            cookie_config["cookiefile"] = str(cookies_path)
            logger.info(f"Using cookies file: {cookies_path}")
        else:
            logger.warning(f"Cookies file not found: {cookies_path}")
    
    # Method 2: Extract cookies from browser (takes precedence if both are set)
    browser = os.getenv("YOUTUBE_COOKIES_FROM_BROWSER")
    if browser:
        # Supported browsers: chrome, chromium, firefox, opera, edge, safari, brave, vivaldi
        browser_lower = browser.lower()
        cookie_config["cookiesfrombrowser"] = (browser_lower,)
        logger.info(f"Using cookies from browser: {browser_lower}")
    
    return cookie_config


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
    
    # Get cookie configuration
    cookie_config = get_cookie_config()
    
    ydl_opts = {
        "format": format_pref,
        "outtmpl": os.path.join(output_dir, "%(id)s.%(ext)s"),
        "quiet": False,
        "no_warnings": False,
        "extract_flat": False,
        "postprocessors": [],  # Disable postprocessors to avoid ffmpeg conversion issues
        "prefer_insecure": False,
    }
    
    # Add cookie configuration if available
    ydl_opts.update(cookie_config)
    
    duration = 0.0
    downloaded_file_path = None
    
    def _download():
        """Synchronous download function to run in executor."""
        nonlocal duration, downloaded_file_path
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            logger.info(f"Downloading YouTube video: {url}")
            
            try:
                info = ydl.extract_info(url, download=True)
            except yt_dlp.utils.PostProcessingError as e:
                # If postprocessing fails but download succeeded, try to find the file
                logger.warning(f"Postprocessing failed, but download may have succeeded: {str(e)}")
                # Extract video ID from URL to find the downloaded file
                match = YOUTUBE_URL_PATTERN.match(url)
                if match:
                    video_id = match.group(1)
                    # Try to find the downloaded file with various extensions
                    for ext in ["mp4", "webm", "mkv", "m4a"]:
                        potential_path = os.path.join(output_dir, f"{video_id}.{ext}")
                        if os.path.exists(potential_path):
                            downloaded_file_path = potential_path
                            logger.info(f"Found downloaded file despite postprocessing error: {downloaded_file_path}")
                            # Try to get duration from file or use a default
                            try:
                                import subprocess
                                result = subprocess.run(
                                    ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", downloaded_file_path],
                                    capture_output=True,
                                    text=True,
                                    timeout=10
                                )
                                if result.returncode == 0:
                                    duration = float(result.stdout.strip())
                                else:
                                    duration = 0.0
                            except Exception:
                                duration = 0.0
                            return downloaded_file_path, duration
                raise Exception(f"Postprocessing failed and could not locate downloaded file: {str(e)}")
            
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
        error_str = str(e)
        logger.warning(f"yt-dlp download error: {error_str}")
        
        # If postprocessing failed but file might exist, try to find it
        if "Postprocessing" in error_str or "Conversion failed" in error_str:
            match = YOUTUBE_URL_PATTERN.match(url)
            if match:
                video_id = match.group(1)
                # Check for downloaded file with various extensions
                for ext in ["mp4", "webm", "mkv", "m4a", "ts"]:
                    potential_path = os.path.join(output_dir, f"{video_id}.{ext}")
                    if os.path.exists(potential_path):
                        file_size = os.path.getsize(potential_path)
                        if file_size > 0:  # File exists and has content
                            logger.info(f"Using downloaded file despite postprocessing error: {potential_path}")
                            downloaded_file_path = potential_path
                            # Try to get duration using ffprobe
                            try:
                                import subprocess
                                result = subprocess.run(
                                    [
                                        "ffprobe", "-v", "error", "-show_entries",
                                        "format=duration", "-of",
                                        "default=noprint_wrappers=1:nokey=1",
                                        downloaded_file_path
                                    ],
                                    capture_output=True,
                                    text=True,
                                    timeout=10
                                )
                                if result.returncode == 0 and result.stdout.strip():
                                    duration = float(result.stdout.strip())
                                else:
                                    duration = 0.0
                            except Exception as probe_error:
                                logger.warning(f"Could not get duration from file: {probe_error}")
                                duration = 0.0
                            return downloaded_file_path, duration
        
        # If we couldn't recover, clean up and raise
        if downloaded_file_path and os.path.exists(downloaded_file_path):
            try:
                os.remove(downloaded_file_path)
            except Exception:
                pass
        raise Exception(f"Failed to download YouTube video: {error_str}")
    except Exception as e:
        logger.error(f"Error downloading YouTube video: {str(e)}", exc_info=True)
        if downloaded_file_path and os.path.exists(downloaded_file_path):
            try:
                os.remove(downloaded_file_path)
            except Exception:
                pass
        raise

