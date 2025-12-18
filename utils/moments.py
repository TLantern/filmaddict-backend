import asyncio
import logging
import os
import tempfile
import urllib.request
import uuid
from typing import List, Tuple, Optional
from uuid import UUID

import ffmpeg
from database import async_session_maker
from db import crud
from models import MomentRecord
from utils.storage import get_video_path, get_storage_instance, S3Storage

logger = logging.getLogger(__name__)

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
MOMENTS_DIR = os.path.join(UPLOAD_DIR, "moments")
THUMBNAILS_DIR = os.path.join(UPLOAD_DIR, "thumbnails")
MAX_CONCURRENT_MOMENTS = 3  # Parallel moment generation limit


def get_aspect_ratio_filter(aspect_ratio: str) -> Optional[str]:
    """Get FFmpeg filter for the given aspect ratio."""
    # Map aspect ratios to width:height
    ratios = {
        "9:16": (9, 16),   # Vertical (TikTok, Reels)
        "16:9": (16, 9),   # Horizontal (YouTube)
        "1:1": (1, 1),     # Square (Instagram)
        "4:5": (4, 5),     # Portrait (Instagram)
    }
    if aspect_ratio not in ratios:
        return None
    
    w, h = ratios[aspect_ratio]
    # Center crop to the target aspect ratio
    return f"crop=ih*{w}/{h}:ih:(iw-ih*{w}/{h})/2:0,scale=1080:-2" if w < h else f"crop=iw:iw*{h}/{w}:0:(ih-iw*{h}/{w})/2,scale=-2:1080"


def generate_moment(video_path: str, start: float, end: float, aspect_ratio: Optional[str] = None) -> Tuple[str, Optional[str]]:
    """
    Generate a video moment by trimming a segment from the video using FFmpeg.
    
    Outputs standard MP4 format (H.264 video, AAC audio).
    Works with both local file paths and HTTP URLs (S3 presigned URLs).
    Stores moment to S3 if configured, otherwise to local uploads/moments/ directory.
    
    Args:
        video_path: Full path to the source video file or HTTP URL (S3 presigned URL)
        start: Start time in seconds
        end: End time in seconds
        aspect_ratio: Optional aspect ratio (9:16, 16:9, 1:1, 4:5, original)
        
    Returns:
        Tuple of (relative_storage_path, temp_file_path)
        - relative_storage_path: Relative storage path (e.g., "moments/uuid.mp4")
        - temp_file_path: Path to temp file (for S3) or None (for local storage)
        
    Raises:
        FileNotFoundError: If video file doesn't exist (for local paths only)
        ValueError: If timestamps are invalid
        Exception: If FFmpeg moment generation fails
    """
    # Check if it's a local file path or HTTP URL (S3 presigned URL)
    is_http_url = video_path.startswith("http://") or video_path.startswith("https://")
    
    # Only check file existence for local paths, not HTTP URLs
    if not is_http_url and not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    if start < 0:
        raise ValueError(f"Start time must be non-negative, got {start}")
    
    if end <= start:
        raise ValueError(f"End time must be greater than start time, got start={start}, end={end}")
    
    duration = end - start
    
    storage = get_storage_instance()
    moment_filename = f"{uuid.uuid4()}.mp4"
    
    # Create temp file for FFmpeg output
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_moment_path = temp_file.name
    temp_file.close()
    
    try:
        # Get aspect ratio filter if needed
        vf_filter = get_aspect_ratio_filter(aspect_ratio) if aspect_ratio and aspect_ratio != "original" else None
        
        logger.info(f"Generating moment from {video_path} at {start:.2f}s to {end:.2f}s (aspect_ratio={aspect_ratio})")
        
        stream = ffmpeg.input(video_path, ss=start, t=duration)
        
        if vf_filter:
            # Re-encode with aspect ratio filter
            stream = ffmpeg.output(
                stream,
                temp_moment_path,
                vf=vf_filter,
                vcodec="libx264",
                acodec="aac",
                preset="fast",
                crf=23,
                **{"movflags": "faststart"},
            )
        else:
            # Use copy codec for faster extraction (no re-encoding)
            stream = ffmpeg.output(
                stream,
                temp_moment_path,
                vcodec="copy",
                acodec="copy",
                **{"movflags": "faststart", "avoid_negative_ts": "make_zero"},
            )
        ffmpeg.run(stream, overwrite_output=True, quiet=True)
        
        if not os.path.exists(temp_moment_path):
            raise Exception(f"Moment generation failed: output file not created")
        
        # Upload to S3 if using S3Storage, otherwise use local path
        if isinstance(storage, S3Storage):
            relative_path = storage.store_moment_from_file(temp_moment_path, moment_filename)
            # Don't clean up temp file yet - it will be used for thumbnail generation
            logger.info(f"Successfully generated moment: {relative_path}")
            return (relative_path, temp_moment_path)
        else:
            # Local storage - move to moments directory
            os.makedirs(MOMENTS_DIR, exist_ok=True)
            moment_path = os.path.join(MOMENTS_DIR, moment_filename)
            os.rename(temp_moment_path, moment_path)
            relative_path = os.path.join("moments", moment_filename)
            logger.info(f"Successfully generated moment: {relative_path}")
            return (relative_path, None)
        
    except ffmpeg.Error as e:
        error_message = e.stderr.decode() if e.stderr else str(e)
        logger.error(f"FFmpeg error generating moment: {error_message}")
        if os.path.exists(temp_moment_path):
            try:
                os.remove(temp_moment_path)
            except:
                pass
        raise Exception(f"Failed to generate moment: {error_message}")
    except Exception as e:
        logger.error(f"Error generating moment: {str(e)}", exc_info=True)
        if os.path.exists(temp_moment_path):
            try:
                os.remove(temp_moment_path)
            except:
                pass
        raise


async def generate_moment_async(video_path: str, start: float, end: float, aspect_ratio: Optional[str] = None) -> Tuple[str, Optional[str]]:
    """
    Async wrapper for generate_moment function.
    
    Args:
        video_path: Full path to the source video file
        start: Start time in seconds
        end: End time in seconds
        aspect_ratio: Optional aspect ratio (9:16, 16:9, 1:1, 4:5, original)
        
    Returns:
        Tuple of (relative_storage_path, temp_file_path)
        - relative_storage_path: Relative storage path (e.g., "moments/uuid.mp4")
        - temp_file_path: Path to temp file (for S3) or None (for local storage)
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, generate_moment, video_path, start, end, aspect_ratio)


def generate_thumbnail(moment_path: str, time_offset: float = 0.0) -> str:
    """
    Generate a thumbnail image by extracting a frame from a video moment using FFmpeg.
    
    Extracts a frame at the specified time offset and saves it as a JPEG image.
    Stores thumbnail to S3 if configured, otherwise to local uploads/thumbnails/ directory.
    
    Args:
        moment_path: Full path to the source moment file (may be local or S3 presigned URL)
        time_offset: Time offset in seconds from the start of the moment (default: 0.0)
        
    Returns:
        Relative storage path (e.g., "thumbnails/uuid.jpg")
        
    Raises:
        FileNotFoundError: If moment file doesn't exist
        ValueError: If time offset is invalid
        Exception: If FFmpeg thumbnail generation fails
    """
    # If moment_path is an S3 presigned URL, download to temp file first
    storage = get_storage_instance()
    temp_moment_path = None
    
    if isinstance(storage, S3Storage) and moment_path.startswith("http"):
        # Download from S3 to temp file for FFmpeg processing
        temp_moment_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_moment_path = temp_moment_file.name
        temp_moment_file.close()
        
        urllib.request.urlretrieve(moment_path, temp_moment_path)
        actual_moment_path = temp_moment_path
    else:
        if not os.path.exists(moment_path):
            raise FileNotFoundError(f"Moment file not found: {moment_path}")
        actual_moment_path = moment_path
    
    if time_offset < 0:
        raise ValueError(f"Time offset must be non-negative, got {time_offset}")
    
    thumbnail_filename = f"{uuid.uuid4()}.jpg"
    
    # Create temp file for FFmpeg output
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    temp_thumbnail_path = temp_file.name
    temp_file.close()
    
    try:
        logger.info(f"Generating thumbnail from {actual_moment_path} at {time_offset:.2f}s")
        
        stream = ffmpeg.input(actual_moment_path, ss=time_offset)
        stream = ffmpeg.output(stream, temp_thumbnail_path, vframes=1, q=2)
        ffmpeg.run(stream, overwrite_output=True, quiet=True)
        
        if not os.path.exists(temp_thumbnail_path):
            raise Exception(f"Thumbnail generation failed: output file not created")
        
        # Upload to S3 if using S3Storage, otherwise use local path
        if isinstance(storage, S3Storage):
            relative_path = storage.store_thumbnail_from_file(temp_thumbnail_path, thumbnail_filename)
            # Clean up temp files after upload
            try:
                os.remove(temp_thumbnail_path)
            except Exception as e:
                logger.warning(f"Failed to clean up temp thumbnail file: {str(e)}")
        else:
            # Local storage - move to thumbnails directory
            os.makedirs(THUMBNAILS_DIR, exist_ok=True)
            thumbnail_path = os.path.join(THUMBNAILS_DIR, thumbnail_filename)
            os.rename(temp_thumbnail_path, thumbnail_path)
            relative_path = os.path.join("thumbnails", thumbnail_filename)
        
        logger.info(f"Successfully generated thumbnail: {relative_path}")
        return relative_path
        
    except ffmpeg.Error as e:
        error_message = e.stderr.decode() if e.stderr else str(e)
        logger.error(f"FFmpeg error generating thumbnail: {error_message}")
        if os.path.exists(temp_thumbnail_path):
            try:
                os.remove(temp_thumbnail_path)
            except:
                pass
        if temp_moment_path and os.path.exists(temp_moment_path):
            try:
                os.remove(temp_moment_path)
            except:
                pass
        raise Exception(f"Failed to generate thumbnail: {error_message}")
    except Exception as e:
        logger.error(f"Error generating thumbnail: {str(e)}", exc_info=True)
        if os.path.exists(temp_thumbnail_path):
            try:
                os.remove(temp_thumbnail_path)
            except:
                pass
        if temp_moment_path and os.path.exists(temp_moment_path):
            try:
                os.remove(temp_moment_path)
            except:
                pass
        raise
    finally:
        # Clean up temp moment file if we downloaded it
        if temp_moment_path and os.path.exists(temp_moment_path):
            try:
                os.remove(temp_moment_path)
            except:
                pass


async def generate_thumbnail_async(moment_path: str, time_offset: float = 0.0) -> str:
    """
    Async wrapper for generate_thumbnail function.
    
    Args:
        moment_path: Full path to the source moment file
        time_offset: Time offset in seconds from the start of the moment (default: 0.0)
        
    Returns:
        Relative storage path (e.g., "thumbnails/uuid.jpg")
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, generate_thumbnail, moment_path, time_offset)


async def delete_moment_file(storage_path: str, thumbnail_path: Optional[str] = None) -> None:
    """
    Delete a moment file and optionally its thumbnail from storage.
    
    Handles both S3 and local storage.
    
    Args:
        storage_path: Relative storage path of the moment file
        thumbnail_path: Optional relative storage path of the thumbnail file
    """
    storage = get_storage_instance()
    
    try:
        if isinstance(storage, S3Storage):
            storage.delete_video(storage_path)
            if thumbnail_path:
                storage.delete_video(thumbnail_path)
        else:
            upload_dir = os.getenv("UPLOAD_DIR", "./uploads")
            moment_full_path = os.path.join(upload_dir, storage_path)
            if os.path.exists(moment_full_path):
                os.remove(moment_full_path)
            
            if thumbnail_path:
                thumbnail_full_path = os.path.join(upload_dir, thumbnail_path)
                if os.path.exists(thumbnail_full_path):
                    os.remove(thumbnail_full_path)
        
        logger.info(f"Deleted moment file: {storage_path}")
        if thumbnail_path:
            logger.info(f"Deleted thumbnail file: {thumbnail_path}")
    except Exception as e:
        logger.error(f"Error deleting moment file {storage_path}: {str(e)}", exc_info=True)
        raise


async def _generate_single_moment(
    video_path: str,
    highlight,
    video_id: UUID,
    semaphore: asyncio.Semaphore,
    aspect_ratio: Optional[str] = None,
) -> Optional[Tuple[float, float, str, Optional[str]]]:
    """Generate a single moment with thumbnail (runs in parallel)."""
    async with semaphore:
        temp_moment_path = None
        try:
            storage_path, temp_moment_path = await generate_moment_async(video_path, highlight.start, highlight.end, aspect_ratio)
            
            thumbnail_path = None
            try:
                if temp_moment_path and os.path.exists(temp_moment_path):
                    thumbnail_path = await generate_thumbnail_async(temp_moment_path, time_offset=0.0)
                else:
                    moment_full_path = os.path.join(UPLOAD_DIR, storage_path)
                    if os.path.exists(moment_full_path):
                        thumbnail_path = await generate_thumbnail_async(moment_full_path, time_offset=0.0)
            except Exception as e:
                logger.warning(f"Failed to generate thumbnail for moment: {str(e)}")
            finally:
                if temp_moment_path and os.path.exists(temp_moment_path):
                    try:
                        os.remove(temp_moment_path)
                    except:
                        pass
            
            logger.info(f"Generated moment [{highlight.start:.2f}-{highlight.end:.2f}s] for video {video_id}")
            return (highlight.start, highlight.end, storage_path, thumbnail_path)
            
        except Exception as e:
            logger.error(f"Failed to generate moment [{highlight.start:.2f}-{highlight.end:.2f}s]: {str(e)}")
            if temp_moment_path and os.path.exists(temp_moment_path):
                try:
                    os.remove(temp_moment_path)
                except:
                    pass
            return None


async def generate_moments_for_video(video_id: UUID) -> List[MomentRecord]:
    """
    Generate moments for all highlights of a video (PARALLEL processing).
    """
    async with async_session_maker() as db:
        video_path = await get_video_path(video_id, db, download_local=False)
        if not video_path:
            raise ValueError(f"Video not found: {video_id}")
        
        # Get video's aspect ratio
        video = await crud.get_video_by_id(db, video_id)
        aspect_ratio = video.aspect_ratio if video else None
        
        highlights = await crud.get_highlights_by_video_id(db, video_id)
        
        if not highlights:
            logger.info(f"No highlights found for video {video_id}, skipping moment generation")
            return []
        
        logger.info(f"Generating {len(highlights)} moments for video {video_id} (parallel, aspect_ratio={aspect_ratio})")
        
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_MOMENTS)
        tasks = [_generate_single_moment(video_path, h, video_id, semaphore, aspect_ratio) for h in highlights]
        results = await asyncio.gather(*tasks)
        
        created_moments = []
        for result in results:
            if result is None:
                continue
            start, end, storage_path, thumbnail_path = result
            try:
                moment = await crud.create_moment(
                    db=db, video_id=video_id, start=start, end=end,
                    storage_path=storage_path, thumbnail_path=thumbnail_path,
                )
                logger.info(f"[Moments] Successfully saved moment {moment.id} to DB for video {video_id} ({start:.2f}s-{end:.2f}s)")
                created_moments.append(MomentRecord(
                    id=moment.id, video_id=moment.video_id, start=moment.start, end=moment.end,
                    storage_path=moment.storage_path, thumbnail_path=moment.thumbnail_path,
                ))
            except Exception as e:
                logger.error(f"[Moments] Failed to save moment to DB for video {video_id}: {str(e)}")
        
        logger.info(f"[Moments] Moment generation completed for video {video_id}: {len(created_moments)} successful, {len(highlights) - len(created_moments)} failed")
        return created_moments
