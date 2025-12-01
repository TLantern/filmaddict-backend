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
from models import ClipRecord
from utils.storage import get_video_path, get_storage_instance, S3Storage

logger = logging.getLogger(__name__)

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
CLIPS_DIR = os.path.join(UPLOAD_DIR, "clips")
THUMBNAILS_DIR = os.path.join(UPLOAD_DIR, "thumbnails")
MAX_CONCURRENT_CLIPS = 3  # Parallel clip generation limit


def generate_clip(video_path: str, start: float, end: float) -> Tuple[str, Optional[str]]:
    """
    Generate a video clip by trimming a segment from the video using FFmpeg.
    
    Outputs standard MP4 format (H.264 video, AAC audio).
    Works with both local file paths and HTTP URLs (S3 presigned URLs).
    Stores clip to S3 if configured, otherwise to local uploads/clips/ directory.
    
    Args:
        video_path: Full path to the source video file or HTTP URL (S3 presigned URL)
        start: Start time in seconds
        end: End time in seconds
        
    Returns:
        Tuple of (relative_storage_path, temp_file_path)
        - relative_storage_path: Relative storage path (e.g., "clips/uuid.mp4")
        - temp_file_path: Path to temp file (for S3) or None (for local storage)
        
    Raises:
        FileNotFoundError: If video file doesn't exist (for local paths only)
        ValueError: If timestamps are invalid
        Exception: If FFmpeg clipping fails
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
    clip_filename = f"{uuid.uuid4()}.mp4"
    
    # Create temp file for FFmpeg output
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_clip_path = temp_file.name
    temp_file.close()
    
    try:
        logger.info(f"Generating clip from {video_path} at {start:.2f}s to {end:.2f}s")
        
        # Use copy codec for 10x faster extraction (no re-encoding)
        stream = ffmpeg.input(video_path, ss=start, t=duration)
        stream = ffmpeg.output(
            stream,
            temp_clip_path,
            vcodec="copy",
            acodec="copy",
            **{"movflags": "faststart", "avoid_negative_ts": "make_zero"},
        )
        ffmpeg.run(stream, overwrite_output=True, quiet=True)
        
        if not os.path.exists(temp_clip_path):
            raise Exception(f"Clip generation failed: output file not created")
        
        # Upload to S3 if using S3Storage, otherwise use local path
        if isinstance(storage, S3Storage):
            relative_path = storage.store_clip_from_file(temp_clip_path, clip_filename)
            # Don't clean up temp file yet - it will be used for thumbnail generation
            logger.info(f"Successfully generated clip: {relative_path}")
            return (relative_path, temp_clip_path)
        else:
            # Local storage - move to clips directory
            os.makedirs(CLIPS_DIR, exist_ok=True)
            clip_path = os.path.join(CLIPS_DIR, clip_filename)
            os.rename(temp_clip_path, clip_path)
            relative_path = os.path.join("clips", clip_filename)
            logger.info(f"Successfully generated clip: {relative_path}")
            return (relative_path, None)
        
    except ffmpeg.Error as e:
        error_message = e.stderr.decode() if e.stderr else str(e)
        logger.error(f"FFmpeg error generating clip: {error_message}")
        if os.path.exists(temp_clip_path):
            try:
                os.remove(temp_clip_path)
            except:
                pass
        raise Exception(f"Failed to generate clip: {error_message}")
    except Exception as e:
        logger.error(f"Error generating clip: {str(e)}", exc_info=True)
        if os.path.exists(temp_clip_path):
            try:
                os.remove(temp_clip_path)
            except:
                pass
        raise


async def generate_clip_async(video_path: str, start: float, end: float) -> Tuple[str, Optional[str]]:
    """
    Async wrapper for generate_clip function.
    
    Args:
        video_path: Full path to the source video file
        start: Start time in seconds
        end: End time in seconds
        
    Returns:
        Tuple of (relative_storage_path, temp_file_path)
        - relative_storage_path: Relative storage path (e.g., "clips/uuid.mp4")
        - temp_file_path: Path to temp file (for S3) or None (for local storage)
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, generate_clip, video_path, start, end)


def generate_thumbnail(clip_path: str, time_offset: float = 0.0) -> str:
    """
    Generate a thumbnail image by extracting a frame from a video clip using FFmpeg.
    
    Extracts a frame at the specified time offset and saves it as a JPEG image.
    Stores thumbnail to S3 if configured, otherwise to local uploads/thumbnails/ directory.
    
    Args:
        clip_path: Full path to the source clip file (may be local or S3 presigned URL)
        time_offset: Time offset in seconds from the start of the clip (default: 0.0)
        
    Returns:
        Relative storage path (e.g., "thumbnails/uuid.jpg")
        
    Raises:
        FileNotFoundError: If clip file doesn't exist
        ValueError: If time offset is invalid
        Exception: If FFmpeg thumbnail generation fails
    """
    # If clip_path is an S3 presigned URL, download to temp file first
    storage = get_storage_instance()
    temp_clip_path = None
    
    if isinstance(storage, S3Storage) and clip_path.startswith("http"):
        # Download from S3 to temp file for FFmpeg processing
        temp_clip_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_clip_path = temp_clip_file.name
        temp_clip_file.close()
        
        urllib.request.urlretrieve(clip_path, temp_clip_path)
        actual_clip_path = temp_clip_path
    else:
        if not os.path.exists(clip_path):
            raise FileNotFoundError(f"Clip file not found: {clip_path}")
        actual_clip_path = clip_path
    
    if time_offset < 0:
        raise ValueError(f"Time offset must be non-negative, got {time_offset}")
    
    thumbnail_filename = f"{uuid.uuid4()}.jpg"
    
    # Create temp file for FFmpeg output
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    temp_thumbnail_path = temp_file.name
    temp_file.close()
    
    try:
        logger.info(f"Generating thumbnail from {actual_clip_path} at {time_offset:.2f}s")
        
        stream = ffmpeg.input(actual_clip_path, ss=time_offset)
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
        if temp_clip_path and os.path.exists(temp_clip_path):
            try:
                os.remove(temp_clip_path)
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
        if temp_clip_path and os.path.exists(temp_clip_path):
            try:
                os.remove(temp_clip_path)
            except:
                pass
        raise
    finally:
        # Clean up temp clip file if we downloaded it
        if temp_clip_path and os.path.exists(temp_clip_path):
            try:
                os.remove(temp_clip_path)
            except:
                pass


async def generate_thumbnail_async(clip_path: str, time_offset: float = 0.0) -> str:
    """
    Async wrapper for generate_thumbnail function.
    
    Args:
        clip_path: Full path to the source clip file
        time_offset: Time offset in seconds from the start of the clip (default: 0.0)
        
    Returns:
        Relative storage path (e.g., "thumbnails/uuid.jpg")
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, generate_thumbnail, clip_path, time_offset)


async def delete_clip_file(storage_path: str, thumbnail_path: Optional[str] = None) -> None:
    """
    Delete a clip file and optionally its thumbnail from storage.
    
    Handles both S3 and local storage.
    
    Args:
        storage_path: Relative storage path of the clip file
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
            clip_full_path = os.path.join(upload_dir, storage_path)
            if os.path.exists(clip_full_path):
                os.remove(clip_full_path)
            
            if thumbnail_path:
                thumbnail_full_path = os.path.join(upload_dir, thumbnail_path)
                if os.path.exists(thumbnail_full_path):
                    os.remove(thumbnail_full_path)
        
        logger.info(f"Deleted clip file: {storage_path}")
        if thumbnail_path:
            logger.info(f"Deleted thumbnail file: {thumbnail_path}")
    except Exception as e:
        logger.error(f"Error deleting clip file {storage_path}: {str(e)}", exc_info=True)
        raise


async def _generate_single_clip(
    video_path: str,
    highlight,
    video_id: UUID,
    semaphore: asyncio.Semaphore,
) -> Optional[Tuple[float, float, str, Optional[str]]]:
    """Generate a single clip with thumbnail (runs in parallel)."""
    async with semaphore:
        temp_clip_path = None
        try:
            storage_path, temp_clip_path = await generate_clip_async(video_path, highlight.start, highlight.end)
            
            thumbnail_path = None
            try:
                if temp_clip_path and os.path.exists(temp_clip_path):
                    thumbnail_path = await generate_thumbnail_async(temp_clip_path, time_offset=0.0)
                else:
                    clip_full_path = os.path.join(UPLOAD_DIR, storage_path)
                    if os.path.exists(clip_full_path):
                        thumbnail_path = await generate_thumbnail_async(clip_full_path, time_offset=0.0)
            except Exception as e:
                logger.warning(f"Failed to generate thumbnail for clip: {str(e)}")
            finally:
                if temp_clip_path and os.path.exists(temp_clip_path):
                    try:
                        os.remove(temp_clip_path)
                    except:
                        pass
            
            logger.info(f"Generated clip [{highlight.start:.2f}-{highlight.end:.2f}s] for video {video_id}")
            return (highlight.start, highlight.end, storage_path, thumbnail_path)
            
        except Exception as e:
            logger.error(f"Failed to generate clip [{highlight.start:.2f}-{highlight.end:.2f}s]: {str(e)}")
            if temp_clip_path and os.path.exists(temp_clip_path):
                try:
                    os.remove(temp_clip_path)
                except:
                    pass
            return None


async def generate_clips_for_video(video_id: UUID) -> List[ClipRecord]:
    """
    Generate clips for all highlights of a video (PARALLEL processing).
    """
    async with async_session_maker() as db:
        video_path = await get_video_path(video_id, db, download_local=False)
        if not video_path:
            raise ValueError(f"Video not found: {video_id}")
        
        highlights = await crud.get_highlights_by_video_id(db, video_id)
        
        if not highlights:
            logger.info(f"No highlights found for video {video_id}, skipping clip generation")
            return []
        
        logger.info(f"Generating {len(highlights)} clips for video {video_id} (parallel)")
        
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_CLIPS)
        tasks = [_generate_single_clip(video_path, h, video_id, semaphore) for h in highlights]
        results = await asyncio.gather(*tasks)
        
        created_clips = []
        for result in results:
            if result is None:
                continue
            start, end, storage_path, thumbnail_path = result
            try:
                clip = await crud.create_clip(
                    db=db, video_id=video_id, start=start, end=end,
                    storage_path=storage_path, thumbnail_path=thumbnail_path,
                )
                created_clips.append(ClipRecord(
                    id=clip.id, video_id=clip.video_id, start=clip.start, end=clip.end,
                    storage_path=clip.storage_path, thumbnail_path=clip.thumbnail_path,
                ))
            except Exception as e:
                logger.error(f"Failed to save clip to DB: {str(e)}")
        
        logger.info(f"Clip generation completed: {len(created_clips)} successful, {len(highlights) - len(created_clips)} failed")
        return created_clips

