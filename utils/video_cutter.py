import logging
import os
import shutil
import tempfile
import uuid
from typing import List, Tuple
from uuid import UUID

import ffmpeg
from sqlalchemy.ext.asyncio import AsyncSession

from db import crud
from utils.storage import get_storage_instance, S3Storage, get_video_path
from utils.preview_cache import get_preview_file_path, set_preview_path

logger = logging.getLogger(__name__)

# Check if FFmpeg is available
def check_ffmpeg_available() -> bool:
    """Check if FFmpeg is installed and available in PATH."""
    return shutil.which("ffmpeg") is not None

if not check_ffmpeg_available():
    logger.warning(
        "FFmpeg not found in PATH. Video cutting will fail. "
        "Please install FFmpeg and add it to your system PATH. "
        "Download from: https://ffmpeg.org/download.html"
    )


def cut_segments_from_video(
    video_path: str,
    segments_to_remove: List[Tuple[float, float]],
    output_path: str,
    use_stream_copy: bool = True,
) -> None:
    """
    Cut specified segments from a video using FFmpeg.
    
    Creates a new video file that excludes the specified time ranges.
    
    Args:
        video_path: Full path to the source video file or HTTP URL (S3 presigned URL)
        segments_to_remove: List of (start_time, end_time) tuples to remove
        output_path: Path where the cut video will be saved
        use_stream_copy: If True, use fast stream copy (may not be frame-accurate).
                        If False, re-encode for precise cuts (much slower).
        
    Raises:
        ValueError: If segments are invalid
        FileNotFoundError: If FFmpeg is not installed or not in PATH
        Exception: If FFmpeg processing fails
    """
    # Check if FFmpeg is available before attempting to use it
    if not check_ffmpeg_available():
        raise FileNotFoundError(
            "FFmpeg is not installed or not found in PATH. "
            "Please install FFmpeg from https://ffmpeg.org/download.html "
            "and add it to your system PATH."
        )
    
    # Check if it's a local file path or HTTP URL (S3 presigned URL)
    is_http_url = video_path.startswith("http://") or video_path.startswith("https://")
    
    # Only check file existence for local paths, not HTTP URLs
    if not is_http_url and not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    if not segments_to_remove:
        raise ValueError("No segments to remove")
    
    # Sort segments by start time
    sorted_segments = sorted(segments_to_remove, key=lambda x: x[0])
    
    # Merge overlapping or adjacent segments
    merged_segments = []
    for start, end in sorted_segments:
        if start < 0:
            raise ValueError(f"Segment start time must be non-negative, got {start}")
        if end <= start:
            raise ValueError(f"Segment end time must be greater than start time, got start={start}, end={end}")
        
        if not merged_segments:
            merged_segments.append((start, end))
        else:
            prev_start, prev_end = merged_segments[-1]
            # If current segment overlaps or is adjacent to previous, merge them
            if start <= prev_end:
                merged_segments[-1] = (prev_start, max(prev_end, end))
            else:
                merged_segments.append((start, end))
    
    sorted_segments = merged_segments
    
    # Get video duration (needed to determine last segment)
    try:
        probe = ffmpeg.probe(video_path)
        video_info = next(s for s in probe['streams'] if s.get('codec_type') == 'video')
        video_duration = float(probe['format'].get('duration', video_info.get('duration', sorted_segments[-1][1] + 10)))
    except Exception as e:
        logger.warning(f"Could not probe video duration, using segments to infer: {str(e)}")
        video_duration = sorted_segments[-1][1] + 10  # Estimate
    
    # Calculate segments to keep (parts between removed segments)
    keep_segments = []
    current_time = 0.0
    
    for start, end in sorted_segments:
        if current_time < start:
            # Keep the segment from current_time to start
            keep_segments.append((current_time, start))
        current_time = max(current_time, end)
    
    # Keep the remaining part after the last removed segment
    if current_time < video_duration:
        keep_segments.append((current_time, video_duration))
    
    if not keep_segments:
        raise ValueError("All video would be removed - cannot create empty video")
    
    logger.info(f"Cutting video: removing {len(sorted_segments)} segments, keeping {len(keep_segments)} segments (stream_copy={use_stream_copy})")
    
    # Common input options for error tolerance
    input_kwargs = {
        "err_detect": "ignore_err",
        "fflags": "+genpts+igndts+discardcorrupt",
    }
    
    # Output options depend on whether we use stream copy or re-encode
    if use_stream_copy:
        # Fast: just copy streams (not frame-accurate, but instant)
        output_kwargs = {
            "c": "copy",
            "movflags": "faststart",
            "avoid_negative_ts": "make_zero",
        }
    else:
        # Slow: re-encode for precise cuts
        output_kwargs = {
            "vcodec": "libx264",
            "acodec": "aac",
            "preset": "fast",
            "crf": 23,
            "movflags": "faststart",
            "max_muxing_queue_size": "1024",
        }
    
    try:
        # If only one segment to keep, use simple trim
        if len(keep_segments) == 1:
            start, end = keep_segments[0]
            duration = end - start
            stream = ffmpeg.input(video_path, ss=start, t=duration, **input_kwargs)
            stream = ffmpeg.output(stream, output_path, **output_kwargs)
            ffmpeg.run(stream, overwrite_output=True, quiet=True, capture_stderr=True)
        else:
            # Multiple segments - use concat demuxer
            temp_files = []
            segment_paths = []
            
            try:
                for i, (start, end) in enumerate(keep_segments):
                    duration = end - start
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f"_segment_{i}.mp4")
                    temp_files.append(temp_file.name)
                    temp_file.close()
                    
                    # Extract segment
                    segment_stream = ffmpeg.input(video_path, ss=start, t=duration, **input_kwargs)
                    segment_stream = ffmpeg.output(segment_stream, temp_files[-1], **output_kwargs)
                    ffmpeg.run(segment_stream, overwrite_output=True, quiet=True, capture_stderr=True)
                    
                    # Validate segment file
                    if os.path.exists(temp_files[-1]) and os.path.getsize(temp_files[-1]) > 1000:
                        segment_paths.append(temp_files[-1])
                    else:
                        logger.warning(f"Segment {i} failed to generate valid output")
                
                if not segment_paths:
                    raise Exception("No valid segments were extracted")
                
                # Create concat file
                concat_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
                for segment_path in segment_paths:
                    concat_file.write(f"file '{segment_path}'\n")
                concat_file.close()
                
                # Concatenate segments (always use stream copy for concat - segments already processed)
                concat_stream = ffmpeg.input(concat_file.name, format='concat', safe=0)
                concat_stream = ffmpeg.output(
                    concat_stream,
                    output_path,
                    c="copy",
                    movflags="faststart",
                )
                ffmpeg.run(concat_stream, overwrite_output=True, quiet=True, capture_stderr=True)
                
                # Clean up concat file
                if os.path.exists(concat_file.name):
                    os.remove(concat_file.name)
                    
            finally:
                # Clean up temporary segment files
                for temp_file in temp_files:
                    if os.path.exists(temp_file):
                        try:
                            os.remove(temp_file)
                        except Exception as e:
                            logger.warning(f"Failed to clean up temp file {temp_file}: {str(e)}")
        
        if not os.path.exists(output_path):
            raise Exception(f"Video cutting failed: output file not created")
        
        logger.info(f"Successfully cut video, output saved to: {output_path}")
        
    except ffmpeg.Error as e:
        error_message = e.stderr.decode() if e.stderr else str(e)
        logger.error(f"FFmpeg error cutting video: {error_message}")
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except:
                pass
        raise Exception(f"Failed to cut video: {error_message}")
    except Exception as e:
        logger.error(f"Error cutting video: {str(e)}", exc_info=True)
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except:
                pass
        raise


async def cut_segments_from_video_async(
    video_path: str,
    segments_to_remove: List[Tuple[float, float]],
    output_path: str,
    use_stream_copy: bool = True,
    timeout_seconds: int = 120,
) -> None:
    """
    Async wrapper for cut_segments_from_video function with timeout.
    
    Args:
        use_stream_copy: If True (default), use fast stream copy. If False, re-encode.
        timeout_seconds: Max time to wait (default 120s, should be fast with stream copy)
    """
    import asyncio
    from functools import partial
    
    loop = asyncio.get_event_loop()
    func = partial(cut_segments_from_video, video_path, segments_to_remove, output_path, use_stream_copy)
    try:
        await asyncio.wait_for(
            loop.run_in_executor(None, func),
            timeout=timeout_seconds
        )
    except asyncio.TimeoutError:
        logger.error(f"FFmpeg timed out after {timeout_seconds}s cutting video")
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except:
                pass
        raise Exception(f"Video cutting timed out after {timeout_seconds} seconds")


async def process_video_cutting(
    video_id: UUID,
    segments_to_remove: List[Tuple[float, float]],
    db: AsyncSession,
) -> str:
    """
    Process video cutting by removing specified segments and replacing the original video.
    
    Args:
        video_id: UUID of the video to cut
        segments_to_remove: List of (start_time, end_time) tuples to remove
        db: Database session
        
    Returns:
        New storage path for the cut video
        
    Raises:
        ValueError: If video not found or segments are invalid
        Exception: If cutting fails
    """
    # Get video
    video = await crud.get_video_by_id(db, video_id)
    if not video:
        raise ValueError(f"Video not found: {video_id}")
    
    # Get video path (uses cache if available)
    video_path = await get_video_path(video_id, db, download_local=True)
    if not video_path:
        raise ValueError(f"Video file not found: {video_id}")
    
    storage = get_storage_instance()
    
    # Create temp file for output
    temp_output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_output_path = temp_output_file.name
    temp_output_file.close()
    
    original_storage_path = video.storage_path
    new_storage_path = original_storage_path
    
    try:
        # video_path is already local (from cache or local storage)
        actual_video_path = video_path
        
        # Cut segments from video
        await cut_segments_from_video_async(actual_video_path, segments_to_remove, temp_output_path)
        
        # Store the new video
        if isinstance(storage, S3Storage):
            # First, backup old video to a backup location (in case upload fails)
            old_backup_path = original_storage_path.replace("videos/", "videos/backup_")
            backup_created = False
            try:
                # Copy old file to backup (S3 doesn't have rename, so we copy then delete)
                from botocore.exceptions import ClientError
                try:
                    storage.s3_client.copy_object(
                        Bucket=storage.bucket_name,
                        CopySource={"Bucket": storage.bucket_name, "Key": original_storage_path},
                        Key=old_backup_path,
                    )
                    logger.info(f"Backed up old video to {old_backup_path}")
                    backup_created = True
                except ClientError as e:
                    logger.warning(f"Failed to backup old video (may not exist): {str(e)}")
                    backup_created = False
                
                # Upload new video
                new_filename = f"{uuid.uuid4()}.mp4"
                new_storage_path = storage.store_video_from_file(temp_output_path, new_filename)
                
                # Only delete old video after successful upload
                try:
                    storage.delete_video(original_storage_path)
                    logger.info(f"Deleted old video {original_storage_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete old video from S3: {str(e)}")
                
                # Delete backup after successful replacement
                try:
                    storage.delete_video(old_backup_path)
                    logger.info(f"Deleted backup video {old_backup_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete backup video: {str(e)}")
                    
            except Exception as e:
                logger.error(f"Error during S3 video swap: {str(e)}")
                # If something failed, try to restore from backup
                if backup_created:
                    try:
                        storage.s3_client.copy_object(
                            Bucket=storage.bucket_name,
                            CopySource={"Bucket": storage.bucket_name, "Key": old_backup_path},
                            Key=original_storage_path,
                        )
                        logger.info(f"Restored video from backup")
                    except Exception as restore_error:
                        logger.error(f"Failed to restore from backup: {str(restore_error)}")
                raise
        else:
            # Local storage - replace the old file
            import shutil
            old_full_path = storage.get_video_path(original_storage_path)
            shutil.move(temp_output_path, old_full_path)
            new_storage_path = original_storage_path
            temp_output_path = None  # File moved, don't delete
        
        # Update database with new storage path (if changed)
        if new_storage_path != original_storage_path:
            try:
                video.storage_path = new_storage_path
                await db.commit()
            except Exception as e:
                # If connection is closed, use a fresh session to update
                error_str = str(e).lower()
                if 'connection is closed' in error_str or 'interfaceerror' in error_str:
                    logger.warning(f"Database connection closed during commit, retrying with fresh session: {str(e)}")
                    from sqlalchemy import update
                    from database import async_session_maker
                    from db.models import Video
                    async with async_session_maker() as fresh_db:
                        stmt = update(Video).where(Video.id == video_id).values(storage_path=new_storage_path)
                        await fresh_db.execute(stmt)
                        await fresh_db.commit()
                        logger.info(f"Successfully updated video storage path using fresh session")
                else:
                    raise
        
        logger.info(f"Successfully cut video {video_id}, new storage path: {new_storage_path}")
        
        # Clear video cache since video was updated
        try:
            from utils.video_cache import clear_cached_video
            clear_cached_video(video_id)
        except Exception as e:
            logger.warning(f"Failed to clear video cache after cutting: {str(e)}")
        
        # Clean up any orphaned audio files after cutting
        try:
            from utils.audio import cleanup_audio_directory
            cleanup_audio_directory()
        except Exception as e:
            logger.warning(f"Failed to cleanup audio directory after video cutting: {str(e)}")
        
        return new_storage_path
        
    finally:
        # Clean up temp files
        if temp_output_path and os.path.exists(temp_output_path):
            try:
                os.remove(temp_output_path)
            except Exception as e:
                logger.warning(f"Failed to clean up temp output file: {str(e)}")


async def generate_local_preview(
    video_id: UUID,
    segments_to_remove: List[Tuple[float, float]],
    db: AsyncSession,
) -> str:
    """
    Generate a local preview version of the video with cuts applied instantly.
    This is used for instant preview before saving to S3.
    
    Args:
        video_id: UUID of the video
        segments_to_remove: List of (start_time, end_time) tuples to remove
        db: Database session
        
    Returns:
        Path to the local preview file
        
    Raises:
        ValueError: If video not found or segments are invalid
        Exception: If preview generation fails
    """
    # Get video
    video = await crud.get_video_by_id(db, video_id)
    if not video:
        raise ValueError(f"Video not found: {video_id}")
    
    if not segments_to_remove:
        # No cuts, return original video path
        storage = get_storage_instance()
        if isinstance(storage, S3Storage):
            # For S3, we still need to download locally for preview
            video_path = await get_video_path(video_id, db, download_local=True)
            preview_path = get_preview_file_path(video_id)
            if video_path and not video_path.startswith("http"):
                import shutil
                shutil.copy2(video_path, preview_path)
                set_preview_path(video_id, preview_path)
                return preview_path
        else:
            # Local storage - use original path
            return storage.get_video_path(video.storage_path)
    
    # Get video path (uses cache if available)
    video_path = await get_video_path(video_id, db, download_local=True)
    if not video_path:
        raise ValueError(f"Video file not found: {video_id}")
    
    preview_path = get_preview_file_path(video_id)
    
    # video_path is already local (from cache or local storage)
    actual_video_path = video_path
    
    # Cut segments from video to preview path
    await cut_segments_from_video_async(actual_video_path, segments_to_remove, preview_path)
    
    # Store preview path in cache
    set_preview_path(video_id, preview_path)
    
    logger.info(f"Generated local preview for video {video_id} at {preview_path}")
    return preview_path

