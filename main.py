import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import UUID

# Disable tokenizers parallelism to avoid deadlocks when forking
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Request, Header, BackgroundTasks
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession

from database import engine, Base, get_db, async_session_maker
from db.models import Video, Transcript, Highlight as HighlightDB, Moment, Timeline
from db import crud
from sqlalchemy import select
from models import (
    VideoStatus,
    TranscriptResponse,
    TranscriptSegment,
    VideoStatusResponse,
    HighlightsResponse,
    MomentResponse,
    MomentsResponse,
    Highlight,
    PromptVersionResponse,
    PromptVersionRequest,
    CalibrationConfigResponse,
    SavedMomentResponse,
    EditMomentRequest,
    MomentDetailResponse,
    ProjectResponse,
    ProjectsResponse,
    SegmentsResponse,
    SegmentAnalysis,
    CutVideoRequest,
    SegmentFeedbackRequest,
    SegmentFeedbackResponse,
)
from typing import List, Optional
from datetime import datetime
from utils.storage import store_video, get_storage_instance, S3Storage, get_video_path
from utils.youtube import download_youtube_video, validate_youtube_url
from utils.jobs import enqueue_video_processing, start_worker, stop_worker, start_learning_worker, stop_learning_worker
from utils.moments import generate_moment_async, generate_thumbnail_async, generate_video_thumbnail_async, delete_moment_file
from utils.learning_logger import (
    get_learning_log,
    get_changes_by_type,
    get_recent_changes,
    log_future_change,
    mark_future_change_completed,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_clerk_user_id(x_clerk_user_id: Optional[str] = Header(None, alias="X-Clerk-User-Id")) -> Optional[str]:
    """Extract Clerk user ID from request headers."""
    return x_clerk_user_id


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting FilmAddict Backend")
    logger.info("Initializing database connection")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database initialized")
    
    # Initialize default prompt version if none exists
    async with async_session_maker() as db:
        active_version = await crud.get_active_prompt_version(db)
        if not active_version:
            default_system_prompt = "You are an expert at identifying engaging moments in video content that would perform well as short-form clips. Return only valid JSON objects with a 'highlights' key containing an array."
            default_user_template = """Chunk time range: {start:.2f} - {end:.2f} seconds
Transcript text: {text}

Identify the most engaging, emotionally intense, or information-dense moments likely to perform well as short-form content. Return 0–2 timestamp ranges per chunk with reason and a score from 1–10.

Return your response as a JSON object with a "highlights" key containing an array. Each highlight must have: start (seconds), end (seconds), reason (string), score (1-10).
Example format:
{{
  "highlights": [
    {{
      "start": 120.5,
      "end": 145.2,
      "reason": "High emotional intensity moment with personal revelation",
      "score": 8.5
    }}
  ]
}}"""
            await crud.create_prompt_version(
                db,
                version_name="v1",
                system_prompt=default_system_prompt,
                user_prompt_template=default_user_template,
                is_active=True,
            )
            logger.info("Created default prompt version v1")
    
    # Start background workers
    logger.info("🚀 Starting background workers...")
    try:
        await start_worker()
        logger.info("✅ Video processing worker started")
    except Exception as e:
        logger.error(f"❌ Failed to start video processing worker: {e}", exc_info=True)
        raise
    
    # Learning pipeline disabled
    # try:
    #     await start_learning_worker()
    #     logger.info("✅ Learning worker started")
    # except Exception as e:
    #     logger.warning(f"⚠️  Failed to start learning worker (non-critical): {e}", exc_info=True)
    
    yield
    
    # Stop background workers
    # await stop_learning_worker()  # Learning pipeline disabled
    await stop_worker()
    logger.info("Shutting down FilmAddict Backend")
    await engine.dispose()


app = FastAPI(
    title="FilmAddict Backend",
    description="Highlight Extractor SaaS API",
    version=os.getenv("APP_VERSION", "0.1.0"),
    lifespan=lifespan,
)

cors_origins = os.getenv(
    "CORS_ORIGINS", "http://localhost:3000,http://localhost:3001"
).split(",")

# Add ngrok domains for development - allow all ngrok-free.app domains
# This allows any ngrok tunnel to work without hardcoding domains
cors_origins.extend([
    "https://26c76978b7a4.ngrok-free.app",
    "https://43271b5099b0.ngrok-free.app",
    "https://3be3763fc0ab.ngrok-free.app",
])

# Remove duplicates
cors_origins = list(set(cors_origins))

# For development, allow all ngrok domains dynamically using regex
# This allows any ngrok tunnel to work without hardcoding domains
ngrok_regex = r"https://.*\.ngrok-free\.app|https://.*\.ngrok\.io"

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_origin_regex=ngrok_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"status": "ok", "message": "YKlipp Backend API"}

@app.get("/health")
async def health_check():
    """Health check endpoint that also reports worker status."""
    from utils.jobs import _worker_task, _video_processing_queue
    
    worker_status = "running"
    queue_size = 0
    
    if _worker_task is None:
        worker_status = "not_started"
    elif _worker_task.done():
        worker_status = "stopped"
        if _worker_task.exception():
            worker_status = f"crashed: {str(_worker_task.exception())}"
    
    if _video_processing_queue is not None:
        queue_size = _video_processing_queue.qsize()
    
    # Test Redis connection
    from utils.redis_cache import test_redis_connection
    redis_status = test_redis_connection()
    
    return {
        "status": "ok",
        "worker_status": worker_status,
        "queue_size": queue_size,
        "redis": redis_status,
    }


@app.get("/version")
async def version():
    return {"version": app.version}


@app.get("/learning/log")
async def get_learning_log_endpoint(
    change_type: Optional[str] = None,
    include_previous: bool = True,
    limit: Optional[int] = None,
):
    """
    Get learning system change log.
    
    Query parameters:
    - change_type: Filter by type (calibration, prompt_version, feedback_patterns)
    - include_previous: Include previous changes (default: true)
    - limit: Limit number of results (default: all)
    """
    try:
        if change_type:
            changes = get_changes_by_type(change_type, include_previous=include_previous)
            if limit:
                changes = changes[:limit]
            return {
                "change_type": change_type,
                "changes": changes,
                "count": len(changes),
            }
        elif limit:
            changes = get_recent_changes(limit=limit)
            return {
                "recent_changes": changes,
                "count": len(changes),
            }
        else:
            log_data = get_learning_log()
            return log_data
    except Exception as e:
        logger.error(f"Error getting learning log: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get learning log: {str(e)}")


@app.post("/learning/log/future")
async def add_future_change(
    change_type: str,
    description: str,
    planned_date: Optional[str] = None,
    metadata: Optional[dict] = None,
):
    """
    Add a planned/future change to the learning log.
    
    Body:
    - change_type: Type of planned change
    - description: Description of the planned change
    - planned_date: Planned date (ISO format, optional)
    - metadata: Additional metadata (optional)
    """
    try:
        log_future_change(
            change_type=change_type,
            description=description,
            planned_date=planned_date,
            metadata=metadata,
        )
        return {"status": "success", "message": "Future change logged"}
    except Exception as e:
        logger.error(f"Error logging future change: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to log future change: {str(e)}")


ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
MAX_FILE_SIZE = 524288000  # 500MB in bytes


class YouTubeRequest(BaseModel):
    youtube_url: str = Field(..., description="YouTube video URL")
    aspect_ratio: Optional[str] = Field("16:9", description="Aspect ratio for generated clips (9:16, 16:9, 1:1, 4:5, original)")


class ExportVideoRequest(BaseModel):
    format: str = Field(..., description="Export format: mp4, mov_prores422, mov_prores4444, webm, xml, edl, aaf")
    segments_to_remove: Optional[List[dict]] = Field(None, description="Optional list of {start_time, end_time} segments to remove")


@app.post("/videos/upload", status_code=201)
async def upload_video(
    file: UploadFile = File(...),
    aspect_ratio: Optional[str] = Form("16:9"),
    db: AsyncSession = Depends(get_db),
    clerk_user_id: Optional[str] = Depends(get_clerk_user_id),
):
    """
    Upload a video file.
    
    Accepts multipart/form-data with a video file and optional aspect_ratio.
    Validates file size (max 500MB) and format (mp4, mov, mkv, avi, webm).
    Stores the file and creates a VideoRecord in the database.
    """
    try:
        # Validate aspect ratio
        valid_ratios = ["9:16", "16:9", "1:1", "4:5", "original"]
        if aspect_ratio and aspect_ratio not in valid_ratios:
            aspect_ratio = "16:9"
        
        # Validate file extension
        if not file.filename:
            raise HTTPException(status_code=400, detail="Filename is required")
        
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in ALLOWED_VIDEO_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"File format not allowed. Allowed formats: {', '.join(ALLOWED_VIDEO_EXTENSIONS)}",
            )
        
        # Validate file size
        file.file.seek(0, 2)  # Seek to end
        file_size = file.file.tell()
        file.file.seek(0)  # Reset to beginning
        
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File size exceeds maximum allowed size of 500MB",
            )
        
        if file_size == 0:
            raise HTTPException(status_code=400, detail="File is empty")
        
        # Store video file
        storage_path = await store_video(file)
        
        # Generate thumbnail from video
        thumbnail_path = None
        try:
            storage = get_storage_instance()
            video_file_path = storage.get_video_path(storage_path)
            thumbnail_path = await generate_video_thumbnail_async(video_file_path, time_offset=1.0)
            logger.info(f"Generated thumbnail for video: {thumbnail_path}")
        except Exception as e:
            logger.warning(f"Failed to generate thumbnail for video: {str(e)}")
            # Continue without thumbnail - not critical
        
        # Create video record in database
        video = await crud.create_video(
            db=db,
            storage_path=storage_path,
            status=VideoStatus.UPLOADED,
            aspect_ratio=aspect_ratio,
            clerk_user_id=clerk_user_id,
            thumbnail_path=thumbnail_path,
        )
        
        # Enqueue background job to process the video
        await enqueue_video_processing(video.id)
        
        # Update status to QUEUED
        video = await crud.update_video_status(db, video.id, VideoStatus.QUEUED)
        
        # Create timeline for this video
        await crud.create_or_update_timeline(
            db=db,
            video_id=video.id,
            clerk_user_id=clerk_user_id,
        )
        
        logger.info(f"Video uploaded successfully and queued for processing: {video.id}")
        
        return {
            "video_id": str(video.id),
            "status": "QUEUED",
            "storage_path": storage_path,
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading video: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during video upload")


@app.post("/videos/youtube", status_code=201)
async def upload_youtube_video(
    request: YouTubeRequest,
    db: AsyncSession = Depends(get_db),
    clerk_user_id: Optional[str] = Depends(get_clerk_user_id),
):
    """
    Download and store a YouTube video.
    
    Accepts a JSON body with youtube_url.
    Downloads the video using yt-dlp, stores it, and creates a VideoRecord.
    """
    import tempfile
    
    temp_dir = None
    downloaded_file_path = None
    
    try:
        # Validate YouTube URL
        if not validate_youtube_url(request.youtube_url):
            raise HTTPException(
                status_code=400,
                detail="Invalid YouTube URL format",
            )
        
        # Create temporary directory for download
        temp_dir = tempfile.mkdtemp()
        
        # Download YouTube video
        try:
            downloaded_file_path, duration = await download_youtube_video(
                request.youtube_url, temp_dir
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Error downloading YouTube video: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to download YouTube video: {str(e)}",
            )
        
        # Upload to S3 and delete immediately after
        try:
            filename = os.path.basename(downloaded_file_path)
            storage = get_storage_instance()
            
            if isinstance(storage, S3Storage):
                storage_path = storage.store_video_from_file(downloaded_file_path, filename)
            else:
                with open(downloaded_file_path, "rb") as f:
                    file_content = f.read()
                storage_path = await storage.store_video(file_content, filename)
            
            # Delete file immediately after upload
            if downloaded_file_path and os.path.exists(downloaded_file_path):
                try:
                    os.remove(downloaded_file_path)
                    logger.info(f"Deleted temp file immediately after upload: {downloaded_file_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete temp file: {str(e)}")
        except Exception as e:
            logger.error(f"Error storing downloaded video: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail="Failed to store downloaded video",
            )
        
        # Validate aspect ratio
        aspect_ratio = request.aspect_ratio
        valid_ratios = ["9:16", "16:9", "1:1", "4:5", "original"]
        if aspect_ratio and aspect_ratio not in valid_ratios:
            aspect_ratio = "16:9"
        
        # Generate thumbnail from video
        thumbnail_path = None
        try:
            storage = get_storage_instance()
            video_file_path = storage.get_video_path(storage_path)
            thumbnail_path = await generate_video_thumbnail_async(video_file_path, time_offset=1.0)
            logger.info(f"Generated thumbnail for YouTube video: {thumbnail_path}")
        except Exception as e:
            logger.warning(f"Failed to generate thumbnail for YouTube video: {str(e)}")
            # Continue without thumbnail - not critical
        
        # Create video record in database
        video = await crud.create_video(
            db=db,
            storage_path=storage_path,
            duration=duration,
            status=VideoStatus.UPLOADED,
            aspect_ratio=aspect_ratio,
            clerk_user_id=clerk_user_id,
            thumbnail_path=thumbnail_path,
        )
        
        # Enqueue background job to process the video
        await enqueue_video_processing(video.id)
        
        # Update status to QUEUED
        video = await crud.update_video_status(db, video.id, VideoStatus.QUEUED)
        
        # Create timeline for this video
        await crud.create_or_update_timeline(
            db=db,
            video_id=video.id,
            clerk_user_id=clerk_user_id,
        )
        
        logger.info(f"YouTube video processed successfully and queued for processing: {video.id}")
        
        return {
            "video_id": str(video.id),
            "status": "QUEUED",
            "storage_path": storage_path,
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing YouTube video: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Internal server error during YouTube video processing"
        )
    finally:
        # Clean up temp directory if it still exists
        if temp_dir and os.path.exists(temp_dir):
            try:
                # Remove directory if empty
                os.rmdir(temp_dir)
            except Exception as e:
                logger.warning(f"Failed to clean up temp directory: {str(e)}")


@app.get("/videos/{video_id}/transcript", response_model=TranscriptResponse)
async def get_transcript(
    video_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Retrieve transcript segments for a video.
    
    Returns transcript segments with start time, end time, and text.
    Returns 404 if transcript doesn't exist.
    """
    try:
        video_uuid = UUID(video_id)
        transcript = await crud.get_transcript_by_video_id(db, video_uuid)
        
        if not transcript:
            raise HTTPException(status_code=404, detail="Transcript not found")
        
        segments = [TranscriptSegment(**segment) for segment in transcript.segments]
        
        return TranscriptResponse(video_id=video_uuid, segments=segments)
    
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid video ID format")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving transcript: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error retrieving transcript")


@app.get("/videos/{video_id}/status", response_model=VideoStatusResponse)
async def get_video_status(
    video_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Get the current status and metadata for a video.
    
    Returns current status (UPLOADED, PROCESSING, TRANSCRIBED, HIGHLIGHTS_FOUND, DONE, FAILED),
    duration, and created_at timestamp.
    Returns 404 if video not found.
    """
    try:
        video_uuid = UUID(video_id)
        logger.info(f"[Backend] /videos/{video_id}/status endpoint called")
        video = await crud.get_video_by_id(db, video_uuid)
        
        if not video:
            logger.warning(f"[Backend] Video {video_id} not found")
            raise HTTPException(status_code=404, detail="Video not found")
        
        # Get segment counts using optimized aggregation query (much faster than loading all segments)
        segment_counts = await crud.get_segment_counts_by_label(db, video_uuid)
        segments_count = sum(segment_counts.values())
        
        # Extract counts by label (default to 0 if label not present)
        fluff_count = segment_counts.get("FLUFF", 0)
        
        logger.info(
            f"[Backend] Video {video_id} status: {video.status}, duration: {video.duration}, "
            f"segments: {segments_count} (FLUFF: {fluff_count})"
        )
        
        return VideoStatusResponse(
            video_id=video.id,
            status=VideoStatus(video.status),
            duration=video.duration,
            created_at=video.created_at,
        )
    
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid video ID format")
    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e)
        if "Authentication timed out" in error_msg or "ProtocolViolationError" in error_msg:
            logger.error(f"Database connection timeout: {error_msg}", exc_info=True)
            raise HTTPException(
                status_code=503,
                detail="Database connection timeout. Please try again in a moment."
            )
        logger.error(f"Error retrieving video status: {error_msg}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error retrieving video status")


@app.post("/videos/{video_id}/reprocess", status_code=200)
async def reprocess_video(
    video_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Manually trigger reprocessing of a video.
    Useful if processing failed or segments weren't saved.
    """
    try:
        from utils.jobs import enqueue_video_processing
        from models import VideoStatus
        
        video_uuid = UUID(video_id)
        video = await crud.get_video_by_id(db, video_uuid)
        
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")
        
        # Reset status and enqueue for processing
        await crud.update_video_status(db, video_uuid, VideoStatus.QUEUED)
        await enqueue_video_processing(video_uuid)
        
        logger.info(f"Manually triggered reprocessing for video: {video_id}")
        
        return {
            "status": "success",
            "video_id": str(video_uuid),
            "message": "Video queued for reprocessing",
        }
    
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid video ID format")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reprocessing video: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/videos/{video_id}/highlights", response_model=HighlightsResponse)
async def get_highlights(
    video_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Retrieve highlights for a video.
    
    Returns list of highlights with start, end, reason, and score.
    Returns empty list if no highlights found.
    Returns 404 if video not found.
    """
    try:
        video_uuid = UUID(video_id)
        
        # Verify video exists
        video = await crud.get_video_by_id(db, video_uuid)
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")
        
        highlights_db = await crud.get_highlights_by_video_id(db, video_uuid)
        logger.info(f"[Backend] /videos/{video_id}/highlights - Found {len(highlights_db)} highlights")
        
        # Get transcript to extract text for each highlight
        transcript = await crud.get_transcript_by_video_id(db, video_uuid)
        transcript_segments = []
        if transcript and transcript.segments:
            transcript_segments = [TranscriptSegment(**seg) for seg in transcript.segments]
        
        # Generate explanations for highlights using LLM
        from utils.explanation_builder import build_highlight_explanation_from_text_async
        highlights = []
        for h in highlights_db:
            explanation = None
            try:
                # Extract text for this highlight segment
                highlight_text = ""
                if transcript_segments:
                    segment_texts = [
                        seg.text for seg in transcript_segments
                        if seg.start >= h.start - 0.5 and seg.end <= h.end + 0.5
                    ]
                    highlight_text = " ".join(segment_texts)
                
                # If no text found, use title as fallback
                if not highlight_text and h.title:
                    highlight_text = h.title
                
                if highlight_text:
                    explanation = await build_highlight_explanation_from_text_async(
                        text=highlight_text,
                        score=h.score,
                    )
                    logger.debug(f"[Backend] Generated explanation for highlight [{h.start:.1f}s-{h.end:.1f}s]: {explanation.confidence} confidence")
                else:
                    logger.warning(f"[Backend] No transcript text found for highlight [{h.start:.1f}s-{h.end:.1f}s]")
            except Exception as e:
                logger.warning(f"[Backend] Failed to generate explanation for highlight [{h.start:.1f}s-{h.end:.1f}s]: {e}", exc_info=True)
                explanation = None
            
            highlights.append(Highlight(
                start=h.start,
                end=h.end,
                title=h.title,
                summary=None,  # Summary no longer used
                score=h.score,
                explanation=explanation,
            ))
        
        logger.info(f"[Backend] Returning {len(highlights)} highlights for video {video_id}")
        return HighlightsResponse(video_id=video_uuid, highlights=highlights)
    
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid video ID format")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving highlights: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error retrieving highlights")


@app.get("/videos/{video_id}/segments", response_model=SegmentsResponse)
async def get_segments(
    video_id: str,
    label: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """
    Retrieve segment analyses for a video.
    
    Optional query parameter:
    - label: Filter by label (FLUFF, REPEATED, USEFUL, NEUTRAL)
    
    Returns list of segments with labels, ratings, and scores.
    Returns empty list if no segments found.
    Returns 404 if video not found.
    """
    try:
        video_uuid = UUID(video_id)
        
        # Verify video exists
        video = await crud.get_video_by_id(db, video_uuid)
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")
        
        # Get segments, optionally filtered by label
        if label:
            if label.upper() != "FLUFF":
                raise HTTPException(status_code=400, detail="Invalid label. Must be FLUFF")
            segments_db = await crud.get_segments_by_label(db, video_uuid, label.upper())
        else:
            segments_db = await crud.get_video_segments(db, video_uuid)
        
        logger.info(f"[Backend] /videos/{video_id}/segments - Found {len(segments_db)} segments")
        
        # Get retention metrics for FLUFF segments
        from utils.explanation_builder import build_fluff_explanation, build_fluff_explanation_from_segment
        from utils.retention_scoring import RETENTION_THRESHOLD
        from models import RetentionMetrics as RetentionMetricsModel
        
        # Fetch all retention metrics for the video and create a lookup map
        retention_metrics_db = await crud.get_retention_metrics_by_video(db, video_uuid)
        retention_metrics_map = {rm.segment_id: rm for rm in retention_metrics_db}
        logger.info(f"[Backend] Found {len(retention_metrics_map)} retention metrics for video {video_id}")
        
        segments = []
        for s in segments_db:
            explanation = None
            
            # Generate explanation for FLUFF segments
            if s.label == "FLUFF":
                segment_id = s.segment_id
                logger.debug(f"[Backend] Processing FLUFF segment {segment_id}")
                
                if segment_id in retention_metrics_map:
                    rm = retention_metrics_map[segment_id]
                    # Convert JSONB metrics to Pydantic model
                    try:
                        metrics_dict = rm.metrics if isinstance(rm.metrics, dict) else rm.metrics
                        retention_metrics = RetentionMetricsModel(**metrics_dict)
                        explanation = build_fluff_explanation(
                            retention_metrics,
                            rm.retention_value,
                            RETENTION_THRESHOLD,
                        )
                        logger.debug(f"[Backend] Generated explanation from retention metrics for segment {segment_id}")
                    except Exception as e:
                        logger.warning(f"Failed to generate explanation for segment {segment_id}: {e}", exc_info=True)
                        # Fallback to segment-based explanation
                        try:
                            explanation = build_fluff_explanation_from_segment(
                                s.label,
                                s.repetition_score,
                                s.filler_density,
                                s.visual_change_score,
                                s.usefulness_score,
                            )
                            logger.debug(f"[Backend] Generated fallback explanation for segment {segment_id}")
                        except Exception as e2:
                            logger.error(f"Failed to generate fallback explanation for segment {segment_id}: {e2}", exc_info=True)
                else:
                    # No retention metrics available, use segment scores
                    logger.debug(f"[Backend] No retention metrics for segment {segment_id}, using segment scores")
                    try:
                        explanation = build_fluff_explanation_from_segment(
                            s.label,
                            s.repetition_score,
                            s.filler_density,
                            s.visual_change_score,
                            s.usefulness_score,
                        )
                        logger.debug(f"[Backend] Generated segment-based explanation for segment {segment_id}")
                    except Exception as e:
                        logger.error(f"Failed to generate segment-based explanation for segment {segment_id}: {e}", exc_info=True)
                
                # If no explanation was generated, create a minimal one
                if not explanation:
                    logger.warning(f"[Backend] No explanation generated for FLUFF segment {segment_id}, creating minimal explanation")
                    from models import VerdictExplanation
                    explanation = VerdictExplanation(
                        verdict="FLUFF",
                        confidence="medium",
                        evidence=[s.reason] if s.reason else ["Low quality segment"],
                        action_hint="Remove",
                    )
                
                if explanation:
                    logger.debug(f"[Backend] Explanation for segment {segment_id}: verdict={explanation.verdict}, confidence={explanation.confidence}, evidence={len(explanation.evidence)} items")
            
            segments.append(
                SegmentAnalysis(
                    id=s.id,
                    start_time=s.start_time,
                    end_time=s.end_time,
                    label=s.label,
                    rating=s.rating,
                    grade=getattr(s, 'grade', 'C'),
                    reason=s.reason,
                    repetition_score=s.repetition_score,
                    filler_density=s.filler_density,
                    visual_change_score=s.visual_change_score,
                    usefulness_score=s.usefulness_score,
                    explanation=explanation,
                )
            )
        
        logger.info(f"[Backend] Returning {len(segments)} segments for video {video_id}")
        return SegmentsResponse(video_id=video_uuid, segments=segments)
    
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid video ID format")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving segments: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error retrieving segments")


@app.get("/videos/{video_id}/retention-metrics")
async def get_retention_metrics(
    video_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Retrieve retention metrics for a video.
    
    Returns list of retention analyses with detailed metrics.
    Returns empty list if no retention metrics found.
    Returns 404 if video not found.
    """
    try:
        from models import RetentionAnalysis
        
        video_uuid = UUID(video_id)
        
        # Verify video exists
        video = await crud.get_video_by_id(db, video_uuid)
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")
        
        # Get retention metrics
        retention_metrics_db = await crud.get_retention_metrics_by_video(db, video_uuid)
        
        logger.info(f"[Backend] /videos/{video_id}/retention-metrics - Found {len(retention_metrics_db)} retention metrics")
        
        # Convert to Pydantic models
        retention_analyses = []
        for rm in retention_metrics_db:
            analysis = RetentionAnalysis(
                video_id=str(rm.video_id),
                segment_id=rm.segment_id,
                time_range=rm.time_range,
                text=rm.text,
                metrics=rm.metrics,
                retention_value=rm.retention_value,
                decision=rm.decision,
            )
            retention_analyses.append(analysis)
        
        logger.info(f"[Backend] Returning {len(retention_analyses)} retention analyses for video {video_id}")
        return {"video_id": str(video_uuid), "retention_analyses": retention_analyses}
    
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid video ID format")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving retention metrics: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error retrieving retention metrics")




@app.post("/videos/{video_id}/segments/{segment_id}/feedback", response_model=SegmentFeedbackResponse, status_code=201)
async def submit_segment_feedback(
    video_id: str,
    segment_id: str,
    feedback_request: SegmentFeedbackRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Submit feedback for a video segment.
    
    Feedback types:
    - GREAT: +0.6 adjustment to usefulness score
    - FINE: -0.4 adjustment to usefulness score
    - WRONG: -0.6 adjustment to usefulness score
    
    Returns feedback record and updates segment rating/usefulness_score.
    
    If segment_id is "lookup", will use start_time and end_time from request body to find segment.
    """
    try:
        from utils.learning import update_segment_calibration_online
        
        video_uuid = UUID(video_id)
        
        # Verify video exists
        video = await crud.get_video_by_id(db, video_uuid)
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")
        
        # Get segment - either by ID or by time range
        if segment_id == "lookup" and feedback_request.start_time is not None and feedback_request.end_time is not None:
            segment = await crud.get_video_segment_by_time_range(
                db,
                video_uuid,
                feedback_request.start_time,
                feedback_request.end_time,
            )
            if not segment:
                raise HTTPException(status_code=404, detail="Segment not found by time range")
        else:
            try:
                segment_uuid = UUID(segment_id)
                segment = await crud.get_video_segment_by_id(db, segment_uuid)
                if not segment:
                    raise HTTPException(status_code=404, detail="Segment not found")
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid segment_id format")
        
        # Verify segment belongs to video
        if segment.video_id != video_uuid:
            raise HTTPException(status_code=400, detail="Segment does not belong to this video")
        
        # Validate feedback type
        feedback_type = feedback_request.feedback_type.upper()
        if feedback_type not in ["GREAT", "FINE", "WRONG"]:
            raise HTTPException(status_code=400, detail="Invalid feedback_type. Must be GREAT, FINE, or WRONG")
        
        # Calculate adjustment
        adjustments = {
            "GREAT": 0.6,
            "FINE": -0.4,
            "WRONG": -0.6,
        }
        adjustment = adjustments[feedback_type]
        
        # Store original rating for calibration
        original_rating = segment.rating
        original_usefulness = segment.usefulness_score
        
        # Calculate new rating and usefulness_score
        new_rating = max(0.0, min(1.0, segment.rating + adjustment))
        new_usefulness_score = max(0.0, min(1.0, segment.usefulness_score + adjustment))
        
        # Create feedback record
        feedback = await crud.create_segment_feedback(
            db,
            video_segment_id=segment.id,
            feedback_type=feedback_type,
        )
        
        # Update segment rating and usefulness_score
        await crud.update_video_segment_rating(
            db,
            segment_id=segment.id,
            new_rating=new_rating,
            new_usefulness_score=new_usefulness_score,
        )
        
        # Feed into calibration system
        await update_segment_calibration_online(
            db,
            segment_id=segment.id,
            predicted_rating=original_rating,
            actual_rating=new_rating,
        )
        
        logger.info(
            f"Segment feedback submitted: segment_id={segment.id}, "
            f"feedback_type={feedback_type}, rating={original_rating:.2f} -> {new_rating:.2f}"
        )
        
        return SegmentFeedbackResponse(
            id=feedback.id,
            video_segment_id=feedback.video_segment_id,
            feedback_type=feedback.feedback_type,
            created_at=feedback.created_at,
        )
    
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid ID format")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting segment feedback: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error submitting segment feedback")


@app.get("/videos/{video_id}/moments", response_model=MomentsResponse)
async def get_moments(
    video_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Retrieve moments for a video.
    
    Returns list of moment metadata with moment_url, start, end, and thumbnail_url.
    Returns empty list if no moments found.
    Returns 404 if video not found.
    """
    try:
        video_uuid = UUID(video_id)
        
        # Verify video exists
        video = await crud.get_video_by_id(db, video_uuid)
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")
        
        moments_db = await crud.get_moments_by_video_id(db, video_uuid)
        logger.info(f"[Backend] /videos/{video_id}/moments - Found {len(moments_db)} moments in database")
        
        moments = [
            MomentResponse(
                id=moment.id,
                moment_url=f"/moments/{moment.id}/download",
                start=moment.start,
                end=moment.end,
                thumbnail_url=f"/moments/{moment.id}/thumbnail",
            )
            for moment in moments_db
        ]
        
        logger.info(f"[Backend] Returning {len(moments)} moments for video {video_id}: {[str(m.id) for m in moments]}")
        return MomentsResponse(video_id=video_uuid, moments=moments)
    
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid video ID format")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving moments: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error retrieving moments")


@app.get("/videos/{video_id}/download")
async def download_video(
    video_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Stream video for playback with Range request support.
    Checks for local preview first, then falls back to S3.
    """
    try:
        video_uuid = UUID(video_id)
        video = await crud.get_video_by_id(db, video_uuid)
        
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")
        
        # Check for local preview first (instant preview with cuts)
        from utils.preview_cache import get_preview_path
        preview_path = get_preview_path(video_uuid)
        if preview_path:
            # Serve local preview file with range support
            from fastapi.responses import FileResponse
            range_header = request.headers.get("range")
            return FileResponse(
                preview_path,
                media_type="video/mp4",
                headers={"Accept-Ranges": "bytes"},
            )
        
        storage = get_storage_instance()
        
        if isinstance(storage, S3Storage):
            # Stream from S3 through backend to avoid CORS issues
            import httpx
            
            presigned_url = storage.get_video_path(video.storage_path)
            
            # Get Range header from request
            range_header = request.headers.get("range")
            
            # Prepare headers for S3 request
            headers = {}
            if range_header:
                headers["Range"] = range_header
            
            # Stream from S3
            client = None
            stream_ctx = None
            try:
                client = httpx.AsyncClient(timeout=300.0)
                stream_ctx = client.stream("GET", presigned_url, headers=headers, follow_redirects=True)
                response = await stream_ctx.__aenter__()
                
                if response.status_code == 404:
                    await stream_ctx.__aexit__(None, None, None)
                    await client.aclose()
                    raise HTTPException(status_code=404, detail="Video file not found in storage")
                
                # Handle 416 Range Not Satisfiable - retry without range header
                if response.status_code == 416:
                    await stream_ctx.__aexit__(None, None, None)
                    # Retry without range header to get full file
                    stream_ctx = client.stream("GET", presigned_url, follow_redirects=True)
                    response = await stream_ctx.__aenter__()
                    range_header = None  # Clear range header since we're getting full file
                
                # Get content info
                content_type = response.headers.get("content-type", "video/mp4")
                content_length = response.headers.get("content-length")
                content_range = response.headers.get("content-range")
                
                # Build response headers
                response_headers = {
                    "Accept-Ranges": "bytes",
                    "Content-Type": content_type,
                }
                
                if content_length:
                    response_headers["Content-Length"] = content_length
                if content_range:
                    response_headers["Content-Range"] = content_range
                
                # Determine status code
                status_code = 206 if range_header and response.status_code == 206 else 200
                
                # Generator to stream chunks
                async def generate():
                    try:
                        async for chunk in response.aiter_bytes():
                            yield chunk
                    except httpx.StreamClosed:
                        pass
                    finally:
                        if stream_ctx:
                            try:
                                await stream_ctx.__aexit__(None, None, None)
                            except Exception:
                                pass
                        if client:
                            try:
                                await client.aclose()
                            except Exception:
                                pass
                
                return StreamingResponse(
                    generate(),
                    status_code=status_code,
                    headers=response_headers,
                    media_type=content_type,
                )
            except HTTPException:
                raise
            except Exception as e:
                if stream_ctx:
                    try:
                        await stream_ctx.__aexit__(None, None, None)
                    except Exception:
                        pass
                if client:
                    try:
                        await client.aclose()
                    except Exception:
                        pass
                logger.error(f"Error streaming video from S3: {str(e)}", exc_info=True)
                raise HTTPException(status_code=500, detail="Error accessing video file")
        else:
            # Local file - use FileResponse which handles Range requests
            local_path = storage.get_video_path(video.storage_path)
            if not os.path.exists(local_path):
                raise HTTPException(status_code=404, detail="Video file not found on disk")
            return FileResponse(
                path=local_path,
                media_type="video/mp4",
                filename=f"video_{video_id}.mp4",
            )
    
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid video ID format")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading video: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error downloading video")


@app.get("/moments/{moment_id}/download")
async def download_moment(
    moment_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Download or stream a moment file.
    
    Supports video playback (range requests) and download.
    For S3 storage, proxies the content to avoid CORS issues.
    For local storage, streams the file directly.
    Returns 404 if moment not found or file doesn't exist.
    """
    from fastapi.responses import StreamingResponse
    
    try:
        moment_uuid = UUID(moment_id)
        moment = await crud.get_moment_by_id(db, moment_uuid)
        
        if not moment:
            raise HTTPException(status_code=404, detail="Moment not found")
        
        storage = get_storage_instance()
        
        # Handle S3 storage - proxy the video to avoid CORS issues
        if isinstance(storage, S3Storage):
            try:
                import httpx
                presigned_url = storage.get_video_path(moment.storage_path)
                
                # Get range header if present (for video seeking)
                range_header = None
                if request and hasattr(request, "headers"):
                    range_header = request.headers.get("range")
                
                # Download from S3 presigned URL with range support
                client = None
                stream_ctx = None
                try:
                    client = httpx.AsyncClient(timeout=300.0)
                    headers = {}
                    if range_header:
                        headers["Range"] = range_header
                    
                    stream_ctx = client.stream("GET", presigned_url, headers=headers, follow_redirects=True)
                    response = await stream_ctx.__aenter__()
                    
                    if response.status_code == 404:
                        await stream_ctx.__aexit__(None, None, None)
                        await client.aclose()
                        raise HTTPException(status_code=404, detail="Moment file not found in storage")
                    
                    # Handle 416 Range Not Satisfiable - retry without range header
                    if response.status_code == 416:
                        await stream_ctx.__aexit__(None, None, None)
                        # Retry without range header to get full file
                        stream_ctx = client.stream("GET", presigned_url, follow_redirects=True)
                        response = await stream_ctx.__aenter__()
                        range_header = None  # Clear range header since we're getting full file
                    
                    # Determine content type
                    content_type = response.headers.get("content-type", "video/mp4")
                    
                    # Get content length and range info
                    content_length = response.headers.get("content-length")
                    content_range = response.headers.get("content-range")
                    
                    # Build response headers
                    response_headers = {
                        "Accept-Ranges": "bytes",
                        "Content-Type": content_type,
                        "Content-Disposition": f'inline; filename="moment_{moment_id}.mp4"',
                    }
                    
                    if content_length:
                        response_headers["Content-Length"] = content_length
                    if content_range:
                        response_headers["Content-Range"] = content_range
                    
                    # Return appropriate status code for range requests
                    status_code = 206 if range_header and response.status_code == 206 else 200
                    
                    async def generate():
                        try:
                            async for chunk in response.aiter_bytes():
                                yield chunk
                        except httpx.StreamClosed:
                            # Client disconnected or stream closed - this is normal
                            pass
                        finally:
                            # Clean up resources when generator completes
                            if stream_ctx:
                                try:
                                    await stream_ctx.__aexit__(None, None, None)
                                except Exception:
                                    pass
                            if client:
                                try:
                                    await client.aclose()
                                except Exception:
                                    pass
                    
                    return StreamingResponse(
                        generate(),
                        status_code=status_code,
                        headers=response_headers,
                        media_type=content_type,
                    )
                except HTTPException:
                    # Re-raise HTTP exceptions after cleanup
                    if stream_ctx:
                        try:
                            await stream_ctx.__aexit__(None, None, None)
                        except Exception:
                            pass
                    if client:
                        try:
                            await client.aclose()
                        except Exception:
                            pass
                    raise
            except httpx.HTTPError as e:
                logger.error(f"Error fetching video from S3 for moment {moment_id}: {str(e)}", exc_info=True)
                raise HTTPException(status_code=500, detail="Error accessing moment file")
            except Exception as e:
                logger.error(f"Error generating presigned URL for moment {moment_id}: {str(e)}", exc_info=True)
                raise HTTPException(status_code=500, detail="Error accessing moment file")
        
        # Handle local storage
        upload_dir = os.getenv("UPLOAD_DIR", "./uploads")
        moment_full_path = os.path.join(upload_dir, moment.storage_path)
        
        if not os.path.exists(moment_full_path):
            logger.error(f"Moment file not found at path: {moment_full_path}")
            raise HTTPException(status_code=404, detail="Moment file not found")
        
        return FileResponse(
            moment_full_path,
            media_type="video/mp4",
            filename=f"moment_{moment_id}.mp4",
            headers={
                "Accept-Ranges": "bytes",
                "Content-Disposition": f'inline; filename="moment_{moment_id}.mp4"',
            },
        )
    
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid moment ID format")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading moment: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error downloading moment")


@app.get("/moments/{moment_id}/thumbnail")
async def get_moment_thumbnail(
    moment_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Get thumbnail image for a moment.
    
    Returns the thumbnail image file.
    For S3 storage, redirects to presigned URL.
    For local storage, streams the file directly.
    If thumbnail doesn't exist, generates it on-demand from the moment.
    Returns 404 if moment not found.
    """
    try:
        moment_uuid = UUID(moment_id)
        moment = await crud.get_moment_by_id(db, moment_uuid)
        
        if not moment:
            raise HTTPException(status_code=404, detail="Moment not found")
        
        storage = get_storage_instance()
        
        # Generate thumbnail on-demand if it doesn't exist
        if not moment.thumbnail_path:
            try:
                logger.info(f"Generating thumbnail on-demand for moment {moment_id}")
                
                # Get moment file path
                moment_file_path = storage.get_video_path(moment.storage_path)
                
                # For S3 storage, moment_file_path is a presigned URL
                # For local storage, it's a file path
                thumbnail_path = await generate_thumbnail_async(moment_file_path, time_offset=0.0)
                
                # Update moment with thumbnail path
                moment.thumbnail_path = thumbnail_path
                await db.commit()
                await db.refresh(moment)
                
                logger.info(f"Successfully generated thumbnail for moment {moment_id}: {thumbnail_path}")
            except Exception as e:
                logger.error(f"Failed to generate thumbnail on-demand for moment {moment_id}: {str(e)}", exc_info=True)
                raise HTTPException(status_code=500, detail="Failed to generate thumbnail")
        
        # Handle S3 storage - proxy the thumbnail to avoid CORB issues
        if isinstance(storage, S3Storage):
            try:
                # Download thumbnail from S3 and stream it directly (same origin, no CORB)
                import httpx
                
                presigned_url = storage.get_video_path(moment.thumbnail_path)
                
                # Download from S3 presigned URL
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.get(presigned_url)
                    
                    # If thumbnail doesn't exist in S3 (404), generate it on-demand
                    if response.status_code == 404:
                        logger.info(f"Thumbnail not found in S3 for moment {moment_id}, generating on-demand")
                        try:
                            # Get moment file path
                            moment_file_path = storage.get_video_path(moment.storage_path)
                            # Generate thumbnail
                            thumbnail_path = await generate_thumbnail_async(moment_file_path, time_offset=0.0)
                            # Update moment with thumbnail path
                            moment.thumbnail_path = thumbnail_path
                            await db.commit()
                            await db.refresh(moment)
                            # Get new presigned URL for the generated thumbnail
                            presigned_url = storage.get_video_path(moment.thumbnail_path)
                            # S3 eventual consistency - retry with small delays
                            import asyncio
                            max_retries = 3
                            for attempt in range(max_retries):
                                await asyncio.sleep(0.5 * (attempt + 1))  # 0.5s, 1s, 1.5s delays
                                response = await client.get(presigned_url)
                                if response.status_code == 200:
                                    thumbnail_data = response.content
                                    break
                                elif attempt == max_retries - 1:
                                    raise HTTPException(status_code=500, detail="Failed to fetch generated thumbnail after retries")
                        except Exception as gen_error:
                            logger.error(f"Failed to generate thumbnail on-demand for moment {moment_id}: {str(gen_error)}", exc_info=True)
                            raise HTTPException(status_code=500, detail="Failed to generate thumbnail")
                    else:
                        response.raise_for_status()
                        thumbnail_data = response.content
                
                # Return the image data directly (same origin, no CORB issues)
                from fastapi.responses import Response
                return Response(
                    content=thumbnail_data,
                    media_type="image/jpeg",
                    headers={
                        "Cache-Control": "public, max-age=3600",
                        "Content-Disposition": f'inline; filename="thumbnail_{moment_id}.jpg"',
                    }
                )
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    # Thumbnail doesn't exist, try to generate it
                    logger.info(f"Thumbnail not found in S3 for moment {moment_id}, generating on-demand")
                    try:
                        moment_file_path = storage.get_video_path(moment.storage_path)
                        thumbnail_path = await generate_thumbnail_async(moment_file_path, time_offset=0.0)
                        moment.thumbnail_path = thumbnail_path
                        await db.commit()
                        await db.refresh(moment)
                        # Retry fetching the newly generated thumbnail
                        presigned_url = storage.get_video_path(moment.thumbnail_path)
                        async with httpx.AsyncClient(timeout=30.0) as client:
                            response = await client.get(presigned_url)
                            response.raise_for_status()
                            thumbnail_data = response.content
                        from fastapi.responses import Response
                        return Response(
                            content=thumbnail_data,
                            media_type="image/jpeg",
                            headers={
                                "Cache-Control": "public, max-age=3600",
                                "Content-Disposition": f'inline; filename="thumbnail_{moment_id}.jpg"',
                            }
                        )
                    except Exception as gen_error:
                        logger.error(f"Failed to generate thumbnail on-demand for moment {moment_id}: {str(gen_error)}", exc_info=True)
                        raise HTTPException(status_code=500, detail="Failed to generate thumbnail")
                else:
                    logger.error(f"Error proxying thumbnail from S3 for moment {moment_id}: {str(e)}", exc_info=True)
                    raise HTTPException(status_code=500, detail="Error accessing thumbnail file")
            except Exception as e:
                logger.error(f"Error proxying thumbnail from S3 for moment {moment_id}: {str(e)}", exc_info=True)
                raise HTTPException(status_code=500, detail="Error accessing thumbnail file")
        
        # Handle local storage
        upload_dir = os.getenv("UPLOAD_DIR", "./uploads")
        thumbnail_full_path = os.path.join(upload_dir, moment.thumbnail_path)
        
        if not os.path.exists(thumbnail_full_path):
            logger.error(f"Thumbnail file not found at path: {thumbnail_full_path}")
            raise HTTPException(status_code=404, detail="Thumbnail file not found")
        
        return FileResponse(
            thumbnail_full_path,
            media_type="image/jpeg",
            filename=f"thumbnail_{moment_id}.jpg",
        )
    
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid moment ID format")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving thumbnail: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error retrieving thumbnail")


@app.get("/moments/{moment_id}", response_model=MomentDetailResponse)
async def get_moment_detail(
    moment_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Get detailed moment information including parent video URL and duration.
    
    Returns moment metadata with parent video information needed for editing.
    Returns 404 if moment not found.
    """
    try:
        moment_uuid = UUID(moment_id)
        moment = await crud.get_moment_by_id(db, moment_uuid)
        
        if not moment:
            raise HTTPException(status_code=404, detail="Moment not found")
        
        video = await crud.get_video_by_id(db, moment.video_id)
        if not video:
            raise HTTPException(status_code=404, detail="Parent video not found")
        
        video_url = await get_video_path(moment.video_id, db, download_local=False)
        if not video_url:
            raise HTTPException(status_code=404, detail="Video file not found")
        
        return MomentDetailResponse(
            id=moment.id,
            video_id=moment.video_id,
            start=moment.start,
            end=moment.end,
            video_url=video_url,
            video_duration=video.duration if video.duration else 0.0,
            thumbnail_url=f"/moments/{moment.id}/thumbnail",
        )
    
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid moment ID format")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving moment detail: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error retrieving moment detail")


@app.post("/moments/{moment_id}/save", status_code=201)
async def save_moment(
    moment_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Save a moment for later.
    
    Creates a saved moment record. Returns existing record if moment is already saved.
    Returns 404 if moment not found.
    """
    try:
        moment_uuid = UUID(moment_id)
        
        # Get moment
        moment = await crud.get_moment_by_id(db, moment_uuid)
        if not moment:
            raise HTTPException(status_code=404, detail="Moment not found")
        
        # Check if already saved
        existing_saved = await crud.get_saved_moment(db, moment_uuid)
        if existing_saved:
            return SavedMomentResponse(
                id=existing_saved.id,
                moment_id=existing_saved.moment_id,
                highlight_id=existing_saved.highlight_id,
                created_at=existing_saved.created_at,
            )
        
        # Find associated highlight
        highlight = await db.execute(
            select(HighlightDB).where(
                HighlightDB.video_id == moment.video_id,
                HighlightDB.start <= moment.start,
                HighlightDB.end >= moment.end,
            ).limit(1)
        )
        highlight_obj = highlight.scalar_one_or_none()
        
        # Create saved moment record
        saved_moment = await crud.create_saved_moment(
            db,
            moment_id=moment_uuid,
            highlight_id=highlight_obj.id if highlight_obj else None,
        )
        
        logger.info(f"Moment saved: {moment_id}")
        
        return SavedMomentResponse(
            id=saved_moment.id,
            moment_id=saved_moment.moment_id,
            highlight_id=saved_moment.highlight_id,
            created_at=saved_moment.created_at,
        )
    
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid moment ID format")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error saving moment: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error saving moment")


@app.post("/moments/{moment_id}/edit", response_model=MomentResponse)
async def edit_moment(
    moment_id: str,
    edit_request: EditMomentRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Edit a moment by updating its start and end timestamps.
    
    Re-cuts the moment using FFmpeg, updates the database record, and deletes the old moment file.
    Returns 404 if moment not found, 400 if timestamps are invalid.
    """
    try:
        moment_uuid = UUID(moment_id)
        moment = await crud.get_moment_by_id(db, moment_uuid)
        
        if not moment:
            raise HTTPException(status_code=404, detail="Moment not found")
        
        video = await crud.get_video_by_id(db, moment.video_id)
        if not video:
            raise HTTPException(status_code=404, detail="Parent video not found")
        
        new_start = edit_request.new_start
        new_end = edit_request.new_end
        
        if new_start < 0:
            raise HTTPException(status_code=400, detail="Start time must be non-negative")
        
        if new_end <= new_start:
            raise HTTPException(status_code=400, detail="End time must be greater than start time")
        
        if video.duration and new_end > video.duration:
            raise HTTPException(
                status_code=400,
                detail=f"End time ({new_end}) exceeds video duration ({video.duration})"
            )
        
        old_storage_path = moment.storage_path
        old_thumbnail_path = moment.thumbnail_path
        
        video_path = await get_video_path(moment.video_id, db, download_local=False)
        if not video_path:
            raise HTTPException(status_code=404, detail="Video file not found")
        
        storage_path, temp_moment_path = await generate_moment_async(video_path, new_start, new_end)
        
        thumbnail_path = None
        try:
            if temp_moment_path and os.path.exists(temp_moment_path):
                thumbnail_path = await generate_thumbnail_async(temp_moment_path, time_offset=0.0)
            else:
                upload_dir = os.getenv("UPLOAD_DIR", "./uploads")
                moment_full_path = os.path.join(upload_dir, storage_path)
                if os.path.exists(moment_full_path):
                    thumbnail_path = await generate_thumbnail_async(moment_full_path, time_offset=0.0)
        except Exception as e:
            logger.warning(f"Failed to generate thumbnail for edited moment {moment_id}: {str(e)}", exc_info=True)
        
        updated_moment = await crud.update_moment(
            db,
            moment_uuid,
            start=new_start,
            end=new_end,
            storage_path=storage_path,
            thumbnail_path=thumbnail_path,
        )
        
        if not updated_moment:
            raise HTTPException(status_code=500, detail="Failed to update moment in database")
        
        try:
            await delete_moment_file(old_storage_path, old_thumbnail_path)
        except Exception as e:
            logger.warning(f"Failed to delete old moment file: {str(e)}", exc_info=True)
        
        if temp_moment_path and os.path.exists(temp_moment_path):
            try:
                os.remove(temp_moment_path)
            except Exception as e:
                logger.warning(f"Failed to clean up temp moment file: {str(e)}")
        
        logger.info(f"Successfully edited moment {moment_id}: {new_start:.2f}s - {new_end:.2f}s")
        
        return MomentResponse(
            id=updated_moment.id,
            moment_url=f"/moments/{updated_moment.id}/download",
            start=updated_moment.start,
            end=updated_moment.end,
            thumbnail_url=f"/moments/{updated_moment.id}/thumbnail",
        )
    
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid moment ID format")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error editing moment: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error editing moment")


@app.delete("/moments/{moment_id}/save")
async def unsave_moment(
    moment_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Remove a moment from saved moments.
    
    Returns 404 if moment is not currently saved.
    """
    try:
        moment_uuid = UUID(moment_id)
        
        deleted = await crud.delete_saved_moment(db, moment_uuid)
        if not deleted:
            raise HTTPException(status_code=404, detail="Moment is not saved")
        
        logger.info(f"Moment unsaved: {moment_id}")
        
        return {"status": "deleted"}
    
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid moment ID format")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error unsaving moment: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error unsaving moment")


@app.get("/moments", response_model=MomentsResponse)
async def get_all_moments(
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    """
    Get all moments from all videos.
    
    Returns list of moments with pagination support.
    """
    try:
        moments_db = await crud.get_all_moments(db, limit=limit, offset=offset)
        
        moments = [
            MomentResponse(
                id=moment.id,
                moment_url=f"/moments/{moment.id}/download",
                start=moment.start,
                end=moment.end,
                thumbnail_url=f"/moments/{moment.id}/thumbnail",
            )
            for moment in moments_db
        ]
        
        # Use a dummy video_id for all moments endpoint
        return MomentsResponse(video_id=UUID("00000000-0000-0000-0000-000000000000"), moments=moments)
    
    except Exception as e:
        logger.error(f"Error retrieving all moments: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error retrieving moments")


@app.get("/moments/saved", response_model=List[MomentResponse])
async def get_saved_moments(
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    """
    Get all saved moments.
    
    Returns list of saved moments with their moment details.
    """
    try:
        saved_moments_db = await crud.get_all_saved_moments(db, limit=limit, offset=offset)
        
        moments = []
        for saved_moment in saved_moments_db:
            moment = await crud.get_moment_by_id(db, saved_moment.moment_id)
            if moment:
                moments.append(
                    MomentResponse(
                        id=moment.id,
                        moment_url=f"/moments/{moment.id}/download",
                        start=moment.start,
                        end=moment.end,
                        thumbnail_url=f"/moments/{moment.id}/thumbnail",
                    )
                )
        
        return moments
    
    except Exception as e:
        logger.error(f"Error retrieving saved moments: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error retrieving saved moments")


@app.delete("/moments")
async def delete_all_moments_endpoint(
    db: AsyncSession = Depends(get_db),
):
    """
    Delete all moments from the database and storage.
    
    This will also delete associated saved moments and feedback due to CASCADE.
    """
    try:
        # Get all moments before deletion to clean up storage files
        result = await db.execute(select(Moment))
        moments = list(result.scalars().all())
        
        storage = get_storage_instance()
        
        # Delete files from storage
        for moment in moments:
            try:
                if isinstance(storage, S3Storage):
                    # Delete moment file
                    storage.delete_video(moment.storage_path)
                    # Delete thumbnail if exists
                    if moment.thumbnail_path:
                        storage.delete_video(moment.thumbnail_path)
                else:
                    # Local storage - delete files
                    upload_dir = os.getenv("UPLOAD_DIR", "./uploads")
                    moment_path = os.path.join(upload_dir, moment.storage_path)
                    if os.path.exists(moment_path):
                        os.remove(moment_path)
                    if moment.thumbnail_path:
                        thumbnail_path = os.path.join(upload_dir, moment.thumbnail_path)
                        if os.path.exists(thumbnail_path):
                            os.remove(thumbnail_path)
            except Exception as e:
                logger.warning(f"Failed to delete storage file for moment {moment.id}: {str(e)}")
        
        # Clean up orphaned audio files after deleting all moments
        try:
            from utils.audio import cleanup_audio_directory
            cleanup_audio_directory()
        except Exception as e:
            logger.warning(f"Failed to cleanup audio directory: {str(e)}")
        
        # Delete all moments from database
        deleted_count = await crud.delete_all_moments(db)
        
        logger.info(f"Deleted {deleted_count} moments")
        
        return {"status": "deleted", "count": deleted_count}
    
    except Exception as e:
        logger.error(f"Error deleting all moments: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error deleting moments")


@app.get("/admin/prompt-versions", response_model=List[PromptVersionResponse])
async def list_prompt_versions(
    db: AsyncSession = Depends(get_db),
):
    """
    List all prompt versions.
    
    Returns list of all prompt versions with their configuration and performance metrics.
    """
    try:
        versions = await crud.get_all_prompt_versions(db)
        
        return [
            PromptVersionResponse(
                id=version.id,
                version_name=version.version_name,
                system_prompt=version.system_prompt,
                user_prompt_template=version.user_prompt_template,
                is_active=version.is_active,
                performance_metrics=version.performance_metrics,
                created_at=version.created_at,
            )
            for version in versions
        ]
    
    except Exception as e:
        logger.error(f"Error listing prompt versions: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error listing prompt versions")


@app.post("/admin/prompt-versions", status_code=201, response_model=PromptVersionResponse)
async def create_prompt_version(
    prompt_version: PromptVersionRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Create a new prompt version.
    
    Creates a new prompt version. Set is_active=False initially for A/B testing.
    """
    try:
        existing = await crud.get_prompt_version_by_name(db, prompt_version.version_name)
        if existing:
            raise HTTPException(status_code=400, detail=f"Prompt version '{prompt_version.version_name}' already exists")
        
        version = await crud.create_prompt_version(
            db,
            version_name=prompt_version.version_name,
            system_prompt=prompt_version.system_prompt,
            user_prompt_template=prompt_version.user_prompt_template,
            is_active=False,
        )
        
        logger.info(f"Created prompt version: {prompt_version.version_name}")
        
        return PromptVersionResponse(
            id=version.id,
            version_name=version.version_name,
            system_prompt=version.system_prompt,
            user_prompt_template=version.user_prompt_template,
            is_active=version.is_active,
            performance_metrics=version.performance_metrics,
            created_at=version.created_at,
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating prompt version: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error creating prompt version")


@app.put("/admin/prompt-versions/{version_id}/activate", response_model=PromptVersionResponse)
async def activate_prompt_version(
    version_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Activate a prompt version.
    
    Deactivates all other versions and activates the specified version.
    Returns 404 if version not found.
    """
    try:
        version_uuid = UUID(version_id)
        
        version = await crud.activate_prompt_version(db, version_uuid)
        
        if not version:
            raise HTTPException(status_code=404, detail="Prompt version not found")
        
        logger.info(f"Activated prompt version: {version.version_name}")
        
        return PromptVersionResponse(
            id=version.id,
            version_name=version.version_name,
            system_prompt=version.system_prompt,
            user_prompt_template=version.user_prompt_template,
            is_active=version.is_active,
            performance_metrics=version.performance_metrics,
            created_at=version.created_at,
        )
    
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid version ID format")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error activating prompt version: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error activating prompt version")


@app.get("/admin/calibration", response_model=CalibrationConfigResponse)
async def get_calibration_config(
    db: AsyncSession = Depends(get_db),
):
    """
    Get current calibration configuration.
    
    Returns the current score offset and sample size used for calibration.
    """
    try:
        config = await crud.get_calibration_config(db)
        
        if not config:
            return CalibrationConfigResponse(
                score_offset=0.0,
                last_updated=datetime.utcnow(),
                sample_size=0,
            )
        
        return CalibrationConfigResponse(
            score_offset=config.score_offset,
            last_updated=config.last_updated,
            sample_size=config.sample_size,
        )
    
    except Exception as e:
        logger.error(f"Error retrieving calibration config: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error retrieving calibration config")


@app.post("/admin/learning/run", status_code=200)
async def run_learning_pipeline(
    db: AsyncSession = Depends(get_db),
):
    """
    Manually trigger the learning pipeline.
    
    Runs the learning pipeline to:
    - Calculate calibration offset from feedback
    - Evaluate prompt version performance
    - Generate improved prompts if sufficient data
    - Promote best performing prompt version
    """
    try:
        from jobs.learning_job import run_learning_pipeline
        
        result = await run_learning_pipeline(db)
        
        return {
            "status": "completed",
            "calibration_updated": result.get("calibration_updated", False),
            "prompt_evaluated": result.get("prompt_evaluated", False),
            "prompt_promoted": result.get("prompt_promoted", False),
        }
    
    except Exception as e:
        logger.error(f"Error running learning pipeline: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error running learning pipeline: {str(e)}")


@app.get("/videos/{video_id}/thumbnail")
async def get_video_thumbnail(
    video_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Get thumbnail image for a video.
    
    Returns the thumbnail image file.
    For S3 storage, proxies the thumbnail to avoid CORB issues.
    For local storage, streams the file directly.
    If thumbnail doesn't exist, generates it on-demand from the video.
    Returns 404 if video not found.
    """
    try:
        video_uuid = UUID(video_id)
        video = await crud.get_video_by_id(db, video_uuid)
        
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")
        
        storage = get_storage_instance()
        
        # Generate thumbnail on-demand if it doesn't exist
        if not video.thumbnail_path:
            try:
                logger.info(f"Generating thumbnail on-demand for video {video_id}")
                
                # Get video file path
                video_file_path = storage.get_video_path(video.storage_path)
                
                # Generate thumbnail
                thumbnail_path = await generate_video_thumbnail_async(video_file_path, time_offset=1.0)
                
                # Update video with thumbnail path
                video.thumbnail_path = thumbnail_path
                await db.commit()
                await db.refresh(video)
                
                logger.info(f"Successfully generated thumbnail for video {video_id}: {thumbnail_path}")
            except Exception as e:
                logger.error(f"Failed to generate thumbnail on-demand for video {video_id}: {str(e)}", exc_info=True)
                raise HTTPException(status_code=500, detail="Failed to generate thumbnail")
        
        # Handle S3 storage - proxy the thumbnail to avoid CORB issues
        if isinstance(storage, S3Storage):
            try:
                import httpx
                
                presigned_url = storage.get_video_path(video.thumbnail_path)
                
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.get(presigned_url)
                    
                    if response.status_code == 404:
                        logger.info(f"Thumbnail not found in S3 for video {video_id}, generating on-demand")
                        try:
                            video_file_path = storage.get_video_path(video.storage_path)
                            thumbnail_path = await generate_video_thumbnail_async(video_file_path, time_offset=1.0)
                            video.thumbnail_path = thumbnail_path
                            await db.commit()
                            await db.refresh(video)
                            presigned_url = storage.get_video_path(video.thumbnail_path)
                            import asyncio
                            max_retries = 3
                            for attempt in range(max_retries):
                                await asyncio.sleep(0.5 * (attempt + 1))
                                response = await client.get(presigned_url)
                                if response.status_code == 200:
                                    thumbnail_data = response.content
                                    break
                                elif attempt == max_retries - 1:
                                    raise HTTPException(status_code=500, detail="Failed to fetch generated thumbnail after retries")
                        except Exception as gen_error:
                            logger.error(f"Failed to generate thumbnail on-demand for video {video_id}: {str(gen_error)}", exc_info=True)
                            raise HTTPException(status_code=500, detail="Failed to generate thumbnail")
                    else:
                        response.raise_for_status()
                        thumbnail_data = response.content
                
                from fastapi.responses import Response
                return Response(
                    content=thumbnail_data,
                    media_type="image/jpeg",
                    headers={
                        "Cache-Control": "public, max-age=3600",
                        "Content-Disposition": f'inline; filename="thumbnail_{video_id}.jpg"',
                    }
                )
            except Exception as e:
                logger.error(f"Error proxying thumbnail from S3 for video {video_id}: {str(e)}", exc_info=True)
                raise HTTPException(status_code=500, detail="Error accessing thumbnail file")
        
        # Handle local storage
        upload_dir = os.getenv("UPLOAD_DIR", "./uploads")
        thumbnail_full_path = os.path.join(upload_dir, video.thumbnail_path)
        
        if not os.path.exists(thumbnail_full_path):
            logger.error(f"Thumbnail file not found at path: {thumbnail_full_path}")
            raise HTTPException(status_code=404, detail="Thumbnail file not found")
        
        return FileResponse(
            thumbnail_full_path,
            media_type="image/jpeg",
            filename=f"thumbnail_{video_id}.jpg",
            headers={
                "Cache-Control": "public, max-age=3600",
            }
        )
    
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid video ID format")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving video thumbnail: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error retrieving video thumbnail")


@app.get("/timelines/{video_id}/thumbnail")
async def get_timeline_thumbnail(
    video_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Get thumbnail image for a timeline.
    
    Returns the thumbnail image file.
    For S3 storage, proxies the thumbnail to avoid CORB issues.
    For local storage, streams the file directly.
    If thumbnail doesn't exist, falls back to video thumbnail or generates it on-demand.
    Returns 404 if timeline/video not found.
    """
    try:
        video_uuid = UUID(video_id)
        timeline = await crud.get_timeline_by_video_id(db, video_uuid)
        
        if not timeline:
            raise HTTPException(status_code=404, detail="Timeline not found")
        
        video = timeline.video
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")
        
        storage = get_storage_instance()
        
        # Use timeline thumbnail if available, otherwise fall back to video thumbnail
        thumbnail_path = timeline.thumbnail_path or video.thumbnail_path
        
        # Generate thumbnail on-demand if it doesn't exist
        if not thumbnail_path:
            try:
                logger.info(f"Generating thumbnail on-demand for timeline {video_id}")
                
                # Get video file path
                video_file_path = storage.get_video_path(video.storage_path)
                
                # Generate thumbnail
                thumbnail_path = await generate_video_thumbnail_async(video_file_path, time_offset=1.0)
                
                # Update timeline with thumbnail path
                timeline.thumbnail_path = thumbnail_path
                await db.commit()
                await db.refresh(timeline)
                
                logger.info(f"Successfully generated thumbnail for timeline {video_id}: {thumbnail_path}")
            except Exception as e:
                logger.error(f"Failed to generate thumbnail on-demand for timeline {video_id}: {str(e)}", exc_info=True)
                raise HTTPException(status_code=500, detail="Failed to generate thumbnail")
        
        # Handle S3 storage - proxy the thumbnail to avoid CORB issues
        if isinstance(storage, S3Storage):
            try:
                import httpx
                
                presigned_url = storage.get_video_path(thumbnail_path)
                
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.get(presigned_url)
                    
                    if response.status_code == 404:
                        logger.info(f"Thumbnail not found in S3 for timeline {video_id}, generating on-demand")
                        try:
                            video_file_path = storage.get_video_path(video.storage_path)
                            thumbnail_path = await generate_video_thumbnail_async(video_file_path, time_offset=1.0)
                            timeline.thumbnail_path = thumbnail_path
                            await db.commit()
                            await db.refresh(timeline)
                            presigned_url = storage.get_video_path(thumbnail_path)
                            import asyncio
                            max_retries = 3
                            for attempt in range(max_retries):
                                await asyncio.sleep(0.5 * (attempt + 1))
                                response = await client.get(presigned_url)
                                if response.status_code == 200:
                                    thumbnail_data = response.content
                                    break
                                elif attempt == max_retries - 1:
                                    raise HTTPException(status_code=500, detail="Failed to fetch generated thumbnail after retries")
                        except Exception as gen_error:
                            logger.error(f"Failed to generate thumbnail on-demand for timeline {video_id}: {str(gen_error)}", exc_info=True)
                            raise HTTPException(status_code=500, detail="Failed to generate thumbnail")
                    else:
                        response.raise_for_status()
                        thumbnail_data = response.content
                
                from fastapi.responses import Response
                return Response(
                    content=thumbnail_data,
                    media_type="image/jpeg",
                    headers={
                        "Cache-Control": "public, max-age=3600",
                        "Content-Disposition": f'inline; filename="thumbnail_{video_id}.jpg"',
                    }
                )
            except Exception as e:
                logger.error(f"Error proxying thumbnail from S3 for timeline {video_id}: {str(e)}", exc_info=True)
                raise HTTPException(status_code=500, detail="Error accessing thumbnail file")
        
        # Handle local storage
        upload_dir = os.getenv("UPLOAD_DIR", "./uploads")
        thumbnail_full_path = os.path.join(upload_dir, thumbnail_path)
        
        if not os.path.exists(thumbnail_full_path):
            logger.error(f"Thumbnail file not found at path: {thumbnail_full_path}")
            raise HTTPException(status_code=404, detail="Thumbnail file not found")
        
        return FileResponse(
            thumbnail_full_path,
            media_type="image/jpeg",
            filename=f"thumbnail_{video_id}.jpg",
            headers={
                "Cache-Control": "public, max-age=3600",
            }
        )
    
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid video ID format")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving timeline thumbnail: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error retrieving timeline thumbnail")


@app.get("/projects", response_model=ProjectsResponse)
async def get_projects(
    db: AsyncSession = Depends(get_db),
    clerk_user_id: Optional[str] = Depends(get_clerk_user_id),
):
    """
    Get all timelines (projects) for the authenticated user.
    
    Returns list of timelines with video_id, created_at, duration, and moment_count.
    A timeline is created when a user starts working on a video.
    Filters by clerk_user_id if provided in headers.
    """
    try:
        logger.info(f"[Backend] /projects endpoint called - fetching timelines for user (user_id: {clerk_user_id})")
        timelines = await crud.get_timelines_by_user(db, clerk_user_id=clerk_user_id)
        logger.info(f"[Backend] Found {len(timelines)} timelines for user {clerk_user_id}")
        
        projects = []
        for timeline in timelines:
            video = timeline.video
            # Use timeline thumbnail if available, fallback to video thumbnail
            # Always return thumbnail URL even if not generated yet - backend will generate on-demand
            thumbnail_url = None
            if timeline.thumbnail_path:
                thumbnail_url = f"/timelines/{video.id}/thumbnail"
            else:
                # Always return video thumbnail URL - it will be generated on-demand if needed
                thumbnail_url = f"/videos/{video.id}/thumbnail"
            
            projects.append(
                ProjectResponse(
                    video_id=video.id,
                    created_at=timeline.created_at,
                    duration=video.duration,
                    moment_count=len(video.moments),
                    thumbnail_url=thumbnail_url,
                    project_name=timeline.project_name,
                )
            )
        
        logger.info(f"[Backend] Returning {len(projects)} projects: {[str(p.video_id) for p in projects]}")
        return ProjectsResponse(projects=projects)
    
    except Exception as e:
        logger.error(f"[Backend] Error retrieving projects: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error retrieving projects")


@app.get("/projects/{video_id}", response_model=MomentsResponse)
async def get_project(
    video_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Get a video by video_id (returns all moments for that video).
    
    This is essentially the same as /videos/{video_id}/moments but with a projects endpoint.
    """
    try:
        video_uuid = UUID(video_id)
        
        video = await crud.get_video_by_id(db, video_uuid)
        if not video:
            raise HTTPException(status_code=404, detail="Project not found")
        
        moments_db = await crud.get_moments_by_video_id(db, video_uuid)
        
        moments = [
            MomentResponse(
                id=moment.id,
                moment_url=f"/moments/{moment.id}/download",
                start=moment.start,
                end=moment.end,
                thumbnail_url=f"/moments/{moment.id}/thumbnail",
            )
            for moment in moments_db
        ]
        
        return MomentsResponse(video_id=video_uuid, moments=moments)
    
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid video ID format")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving project: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error retrieving project")


@app.post("/videos/{video_id}/cut", status_code=200)
async def store_pending_cuts(
    video_id: str,
    cut_request: CutVideoRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """
    Store pending cuts for a video without processing.
    This allows users to accumulate multiple cuts and apply them all at once when saving.
    """
    try:
        video_uuid = UUID(video_id)
        video = await crud.get_video_by_id(db, video_uuid)
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")
        
        # Convert request segments to dict format for JSON storage
        new_segments = [{"start_time": seg.start_time, "end_time": seg.end_time} for seg in cut_request.segments_to_remove]
        
        if not new_segments:
            raise HTTPException(status_code=400, detail="No segments to remove")
        
        # Merge with existing pending cuts (if any)
        existing_cuts = video.pending_cuts if video.pending_cuts else []
        merged_cuts = existing_cuts + new_segments
        
        # Remove duplicates (same start_time and end_time)
        seen = set()
        unique_cuts = []
        for cut in merged_cuts:
            key = (cut["start_time"], cut["end_time"])
            if key not in seen:
                seen.add(key)
                unique_cuts.append(cut)
        
        # Sort by start_time
        unique_cuts.sort(key=lambda x: x["start_time"])
        
        # Store pending cuts
        video.pending_cuts = unique_cuts
        await db.commit()
        
        # Generate local preview instantly in background (non-blocking)
        try:
            from utils.video_cutter import generate_local_preview
            segments_to_remove = [(cut["start_time"], cut["end_time"]) for cut in unique_cuts]
            # Generate preview in background
            background_tasks.add_task(generate_local_preview, video_uuid, segments_to_remove, db)
            logger.info(f"Started generating local preview for video {video_uuid}")
        except Exception as e:
            logger.warning(f"Failed to schedule local preview generation (non-critical): {str(e)}")
        
        return {
            "status": "success",
            "video_id": str(video_uuid),
            "pending_cuts": unique_cuts,
            "total_pending": len(unique_cuts),
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error storing pending cuts: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/videos/{video_id}/cut", status_code=200)
async def get_pending_cuts(
    video_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get pending cuts for a video."""
    try:
        video_uuid = UUID(video_id)
        video = await crud.get_video_by_id(db, video_uuid)
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")
        
        pending_cuts = video.pending_cuts if video.pending_cuts else []
        
        return {
            "status": "success",
            "video_id": str(video_uuid),
            "pending_cuts": pending_cuts,
            "total_pending": len(pending_cuts),
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting pending cuts: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.delete("/videos/{video_id}/cut", status_code=200)
async def clear_pending_cuts(
    video_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Clear all pending cuts for a video."""
    try:
        video_uuid = UUID(video_id)
        video = await crud.get_video_by_id(db, video_uuid)
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")
        
        video.pending_cuts = None
        await db.commit()
        
        # Clear local preview if it exists
        try:
            from utils.preview_cache import clear_preview
            clear_preview(video_uuid)
            logger.info(f"Cleared preview for video {video_uuid}")
        except Exception as e:
            logger.warning(f"Failed to clear preview (non-critical): {str(e)}")
        
        return {
            "status": "success",
            "video_id": str(video_uuid),
            "message": "Pending cuts cleared",
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing pending cuts: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/videos/{video_id}/cut/save", status_code=200)
async def save_video_cuts(
    video_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Apply all pending cuts to the video and save.
    This processes the video cutting and replaces the original video file.
    """
    try:
        from utils.video_cutter import process_video_cutting
        
        video_uuid = UUID(video_id)
        video = await crud.get_video_by_id(db, video_uuid)
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")
        
        pending_cuts = video.pending_cuts if video.pending_cuts else []
        if not pending_cuts:
            raise HTTPException(status_code=400, detail="No pending cuts to apply")
        
        # Convert to tuples for processing
        segments_to_remove = [(cut["start_time"], cut["end_time"]) for cut in pending_cuts]
        
        # Check if we have a local preview (much faster to upload)
        from utils.preview_cache import get_preview_path, clear_preview
        preview_path = get_preview_path(video_uuid)
        
        if preview_path:
            # Use local preview - upload it to S3 and replace old version
            logger.info(f"Using local preview for video {video_uuid} from {preview_path}")
            storage = get_storage_instance()
            
            if isinstance(storage, S3Storage):
                original_storage_path = video.storage_path
                
                # Backup old video
                old_backup_path = original_storage_path.replace("videos/", "videos/backup_")
                backup_created = False
                try:
                    storage.s3_client.copy_object(
                        Bucket=storage.bucket_name,
                        CopySource={"Bucket": storage.bucket_name, "Key": original_storage_path},
                        Key=old_backup_path,
                    )
                    logger.info(f"Backed up old video to {old_backup_path}")
                    backup_created = True
                except Exception as e:
                    logger.warning(f"Failed to backup old video (may not exist): {str(e)}")
                
                # Upload preview to S3
                import uuid
                new_filename = f"{uuid.uuid4()}.mp4"
                new_storage_path = storage.store_video_from_file(preview_path, new_filename)
                
                # Update database
                video.storage_path = new_storage_path
                await db.commit()
                
                # Delete old video after successful upload
                try:
                    storage.delete_video(original_storage_path)
                    logger.info(f"Deleted old video {original_storage_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete old video: {str(e)}")
                
                # Delete backup
                try:
                    storage.delete_video(old_backup_path)
                    logger.info(f"Deleted backup video {old_backup_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete backup video: {str(e)}")
                
                # Clear preview cache
                clear_preview(video_uuid)
            else:
                # Local storage - move preview to replace original
                import shutil
                old_full_path = storage.get_video_path(video.storage_path)
                shutil.move(preview_path, old_full_path)
                clear_preview(video_uuid)
                new_storage_path = video.storage_path
        else:
            # No preview available, process video cutting normally
            logger.info(f"No local preview found, processing video cutting for {video_uuid}")
            new_storage_path = await process_video_cutting(video_uuid, segments_to_remove, db)
        
        # Refresh video object and clear pending cuts after successful save
        await db.refresh(video)
        video.pending_cuts = None
        await db.commit()
        
        return {
            "status": "success",
            "video_id": str(video_uuid),
            "storage_path": new_storage_path,
            "segments_removed": len(segments_to_remove),
            "message": "Video cuts saved successfully",
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error saving video cuts: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/videos/{video_id}/export")
async def export_video(
    video_id: str,
    request: ExportVideoRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """
    Export video in specified format with optional cuts applied.
    
    Formats:
    - mp4: MP4 (H.264) - Default delivery
    - mov_prores422: MOV (ProRes 422) - For real editors & post houses
    - mov_prores4444: MOV (ProRes 4444) - For real editors & post houses
    - webm: WebM (VP9) - YouTube optimization / modern web
    - xml: XML (Final Cut Pro)
    - edl: EDL (Premiere / Resolve)
    - aaf: AAF (Avid / Pro pipelines)
    """
    try:
        from utils.video_export import export_video as export_video_func, EXPORT_FORMATS
        
        video_uuid = UUID(video_id)
        
        if request.format not in EXPORT_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported format. Supported formats: {', '.join(EXPORT_FORMATS.keys())}"
            )
        
        segments_to_remove = None
        if request.segments_to_remove:
            segments_to_remove = [
                (seg["start_time"], seg["end_time"])
                for seg in request.segments_to_remove
            ]
        
        output_path, mime_type = await export_video_func(
            video_uuid,
            request.format,
            segments_to_remove,
            db
        )
        
        format_info = EXPORT_FORMATS[request.format]
        filename = f"export_{video_id}.{format_info['extension']}"
        
        def cleanup_file():
            try:
                if os.path.exists(output_path):
                    os.remove(output_path)
            except Exception:
                pass
        
        background_tasks.add_task(cleanup_file)
        
        return FileResponse(
            path=output_path,
            media_type=mime_type,
            filename=filename,
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting video: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/timelines/{video_id}", status_code=200)
async def save_timeline(
    video_id: str,
    request: dict,
    db: AsyncSession = Depends(get_db),
    clerk_user_id: Optional[str] = Depends(get_clerk_user_id),
):
    """
    Save timeline state for a video.
    
    Creates or updates a timeline with editing state (markers, selections, etc.).
    """
    try:
        video_uuid = UUID(video_id)
        video = await crud.get_video_by_id(db, video_uuid)
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")
        
        timeline = await crud.create_or_update_timeline(
            db=db,
            video_id=video_uuid,
            clerk_user_id=clerk_user_id,
            project_name=request.get("projectName"),
            markers=request.get("markers"),
            selections=request.get("selections"),
            current_time=request.get("currentTime"),
            in_point=request.get("inPoint"),
            out_point=request.get("outPoint"),
            zoom=request.get("zoom"),
            view_preferences=request.get("viewPreferences"),
        )
        
        logger.info(f"Timeline saved for video {video_id}")
        
        return {
            "status": "success",
            "video_id": str(video_uuid),
            "timeline_id": str(timeline.id),
        }
    
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid video ID format")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error saving timeline: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/timelines/{video_id}", status_code=200)
async def get_timeline(
    video_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Get timeline state for a video.
    """
    try:
        video_uuid = UUID(video_id)
        timeline = await crud.get_timeline_by_video_id(db, video_uuid)
        
        if not timeline:
            # Return default empty timeline state
            return {
                "video_id": str(video_uuid),
                "project_name": None,
                "markers": [],
                "selections": [],
                "current_time": 0.0,
                "in_point": None,
                "out_point": None,
                "zoom": 1.0,
                "view_preferences": {
                    "snapEnabled": True,
                    "loopPlayback": False,
                },
            }
        
        return {
            "video_id": str(video_uuid),
            "project_name": timeline.project_name,
            "markers": timeline.markers or [],
            "selections": timeline.selections or [],
            "current_time": timeline.current_time or 0.0,
            "in_point": timeline.in_point,
            "out_point": timeline.out_point,
            "zoom": timeline.zoom or 1.0,
            "view_preferences": timeline.view_preferences or {
                "snapEnabled": True,
                "loopPlayback": False,
            },
        }
    
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid video ID format")
    except Exception as e:
        logger.error(f"Error retrieving timeline: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

