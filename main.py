import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import UUID

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Request
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession

from database import engine, Base, get_db, async_session_maker
from db.models import Video, Transcript, Highlight as HighlightDB, Moment
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
    HighlightFeedbackRequest,
    FeedbackAnalyticsResponse,
    FeedbackType,
    PromptVersionResponse,
    PromptVersionRequest,
    CalibrationConfigResponse,
    MomentFeedbackRequest,
    SavedMomentResponse,
    EditMomentRequest,
    MomentDetailResponse,
    ProjectResponse,
    ProjectsResponse,
)
from typing import List, Optional
from datetime import datetime
from utils.storage import store_video, get_storage_instance, S3Storage, get_video_path
from utils.youtube import download_youtube_video, validate_youtube_url
from utils.jobs import enqueue_video_processing, start_worker, stop_worker, start_learning_worker, stop_learning_worker
from utils.moments import generate_moment_async, generate_thumbnail_async, delete_moment_file
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
    await start_worker()
    await start_learning_worker()
    
    yield
    
    # Stop background workers
    await stop_learning_worker()
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
    return {"status": "ok"}


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


@app.post("/videos/upload", status_code=201)
async def upload_video(
    file: UploadFile = File(...),
    aspect_ratio: Optional[str] = Form("16:9"),
    db: AsyncSession = Depends(get_db),
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
        
        # Create video record in database
        video = await crud.create_video(
            db=db,
            storage_path=storage_path,
            status=VideoStatus.UPLOADED,
            aspect_ratio=aspect_ratio,
        )
        
        # Enqueue background job to process the video
        await enqueue_video_processing(video.id)
        
        # Update status to QUEUED
        video = await crud.update_video_status(db, video.id, VideoStatus.QUEUED)
        
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
        
        # Create video record in database
        video = await crud.create_video(
            db=db,
            storage_path=storage_path,
            duration=duration,
            status=VideoStatus.UPLOADED,
            aspect_ratio=aspect_ratio,
        )
        
        # Enqueue background job to process the video
        await enqueue_video_processing(video.id)
        
        # Update status to QUEUED
        video = await crud.update_video_status(db, video.id, VideoStatus.QUEUED)
        
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
        
        # Get moment count for logging
        moments_db = await crud.get_moments_by_video_id(db, video_uuid)
        moments_count = len(moments_db)
        logger.info(f"[Backend] Video {video_id} status: {video.status}, duration: {video.duration}, moments: {moments_count}")
        
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
        logger.error(f"Error retrieving video status: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error retrieving video status")


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
        
        highlights = [
            Highlight(
                start=h.start,
                end=h.end,
                title=h.title,
                summary=h.summary,
                score=h.score,
            )
            for h in highlights_db
        ]
        
        logger.info(f"[Backend] Returning {len(highlights)} highlights for video {video_id}")
        return HighlightsResponse(video_id=video_uuid, highlights=highlights)
    
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid video ID format")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving highlights: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error retrieving highlights")


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
    """
    try:
        video_uuid = UUID(video_id)
        video = await crud.get_video_by_id(db, video_uuid)
        
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")
        
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


@app.post("/moments/{moment_id}/feedback", status_code=201)
async def submit_moment_feedback(
    moment_id: str,
    feedback: MomentFeedbackRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Submit feedback for a moment.
    
    Finds the associated highlight and creates feedback with confidence score (0-100) and optional text.
    Returns 404 if moment not found.
    """
    try:
        moment_uuid = UUID(moment_id)
        
        # Get moment
        moment = await crud.get_moment_by_id(db, moment_uuid)
        if not moment:
            raise HTTPException(status_code=404, detail="Moment not found")
        
        # Find associated highlight by matching start/end times
        highlight = await db.execute(
            select(HighlightDB).where(
                HighlightDB.video_id == moment.video_id,
                HighlightDB.start <= moment.start,
                HighlightDB.end >= moment.end,
            ).limit(1)
        )
        highlight_obj = highlight.scalar_one_or_none()
        
        if not highlight_obj:
            raise HTTPException(status_code=404, detail="No associated highlight found for this moment")
        
        # Create feedback with confidence score and text
        feedback_record = await crud.create_feedback(
            db,
            highlight_id=highlight_obj.id,
            feedback_type=FeedbackType.CONFIDENCE_SCORE.value,
            confidence_score=feedback.confidence_score,
            text_feedback=feedback.text_feedback,
        )
        
        logger.info(f"Moment feedback submitted for moment {moment_id}: confidence_score={feedback.confidence_score}")
        
        return {"id": str(feedback_record.id), "status": "created"}
    
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid moment ID format")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting moment feedback: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error submitting moment feedback")


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
        
        # Delete all moments from database
        deleted_count = await crud.delete_all_moments(db)
        
        logger.info(f"Deleted {deleted_count} moments")
        
        return {"status": "deleted", "count": deleted_count}
    
    except Exception as e:
        logger.error(f"Error deleting all moments: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error deleting moments")


@app.post("/highlights/{highlight_id}/feedback", status_code=201)
async def submit_feedback(
    highlight_id: str,
    feedback: HighlightFeedbackRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Submit feedback for a highlight.
    
    Accepts explicit feedback (positive, negative, confidence_score) for a highlight.
    Returns 404 if highlight not found.
    """
    try:
        highlight_uuid = UUID(highlight_id)
        
        highlight = await db.execute(select(HighlightDB).where(HighlightDB.id == highlight_uuid))
        highlight_obj = highlight.scalar_one_or_none()
        
        if not highlight_obj:
            raise HTTPException(status_code=404, detail="Highlight not found")
        
        if feedback.feedback_type == FeedbackType.CONFIDENCE_SCORE.value and feedback.confidence_score is None:
            raise HTTPException(status_code=400, detail="Confidence score is required for confidence_score feedback type")
        
        feedback_record = await crud.create_feedback(
            db,
            highlight_id=highlight_uuid,
            feedback_type=feedback.feedback_type.value,
            confidence_score=feedback.confidence_score,
            text_feedback=feedback.text_feedback,
        )
        
        logger.info(f"Feedback submitted for highlight {highlight_id}: {feedback.feedback_type}")
        
        return {"id": str(feedback_record.id), "status": "created"}
    
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid highlight ID format")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting feedback: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error submitting feedback")


@app.post("/highlights/{highlight_id}/view", status_code=201)
async def track_highlight_view(
    highlight_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Track that a user viewed a highlight (implicit positive feedback).
    
    Returns 404 if highlight not found.
    """
    try:
        highlight_uuid = UUID(highlight_id)
        
        highlight = await db.execute(select(HighlightDB).where(HighlightDB.id == highlight_uuid))
        highlight_obj = highlight.scalar_one_or_none()
        
        if not highlight_obj:
            raise HTTPException(status_code=404, detail="Highlight not found")
        
        feedback_record = await crud.create_feedback(
            db,
            highlight_id=highlight_uuid,
            feedback_type=FeedbackType.VIEW.value,
            confidence_score=None,
        )
        
        return {"id": str(feedback_record.id), "status": "created"}
    
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid highlight ID format")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error tracking view: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error tracking view")


@app.post("/highlights/{highlight_id}/skip", status_code=201)
async def track_highlight_skip(
    highlight_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Track that a user skipped a highlight (implicit negative feedback).
    
    Returns 404 if highlight not found.
    """
    try:
        highlight_uuid = UUID(highlight_id)
        
        highlight = await db.execute(select(HighlightDB).where(HighlightDB.id == highlight_uuid))
        highlight_obj = highlight.scalar_one_or_none()
        
        if not highlight_obj:
            raise HTTPException(status_code=404, detail="Highlight not found")
        
        feedback_record = await crud.create_feedback(
            db,
            highlight_id=highlight_uuid,
            feedback_type=FeedbackType.SKIP.value,
            confidence_score=None,
        )
        
        return {"id": str(feedback_record.id), "status": "created"}
    
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid highlight ID format")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error tracking skip: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error tracking skip")


@app.post("/moments/{moment_id}/view", status_code=201)
async def track_moment_view(
    moment_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Track that a user viewed a moment (implicit positive feedback for associated highlight).
    
    Returns 404 if moment not found.
    """
    try:
        moment_uuid = UUID(moment_id)
        moment = await crud.get_moment_by_id(db, moment_uuid)
        
        if not moment:
            raise HTTPException(status_code=404, detail="Moment not found")
        
        highlight = await db.execute(
            select(HighlightDB).where(
                HighlightDB.video_id == moment.video_id,
                HighlightDB.start <= moment.start,
                HighlightDB.end >= moment.end,
            ).limit(1)
        )
        highlight_obj = highlight.scalar_one_or_none()
        
        if highlight_obj:
            feedback_record = await crud.create_feedback(
                db,
                highlight_id=highlight_obj.id,
                feedback_type=FeedbackType.VIEW.value,
                rating=None,
            )
            return {"id": str(feedback_record.id), "status": "created"}
        else:
            return {"status": "created", "note": "No associated highlight found"}
    
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid moment ID format")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error tracking moment view: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error tracking moment view")


@app.get("/highlights/{highlight_id}/analytics", response_model=FeedbackAnalyticsResponse)
async def get_highlight_analytics(
    highlight_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Get aggregated feedback metrics for a highlight.
    
    Returns analytics including view count, skip count, ratings, etc.
    Returns 404 if highlight not found.
    """
    try:
        highlight_uuid = UUID(highlight_id)
        
        highlight = await db.execute(select(HighlightDB).where(HighlightDB.id == highlight_uuid))
        highlight_obj = highlight.scalar_one_or_none()
        
        if not highlight_obj:
            raise HTTPException(status_code=404, detail="Highlight not found")
        
        stats = await crud.get_feedback_stats(db, highlight_uuid)
        
        return FeedbackAnalyticsResponse(
            highlight_id=highlight_uuid,
            total_feedback=stats["total_feedback"],
            positive_count=stats["positive_count"],
            negative_count=stats["negative_count"],
            view_count=stats["view_count"],
            skip_count=stats["skip_count"],
            share_count=stats["share_count"],
            average_rating=stats["average_rating"],
            rating_count=stats["rating_count"],
            created_at=highlight_obj.created_at,
        )
    
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid highlight ID format")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving analytics: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error retrieving analytics")


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


@app.get("/projects", response_model=ProjectsResponse)
async def get_projects(
    db: AsyncSession = Depends(get_db),
):
    """
    Get all projects (videos that have moments).
    
    Returns list of projects with video_id, created_at, duration, and moment_count.
    """
    try:
        logger.info("[Backend] /projects endpoint called - fetching videos with moments")
        videos = await crud.get_videos_with_moments(db)
        logger.info(f"[Backend] Found {len(videos)} videos with moments")
        
        projects = [
            ProjectResponse(
                video_id=video.id,
                created_at=video.created_at,
                duration=video.duration,
                moment_count=len(video.moments),
            )
            for video in videos
        ]
        
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


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

