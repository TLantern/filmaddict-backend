from datetime import datetime
from enum import Enum
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict


class VideoStatus(str, Enum):
    UPLOADED = "UPLOADED"
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    TRANSCRIBED = "TRANSCRIBED"
    HIGHLIGHTS_FOUND = "HIGHLIGHTS_FOUND"
    DONE = "DONE"
    FAILED = "FAILED"


class VideoSource(str, Enum):
    FILE = "file"
    YOUTUBE = "YouTube"


class VideoInput(BaseModel):
    type: VideoSource
    source: VideoSource = Field(..., description="Source type: file or YouTube")
    url: Optional[str] = Field(None, description="YouTube URL if source is YouTube")
    file_id: Optional[str] = Field(None, description="File ID if source is file upload")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "type": "YouTube",
                "source": "YouTube",
                "url": "https://www.youtube.com/watch?v=example",
            }
        }
    )


class VideoRecord(BaseModel):
    id: UUID
    storage_path: str
    duration: Optional[float] = Field(None, ge=0, description="Duration in seconds")
    status: VideoStatus
    created_at: datetime

    model_config = ConfigDict(
        json_encoders={UUID: str},
        json_schema_extra={
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "storage_path": "/videos/video_123.mp4",
                "duration": 3600.5,
                "status": "PROCESSING",
                "created_at": "2024-01-01T00:00:00Z",
            }
        },
    )


class TranscriptSegment(BaseModel):
    start: float = Field(..., ge=0, description="Start time in seconds")
    end: float = Field(..., ge=0, description="End time in seconds")
    text: str

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "start": 0.0,
                "end": 5.5,
                "text": "Hello, welcome to this video.",
            }
        }
    )


class TranscriptChunk(BaseModel):
    start: float = Field(..., ge=0, description="Start time in seconds")
    end: float = Field(..., ge=0, description="End time in seconds")
    text: str = Field(..., description="Combined text from segments")
    segments: List[TranscriptSegment] = Field(..., description="Original segments in this chunk")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "start": 0.0,
                "end": 45.2,
                "text": "Hello, welcome to this video. Today we'll be discussing highlights.",
                "segments": [
                    {"start": 0.0, "end": 5.5, "text": "Hello, welcome to this video."},
                    {"start": 5.5, "end": 45.2, "text": "Today we'll be discussing highlights."},
                ],
            }
        }
    )


class Highlight(BaseModel):
    start: float = Field(..., ge=0, description="Start time in seconds")
    end: float = Field(..., ge=0, description="End time in seconds")
    reason: str = Field(..., description="Reason why this is a highlight")
    score: float = Field(..., ge=1, le=10, description="Score from 1 to 10")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "start": 120.0,
                "end": 180.0,
                "reason": "High emotional intensity moment",
                "score": 8.5,
            }
        }
    )


class ClipRecord(BaseModel):
    id: UUID
    video_id: UUID
    start: float = Field(..., ge=0, description="Start time in seconds")
    end: float = Field(..., ge=0, description="End time in seconds")
    storage_path: str
    thumbnail_path: Optional[str] = Field(None, description="Path to thumbnail image")

    model_config = ConfigDict(
        json_encoders={UUID: str},
        json_schema_extra={
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174001",
                "video_id": "123e4567-e89b-12d3-a456-426614174000",
                "start": 120.0,
                "end": 180.0,
                "storage_path": "/clips/clip_123.mp4",
                "thumbnail_path": "/thumbnails/thumb_123.jpg",
            }
        },
    )


class TranscriptResponse(BaseModel):
    video_id: UUID
    segments: list[TranscriptSegment]

    model_config = ConfigDict(
        json_encoders={UUID: str},
        json_schema_extra={
            "example": {
                "video_id": "123e4567-e89b-12d3-a456-426614174000",
                "segments": [
                    {
                        "start": 0.0,
                        "end": 5.5,
                        "text": "Hello, welcome to this video.",
                    },
                    {
                        "start": 5.5,
                        "end": 10.2,
                        "text": "Today we'll be discussing highlights.",
                    },
                ],
            }
        },
    )


class VideoStatusResponse(BaseModel):
    video_id: UUID
    status: VideoStatus
    duration: Optional[float] = Field(None, ge=0, description="Duration in seconds")
    created_at: datetime

    model_config = ConfigDict(
        json_encoders={UUID: str},
        json_schema_extra={
            "example": {
                "video_id": "123e4567-e89b-12d3-a456-426614174000",
                "status": "PROCESSING",
                "duration": 3600.5,
                "created_at": "2024-01-01T00:00:00Z",
            }
        },
    )


class HighlightsResponse(BaseModel):
    video_id: UUID
    highlights: List[Highlight]

    model_config = ConfigDict(
        json_encoders={UUID: str},
        json_schema_extra={
            "example": {
                "video_id": "123e4567-e89b-12d3-a456-426614174000",
                "highlights": [
                    {
                        "start": 120.0,
                        "end": 180.0,
                        "reason": "High emotional intensity moment",
                        "score": 8.5,
                    }
                ],
            }
        },
    )


class ClipResponse(BaseModel):
    id: UUID
    clip_url: str = Field(..., description="URL to download the clip")
    start: float = Field(..., ge=0, description="Start time in seconds")
    end: float = Field(..., ge=0, description="End time in seconds")
    thumbnail_url: Optional[str] = Field(None, description="URL to the thumbnail image")

    model_config = ConfigDict(
        json_encoders={UUID: str},
        json_schema_extra={
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174001",
                "clip_url": "/clips/123e4567-e89b-12d3-a456-426614174001/download",
                "start": 120.0,
                "end": 180.0,
                "thumbnail_url": "/clips/123e4567-e89b-12d3-a456-426614174001/thumbnail",
            }
        },
    )


class ClipsResponse(BaseModel):
    video_id: UUID
    clips: List[ClipResponse]

    model_config = ConfigDict(
        json_encoders={UUID: str},
        json_schema_extra={
            "example": {
                "video_id": "123e4567-e89b-12d3-a456-426614174000",
                "clips": [
                    {
                        "id": "123e4567-e89b-12d3-a456-426614174001",
                        "clip_url": "/clips/123e4567-e89b-12d3-a456-426614174001/download",
                        "start": 120.0,
                        "end": 180.0,
                        "thumbnail_url": "/clips/123e4567-e89b-12d3-a456-426614174001/thumbnail",
                    }
                ],
            }
        },
    )


class FeedbackType(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    VIEW = "view"
    SKIP = "skip"
    SHARE = "share"
    RATING = "rating"


class HighlightFeedbackRequest(BaseModel):
    feedback_type: FeedbackType = Field(..., description="Type of feedback")
    rating: Optional[float] = Field(None, ge=0, le=100, description="Rating from 0-100 (required for rating type)")
    text_feedback: Optional[str] = Field(None, description="Optional text feedback")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "feedback_type": "rating",
                "rating": 85.0,
                "text_feedback": "Great moment!",
            }
        }
    )


class FeedbackAnalyticsResponse(BaseModel):
    highlight_id: UUID
    total_feedback: int
    positive_count: int
    negative_count: int
    view_count: int
    skip_count: int
    share_count: int
    average_rating: Optional[float] = Field(None, description="Average rating if ratings exist")
    rating_count: int
    created_at: datetime

    model_config = ConfigDict(
        json_encoders={UUID: str},
        json_schema_extra={
            "example": {
                "highlight_id": "123e4567-e89b-12d3-a456-426614174000",
                "total_feedback": 25,
                "positive_count": 15,
                "negative_count": 2,
                "view_count": 20,
                "skip_count": 5,
                "share_count": 3,
                "average_rating": 7.8,
                "rating_count": 10,
                "created_at": "2024-01-01T00:00:00Z",
            }
        },
    )


class PromptVersionResponse(BaseModel):
    id: UUID
    version_name: str
    system_prompt: str
    user_prompt_template: str
    is_active: bool
    performance_metrics: Optional[dict] = None
    created_at: datetime

    model_config = ConfigDict(
        json_encoders={UUID: str},
        json_schema_extra={
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "version_name": "v1",
                "system_prompt": "You are an expert...",
                "user_prompt_template": "Chunk time range...",
                "is_active": True,
                "performance_metrics": {"avg_rating": 75.0, "save_rate": 0.2, "sample_size": 50},
                "created_at": "2024-01-01T00:00:00Z",
            }
        },
    )


class PromptVersionRequest(BaseModel):
    version_name: str = Field(..., description="Unique version name")
    system_prompt: str = Field(..., description="System prompt for GPT")
    user_prompt_template: str = Field(..., description="User prompt template")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "version_name": "v2",
                "system_prompt": "You are an expert at identifying engaging moments...",
                "user_prompt_template": "Chunk time range: {start} - {end} seconds\nTranscript text: {text}\n...",
            }
        }
    )


class ClipFeedbackRequest(BaseModel):
    rating: float = Field(..., ge=0, le=100, description="Rating from 0-100")
    text_feedback: Optional[str] = Field(None, description="Optional text feedback")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "rating": 85.0,
                "text_feedback": "Great clip!",
            }
        }
    )


class EditClipRequest(BaseModel):
    new_start: float = Field(..., ge=0, description="New start time in seconds")
    new_end: float = Field(..., ge=0, description="New end time in seconds")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "new_start": 120.0,
                "new_end": 180.0,
            }
        }
    )


class ClipDetailResponse(BaseModel):
    id: UUID
    video_id: UUID
    start: float = Field(..., ge=0, description="Start time in seconds")
    end: float = Field(..., ge=0, description="End time in seconds")
    video_url: str = Field(..., description="Parent video URL for playback")
    video_duration: float = Field(..., ge=0, description="Parent video duration in seconds")
    thumbnail_url: Optional[str] = Field(None, description="URL to the thumbnail image")

    model_config = ConfigDict(
        json_encoders={UUID: str},
        json_schema_extra={
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174001",
                "video_id": "123e4567-e89b-12d3-a456-426614174000",
                "start": 120.0,
                "end": 180.0,
                "video_url": "https://s3.amazonaws.com/bucket/videos/video.mp4",
                "video_duration": 3600.0,
                "thumbnail_url": "/clips/123e4567-e89b-12d3-a456-426614174001/thumbnail",
            }
        }
    )


class SavedClipResponse(BaseModel):
    id: UUID
    clip_id: UUID
    highlight_id: Optional[UUID] = None
    created_at: datetime

    model_config = ConfigDict(
        json_encoders={UUID: str},
        json_schema_extra={
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174001",
                "clip_id": "123e4567-e89b-12d3-a456-426614174002",
                "highlight_id": "123e4567-e89b-12d3-a456-426614174003",
                "created_at": "2024-01-01T00:00:00Z",
            }
        }
    )


class CalibrationConfigResponse(BaseModel):
    score_offset: float = Field(..., description="Calibration offset to apply to scores")
    last_updated: datetime
    sample_size: int = Field(..., description="Number of samples used for calibration")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "score_offset": 0.5,
                "last_updated": "2024-01-01T00:00:00Z",
                "sample_size": 150,
            }
        },
    )


class ProjectResponse(BaseModel):
    video_id: UUID
    created_at: datetime
    duration: Optional[float] = Field(None, description="Video duration in seconds")
    clip_count: int = Field(..., description="Number of clips in this project")

    model_config = ConfigDict(
        json_encoders={UUID: str},
        json_schema_extra={
            "example": {
                "video_id": "123e4567-e89b-12d3-a456-426614174000",
                "created_at": "2024-01-01T00:00:00Z",
                "duration": 3600.5,
                "clip_count": 5,
            }
        },
    )


class ProjectsResponse(BaseModel):
    projects: List[ProjectResponse]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "projects": [
                    {
                        "video_id": "123e4567-e89b-12d3-a456-426614174000",
                        "created_at": "2024-01-01T00:00:00Z",
                        "duration": 3600.5,
                        "clip_count": 5,
                    }
                ]
            }
        },
    )

