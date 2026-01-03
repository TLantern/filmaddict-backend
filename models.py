from datetime import datetime
from enum import Enum
from typing import List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict


class FeedbackType(str, Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    CONFIDENCE_SCORE = "CONFIDENCE_SCORE"
    SKIP = "SKIP"


class VideoStatus(str, Enum):
    UPLOADED = "UPLOADED"
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    TRANSCRIBED = "TRANSCRIBED"
    HIGHLIGHTS_FOUND = "HIGHLIGHTS_FOUND"
    SEGMENTS_ANALYZED = "SEGMENTS_ANALYZED"
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


class SpeechSegment(BaseModel):
    """Speech segment with transcript text."""
    type: str = Field(default="speech", description="Segment type identifier")
    start: float = Field(..., ge=0, description="Start time in seconds")
    end: float = Field(..., ge=0, description="End time in seconds")
    text: str = Field(..., description="Transcript text")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "type": "speech",
                "start": 0.0,
                "end": 5.5,
                "text": "Hello, welcome to this video.",
            }
        }
    )


class SilenceSegment(BaseModel):
    """Silence segment with no transcript text."""
    type: str = Field(default="silence", description="Segment type identifier")
    start: float = Field(..., ge=0, description="Start time in seconds")
    end: float = Field(..., ge=0, description="End time in seconds")
    duration: float = Field(..., ge=0, description="Duration in seconds")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "type": "silence",
                "start": 5.5,
                "end": 6.34,
                "duration": 0.84,
            }
        }
    )


TimelineItem = Union[SpeechSegment, SilenceSegment]
"""Unified timeline item representing either speech or silence."""


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


class VerdictExplanation(BaseModel):
    verdict: str = Field(..., description="Verdict type: FLUFF or HIGHLIGHT")
    confidence: str = Field(..., description="Confidence level: low, medium, or high")
    evidence: List[str] = Field(..., max_length=3, description="List of evidence strings (max 3)")
    action_hint: str = Field(..., description="Action hint for the user")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "verdict": "FLUFF",
                "confidence": "high",
                "evidence": [
                    "Repeats earlier content",
                    "Low information density",
                    "Flat emotional delivery"
                ],
                "action_hint": "Remove"
            }
        }
    )


class Highlight(BaseModel):
    start: float = Field(..., ge=0, description="Start time in seconds")
    end: float = Field(..., ge=0, description="End time in seconds")
    title: Optional[str] = Field(None, description="Short catchy title for the moment")
    summary: Optional[str] = Field(None, description="2-3 sentence summary of what happens in the moment")
    score: float = Field(..., ge=1, le=10, description="Score from 1 to 10")
    explanation: Optional[VerdictExplanation] = Field(None, description="Unified verdict explanation")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "start": 120.0,
                "end": 180.0,
                "title": "The Moment Everything Changed",
                "summary": "The speaker shares a pivotal moment from their childhood. They describe how this experience shaped their worldview and led to their career path.",
                "score": 8.5,
            }
        }
    )


class MomentRecord(BaseModel):
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
                "storage_path": "/moments/moment_123.mp4",
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
    error_message: Optional[str] = Field(None, description="Error message if processing failed")

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


class MomentResponse(BaseModel):
    id: UUID
    moment_url: str = Field(..., description="URL to download the moment")
    start: float = Field(..., ge=0, description="Start time in seconds")
    end: float = Field(..., ge=0, description="End time in seconds")
    thumbnail_url: Optional[str] = Field(None, description="URL to the thumbnail image")

    model_config = ConfigDict(
        json_encoders={UUID: str},
        json_schema_extra={
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174001",
                "moment_url": "/moments/123e4567-e89b-12d3-a456-426614174001/download",
                "start": 120.0,
                "end": 180.0,
                "thumbnail_url": "/moments/123e4567-e89b-12d3-a456-426614174001/thumbnail",
            }
        },
    )


class MomentsResponse(BaseModel):
    video_id: UUID
    moments: List[MomentResponse]

    model_config = ConfigDict(
        json_encoders={UUID: str},
        json_schema_extra={
            "example": {
                "video_id": "123e4567-e89b-12d3-a456-426614174000",
                "moments": [
                    {
                        "id": "123e4567-e89b-12d3-a456-426614174001",
                        "moment_url": "/moments/123e4567-e89b-12d3-a456-426614174001/download",
                        "start": 120.0,
                        "end": 180.0,
                        "thumbnail_url": "/moments/123e4567-e89b-12d3-a456-426614174001/thumbnail",
                    }
                ],
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
                "performance_metrics": {"avg_confidence_score": 75.0, "save_rate": 0.2, "sample_size": 50},
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


class EditMomentRequest(BaseModel):
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


class MomentDetailResponse(BaseModel):
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
                "thumbnail_url": "/moments/123e4567-e89b-12d3-a456-426614174001/thumbnail",
            }
        }
    )


class SavedMomentResponse(BaseModel):
    id: UUID
    moment_id: UUID
    highlight_id: Optional[UUID] = None
    created_at: datetime

    model_config = ConfigDict(
        json_encoders={UUID: str},
        json_schema_extra={
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174001",
                "moment_id": "123e4567-e89b-12d3-a456-426614174002",
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
    moment_count: int = Field(..., description="Number of moments in this project")
    thumbnail_url: Optional[str] = Field(None, description="URL to the video thumbnail image")
    project_name: Optional[str] = Field(None, description="Name of the project/timeline")

    model_config = ConfigDict(
        json_encoders={UUID: str},
        json_schema_extra={
            "example": {
                "video_id": "123e4567-e89b-12d3-a456-426614174000",
                "created_at": "2024-01-01T00:00:00Z",
                "duration": 3600.5,
                "moment_count": 5,
                "thumbnail_url": "/videos/123e4567-e89b-12d3-a456-426614174000/thumbnail",
                "project_name": "My Video Project",
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
                        "moment_count": 5,
                    }
                ]
            }
        },
    )


class Word(BaseModel):
    start: float = Field(..., ge=0, description="Start time in seconds")
    end: float = Field(..., ge=0, description="End time in seconds")
    word: str = Field(..., description="Word text")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score 0-1")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "start": 0.0,
                "end": 0.5,
                "word": "Hello",
                "confidence": 0.95,
            }
        }
    )


class Sentence(BaseModel):
    start: float = Field(..., ge=0, description="Start time in seconds")
    end: float = Field(..., ge=0, description="End time in seconds")
    text: str = Field(..., description="Sentence text")
    words: List[Word] = Field(..., description="Word-level timestamps")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "start": 0.0,
                "end": 5.5,
                "text": "Hello, welcome to this video.",
                "words": [
                    {"start": 0.0, "end": 0.5, "word": "Hello", "confidence": 0.95},
                ],
            }
        }
    )


class SemanticSegment(BaseModel):
    segment_id: int = Field(..., description="Sequential segment number")
    start_time: float = Field(..., ge=0, description="Start time in seconds")
    end_time: float = Field(..., ge=0, description="End time in seconds")
    text: str = Field(..., description="Segment transcript text")
    embedding: Optional[List[float]] = Field(None, description="BGE embedding vector")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "segment_id": 1,
                "start_time": 0.0,
                "end_time": 30.5,
                "text": "Hello, welcome to this video. Today we'll be discussing highlights.",
                "embedding": None,
            }
        }
    )


class SegmentAnalysis(BaseModel):
    id: Optional[UUID] = Field(None, description="Segment ID")
    start_time: float = Field(..., ge=0, description="Start time in seconds")
    end_time: float = Field(..., ge=0, description="End time in seconds")
    label: str = Field(..., description="Label: FLUFF")
    rating: float = Field(..., ge=0, le=1, description="Usefulness rating 0.0-1.0")
    grade: str = Field(..., description="Grade: A, B, C, D, or F")
    reason: str = Field(..., description="Explanation for the classification")
    repetition_score: float = Field(..., ge=0, le=1, description="Repetition score 0.0-1.0")
    filler_density: float = Field(..., ge=0, le=1, description="Filler word density 0.0-1.0")
    visual_change_score: float = Field(..., ge=0, le=1, description="Visual change score 0.0-1.0")
    usefulness_score: float = Field(..., ge=0, le=1, description="Overall usefulness score 0.0-1.0")
    explanation: Optional[VerdictExplanation] = Field(None, description="Unified verdict explanation")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "start_time": 312.4,
                "end_time": 347.9,
                "label": "FLUFF",
                "rating": 0.18,
                "grade": "F",
                "reason": "Repeats earlier explanation with no visual change",
                "repetition_score": 0.95,
                "filler_density": 0.3,
                "visual_change_score": 0.1,
                "usefulness_score": 0.18,
            }
        }
    )


class SegmentsResponse(BaseModel):
    video_id: UUID
    segments: List[SegmentAnalysis]

    model_config = ConfigDict(
        json_encoders={UUID: str},
        json_schema_extra={
            "example": {
                "video_id": "123e4567-e89b-12d3-a456-426614174000",
                "segments": [
                    {
                        "start_time": 312.4,
                        "end_time": 347.9,
                        "label": "FLUFF",
                        "rating": 0.18,
                        "reason": "Repeats earlier explanation with no visual change",
                        "repetition_score": 0.95,
                        "filler_density": 0.3,
                        "visual_change_score": 0.1,
                        "usefulness_score": 0.18,
                    }
                ],
            }
        },
    )


class SegmentFeedbackRequest(BaseModel):
    feedback_type: str = Field(..., description="Feedback type: GREAT, FINE, or WRONG")
    start_time: Optional[float] = Field(None, description="Start time in seconds (fallback if segment_id not provided)")
    end_time: Optional[float] = Field(None, description="End time in seconds (fallback if segment_id not provided)")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "feedback_type": "GREAT",
                "start_time": 55.7,
                "end_time": 66.6,
            }
        },
    )


class SegmentFeedbackResponse(BaseModel):
    id: UUID
    video_segment_id: UUID
    feedback_type: str
    created_at: datetime

    model_config = ConfigDict(
        json_encoders={UUID: str},
        json_schema_extra={
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "video_segment_id": "123e4567-e89b-12d3-a456-426614174001",
                "feedback_type": "GREAT",
                "created_at": "2024-01-01T00:00:00Z",
            }
        },
    )


class CutSegmentRequest(BaseModel):
    start_time: float = Field(..., ge=0, description="Start time in seconds")
    end_time: float = Field(..., ge=0, description="End time in seconds")


class CutVideoRequest(BaseModel):
    segments_to_remove: List[CutSegmentRequest] = Field(..., description="List of segments to remove from the video")

    model_config = ConfigDict(
        json_encoders={UUID: str},
        json_schema_extra={
            "example": {
                "video_id": "123e4567-e89b-12d3-a456-426614174000",
                "segments": [
                    {
                        "start_time": 312.4,
                        "end_time": 347.9,
                        "label": "FLUFF",
                        "rating": 0.18,
                        "reason": "Repeats earlier explanation with no visual change",
                        "repetition_score": 0.95,
                        "filler_density": 0.3,
                        "visual_change_score": 0.1,
                        "usefulness_score": 0.18,
                    }
                ],
            }
        },
    )


class TimeRange(BaseModel):
    start: float = Field(..., ge=0, description="Start time in seconds")
    end: float = Field(..., ge=0, description="End time in seconds")
    duration: float = Field(..., ge=0, description="Duration in seconds")


class SemanticNoveltyMetrics(BaseModel):
    value: float = Field(..., ge=0, le=1, description="Semantic novelty score 0.0-1.0")
    max_similarity_to_history: float = Field(..., ge=0, le=1, description="Maximum similarity to recent history")
    window_size: int = Field(..., ge=1, description="Number of segments in comparison window")


class InformationDensityMetrics(BaseModel):
    value: float = Field(..., ge=0, description="Information density score")
    meaningful_token_count: int = Field(..., ge=0, description="Count of meaningful tokens")
    tfidf_weight_sum: float = Field(..., ge=0, description="Sum of TF-IDF weights")
    duration_seconds: float = Field(..., ge=0, description="Segment duration in seconds")


class EmotionalDeltaMetrics(BaseModel):
    value: float = Field(..., description="Emotional delta score (can be negative)")
    sentiment_prev: float = Field(..., description="Previous segment sentiment")
    sentiment_curr: float = Field(..., description="Current segment sentiment")
    intensity_prev: float = Field(..., ge=0, description="Previous segment intensity")
    intensity_curr: float = Field(..., ge=0, description="Current segment intensity")


class NarrativeMomentumMetrics(BaseModel):
    value: float = Field(..., ge=0, description="Narrative momentum score")
    new_entities: int = Field(..., ge=0, description="Count of new entities")
    new_events: int = Field(..., ge=0, description="Count of new events")
    new_goals: int = Field(..., ge=0, description="Count of new goals")
    new_stakes: int = Field(..., ge=0, description="Count of new stakes")


class RetentionMetrics(BaseModel):
    semantic_novelty: SemanticNoveltyMetrics
    information_density: InformationDensityMetrics
    emotional_delta: EmotionalDeltaMetrics
    narrative_momentum: NarrativeMomentumMetrics


class RetentionDecision(BaseModel):
    action: str = Field(..., description="Decision: KEEP or CUT")
    reason: str = Field(..., description="Explanation for the decision")


class RetentionAnalysis(BaseModel):
    video_id: str = Field(..., description="Video ID")
    segment_id: int = Field(..., ge=1, description="Segment ID")
    time_range: TimeRange
    text: str = Field(..., description="Segment text")
    metrics: RetentionMetrics
    retention_value: float = Field(..., ge=0, le=1, description="Retention value 0.0-1.0")
    decision: RetentionDecision

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "video_id": "vid_0482",
                "segment_id": 17,
                "time_range": {"start": 42.5, "end": 45.6, "duration": 3.1},
                "text": "We realized the server crash happened because memory was leaking every hour.",
                "metrics": {
                    "semantic_novelty": {
                        "value": 0.81,
                        "max_similarity_to_history": 0.19,
                        "window_size": 20
                    },
                    "information_density": {
                        "value": 0.74,
                        "meaningful_token_count": 9,
                        "tfidf_weight_sum": 5.6,
                        "duration_seconds": 3.1
                    },
                    "emotional_delta": {
                        "value": 0.43,
                        "sentiment_prev": -0.12,
                        "sentiment_curr": -0.41,
                        "intensity_prev": 0.34,
                        "intensity_curr": 0.62
                    },
                    "narrative_momentum": {
                        "value": 0.67,
                        "new_entities": 1,
                        "new_events": 1,
                        "new_goals": 0,
                        "new_stakes": 1
                    }
                },
                "retention_value": 0.67,
                "decision": {
                    "action": "KEEP",
                    "reason": "High semantic novelty and strong narrative progression."
                }
            }
        }
    )

