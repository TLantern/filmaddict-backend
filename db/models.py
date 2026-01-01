from datetime import datetime
from uuid import uuid4

from sqlalchemy import Column, String, Float, ForeignKey, DateTime, JSON, Index, Text, Boolean, Integer, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship

from database import Base
from models import VideoStatus


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    email = Column(String, nullable=False, unique=True, index=True)
    name = Column(String, nullable=True)
    picture_url = Column(String, nullable=True)
    google_id = Column(String, nullable=False, unique=True, index=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("idx_users_email", "email"),
        Index("idx_users_google_id", "google_id"),
    )


class Video(Base):
    __tablename__ = "videos"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    storage_path = Column(String, nullable=False)
    duration = Column(Float, nullable=True)
    aspect_ratio = Column(String, nullable=True, default="16:9")  # 9:16, 16:9, 1:1, 4:5, original
    status = Column(String, nullable=False, default=VideoStatus.UPLOADED.value)
    pending_cuts = Column(JSONB, nullable=True)  # List of {start_time, end_time} segments to remove
    clerk_user_id = Column(String, nullable=True, index=True)  # Clerk user ID for user-based filtering
    thumbnail_path = Column(String, nullable=True)  # Path to thumbnail image in storage
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    transcripts = relationship("Transcript", back_populates="video", cascade="all, delete-orphan")
    highlights = relationship("Highlight", back_populates="video", cascade="all, delete-orphan")
    moments = relationship("Moment", back_populates="video", cascade="all, delete-orphan")
    video_segments = relationship("VideoSegment", back_populates="video", cascade="all, delete-orphan")
    retention_metrics = relationship("RetentionMetrics", back_populates="video", cascade="all, delete-orphan")
    timeline = relationship("Timeline", back_populates="video", uselist=False, cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_videos_status", "status"),
        Index("idx_videos_created_at", "created_at"),
        Index("idx_videos_clerk_user_id", "clerk_user_id"),
    )


class Transcript(Base):
    __tablename__ = "transcripts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    video_id = Column(UUID(as_uuid=True), ForeignKey("videos.id", ondelete="CASCADE"), nullable=False)
    segments = Column(JSONB, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    video = relationship("Video", back_populates="transcripts")

    __table_args__ = (Index("idx_transcripts_video_id", "video_id"),)


class Highlight(Base):
    __tablename__ = "highlights"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    video_id = Column(UUID(as_uuid=True), ForeignKey("videos.id", ondelete="CASCADE"), nullable=False)
    start = Column(Float, nullable=False)
    end = Column(Float, nullable=False)
    title = Column(String, nullable=True)  # Short catchy title for the moment
    summary = Column(Text, nullable=True)  # 2-3 sentence summary of moment content
    score = Column(Float, nullable=False)
    prompt_version_id = Column(UUID(as_uuid=True), ForeignKey("prompt_versions.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    video = relationship("Video", back_populates="highlights")
    prompt_version = relationship("PromptVersion", back_populates="highlights")
    feedback = relationship("HighlightFeedback", back_populates="highlight", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_highlights_video_id", "video_id"),
        Index("idx_highlights_score", "score"),
        Index("idx_highlights_prompt_version_id", "prompt_version_id"),
    )


class Moment(Base):
    __tablename__ = "moments"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    video_id = Column(UUID(as_uuid=True), ForeignKey("videos.id", ondelete="CASCADE"), nullable=False)
    start = Column(Float, nullable=False)
    end = Column(Float, nullable=False)
    storage_path = Column(String, nullable=False)
    thumbnail_path = Column(String, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    video = relationship("Video", back_populates="moments")
    saved_moments = relationship("SavedMoment", back_populates="moment", cascade="all, delete-orphan")

    __table_args__ = (Index("idx_moments_video_id", "video_id"),)


class HighlightFeedback(Base):
    __tablename__ = "highlight_feedback"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    highlight_id = Column(UUID(as_uuid=True), ForeignKey("highlights.id", ondelete="CASCADE"), nullable=False)
    feedback_type = Column(String, nullable=False)
    confidence_score = Column(Float, nullable=True)  # Confidence score 0-100 scale
    text_feedback = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    highlight = relationship("Highlight", back_populates="feedback")

    __table_args__ = (
        Index("idx_highlight_feedback_highlight_id", "highlight_id"),
        Index("idx_highlight_feedback_created_at", "created_at"),
        Index("idx_highlight_feedback_type", "feedback_type"),
    )


class PromptVersion(Base):
    __tablename__ = "prompt_versions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    version_name = Column(String, nullable=False, unique=True)
    system_prompt = Column(Text, nullable=False)
    user_prompt_template = Column(Text, nullable=False)
    is_active = Column(Boolean, nullable=False, default=False)
    performance_metrics = Column(JSONB, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    total_rated = Column(Integer, nullable=False, default=0)
    sum_confidence_scores = Column(Float, nullable=False, default=0.0)
    avg_confidence_score = Column(Float, nullable=False, default=0.0)
    num_positive = Column(Integer, nullable=False, default=0)
    num_negative = Column(Integer, nullable=False, default=0)
    positive_rate = Column(Float, nullable=False, default=0.0)
    negative_rate = Column(Float, nullable=False, default=0.0)
    total_saves = Column(Integer, nullable=False, default=0)
    save_rate = Column(Float, nullable=False, default=0.0)

    highlights = relationship("Highlight", back_populates="prompt_version")

    __table_args__ = (
        Index("idx_prompt_versions_version_name", "version_name"),
        Index("idx_prompt_versions_is_active", "is_active"),
    )


class SavedMoment(Base):
    __tablename__ = "saved_moments"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    moment_id = Column(UUID(as_uuid=True), ForeignKey("moments.id", ondelete="CASCADE"), nullable=False)
    highlight_id = Column(UUID(as_uuid=True), ForeignKey("highlights.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    moment = relationship("Moment", back_populates="saved_moments")
    highlight = relationship("Highlight")

    __table_args__ = (
        Index("idx_saved_moments_moment_id", "moment_id"),
        Index("idx_saved_moments_highlight_id", "highlight_id"),
    )


class CalibrationConfig(Base):
    __tablename__ = "calibration_config"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    score_offset = Column(Float, nullable=False, default=0.0)
    last_updated = Column(DateTime, nullable=False, default=datetime.utcnow)
    sample_size = Column(Integer, nullable=False, default=0)
    feedback_count = Column(Integer, nullable=False, default=0)
    sum_predicted = Column(Float, nullable=False, default=0.0)
    sum_actual = Column(Float, nullable=False, default=0.0)


class VideoSegment(Base):
    __tablename__ = "video_segments"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    video_id = Column(UUID(as_uuid=True), ForeignKey("videos.id", ondelete="CASCADE"), nullable=False)
    segment_id = Column(Integer, nullable=False)
    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=False)
    text = Column(Text, nullable=False)
    label = Column(String, nullable=False)  # FLUFF
    rating = Column(Float, nullable=False)  # 0.0-1.0
    grade = Column(String, nullable=False, default="C")  # A, B, C, D, F
    reason = Column(Text, nullable=False)
    repetition_score = Column(Float, nullable=False)  # 0.0-1.0
    filler_density = Column(Float, nullable=False)  # 0.0-1.0
    visual_change_score = Column(Float, nullable=False)  # 0.0-1.0
    usefulness_score = Column(Float, nullable=False)  # 0.0-1.0
    embedding = Column(JSONB, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    video = relationship("Video", back_populates="video_segments")
    feedback = relationship("SegmentFeedback", back_populates="video_segment", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_video_segments_video_id", "video_id"),
        Index("idx_video_segments_label", "label"),
        Index("idx_video_segments_rating", "rating"),
    )


class SegmentFeedback(Base):
    __tablename__ = "segment_feedback"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    video_segment_id = Column(UUID(as_uuid=True), ForeignKey("video_segments.id", ondelete="CASCADE"), nullable=False)
    feedback_type = Column(String, nullable=False)  # GREAT, FINE, WRONG
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    video_segment = relationship("VideoSegment", back_populates="feedback")

    __table_args__ = (
        Index("idx_segment_feedback_video_segment_id", "video_segment_id"),
        Index("idx_segment_feedback_created_at", "created_at"),
        Index("idx_segment_feedback_type", "feedback_type"),
    )


class RetentionMetrics(Base):
    __tablename__ = "retention_metrics"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    video_id = Column(UUID(as_uuid=True), ForeignKey("videos.id", ondelete="CASCADE"), nullable=False)
    segment_id = Column(Integer, nullable=False)
    time_range = Column(JSONB, nullable=False)  # {start, end, duration}
    text = Column(Text, nullable=False)
    metrics = Column(JSONB, nullable=False)  # {semantic_novelty, information_density, emotional_delta, narrative_momentum}
    retention_value = Column(Float, nullable=False)
    decision = Column(JSONB, nullable=False)  # {action, reason}
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    video = relationship("Video", back_populates="retention_metrics")

    __table_args__ = (
        Index("idx_retention_metrics_video_id", "video_id"),
        Index("idx_retention_metrics_segment_id", "segment_id"),
        Index("idx_retention_metrics_retention_value", "retention_value"),
    )


class Timeline(Base):
    __tablename__ = "timelines"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    video_id = Column(UUID(as_uuid=True), ForeignKey("videos.id", ondelete="CASCADE"), nullable=False)
    clerk_user_id = Column(String, nullable=True)
    project_name = Column(String, nullable=True)
    markers = Column(JSONB, nullable=True)
    selections = Column(JSONB, nullable=True)
    current_time = Column(Float, nullable=True, default=0.0)
    in_point = Column(Float, nullable=True)
    out_point = Column(Float, nullable=True)
    zoom = Column(Float, nullable=True, default=1.0)
    view_preferences = Column(JSONB, nullable=True)
    thumbnail_path = Column(String, nullable=True)  # Path to thumbnail image in storage
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    video = relationship("Video", back_populates="timeline")

    __table_args__ = (
        UniqueConstraint("video_id", name="uq_timelines_video_id"),
        Index("idx_timelines_video_id", "video_id"),
        Index("idx_timelines_clerk_user_id", "clerk_user_id"),
        Index("idx_timelines_created_at", "created_at"),
    )

