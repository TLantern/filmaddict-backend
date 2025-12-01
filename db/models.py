from datetime import datetime
from uuid import uuid4

from sqlalchemy import Column, String, Float, ForeignKey, DateTime, JSON, Index, Text, Boolean, Integer
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship

from database import Base
from models import VideoStatus


class Video(Base):
    __tablename__ = "videos"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    storage_path = Column(String, nullable=False)
    duration = Column(Float, nullable=True)
    status = Column(String, nullable=False, default=VideoStatus.UPLOADED.value)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    transcripts = relationship("Transcript", back_populates="video", cascade="all, delete-orphan")
    highlights = relationship("Highlight", back_populates="video", cascade="all, delete-orphan")
    clips = relationship("Clip", back_populates="video", cascade="all, delete-orphan")

    __table_args__ = (Index("idx_videos_status", "status"), Index("idx_videos_created_at", "created_at"))


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
    reason = Column(String, nullable=False)
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


class Clip(Base):
    __tablename__ = "clips"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    video_id = Column(UUID(as_uuid=True), ForeignKey("videos.id", ondelete="CASCADE"), nullable=False)
    start = Column(Float, nullable=False)
    end = Column(Float, nullable=False)
    storage_path = Column(String, nullable=False)
    thumbnail_path = Column(String, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    video = relationship("Video", back_populates="clips")
    saved_clips = relationship("SavedClip", back_populates="clip", cascade="all, delete-orphan")

    __table_args__ = (Index("idx_clips_video_id", "video_id"),)


class HighlightFeedback(Base):
    __tablename__ = "highlight_feedback"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    highlight_id = Column(UUID(as_uuid=True), ForeignKey("highlights.id", ondelete="CASCADE"), nullable=False)
    feedback_type = Column(String, nullable=False)
    rating = Column(Float, nullable=True)  # Now supports 0-100 scale
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
    sum_ratings = Column(Float, nullable=False, default=0.0)
    avg_rating = Column(Float, nullable=False, default=0.0)
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


class SavedClip(Base):
    __tablename__ = "saved_clips"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    clip_id = Column(UUID(as_uuid=True), ForeignKey("clips.id", ondelete="CASCADE"), nullable=False)
    highlight_id = Column(UUID(as_uuid=True), ForeignKey("highlights.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    clip = relationship("Clip", back_populates="saved_clips")
    highlight = relationship("Highlight")

    __table_args__ = (
        Index("idx_saved_clips_clip_id", "clip_id"),
        Index("idx_saved_clips_highlight_id", "highlight_id"),
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

