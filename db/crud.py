import logging
from typing import List, Optional, Dict, Tuple, TYPE_CHECKING
from uuid import UUID

if TYPE_CHECKING:
    from db.models import RetentionMetrics

from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from datetime import datetime, timedelta

from db.models import Video, Transcript, Highlight, Moment, PromptVersion, CalibrationConfig, SavedMoment, VideoSegment, SegmentFeedback, Timeline
from models import VideoStatus

logger = logging.getLogger(__name__)


async def create_video(
    db: AsyncSession, storage_path: str, duration: Optional[float] = None, status: VideoStatus = VideoStatus.UPLOADED, aspect_ratio: Optional[str] = "16:9", clerk_user_id: Optional[str] = None, thumbnail_path: Optional[str] = None
) -> Video:
    video = Video(storage_path=storage_path, duration=duration, aspect_ratio=aspect_ratio, status=status.value, clerk_user_id=clerk_user_id, thumbnail_path=thumbnail_path)
    db.add(video)
    await db.flush()
    await db.commit()
    await db.refresh(video)
    return video


async def get_video_by_id(db: AsyncSession, video_id: UUID) -> Optional[Video]:
    result = await db.execute(select(Video).where(Video.id == video_id))
    return result.scalar_one_or_none()


async def update_video_status(db: AsyncSession, video_id: UUID, status: VideoStatus) -> Optional[Video]:
    await db.execute(update(Video).where(Video.id == video_id).values(status=status.value))
    await db.commit()
    return await get_video_by_id(db, video_id)


async def get_all_videos(db: AsyncSession, limit: int = 100, offset: int = 0) -> List[Video]:
    result = await db.execute(select(Video).limit(limit).offset(offset).order_by(Video.created_at.desc()))
    return list(result.scalars().all())


async def create_transcript(db: AsyncSession, video_id: UUID, segments: list) -> Transcript:
    # Delete old transcripts for this video to avoid duplicates
    await db.execute(delete(Transcript).where(Transcript.video_id == video_id))
    await db.flush()
    
    # Create new transcript
    transcript = Transcript(video_id=video_id, segments=segments)
    db.add(transcript)
    await db.commit()
    await db.refresh(transcript)
    return transcript


async def get_transcript_by_video_id(db: AsyncSession, video_id: UUID) -> Optional[Transcript]:
    # Return the most recent transcript if multiple exist
    result = await db.execute(
        select(Transcript)
        .where(Transcript.video_id == video_id)
        .order_by(Transcript.created_at.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


async def update_transcript_after_cuts(
    db: AsyncSession,
    video_id: UUID,
    segments_to_remove: List[Tuple[float, float]],
) -> Optional[Transcript]:
    """
    Update transcript segments after video cuts:
    1. Remove transcript segments that overlap with removed time ranges
    2. Adjust timestamps of remaining segments to account for cumulative removed time
    3. Update the transcript in the database
    4. Clear Redis cache for the transcript
    
    Args:
        db: Database session
        video_id: UUID of the video
        segments_to_remove: List of (start_time, end_time) tuples for segments to remove
        
    Returns:
        Updated Transcript object, or None if no transcript exists
    """
    if not segments_to_remove:
        return None
    
    # Get current transcript
    transcript = await get_transcript_by_video_id(db, video_id)
    if not transcript:
        logger.warning(f"No transcript found for video {video_id}, skipping transcript update")
        return None
    
    if not transcript.segments:
        logger.warning(f"Transcript for video {video_id} has no segments, skipping update")
        return None
    
    # Sort removed segments by start time
    sorted_removed = sorted(segments_to_remove, key=lambda x: x[0])
    
    def get_cumulative_removed_time(timestamp: float) -> float:
        """Calculate cumulative time removed before a given timestamp."""
        total = 0.0
        for removed_start, removed_end in sorted_removed:
            if removed_end <= timestamp:
                # Entire removed segment is before this timestamp
                total += removed_end - removed_start
            elif removed_start < timestamp:
                # Partial overlap - count only the part before timestamp
                total += timestamp - removed_start
        return total
    
    # Process transcript segments
    updated_segments = []
    segments_removed = 0
    
    for segment in transcript.segments:
        # Check if segment overlaps with any removed time range
        # A segment overlaps if: start < removed_end AND end > removed_start
        overlaps = False
        for removed_start, removed_end in sorted_removed:
            if segment.get("start", segment.get("start_time", 0)) < removed_end and \
               segment.get("end", segment.get("end_time", 0)) > removed_start:
                overlaps = True
                break
        
        if overlaps:
            segments_removed += 1
            continue  # Skip segments that overlap with removed ranges
        
        # Adjust timestamps by subtracting cumulative removed time
        old_start = segment.get("start", segment.get("start_time", 0))
        old_end = segment.get("end", segment.get("end_time", 0))
        
        cumulative_before_start = get_cumulative_removed_time(old_start)
        cumulative_before_end = get_cumulative_removed_time(old_end)
        
        new_start = max(0.0, old_start - cumulative_before_start)
        new_end = max(new_start, old_end - cumulative_before_end)
        
        # Create updated segment dict
        updated_segment = dict(segment)
        if "start" in updated_segment:
            updated_segment["start"] = new_start
        if "start_time" in updated_segment:
            updated_segment["start_time"] = new_start
        if "end" in updated_segment:
            updated_segment["end"] = new_end
        if "end_time" in updated_segment:
            updated_segment["end_time"] = new_end
        
        updated_segments.append(updated_segment)
    
    # Update transcript in database
    await create_transcript(db, video_id, updated_segments)
    
    # Clear Redis cache
    try:
        from utils.redis_cache import delete_transcript_from_redis
        delete_transcript_from_redis(video_id)
        logger.info(f"Cleared Redis cache for transcript of video {video_id}")
    except Exception as e:
        logger.warning(f"Failed to clear Redis cache for transcript: {str(e)}")
    
    logger.info(f"Updated transcript for video {video_id}: removed {segments_removed} segments, {len(updated_segments)} segments remaining")
    
    return await get_transcript_by_video_id(db, video_id)


async def create_highlight(
    db: AsyncSession, video_id: UUID, start: float, end: float, score: float, title: Optional[str] = None, summary: Optional[str] = None, prompt_version_id: Optional[UUID] = None
) -> Highlight:
    highlight = Highlight(video_id=video_id, start=start, end=end, title=title, summary=summary, score=score, prompt_version_id=prompt_version_id)
    db.add(highlight)
    await db.commit()
    await db.refresh(highlight)
    return highlight


async def create_highlights_batch(
    db: AsyncSession, 
    highlights_data: List[dict]
) -> List[Highlight]:
    """Batch create highlights with a single commit for better performance."""
    if not highlights_data:
        return []
    
    highlights = [
        Highlight(
            video_id=data["video_id"],
            start=data["start"],
            end=data["end"],
            title=data.get("title"),
            summary=data.get("summary"),
            score=data["score"],
            prompt_version_id=data.get("prompt_version_id")
        )
        for data in highlights_data
    ]
    db.add_all(highlights)
    await db.commit()
    for highlight in highlights:
        await db.refresh(highlight)
    return highlights


async def get_highlights_by_video_id(db: AsyncSession, video_id: UUID) -> List[Highlight]:
    result = await db.execute(
        select(Highlight).where(Highlight.video_id == video_id).order_by(Highlight.score.desc())
    )
    return list(result.scalars().all())


async def create_moment(
    db: AsyncSession,
    video_id: UUID,
    start: float,
    end: float,
    storage_path: str,
    thumbnail_path: Optional[str] = None,
) -> Moment:
    moment = Moment(
        video_id=video_id, start=start, end=end, storage_path=storage_path, thumbnail_path=thumbnail_path
    )
    db.add(moment)
    await db.commit()
    await db.refresh(moment)
    return moment


async def get_moments_by_video_id(db: AsyncSession, video_id: UUID) -> List[Moment]:
    result = await db.execute(select(Moment).where(Moment.video_id == video_id).order_by(Moment.start))
    return list(result.scalars().all())


async def get_all_moments(db: AsyncSession, limit: int = 100, offset: int = 0) -> List[Moment]:
    result = await db.execute(
        select(Moment).limit(limit).offset(offset).order_by(Moment.created_at.desc())
    )
    return list(result.scalars().all())


async def get_videos_by_user(db: AsyncSession, clerk_user_id: Optional[str] = None) -> List[Video]:
    """Get all videos for a user, optionally filtered by clerk_user_id."""
    try:
        logger.info(f"[CRUD] Querying for videos by user (user_id: {clerk_user_id})...")
        
        if not clerk_user_id:
            logger.info("[CRUD] No user_id provided, returning empty list")
            return []
        
        # Build query for all videos by user
        query = select(Video).where(Video.clerk_user_id == clerk_user_id)
        
        result = await db.execute(
            query.order_by(Video.created_at.desc())
            .options(selectinload(Video.moments))
        )
        videos = list(result.scalars().all())
        logger.info(f"[CRUD] Retrieved {len(videos)} videos for user {clerk_user_id}")
        for video in videos:
            logger.info(f"[CRUD] Video {video.id}: {len(video.moments)} moments, status={video.status}, duration={video.duration}")
        return videos
    except Exception as e:
        logger.error(f"[CRUD] Error getting videos by user: {str(e)}", exc_info=True)
        raise


async def get_videos_with_moments(db: AsyncSession, clerk_user_id: Optional[str] = None) -> List[Video]:
    """Get all videos that have at least one moment (projects), optionally filtered by clerk_user_id."""
    from sqlalchemy import distinct
    try:
        logger.info(f"[CRUD] Querying for videos with moments (user_id: {clerk_user_id})...")
        
        # Build base query for video IDs with moments
        video_ids_query = select(distinct(Moment.video_id))
        
        # If user_id is provided, filter videos by user_id first
        if clerk_user_id:
            user_videos_result = await db.execute(
                select(Video.id).where(Video.clerk_user_id == clerk_user_id)
            )
            user_video_ids = [row[0] for row in user_videos_result.all()]
            if not user_video_ids:
                logger.info(f"[CRUD] No videos found for user {clerk_user_id}")
                return []
            video_ids_query = video_ids_query.where(Moment.video_id.in_(user_video_ids))
        
        video_ids_result = await db.execute(video_ids_query)
        video_ids = [row[0] for row in video_ids_result.all()]
        logger.info(f"[CRUD] Found {len(video_ids)} distinct video IDs with moments: {[str(vid) for vid in video_ids]}")
        
        if not video_ids:
            logger.info("[CRUD] No videos with moments found, returning empty list")
            return []
        
        # Build final query
        query = select(Video).where(Video.id.in_(video_ids))
        
        # Apply user filter again for safety
        if clerk_user_id:
            query = query.where(Video.clerk_user_id == clerk_user_id)
        
        result = await db.execute(
            query.order_by(Video.created_at.desc())
            .options(selectinload(Video.moments))
        )
        videos = list(result.scalars().all())
        logger.info(f"[CRUD] Retrieved {len(videos)} videos with moments loaded")
        for video in videos:
            logger.info(f"[CRUD] Video {video.id}: {len(video.moments)} moments, status={video.status}, duration={video.duration}")
        return videos
    except Exception as e:
        logger.error(f"[CRUD] Error getting videos with moments: {str(e)}", exc_info=True)
        raise
        return []


async def delete_all_moments(db: AsyncSession) -> int:
    """Delete all videos from the database. Returns the number of videos deleted."""
    # Get count before deletion
    result = await db.execute(select(Moment))
    moments = list(result.scalars().all())
    deleted_count = len(moments)
    
    # Delete all moments (cascade will handle saved_moments and feedback)
    await db.execute(delete(Moment))
    await db.commit()
    
    return deleted_count


async def get_moment_by_id(db: AsyncSession, moment_id: UUID) -> Optional[Moment]:
    result = await db.execute(select(Moment).where(Moment.id == moment_id))
    return result.scalar_one_or_none()


async def update_moment(
    db: AsyncSession,
    moment_id: UUID,
    start: float,
    end: float,
    storage_path: str,
    thumbnail_path: Optional[str] = None,
) -> Optional[Moment]:
    """Update moment metadata."""
    moment = await get_moment_by_id(db, moment_id)
    if not moment:
        return None
    
    moment.start = start
    moment.end = end
    moment.storage_path = storage_path
    if thumbnail_path is not None:
        moment.thumbnail_path = thumbnail_path
    
    await db.commit()
    await db.refresh(moment)
    return moment


async def delete_video(db: AsyncSession, video_id: UUID) -> bool:
    result = await db.execute(delete(Video).where(Video.id == video_id))
    await db.commit()
    return result.rowcount > 0


async def create_prompt_version(
    db: AsyncSession,
    version_name: str,
    system_prompt: str,
    user_prompt_template: str,
    is_active: bool = False,
) -> PromptVersion:
    prompt_version = PromptVersion(
        version_name=version_name,
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
        is_active=is_active,
    )
    db.add(prompt_version)
    await db.commit()
    await db.refresh(prompt_version)
    return prompt_version


async def get_prompt_version_by_id(db: AsyncSession, version_id: UUID) -> Optional[PromptVersion]:
    result = await db.execute(select(PromptVersion).where(PromptVersion.id == version_id))
    return result.scalar_one_or_none()


async def get_prompt_version_by_name(db: AsyncSession, version_name: str) -> Optional[PromptVersion]:
    result = await db.execute(select(PromptVersion).where(PromptVersion.version_name == version_name))
    return result.scalar_one_or_none()


async def get_active_prompt_version(db: AsyncSession) -> Optional[PromptVersion]:
    result = await db.execute(select(PromptVersion).where(PromptVersion.is_active == True))
    return result.scalar_one_or_none()


async def get_all_prompt_versions(db: AsyncSession) -> List[PromptVersion]:
    result = await db.execute(select(PromptVersion).order_by(PromptVersion.created_at.desc()))
    return list(result.scalars().all())


async def activate_prompt_version(db: AsyncSession, version_id: UUID) -> Optional[PromptVersion]:
    await db.execute(
        update(PromptVersion).where(PromptVersion.is_active == True).values(is_active=False)
    )
    await db.execute(
        update(PromptVersion).where(PromptVersion.id == version_id).values(is_active=True)
    )
    await db.commit()
    return await get_prompt_version_by_id(db, version_id)


async def update_prompt_version_metrics(
    db: AsyncSession,
    version_id: UUID,
    metrics: dict,
) -> Optional[PromptVersion]:
    await db.execute(
        update(PromptVersion).where(PromptVersion.id == version_id).values(performance_metrics=metrics)
    )
    await db.commit()
    return await get_prompt_version_by_id(db, version_id)


async def update_prompt_version_metrics_online(
    db: AsyncSession,
    prompt_version_id: UUID,
    feedback_type: Optional[str] = None,
    confidence_score: Optional[float] = None,
    is_save: bool = False,
) -> Optional[PromptVersion]:
    """
    Update prompt version rolling metrics online with atomic operations.
    
    Args:
        db: Database session
        prompt_version_id: Prompt version ID
        feedback_type: Feedback type (CONFIDENCE_SCORE, POSITIVE, NEGATIVE) or None
        confidence_score: Confidence score value (0-100) if feedback_type is CONFIDENCE_SCORE
        is_save: True if this is a save event
        
    Returns:
        Updated PromptVersion or None if not found
    """
    from models import FeedbackType
    
    prompt_version = await get_prompt_version_by_id(db, prompt_version_id)
    if not prompt_version:
        return None
    
    # Build update values dictionary
    update_values = {}
    
    # Update metrics based on feedback type
    if feedback_type == FeedbackType.CONFIDENCE_SCORE.value and confidence_score is not None:
        update_values["total_rated"] = PromptVersion.total_rated + 1
        update_values["sum_confidence_scores"] = PromptVersion.sum_confidence_scores + confidence_score
    
    if feedback_type == FeedbackType.POSITIVE.value:
        update_values["num_positive"] = PromptVersion.num_positive + 1
    
    if feedback_type == FeedbackType.NEGATIVE.value:
        update_values["num_negative"] = PromptVersion.num_negative + 1
    
    if is_save:
        update_values["total_saves"] = PromptVersion.total_saves + 1
    
    # Apply atomic updates
    if update_values:
        await db.execute(
            update(PromptVersion)
            .where(PromptVersion.id == prompt_version_id)
            .values(**update_values)
        )
        await db.commit()
        await db.refresh(prompt_version)
    
    # Recalculate derived fields
    if prompt_version.total_rated > 0:
        prompt_version.avg_confidence_score = prompt_version.sum_confidence_scores / prompt_version.total_rated
        prompt_version.positive_rate = prompt_version.num_positive / prompt_version.total_rated
        prompt_version.negative_rate = prompt_version.num_negative / prompt_version.total_rated
        if prompt_version.total_saves > 0:
            prompt_version.save_rate = prompt_version.total_saves / prompt_version.total_rated
        else:
            prompt_version.save_rate = 0.0
    else:
        prompt_version.avg_confidence_score = 0.0
        prompt_version.positive_rate = 0.0
        prompt_version.negative_rate = 0.0
        prompt_version.save_rate = 0.0
    
    # Update derived fields
    await db.execute(
        update(PromptVersion)
        .where(PromptVersion.id == prompt_version_id)
        .values(
            avg_confidence_score=prompt_version.avg_confidence_score,
            positive_rate=prompt_version.positive_rate,
            negative_rate=prompt_version.negative_rate,
            save_rate=prompt_version.save_rate,
        )
    )
    await db.commit()
    await db.refresh(prompt_version)
    
    # Log metric update if metrics were actually updated
    if update_values:
        logger.info(f"[METRIC-UPDATE] version={prompt_version_id} | "
                    f"avg_confidence_score={prompt_version.avg_confidence_score:.2f} | "
                    f"sample_size={prompt_version.total_rated} | "
                    f"pos={prompt_version.num_positive} | neg={prompt_version.num_negative}")
    
    return prompt_version


async def get_calibration_config(db: AsyncSession) -> Optional[CalibrationConfig]:
    result = await db.execute(select(CalibrationConfig).limit(1))
    return result.scalar_one_or_none()


async def create_or_update_calibration_config(
    db: AsyncSession,
    score_offset: float,
    sample_size: int,
) -> CalibrationConfig:
    existing = await get_calibration_config(db)
    if existing:
        await db.execute(
            update(CalibrationConfig)
            .where(CalibrationConfig.id == existing.id)
            .values(
                score_offset=score_offset,
                sample_size=sample_size,
                last_updated=datetime.utcnow(),
            )
        )
        await db.commit()
        await db.refresh(existing)
        return existing
    else:
        config = CalibrationConfig(
            score_offset=score_offset,
            sample_size=sample_size,
            last_updated=datetime.utcnow(),
        )
        db.add(config)
        await db.commit()
        await db.refresh(config)
        return config


async def update_calibration_config_online(
    db: AsyncSession,
    predicted_score: float,
    actual_confidence_score: float,
) -> Optional[CalibrationConfig]:
    """
    Update calibration config online with atomic increment operations.
    
    Args:
        db: Database session
        predicted_score: Model's predicted score (1-10 scale)
        actual_confidence_score: User confidence score scaled to 1-10 (from 0-100)
        
    Returns:
        Updated CalibrationConfig or None if error
    """
    from sqlalchemy import func
    from db.models import CalibrationConfig
    
    # Load or create calibration config
    existing = await get_calibration_config(db)
    
    if not existing:
        # Create new config if it doesn't exist
        config = CalibrationConfig(
            feedback_count=1,
            sum_predicted=predicted_score,
            sum_actual=actual_confidence_score,
            score_offset=0.0,
            sample_size=1,
            last_updated=datetime.utcnow(),
        )
        db.add(config)
        await db.commit()
        await db.refresh(config)
        avg_pred = predicted_score
        avg_actual = actual_confidence_score
        logger.info(f"[CALIBRATION-UPDATE] count={config.feedback_count} | "
                    f"offset={config.score_offset:.3f} | "
                    f"avg_pred={avg_pred:.2f} | avg_actual={avg_actual:.2f}")
        return config
    
    # Atomic update using SQL expressions to avoid race conditions
    await db.execute(
        update(CalibrationConfig)
        .where(CalibrationConfig.id == existing.id)
        .values(
            feedback_count=CalibrationConfig.feedback_count + 1,
            sum_predicted=CalibrationConfig.sum_predicted + predicted_score,
            sum_actual=CalibrationConfig.sum_actual + actual_confidence_score,
            last_updated=datetime.utcnow(),
        )
    )
    await db.commit()
    
    # Reload to calculate derived fields
    await db.refresh(existing)
    
    # Calculate averages and offset
    avg_pred = existing.sum_predicted / existing.feedback_count
    avg_actual = existing.sum_actual / existing.feedback_count
    
    # Calculate offset (only if we have enough feedback)
    import os
    min_feedback_threshold = int(os.getenv("MIN_FEEDBACK_THRESHOLD", "50"))
    if existing.feedback_count >= min_feedback_threshold:
        score_offset = avg_actual - avg_pred
    else:
        score_offset = 0.0
    
    # Update offset and sample_size
    await db.execute(
        update(CalibrationConfig)
        .where(CalibrationConfig.id == existing.id)
        .values(
            score_offset=score_offset,
            sample_size=existing.feedback_count,
        )
    )
    await db.commit()
    await db.refresh(existing)
    
    logger.info(f"[CALIBRATION-UPDATE] count={existing.feedback_count} | "
                f"offset={existing.score_offset:.3f} | "
                f"avg_pred={avg_pred:.2f} | avg_actual={avg_actual:.2f}")
    
    return existing


async def update_highlight_prompt_version(
    db: AsyncSession,
    highlight_id: UUID,
    prompt_version_id: Optional[UUID],
) -> Optional[Highlight]:
    await db.execute(
        update(Highlight)
        .where(Highlight.id == highlight_id)
        .values(prompt_version_id=prompt_version_id)
    )
    await db.commit()
    result = await db.execute(select(Highlight).where(Highlight.id == highlight_id))
    return result.scalar_one_or_none()


async def update_highlight_explanation(
    db: AsyncSession,
    highlight_id: UUID,
    explanation: dict,
) -> Optional[Highlight]:
    """Update the cached explanation for a highlight."""
    await db.execute(
        update(Highlight)
        .where(Highlight.id == highlight_id)
        .values(explanation=explanation)
    )
    await db.commit()
    result = await db.execute(select(Highlight).where(Highlight.id == highlight_id))
    return result.scalar_one_or_none()


async def create_saved_moment(
    db: AsyncSession,
    moment_id: UUID,
    highlight_id: Optional[UUID] = None,
) -> SavedMoment:
    saved_moment = SavedMoment(
        moment_id=moment_id,
        highlight_id=highlight_id,
    )
    db.add(saved_moment)
    await db.commit()
    await db.refresh(saved_moment)
    
    
    return saved_moment


async def delete_saved_moment(db: AsyncSession, moment_id: UUID) -> bool:
    result = await db.execute(delete(SavedMoment).where(SavedMoment.moment_id == moment_id))
    await db.commit()
    return result.rowcount > 0


async def get_saved_moment(db: AsyncSession, moment_id: UUID) -> Optional[SavedMoment]:
    result = await db.execute(select(SavedMoment).where(SavedMoment.moment_id == moment_id))
    return result.scalar_one_or_none()


async def is_moment_saved(db: AsyncSession, moment_id: UUID) -> bool:
    saved_moment = await get_saved_moment(db, moment_id)
    return saved_moment is not None


async def get_all_saved_moments(db: AsyncSession, limit: int = 100, offset: int = 0) -> List[SavedMoment]:
    result = await db.execute(
        select(SavedMoment).limit(limit).offset(offset).order_by(SavedMoment.created_at.desc())
    )
    return list(result.scalars().all())


async def create_video_segment(
    db: AsyncSession,
    video_id: UUID,
    segment_id: int,
    start_time: float,
    end_time: float,
    text: str,
    label: str,
    rating: float,
    grade: str,
    reason: str,
    repetition_score: float,
    filler_density: float,
    visual_change_score: float,
    usefulness_score: float,
    embedding: Optional[List[float]] = None,
) -> VideoSegment:
    """Create a video segment analysis record."""
    segment = VideoSegment(
        video_id=video_id,
        segment_id=segment_id,
        start_time=start_time,
        end_time=end_time,
        text=text,
        label=label,
        rating=rating,
        grade=grade,
        reason=reason,
        repetition_score=repetition_score,
        filler_density=filler_density,
        visual_change_score=visual_change_score,
        usefulness_score=usefulness_score,
        embedding=embedding,
    )
    db.add(segment)
    await db.commit()
    await db.refresh(segment)
    return segment


async def create_retention_metrics(
    db: AsyncSession,
    video_id: UUID,
    segment_id: int,
    time_range: Dict,
    text: str,
    metrics: Dict,
    retention_value: float,
    decision: Dict,
) -> "RetentionMetrics":
    """Create a retention metrics record."""
    from db.models import RetentionMetrics  # noqa: F401
    
    retention_metric = RetentionMetrics(
        video_id=video_id,
        segment_id=segment_id,
        time_range=time_range,
        text=text,
        metrics=metrics,
        retention_value=retention_value,
        decision=decision,
    )
    db.add(retention_metric)
    await db.commit()
    await db.refresh(retention_metric)
    return retention_metric


async def get_retention_metrics_by_video(db: AsyncSession, video_id: UUID) -> List["RetentionMetrics"]:
    """Get all retention metrics for a video."""
    from db.models import RetentionMetrics
    
    result = await db.execute(
        select(RetentionMetrics)
        .where(RetentionMetrics.video_id == video_id)
        .order_by(RetentionMetrics.segment_id)
    )
    return list(result.scalars().all())


async def get_retention_metrics_by_segment_id(
    db: AsyncSession,
    video_id: UUID,
    segment_id: int,
) -> Optional["RetentionMetrics"]:
    """Get retention metrics for a specific segment."""
    from db.models import RetentionMetrics
    
    result = await db.execute(
        select(RetentionMetrics)
        .where(RetentionMetrics.video_id == video_id)
        .where(RetentionMetrics.segment_id == segment_id)
    )
    return result.scalar_one_or_none()


async def create_video_segments_batch(
    db: AsyncSession,
    video_id: UUID,
    segments_data: List[Dict],
) -> List[VideoSegment]:
    """Create multiple video segments in a single batch operation."""
    segments = []
    for seg_data in segments_data:
        segment = VideoSegment(
            video_id=video_id,
            segment_id=seg_data["segment_id"],
            start_time=seg_data["start_time"],
            end_time=seg_data["end_time"],
            text=seg_data["text"],
            label=seg_data["label"],
            rating=seg_data["rating"],
            grade=seg_data["grade"],
            reason=seg_data["reason"],
            repetition_score=seg_data["repetition_score"],
            filler_density=seg_data["filler_density"],
            visual_change_score=seg_data["visual_change_score"],
            usefulness_score=seg_data["usefulness_score"],
            embedding=seg_data.get("embedding"),
        )
        segments.append(segment)
        db.add(segment)
    
    await db.commit()
    for segment in segments:
        await db.refresh(segment)
    return segments


async def create_retention_metrics_batch(
    db: AsyncSession,
    video_id: UUID,
    metrics_data: List[Dict],
) -> List["RetentionMetrics"]:
    """Create multiple retention metrics in a single batch operation."""
    from db.models import RetentionMetrics
    
    retention_metrics = []
    for metric_data in metrics_data:
        retention_metric = RetentionMetrics(
            video_id=video_id,
            segment_id=metric_data["segment_id"],
            time_range=metric_data["time_range"],
            text=metric_data["text"],
            metrics=metric_data["metrics"],
            retention_value=metric_data["retention_value"],
            decision=metric_data["decision"],
        )
        retention_metrics.append(retention_metric)
        db.add(retention_metric)
    
    await db.commit()
    for metric in retention_metrics:
        await db.refresh(metric)
    return retention_metrics


async def get_video_segments(db: AsyncSession, video_id: UUID) -> List[VideoSegment]:
    """Get all segments for a video."""
    result = await db.execute(
        select(VideoSegment)
        .where(VideoSegment.video_id == video_id)
        .order_by(VideoSegment.segment_id)
    )
    return list(result.scalars().all())


async def get_segments_by_label(db: AsyncSession, video_id: UUID, label: str) -> List[VideoSegment]:
    """Get segments filtered by label (FLUFF)."""
    result = await db.execute(
        select(VideoSegment)
        .where(VideoSegment.video_id == video_id)
        .where(VideoSegment.label == label)
        .order_by(VideoSegment.segment_id)
    )
    return list(result.scalars().all())


async def get_segment_counts_by_label(db: AsyncSession, video_id: UUID) -> dict:
    """
    Get segment counts grouped by label using SQL aggregation.
    
    This is much more efficient than loading all segments and counting in Python.
    
    Returns:
        Dictionary mapping label to count, e.g., {'FLUFF': 10}
    """
    from sqlalchemy import func
    result = await db.execute(
        select(VideoSegment.label, func.count(VideoSegment.id))
        .where(VideoSegment.video_id == video_id)
        .group_by(VideoSegment.label)
    )
    return {row[0]: row[1] for row in result.all()}


async def get_video_segment_by_time_range(
    db: AsyncSession,
    video_id: UUID,
    start_time: float,
    end_time: float,
    tolerance: float = 0.1,
) -> Optional[VideoSegment]:
    """Get a video segment by video_id, start_time, and end_time (with tolerance for floating point comparison)."""
    result = await db.execute(
        select(VideoSegment)
        .where(VideoSegment.video_id == video_id)
        .where(VideoSegment.start_time >= start_time - tolerance)
        .where(VideoSegment.start_time <= start_time + tolerance)
        .where(VideoSegment.end_time >= end_time - tolerance)
        .where(VideoSegment.end_time <= end_time + tolerance)
        .limit(1)
    )
    return result.scalar_one_or_none()


async def update_video_segment_label(
    db: AsyncSession,
    video_id: UUID,
    segment_id: int,
    new_label: str,
    new_grade: Optional[str] = None,
) -> bool:
    """Update a video segment's label and optionally grade."""
    update_values = {"label": new_label}
    if new_grade:
        update_values["grade"] = new_grade
    
    result = await db.execute(
        update(VideoSegment)
        .where(VideoSegment.video_id == video_id)
        .where(VideoSegment.segment_id == segment_id)
        .values(**update_values)
    )
    await db.commit()
    return result.rowcount > 0


async def get_video_segment_by_id(db: AsyncSession, segment_id: UUID) -> Optional[VideoSegment]:
    """Get a video segment by its ID."""
    result = await db.execute(select(VideoSegment).where(VideoSegment.id == segment_id))
    return result.scalar_one_or_none()


async def update_video_segment_rating(
    db: AsyncSession,
    segment_id: UUID,
    new_rating: float,
    new_usefulness_score: Optional[float] = None,
) -> Optional[VideoSegment]:
    """Update a video segment's rating and optionally usefulness_score."""
    update_values = {"rating": max(0.0, min(1.0, new_rating))}
    if new_usefulness_score is not None:
        update_values["usefulness_score"] = max(0.0, min(1.0, new_usefulness_score))
    
    await db.execute(
        update(VideoSegment)
        .where(VideoSegment.id == segment_id)
        .values(**update_values)
    )
    await db.commit()
    
    return await get_video_segment_by_id(db, segment_id)


async def delete_segments_by_time_range(
    db: AsyncSession,
    video_id: UUID,
    segments_to_remove: List[Tuple[float, float]],
) -> int:
    """
    Delete VideoSegment records that overlap with any of the removed time ranges.
    
    Args:
        db: Database session
        video_id: UUID of the video
        segments_to_remove: List of (start_time, end_time) tuples for segments to remove
        
    Returns:
        Number of segments deleted
    """
    if not segments_to_remove:
        return 0
    
    from sqlalchemy import or_
    
    # Build OR conditions for overlapping segments
    # A segment overlaps if: start_time < removed_end AND end_time > removed_start
    conditions = []
    for removed_start, removed_end in segments_to_remove:
        conditions.append(
            (VideoSegment.start_time < removed_end) & (VideoSegment.end_time > removed_start)
        )
    
    # Combine all conditions with OR
    combined_condition = or_(*conditions)
    
    # Delete segments that match any condition
    result = await db.execute(
        delete(VideoSegment)
        .where(VideoSegment.video_id == video_id)
        .where(combined_condition)
    )
    await db.commit()
    
    deleted_count = result.rowcount
    logger.info(f"Deleted {deleted_count} video segments for video {video_id} that overlapped with removed time ranges")
    
    return deleted_count


async def create_segment_feedback(
    db: AsyncSession,
    video_segment_id: UUID,
    feedback_type: str,
) -> SegmentFeedback:
    """Create a segment feedback record."""
    feedback = SegmentFeedback(
        video_segment_id=video_segment_id,
        feedback_type=feedback_type,
    )
    db.add(feedback)
    await db.commit()
    await db.refresh(feedback)
    return feedback


async def create_or_update_timeline(
    db: AsyncSession,
    video_id: UUID,
    clerk_user_id: Optional[str] = None,
    project_name: Optional[str] = None,
    markers: Optional[List] = None,
    selections: Optional[List] = None,
    sequences: Optional[List] = None,
    current_time: Optional[float] = None,
    in_point: Optional[float] = None,
    out_point: Optional[float] = None,
    zoom: Optional[float] = None,
    view_preferences: Optional[Dict] = None,
) -> Timeline:
    """Create or update a timeline for a video."""
    # Check if timeline exists first
    result = await db.execute(select(Timeline).where(Timeline.video_id == video_id))
    timeline = result.scalar_one_or_none()
    
    if timeline:
        # Update existing timeline - allow even if video doesn't exist
        if project_name is not None:
            timeline.project_name = project_name
        if markers is not None:
            timeline.markers = markers
        if selections is not None:
            timeline.selections = selections
        if sequences is not None:
            timeline.sequences = sequences
        if current_time is not None:
            timeline.current_time = current_time
        if in_point is not None:
            timeline.in_point = in_point
        if out_point is not None:
            timeline.out_point = out_point
        if zoom is not None:
            timeline.zoom = zoom
        if view_preferences is not None:
            timeline.view_preferences = view_preferences
        if clerk_user_id is not None:
            timeline.clerk_user_id = clerk_user_id
        # Try to get video to update thumbnail if missing
        video = await get_video_by_id(db, video_id)
        if video and not timeline.thumbnail_path and video.thumbnail_path:
            timeline.thumbnail_path = video.thumbnail_path
        timeline.updated_at = datetime.utcnow()
    else:
        # Create new timeline - video must exist
        video = await get_video_by_id(db, video_id)
        if not video:
            raise ValueError(f"Video with id {video_id} not found")
        
        timeline = Timeline(
            video_id=video_id,
            clerk_user_id=clerk_user_id,
            project_name=project_name,
            markers=markers or [],
            selections=selections or [],
            sequences=sequences or [],
            current_time=current_time or 0.0,
            in_point=in_point,
            out_point=out_point,
            zoom=zoom or 1.0,
            view_preferences=view_preferences or {},
            thumbnail_path=video.thumbnail_path if video else None,
        )
        db.add(timeline)
    
    await db.commit()
    await db.refresh(timeline)
    return timeline


async def get_timeline_by_video_id(db: AsyncSession, video_id: UUID) -> Optional[Timeline]:
    """Get timeline by video ID with video relationship eagerly loaded."""
    result = await db.execute(
        select(Timeline)
        .where(Timeline.video_id == video_id)
        .options(selectinload(Timeline.video))
    )
    return result.scalar_one_or_none()


async def get_timelines_by_user(db: AsyncSession, clerk_user_id: Optional[str] = None) -> List[Timeline]:
    """Get all timelines for a user."""
    if not clerk_user_id:
        return []
    
    result = await db.execute(
        select(Timeline)
        .where(Timeline.clerk_user_id == clerk_user_id)
        .order_by(Timeline.updated_at.desc())
        .options(selectinload(Timeline.video).selectinload(Video.moments))
    )
    return list(result.scalars().all())

