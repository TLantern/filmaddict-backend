import logging
from typing import List, Optional
from uuid import UUID

from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from datetime import datetime, timedelta

from db.models import Video, Transcript, Highlight, Moment, HighlightFeedback, PromptVersion, CalibrationConfig, SavedMoment
from models import VideoStatus, FeedbackType

logger = logging.getLogger(__name__)


async def create_video(
    db: AsyncSession, storage_path: str, duration: Optional[float] = None, status: VideoStatus = VideoStatus.UPLOADED, aspect_ratio: Optional[str] = "16:9"
) -> Video:
    video = Video(storage_path=storage_path, duration=duration, aspect_ratio=aspect_ratio, status=status.value)
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
    transcript = Transcript(video_id=video_id, segments=segments)
    db.add(transcript)
    await db.commit()
    await db.refresh(transcript)
    return transcript


async def get_transcript_by_video_id(db: AsyncSession, video_id: UUID) -> Optional[Transcript]:
    result = await db.execute(select(Transcript).where(Transcript.video_id == video_id))
    return result.scalar_one_or_none()


async def create_highlight(
    db: AsyncSession, video_id: UUID, start: float, end: float, score: float, title: Optional[str] = None, summary: Optional[str] = None, prompt_version_id: Optional[UUID] = None
) -> Highlight:
    highlight = Highlight(video_id=video_id, start=start, end=end, title=title, summary=summary, score=score, prompt_version_id=prompt_version_id)
    db.add(highlight)
    await db.commit()
    await db.refresh(highlight)
    return highlight


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


async def get_videos_with_moments(db: AsyncSession) -> List[Video]:
    """Get all videos that have at least one moment (projects)."""
    from sqlalchemy import distinct
    try:
        logger.info("[CRUD] Querying for videos with moments...")
        # First check total moments count
        total_moments_result = await db.execute(select(Moment))
        total_moments = list(total_moments_result.scalars().all())
        logger.info(f"[CRUD] Total moments in database: {len(total_moments)}")
        if total_moments:
            logger.info(f"[CRUD] Moments by video_id: {[(str(m.video_id), m.id) for m in total_moments[:10]]}")
        
        video_ids_result = await db.execute(
            select(distinct(Moment.video_id))
        )
        video_ids = [row[0] for row in video_ids_result.all()]
        logger.info(f"[CRUD] Found {len(video_ids)} distinct video IDs with moments: {[str(vid) for vid in video_ids]}")
        
        if not video_ids:
            logger.info("[CRUD] No videos with moments found, returning empty list")
            return []
        
        result = await db.execute(
            select(Video)
            .where(Video.id.in_(video_ids))
            .order_by(Video.created_at.desc())
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


async def create_feedback(
    db: AsyncSession,
    highlight_id: UUID,
    feedback_type: str,
    confidence_score: Optional[float] = None,
    text_feedback: Optional[str] = None,
) -> HighlightFeedback:
    feedback = HighlightFeedback(
        highlight_id=highlight_id,
        feedback_type=feedback_type,
        confidence_score=confidence_score,
        text_feedback=text_feedback,
    )
    db.add(feedback)
    await db.commit()
    await db.refresh(feedback)
    
    # Load highlight to get prompt_version_id for logging
    highlight = await db.execute(select(Highlight).where(Highlight.id == highlight_id))
    highlight_obj = highlight.scalar_one_or_none()
    prompt_version_id = highlight_obj.prompt_version_id if highlight_obj else None
    
    # Log feedback received
    logger.info("[ONLINE-LEARNING] Feedback received | "
                f"type={feedback.feedback_type} | confidence_score={feedback.confidence_score} | "
                f"highlight_id={feedback.highlight_id} | prompt_version={prompt_version_id}")
    
    # Hook online learning updates
    try:
        from models import FeedbackType
        from utils.learning import update_calibration_online, update_prompt_version_metrics_online
        
        if highlight_obj:
            # Update calibration for CONFIDENCE_SCORE feedback
            if feedback_type == FeedbackType.CONFIDENCE_SCORE.value and confidence_score is not None and highlight_obj.score is not None:
                await update_calibration_online(
                    db,
                    highlight_id=highlight_id,
                    confidence_score=confidence_score,
                    predicted_score=highlight_obj.score,
                )
            
            # Update prompt version metrics
            if highlight_obj.prompt_version_id:
                await update_prompt_version_metrics_online(
                    db,
                    prompt_version_id=highlight_obj.prompt_version_id,
                    feedback_type=feedback_type,
                    confidence_score=confidence_score,
                    is_save=False,
                )
    except Exception as e:
        # Log error but don't fail the feedback creation
        logger.error(f"Error updating online learning metrics: {str(e)}", exc_info=True)
    
    return feedback


async def get_feedback_by_highlight_id(db: AsyncSession, highlight_id: UUID) -> List[HighlightFeedback]:
    result = await db.execute(
        select(HighlightFeedback)
        .options(selectinload(HighlightFeedback.highlight))
        .where(HighlightFeedback.highlight_id == highlight_id)
        .order_by(HighlightFeedback.created_at.desc())
    )
    return list(result.scalars().all())


async def get_feedback_stats(db: AsyncSession, highlight_id: UUID) -> dict:
    feedback_list = await get_feedback_by_highlight_id(db, highlight_id)
    
    stats = {
        "total_feedback": len(feedback_list),
        "positive_count": 0,
        "negative_count": 0,
        "view_count": 0,
        "skip_count": 0,
        "share_count": 0,
        "average_confidence_score": None,
        "confidence_score_count": 0,
    }
    
    confidence_scores = []
    for feedback in feedback_list:
        if feedback.feedback_type == FeedbackType.POSITIVE.value:
            stats["positive_count"] += 1
        elif feedback.feedback_type == FeedbackType.NEGATIVE.value:
            stats["negative_count"] += 1
        elif feedback.feedback_type == FeedbackType.VIEW.value:
            stats["view_count"] += 1
        elif feedback.feedback_type == FeedbackType.SKIP.value:
            stats["skip_count"] += 1
        elif feedback.feedback_type == FeedbackType.SHARE.value:
            stats["share_count"] += 1
        
        if feedback.confidence_score is not None:
            confidence_scores.append(feedback.confidence_score)
    
    if confidence_scores:
        stats["average_confidence_score"] = sum(confidence_scores) / len(confidence_scores)
        stats["confidence_score_count"] = len(confidence_scores)
    
    return stats


async def get_recent_feedback(
    db: AsyncSession,
    days: int = 7,
    limit: Optional[int] = None,
) -> List[HighlightFeedback]:
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    query = (
        select(HighlightFeedback)
        .options(selectinload(HighlightFeedback.highlight))
        .where(HighlightFeedback.created_at >= cutoff_date)
    )
    if limit:
        query = query.limit(limit)
    result = await db.execute(query.order_by(HighlightFeedback.created_at.desc()))
    return list(result.scalars().all())


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
    return await db.execute(select(Highlight).where(Highlight.id == highlight_id)).scalar_one_or_none()


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
    
    # Hook online learning updates for prompt metrics
    if highlight_id:
        try:
            from utils.learning import update_prompt_version_metrics_online
            
            # Load highlight to get prompt_version_id
            highlight = await db.execute(select(Highlight).where(Highlight.id == highlight_id))
            highlight_obj = highlight.scalar_one_or_none()
            
            if highlight_obj and highlight_obj.prompt_version_id:
                await update_prompt_version_metrics_online(
                    db,
                    prompt_version_id=highlight_obj.prompt_version_id,
                    feedback_type=None,
                    rating=None,
                    is_save=True,
                )
        except Exception as e:
            # Log error but don't fail the save creation
            logger.error(f"Error updating online learning metrics for save: {str(e)}", exc_info=True)
    
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

