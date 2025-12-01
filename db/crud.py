import logging
from typing import List, Optional
from uuid import UUID

from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from datetime import datetime, timedelta

from db.models import Video, Transcript, Highlight, Clip, HighlightFeedback, PromptVersion, CalibrationConfig, SavedClip
from models import VideoStatus, FeedbackType

logger = logging.getLogger(__name__)


async def create_video(
    db: AsyncSession, storage_path: str, duration: Optional[float] = None, status: VideoStatus = VideoStatus.UPLOADED
) -> Video:
    video = Video(storage_path=storage_path, duration=duration, status=status.value)
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
    db: AsyncSession, video_id: UUID, start: float, end: float, reason: str, score: float, prompt_version_id: Optional[UUID] = None
) -> Highlight:
    highlight = Highlight(video_id=video_id, start=start, end=end, reason=reason, score=score, prompt_version_id=prompt_version_id)
    db.add(highlight)
    await db.commit()
    await db.refresh(highlight)
    return highlight


async def get_highlights_by_video_id(db: AsyncSession, video_id: UUID) -> List[Highlight]:
    result = await db.execute(
        select(Highlight).where(Highlight.video_id == video_id).order_by(Highlight.score.desc())
    )
    return list(result.scalars().all())


async def create_clip(
    db: AsyncSession,
    video_id: UUID,
    start: float,
    end: float,
    storage_path: str,
    thumbnail_path: Optional[str] = None,
) -> Clip:
    clip = Clip(
        video_id=video_id, start=start, end=end, storage_path=storage_path, thumbnail_path=thumbnail_path
    )
    db.add(clip)
    await db.commit()
    await db.refresh(clip)
    return clip


async def get_clips_by_video_id(db: AsyncSession, video_id: UUID) -> List[Clip]:
    result = await db.execute(select(Clip).where(Clip.video_id == video_id).order_by(Clip.start))
    return list(result.scalars().all())


async def get_all_clips(db: AsyncSession, limit: int = 100, offset: int = 0) -> List[Clip]:
    result = await db.execute(
        select(Clip).limit(limit).offset(offset).order_by(Clip.created_at.desc())
    )
    return list(result.scalars().all())


async def get_videos_with_clips(db: AsyncSession) -> List[Video]:
    """Get all videos that have at least one clip (projects)."""
    from sqlalchemy import distinct
    try:
        video_ids_result = await db.execute(
            select(distinct(Clip.video_id))
        )
        video_ids = [row[0] for row in video_ids_result.all()]
        
        if not video_ids:
            return []
        
        result = await db.execute(
            select(Video)
            .where(Video.id.in_(video_ids))
            .order_by(Video.created_at.desc())
            .options(selectinload(Video.clips))
        )
        return list(result.scalars().all())
    except Exception as e:
        logger.error(f"Error getting videos with clips: {str(e)}", exc_info=True)
        return []


async def delete_all_clips(db: AsyncSession) -> int:
    """Delete all clips from the database. Returns the number of clips deleted."""
    # Get count before deletion
    result = await db.execute(select(Clip))
    clips = list(result.scalars().all())
    deleted_count = len(clips)
    
    # Delete all clips (cascade will handle saved_clips and feedback)
    await db.execute(delete(Clip))
    await db.commit()
    
    return deleted_count


async def get_clip_by_id(db: AsyncSession, clip_id: UUID) -> Optional[Clip]:
    result = await db.execute(select(Clip).where(Clip.id == clip_id))
    return result.scalar_one_or_none()


async def update_clip(
    db: AsyncSession,
    clip_id: UUID,
    start: float,
    end: float,
    storage_path: str,
    thumbnail_path: Optional[str] = None,
) -> Optional[Clip]:
    """Update clip metadata."""
    clip = await get_clip_by_id(db, clip_id)
    if not clip:
        return None
    
    clip.start = start
    clip.end = end
    clip.storage_path = storage_path
    if thumbnail_path is not None:
        clip.thumbnail_path = thumbnail_path
    
    await db.commit()
    await db.refresh(clip)
    return clip


async def delete_video(db: AsyncSession, video_id: UUID) -> bool:
    result = await db.execute(delete(Video).where(Video.id == video_id))
    await db.commit()
    return result.rowcount > 0


async def create_feedback(
    db: AsyncSession,
    highlight_id: UUID,
    feedback_type: str,
    rating: Optional[float] = None,
    text_feedback: Optional[str] = None,
) -> HighlightFeedback:
    feedback = HighlightFeedback(
        highlight_id=highlight_id,
        feedback_type=feedback_type,
        rating=rating,
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
                f"type={feedback.feedback_type} | rating={feedback.rating} | "
                f"highlight_id={feedback.highlight_id} | prompt_version={prompt_version_id}")
    
    # Hook online learning updates
    try:
        from models import FeedbackType
        from utils.learning import update_calibration_online, update_prompt_version_metrics_online
        
        if highlight_obj:
            # Update calibration for RATING feedback
            if feedback_type == FeedbackType.RATING.value and rating is not None and highlight_obj.score is not None:
                await update_calibration_online(
                    db,
                    highlight_id=highlight_id,
                    rating=rating,
                    predicted_score=highlight_obj.score,
                )
            
            # Update prompt version metrics
            if highlight_obj.prompt_version_id:
                await update_prompt_version_metrics_online(
                    db,
                    prompt_version_id=highlight_obj.prompt_version_id,
                    feedback_type=feedback_type,
                    rating=rating,
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
        "average_rating": None,
        "rating_count": 0,
    }
    
    ratings = []
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
        
        if feedback.rating is not None:
            ratings.append(feedback.rating)
    
    if ratings:
        stats["average_rating"] = sum(ratings) / len(ratings)
        stats["rating_count"] = len(ratings)
    
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
    rating: Optional[float] = None,
    is_save: bool = False,
) -> Optional[PromptVersion]:
    """
    Update prompt version rolling metrics online with atomic operations.
    
    Args:
        db: Database session
        prompt_version_id: Prompt version ID
        feedback_type: Feedback type (RATING, POSITIVE, NEGATIVE) or None
        rating: Rating value (0-100) if feedback_type is RATING
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
    if feedback_type == FeedbackType.RATING.value and rating is not None:
        update_values["total_rated"] = PromptVersion.total_rated + 1
        update_values["sum_ratings"] = PromptVersion.sum_ratings + rating
    
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
        prompt_version.avg_rating = prompt_version.sum_ratings / prompt_version.total_rated
        prompt_version.positive_rate = prompt_version.num_positive / prompt_version.total_rated
        prompt_version.negative_rate = prompt_version.num_negative / prompt_version.total_rated
        if prompt_version.total_saves > 0:
            prompt_version.save_rate = prompt_version.total_saves / prompt_version.total_rated
        else:
            prompt_version.save_rate = 0.0
    else:
        prompt_version.avg_rating = 0.0
        prompt_version.positive_rate = 0.0
        prompt_version.negative_rate = 0.0
        prompt_version.save_rate = 0.0
    
    # Update derived fields
    await db.execute(
        update(PromptVersion)
        .where(PromptVersion.id == prompt_version_id)
        .values(
            avg_rating=prompt_version.avg_rating,
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
                    f"avg_rating={prompt_version.avg_rating:.2f} | "
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
    actual_rating: float,
) -> Optional[CalibrationConfig]:
    """
    Update calibration config online with atomic increment operations.
    
    Args:
        db: Database session
        predicted_score: Model's predicted score (1-10 scale)
        actual_rating: User rating scaled to 1-10 (from 0-100)
        
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
            sum_actual=actual_rating,
            score_offset=0.0,
            sample_size=1,
            last_updated=datetime.utcnow(),
        )
        db.add(config)
        await db.commit()
        await db.refresh(config)
        avg_pred = predicted_score
        avg_actual = actual_rating
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
            sum_actual=CalibrationConfig.sum_actual + actual_rating,
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


async def create_saved_clip(
    db: AsyncSession,
    clip_id: UUID,
    highlight_id: Optional[UUID] = None,
) -> SavedClip:
    saved_clip = SavedClip(
        clip_id=clip_id,
        highlight_id=highlight_id,
    )
    db.add(saved_clip)
    await db.commit()
    await db.refresh(saved_clip)
    
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
    
    return saved_clip


async def delete_saved_clip(db: AsyncSession, clip_id: UUID) -> bool:
    result = await db.execute(delete(SavedClip).where(SavedClip.clip_id == clip_id))
    await db.commit()
    return result.rowcount > 0


async def get_saved_clip(db: AsyncSession, clip_id: UUID) -> Optional[SavedClip]:
    result = await db.execute(select(SavedClip).where(SavedClip.clip_id == clip_id))
    return result.scalar_one_or_none()


async def is_clip_saved(db: AsyncSession, clip_id: UUID) -> bool:
    saved_clip = await get_saved_clip(db, clip_id)
    return saved_clip is not None


async def get_all_saved_clips(db: AsyncSession, limit: int = 100, offset: int = 0) -> List[SavedClip]:
    result = await db.execute(
        select(SavedClip).limit(limit).offset(offset).order_by(SavedClip.created_at.desc())
    )
    return list(result.scalars().all())

