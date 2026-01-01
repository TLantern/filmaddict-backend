#!/usr/bin/env python3
"""
Script to backfill thumbnails for existing timelines.
Copies thumbnails from videos or generates them if missing.
"""
import asyncio
import logging
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from database import async_session_maker
from db.models import Timeline, Video
from utils.storage import get_storage_instance
from utils.moments import generate_video_thumbnail_async

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def backfill_timeline_thumbnails():
    """Backfill thumbnails for all timelines that don't have one."""
    storage = get_storage_instance()
    
    async with async_session_maker() as db:
        # Get all timelines without thumbnails
        result = await db.execute(
            select(Timeline)
            .where(Timeline.thumbnail_path.is_(None))
            .options(selectinload(Timeline.video))
        )
        timelines = result.scalars().all()
        
        total_timelines = len(timelines)
        logger.info(f"Found {total_timelines} timelines without thumbnails")
        
        if total_timelines == 0:
            logger.info("All timelines already have thumbnails")
            return
        
        updated = 0
        errors = 0
        
        for timeline in timelines:
            try:
                video = timeline.video
                if not video:
                    logger.warning(f"Timeline {timeline.id} has no associated video, skipping")
                    errors += 1
                    continue
                
                # If video has a thumbnail, copy it
                if video.thumbnail_path:
                    timeline.thumbnail_path = video.thumbnail_path
                    await db.commit()
                    await db.refresh(timeline)
                    updated += 1
                    logger.info(f"Copied thumbnail from video {video.id} to timeline {timeline.id}")
                else:
                    # Generate thumbnail from video
                    try:
                        logger.info(f"Generating thumbnail for timeline {timeline.id} from video {video.id}")
                        video_file_path = storage.get_video_path(video.storage_path)
                        thumbnail_path = await generate_video_thumbnail_async(video_file_path, time_offset=1.0)
                        
                        # Update both video and timeline with the generated thumbnail
                        video.thumbnail_path = thumbnail_path
                        timeline.thumbnail_path = thumbnail_path
                        await db.commit()
                        await db.refresh(timeline)
                        updated += 1
                        logger.info(f"Generated and assigned thumbnail for timeline {timeline.id}: {thumbnail_path}")
                    except Exception as e:
                        logger.error(f"Failed to generate thumbnail for timeline {timeline.id}: {str(e)}")
                        errors += 1
                        await db.rollback()
                        
            except Exception as e:
                logger.error(f"Error processing timeline {timeline.id}: {str(e)}", exc_info=True)
                errors += 1
                await db.rollback()
        
        logger.info(
            f"Backfill complete: {updated} timelines updated, {errors} errors"
        )


if __name__ == "__main__":
    asyncio.run(backfill_timeline_thumbnails())

