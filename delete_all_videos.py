#!/usr/bin/env python3
"""
Script to delete all videos from storage and database.
"""
import asyncio
import logging
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from database import async_session_maker
from db.models import Video, Moment
from utils.storage import get_storage_instance

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def delete_all_videos():
    """Delete all videos from storage and database."""
    storage = get_storage_instance()
    
    async with async_session_maker() as db:
        # Get all videos
        result = await db.execute(select(Video))
        videos = result.scalars().all()
        
        total_videos = len(videos)
        logger.info(f"Found {total_videos} videos to delete")
        
        if total_videos == 0:
            logger.info("No videos to delete")
            return
        
        deleted_storage = 0
        deleted_moments = 0
        errors = 0
        
        # Delete video files from storage
        for video in videos:
            try:
                # Delete main video file
                if storage.delete_video(video.storage_path):
                    deleted_storage += 1
                    logger.info(f"Deleted video file: {video.storage_path}")
                else:
                    logger.warning(f"Failed to delete video file: {video.storage_path}")
                    errors += 1
                
                # Delete video thumbnail if exists
                if video.thumbnail_path:
                    if storage.delete_video(video.thumbnail_path):
                        logger.info(f"Deleted video thumbnail: {video.thumbnail_path}")
                    else:
                        logger.warning(f"Failed to delete video thumbnail: {video.thumbnail_path}")
            except Exception as e:
                logger.error(f"Error deleting video {video.id} from storage: {str(e)}")
                errors += 1
            
            # Delete moments for this video
            try:
                moments_result = await db.execute(
                    select(Moment).where(Moment.video_id == video.id)
                )
                moments = moments_result.scalars().all()
                
                for moment in moments:
                    try:
                        if storage.delete_video(moment.storage_path):
                            deleted_moments += 1
                            logger.info(f"Deleted moment file: {moment.storage_path}")
                        if moment.thumbnail_path and storage.delete_video(moment.thumbnail_path):
                            logger.info(f"Deleted thumbnail: {moment.thumbnail_path}")
                    except Exception as e:
                        logger.error(f"Error deleting moment {moment.id} from storage: {str(e)}")
                        errors += 1
            except Exception as e:
                logger.error(f"Error fetching moments for video {video.id}: {str(e)}")
                errors += 1
        
        # Delete all videos from database (cascade will delete related records)
        try:
            await db.execute(delete(Video))
            await db.commit()
            logger.info(f"Deleted {total_videos} videos from database")
        except Exception as e:
            logger.error(f"Error deleting videos from database: {str(e)}")
            await db.rollback()
            errors += 1
        
        logger.info(
            f"Deletion complete: {deleted_storage} video files, {deleted_moments} moment files, "
            f"{total_videos} database records, {errors} errors"
        )


if __name__ == "__main__":
    asyncio.run(delete_all_videos())

