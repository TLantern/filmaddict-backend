import asyncio
import logging
import os
from uuid import UUID

logger = logging.getLogger(__name__)

# Async task queue for video processing
_video_processing_queue: asyncio.Queue = None
_worker_task: asyncio.Task = None
_learning_task: asyncio.Task = None

LEARNING_JOB_INTERVAL_HOURS = int(os.getenv("LEARNING_JOB_INTERVAL_HOURS", "24"))


def init_task_queue() -> None:
    """Initialize the async task queue."""
    global _video_processing_queue
    if _video_processing_queue is None:
        _video_processing_queue = asyncio.Queue()


async def _process_video_worker() -> None:
    """Background worker that processes videos from the queue."""
    from jobs.process_video import _process_video_async
    
    logger.info("Video processing worker started")
    while True:
        try:
            video_id = await _video_processing_queue.get()
            logger.info(f"Processing video from queue: {video_id}")
            try:
                await _process_video_async(video_id)
                logger.info(f"Successfully processed video: {video_id}")
            except Exception as e:
                logger.error(f"Error processing video {video_id}: {str(e)}", exc_info=True)
            finally:
                _video_processing_queue.task_done()
        except asyncio.CancelledError:
            logger.info("Video processing worker cancelled")
            break
        except Exception as e:
            logger.error(f"Unexpected error in video processing worker: {str(e)}", exc_info=True)


async def start_worker() -> None:
    """Start the background worker task."""
    global _worker_task
    if _worker_task is None or _worker_task.done():
        init_task_queue()
        _worker_task = asyncio.create_task(_process_video_worker())
        logger.info("Video processing worker task started")


async def stop_worker() -> None:
    """Stop the background worker task."""
    global _worker_task
    if _worker_task and not _worker_task.done():
        _worker_task.cancel()
        try:
            await _worker_task
        except asyncio.CancelledError:
            pass
        logger.info("Video processing worker stopped")


async def enqueue_video_processing(video_id: UUID) -> None:
    """
    Enqueue a video processing job.
    
    Args:
        video_id: UUID of the video to process
    """
    init_task_queue()
    await _video_processing_queue.put(video_id)
    logger.info(f"Enqueued video for processing: {video_id}")


async def _learning_worker() -> None:
    """Background worker that runs learning pipeline periodically."""
    from database import async_session_maker
    from jobs.learning_job import run_learning_pipeline
    
    logger.info("Learning worker started")
    
    while True:
        try:
            await asyncio.sleep(LEARNING_JOB_INTERVAL_HOURS * 3600)
            
            logger.info("Running scheduled learning pipeline")
            try:
                async with async_session_maker() as db:
                    result = await run_learning_pipeline(db)
                    logger.info(f"Learning pipeline completed: {result}")
            except Exception as e:
                logger.error(f"Error in scheduled learning pipeline: {str(e)}", exc_info=True)
        
        except asyncio.CancelledError:
            logger.info("Learning worker cancelled")
            break
        except Exception as e:
            logger.error(f"Unexpected error in learning worker: {str(e)}", exc_info=True)
            await asyncio.sleep(60)


async def start_learning_worker() -> None:
    """Start the background learning worker task."""
    global _learning_task
    if _learning_task is None or _learning_task.done():
        _learning_task = asyncio.create_task(_learning_worker())
        logger.info("Learning worker task started")


async def stop_learning_worker() -> None:
    """Stop the background learning worker task."""
    global _learning_task
    if _learning_task and not _learning_task.done():
        _learning_task.cancel()
        try:
            await _learning_task
        except asyncio.CancelledError:
            pass
        logger.info("Learning worker stopped")


# RQ (Redis Queue) setup for separate worker process
try:
    from rq import Queue
    from redis import Redis
    import redis
    
    # Get Redis URL from environment
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    
    # Parse Redis URL and create connection
    # Handle both redis:// and rediss:// (SSL) URLs
    if redis_url.startswith("rediss://"):
        # SSL connection
        redis_conn = redis.from_url(
            redis_url,
            ssl_cert_reqs=None,
            decode_responses=False
        )
    else:
        # Non-SSL connection
        redis_conn = redis.from_url(redis_url, decode_responses=False)
    
    # Create RQ queue for video processing
    video_processing_queue = Queue("video_processing", connection=redis_conn)
    
except ImportError:
    # RQ not available, set to None
    redis_conn = None
    video_processing_queue = None
    logger.warning("RQ not available - worker.py will not work without rq and redis packages")

