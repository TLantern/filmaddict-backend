import logging
import os
from uuid import UUID

logger = logging.getLogger(__name__)


async def _process_video_async(video_id: UUID) -> None:
    """
    Async implementation of video processing pipeline.
    
    This function handles:
    - Audio extraction (section 4.1)
    - Transcription (section 4.2)
    - Highlight discovery (section 5)
    - Clip generation (section 6)
    
    Args:
        video_id: UUID of the video to process
    """
    # Import here to avoid fork issues - these imports happen after fork
    from database import async_session_maker
    from db import crud
    from models import VideoStatus
    from utils.audio import extract_audio_async
    from utils.clips import generate_clips_for_video
    from utils.highlights import aggregate_and_rank_highlights, chunk_transcript, find_highlights_async
    from utils.storage import get_video_path
    from utils.transcription import transcribe_audio_async, cleanup_audio_file
    
    async with async_session_maker() as db:
        audio_path = None
        try:
            # Update status to PROCESSING
            await crud.update_video_status(db, video_id, VideoStatus.PROCESSING)
            logger.info(f"Started processing video: {video_id}")
            
            # Get video path
            video_path = await get_video_path(video_id, db)
            if not video_path:
                raise ValueError(f"Video not found: {video_id}")
            
            # Section 4.1 - Extract audio
            audio_path = await extract_audio_async(video_path)
            
            # Section 4.2 - Transcribe audio
            transcript_segments = await transcribe_audio_async(audio_path)
            
            # Convert Pydantic models to dicts for JSONB storage
            segments_data = [segment.model_dump() for segment in transcript_segments]
            await crud.create_transcript(db, video_id, segments_data)
            await crud.update_video_status(db, video_id, VideoStatus.TRANSCRIBED)
            
            # Section 5 - Find highlights
            # 5.1 - Chunk transcript
            chunks = chunk_transcript(transcript_segments)
            logger.info(f"Created {len(chunks)} chunks from transcript")
            
            # 5.2 - Get active prompt version
            prompt_version = await crud.get_active_prompt_version(db)
            system_prompt = None
            user_prompt_template = None
            prompt_version_id = None
            
            if prompt_version:
                system_prompt = prompt_version.system_prompt
                user_prompt_template = prompt_version.user_prompt_template
                prompt_version_id = prompt_version.id
                logger.info(f"Using prompt version: {prompt_version.version_name}")
            else:
                logger.info("No active prompt version found, using defaults")
            
            # 5.3 - Find highlights using GPT (parallel processing)
            highlight_candidates = await find_highlights_async(chunks, system_prompt, user_prompt_template)
            logger.info(f"Found {len(highlight_candidates)} highlight candidates")
            
            # 5.4 - Aggregate and rank highlights (with calibration)
            highlights = await aggregate_and_rank_highlights(highlight_candidates, max_results=10, db_session=db)
            logger.info(f"Final highlights after aggregation: {len(highlights)}")
            
            # 5.5 - Store highlights with prompt version tracking
            if highlights:
                saved_count = 0
                failed_count = 0
                last_error = None
                for highlight in highlights:
                    try:
                        # Rollback any previous failed transaction
                        try:
                            await db.rollback()
                        except:
                            pass
                        
                        highlight_db = await crud.create_highlight(
                            db, video_id, highlight.start, highlight.end, highlight.reason, highlight.score, prompt_version_id
                        )
                        saved_count += 1
                    except Exception as e:
                        failed_count += 1
                        last_error = str(e)
                        logger.error(
                            f"Failed to save highlight [{highlight.start:.2f}-{highlight.end:.2f}s] "
                            f"for video {video_id}: {str(e)}",
                            exc_info=True
                        )
                        # Rollback failed transaction
                        try:
                            await db.rollback()
                        except:
                            pass
                
                if saved_count > 0:
                    await crud.update_video_status(db, video_id, VideoStatus.HIGHLIGHTS_FOUND)
                    logger.info(f"Saved {saved_count} highlights for video {video_id}")
                    if failed_count > 0:
                        logger.warning(f"Failed to save {failed_count} highlights for video {video_id}")
                else:
                    error_msg = f"Failed to save any highlights for video {video_id}"
                    if last_error:
                        error_msg += f": {last_error}"
                    raise Exception(error_msg)
            else:
                logger.warning(f"No highlights found for video {video_id}")
                # Still mark as HIGHLIGHTS_FOUND (empty list is valid)
                await crud.update_video_status(db, video_id, VideoStatus.HIGHLIGHTS_FOUND)
            
            # Section 6 - Generate clips (optional)
            generate_clips = os.getenv("GENERATE_CLIPS", "true").lower() == "true"
            if generate_clips:
                try:
                    clips = await generate_clips_for_video(video_id)
                    logger.info(f"Generated {len(clips)} clips for video {video_id}")
                except Exception as e:
                    logger.error(f"Error generating clips for video {video_id}: {str(e)}", exc_info=True)
                    # Continue processing even if clip generation fails
            else:
                logger.info(f"Clip generation disabled for video {video_id} (GENERATE_CLIPS=false)")
            
            # Update status to DONE
            await crud.update_video_status(db, video_id, VideoStatus.DONE)
            logger.info(f"Completed processing video: {video_id}")
            
            # Run learning pipeline after successful processing
            try:
                from jobs.learning_job import run_learning_pipeline
                async with async_session_maker() as learning_db:
                    result = await run_learning_pipeline(learning_db)
                    if result.get("calibration_updated") or result.get("prompt_promoted") or result.get("new_prompt_created"):
                        logger.info(f"Learning pipeline updated system: {result}")
            except Exception as e:
                logger.warning(f"Learning pipeline failed (non-blocking): {str(e)}", exc_info=True)
            
        except Exception as e:
            logger.error(f"Error processing video {video_id}: {str(e)}", exc_info=True)
            async with async_session_maker() as error_db:
                await crud.update_video_status(error_db, video_id, VideoStatus.FAILED)
            raise
        finally:
            # Clean up audio file after transcription (or on error)
            if audio_path:
                cleanup_audio_file(audio_path)

