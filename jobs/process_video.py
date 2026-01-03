import logging
from typing import Optional
from uuid import UUID

logger = logging.getLogger(__name__)


async def _process_video_async(video_id: UUID, clerk_user_id: Optional[str] = None) -> None:
    """
    Async implementation of video processing pipeline.
    
    This function handles:
    - Audio extraction
    - Transcription
    - Fluff/Repeated/Useful detection (8-step pipeline)
    - Segment analysis and storage
    
    Args:
        video_id: UUID of the video to process
        clerk_user_id: Optional Clerk user ID (fetched from video if not provided)
    """
    # Import here to avoid fork issues - these imports happen after fork
    from database import async_session_maker
    from db import crud
    from models import VideoStatus
    from utils.audio import extract_audio_async
    from utils.storage import get_video_path
    from utils.transcription import transcribe_audio_async, cleanup_audio_file
    
    audio_path = None
    clerk_user_id_local = clerk_user_id
    try:
        # Use short-lived sessions for each database operation to avoid connection timeouts
        # Get video record to fetch clerk_user_id if not provided
        async with async_session_maker() as init_db:
            video = await crud.get_video_by_id(init_db, video_id)
            if not video:
                raise ValueError(f"Video not found: {video_id}")
            
            # Use provided clerk_user_id or fetch from video record
            if clerk_user_id_local is None:
                clerk_user_id_local = video.clerk_user_id
            
            # Update status to PROCESSING
            await crud.update_video_status(init_db, video_id, VideoStatus.PROCESSING)
            logger.info(f"[ProcessVideo {video_id}] ✅ Started processing video (user_id: {clerk_user_id_local})")
            
            # Get video path - use presigned URL (don't download full file)
            logger.info(f"[ProcessVideo {video_id} (user: {clerk_user_id_local})] 📍 Step 1/5: Getting video path...")
            video_path = await get_video_path(video_id, init_db, download_local=False)
            if not video_path:
                raise ValueError(f"Video not found: {video_id}")
            logger.info(f"[ProcessVideo {video_id} (user: {clerk_user_id_local})] ✅ Got video path: {video_path[:80]}...")
        
        # Local processing: Extract audio and transcribe (no DB session needed)
        logger.info(f"[ProcessVideo {video_id} (user: {clerk_user_id_local})] 📍 Step 2/5: Extracting audio from video...")
        audio_path = await extract_audio_async(video_path)
        logger.info(f"[ProcessVideo {video_id} (user: {clerk_user_id_local})] ✅ Audio extracted to: {audio_path}")
        
        logger.info(f"[ProcessVideo {video_id} (user: {clerk_user_id_local})] 📍 Step 3/5: Transcribing audio with OpenAI Whisper API...")
        from models import SpeechSegment, TranscriptSegment
        timeline_items = await transcribe_audio_async(audio_path)
        
        # Validate transcript was received
        if not timeline_items:
            raise Exception("OpenAI Whisper API returned empty transcript - no segments found")
        
        # Filter out silence segments and convert SpeechSegment to TranscriptSegment for database storage
        speech_segments = [
            item for item in timeline_items 
            if item.type == "speech" and isinstance(item, SpeechSegment)
        ]
        silence_count = len(timeline_items) - len(speech_segments)
        
        logger.info(
            f"[ProcessVideo {video_id} (user: {clerk_user_id_local})] ✅ Transcription complete: "
            f"{len(speech_segments)} speech segments, {silence_count} silence segments "
            f"(total {len(timeline_items)} timeline items)"
        )
        
        # Convert SpeechSegment to TranscriptSegment for database storage
        transcript_segments = [
            TranscriptSegment(start=seg.start, end=seg.end, text=seg.text)
            for seg in speech_segments
        ]
        
        # Save transcript (database and Redis) - use fresh session
        logger.info(f"[ProcessVideo {video_id} (user: {clerk_user_id_local})] 📍 Saving {len(transcript_segments)} transcript segments to database and Redis...")
        segments_data = [segment.model_dump() for segment in transcript_segments]
        
        # Validate segments data before saving
        if not segments_data:
            raise Exception("No transcript segment data to save")
        
        logger.info(f"[ProcessVideo {video_id} (user: {clerk_user_id_local})] 💾 Saving transcript with {len(segments_data)} segments to database...")
        async with async_session_maker() as transcript_db:
            await crud.create_transcript(transcript_db, video_id, segments_data)
            await crud.update_video_status(transcript_db, video_id, VideoStatus.TRANSCRIBED)
            logger.info(f"[ProcessVideo {video_id} (user: {clerk_user_id_local})] ✅ Transcript saved to database: {len(segments_data)} segments, status: TRANSCRIBED")
        
        # Also save to Redis cache
        from utils.redis_cache import save_transcript_to_redis
        save_transcript_to_redis(video_id, segments_data, ttl=86400 * 7)  # 7 days TTL
            
        # Section 5 - Fluff/Repeated/Useful Detection and Highlights (run in parallel)
        logger.info(f"[ProcessVideo {video_id} (user: {clerk_user_id_local})] 📍 Step 4/5: Starting fluff detection and highlights generation in parallel...")
        from utils.fluff_detection import analyze_video_for_fluff
        from utils.highlights import find_highlights_from_sentences_async, aggregate_and_rank_highlights
        from models import Sentence, Word
        
        # Convert transcript segments to sentences for highlights generation
        sentences_for_highlights = []
        for seg in transcript_segments:
            segment_words = []
            word_start = seg.start
            word_duration = (seg.end - seg.start) / max(1, len(seg.text.split()))
            
            for word_text in seg.text.split():
                word = Word(
                    word=word_text,
                    start=word_start,
                    end=word_start + word_duration,
                    confidence=1.0
                )
                segment_words.append(word)
                word_start += word_duration
            
            sentence = Sentence(
                start=seg.start,
                end=seg.end,
                text=seg.text,
                words=segment_words
            )
            sentences_for_highlights.append(sentence)
        
        # Helper function to generate highlights (without saving)
        async def generate_highlights():
            try:
                logger.info(f"[ProcessVideo {video_id} (user: {clerk_user_id_local})] 🔄 Generating highlights...")
                highlights = await find_highlights_from_sentences_async(sentences_for_highlights)
                logger.info(f"[ProcessVideo {video_id} (user: {clerk_user_id_local})] ✅ Found {len(highlights)} highlight candidates")
                return highlights
            except Exception as e:
                logger.error(f"[ProcessVideo {video_id} (user: {clerk_user_id_local})] ❌ Failed to generate highlights: {e}", exc_info=True)
                logger.warning(f"[ProcessVideo {video_id} (user: {clerk_user_id_local})] ⚠️  Continuing without highlights")
                return []
        
        # Run fluff detection and highlights generation in parallel
        import asyncio
        fluff_task = analyze_video_for_fluff(
            video_id,
            transcript_segments=transcript_segments,
            video_path=video_path,
            skip_highlights=True  # Skip highlights in fluff detection since we're doing it in parallel
        )
        highlights_task = generate_highlights()
        
        segment_analyses, highlight_candidates = await asyncio.gather(fluff_task, highlights_task, return_exceptions=True)
        
        # Handle exceptions
        if isinstance(segment_analyses, Exception):
            logger.error(f"[ProcessVideo {video_id} (user: {clerk_user_id_local})] ❌ Fluff detection failed: {segment_analyses}", exc_info=True)
            raise segment_analyses
        
        if isinstance(highlight_candidates, Exception):
            logger.error(f"[ProcessVideo {video_id} (user: {clerk_user_id_local})] ❌ Highlight generation failed: {highlight_candidates}", exc_info=True)
            highlight_candidates = []
        
        logger.info(f"[ProcessVideo {video_id} (user: {clerk_user_id_local})] ✅ Fluff detection and highlights generation complete: {len(segment_analyses)} segment analyses, {len(highlight_candidates)} highlight candidates")
        
        # Trim fluff from highlights and aggregate
        from models import SegmentAnalysis
        fluff_segments = [s for s in segment_analyses if s.label == "FLUFF"]
        
        if highlight_candidates:
            async with async_session_maker() as db:
                final_highlights = await aggregate_and_rank_highlights(
                    highlight_candidates,
                    max_results=10,
                    db_session=db,
                    fluff_segments=fluff_segments if fluff_segments else None
                )
                logger.info(f"[ProcessVideo {video_id} (user: {clerk_user_id_local})] ✅ Aggregated to {len(final_highlights)} final highlights (after fluff trimming)")
                
                # Save highlights to database (batch insert for performance)
                if final_highlights:
                    logger.info(f"[ProcessVideo {video_id} (user: {clerk_user_id_local})] 💾 Saving {len(final_highlights)} highlights to database...")
                    highlights_data = [
                        {
                            "video_id": video_id,
                            "start": highlight.start,
                            "end": highlight.end,
                            "title": highlight.title,
                            "summary": highlight.summary,
                            "score": highlight.score,
                        }
                        for highlight in final_highlights
                    ]
                    await crud.create_highlights_batch(db, highlights_data)
                    logger.info(f"[ProcessVideo {video_id} (user: {clerk_user_id_local})] ✅ Saved {len(final_highlights)} highlights to database")
                else:
                    logger.warning(f"[ProcessVideo {video_id} (user: {clerk_user_id_local})] ⚠️  No highlights found to save after fluff trimming")
            
        # Store segments in database (batch insert for performance)
        logger.info(f"[ProcessVideo {video_id} (user: {clerk_user_id_local})] 📍 Step 5/5: Saving {len(segment_analyses)} segments to database (batch)...")
        if segment_analyses:
                try:
                    # Prepare batch data for segments
                    segments_data = []
                    retention_metrics_data = []
                    
                    for i, segment in enumerate(segment_analyses):
                        # Get embedding from corresponding semantic segment if available
                        embedding = None
                        # Note: embedding is stored in SemanticSegment but we need to get it from the pipeline
                        # For now, we'll store None and can enhance later if needed
                        
                        # Get segment text and ID if available
                        segment_text = getattr(segment, '_segment_text', '')
                        segment_id = getattr(segment, '_segment_id', i + 1)
                        
                        segments_data.append({
                            "segment_id": segment_id,
                            "start_time": segment.start_time,
                            "end_time": segment.end_time,
                            "text": segment_text,
                            "label": segment.label,
                            "rating": segment.rating,
                            "grade": segment.grade,
                            "reason": segment.reason,
                            "repetition_score": segment.repetition_score,
                            "filler_density": segment.filler_density,
                            "visual_change_score": segment.visual_change_score,
                            "usefulness_score": segment.usefulness_score,
                            "embedding": embedding,
                        })
                        
                        # Prepare retention metrics if available
                        retention_analysis = getattr(segment, '_retention_analysis', None)
                        if retention_analysis:
                            try:
                                from models import RetentionAnalysis
                                time_range_dict = {
                                    "start": retention_analysis.time_range.start,
                                    "end": retention_analysis.time_range.end,
                                    "duration": retention_analysis.time_range.duration,
                                }
                                metrics_dict = {
                                    "semantic_novelty": {
                                        "value": retention_analysis.metrics.semantic_novelty.value,
                                        "max_similarity_to_history": retention_analysis.metrics.semantic_novelty.max_similarity_to_history,
                                        "window_size": retention_analysis.metrics.semantic_novelty.window_size,
                                    },
                                    "information_density": {
                                        "value": retention_analysis.metrics.information_density.value,
                                        "meaningful_token_count": retention_analysis.metrics.information_density.meaningful_token_count,
                                        "tfidf_weight_sum": retention_analysis.metrics.information_density.tfidf_weight_sum,
                                        "duration_seconds": retention_analysis.metrics.information_density.duration_seconds,
                                    },
                                    "emotional_delta": {
                                        "value": retention_analysis.metrics.emotional_delta.value,
                                        "sentiment_prev": retention_analysis.metrics.emotional_delta.sentiment_prev,
                                        "sentiment_curr": retention_analysis.metrics.emotional_delta.sentiment_curr,
                                        "intensity_prev": retention_analysis.metrics.emotional_delta.intensity_prev,
                                        "intensity_curr": retention_analysis.metrics.emotional_delta.intensity_curr,
                                    },
                                    "narrative_momentum": {
                                        "value": retention_analysis.metrics.narrative_momentum.value,
                                        "new_entities": retention_analysis.metrics.narrative_momentum.new_entities,
                                        "new_events": retention_analysis.metrics.narrative_momentum.new_events,
                                        "new_goals": retention_analysis.metrics.narrative_momentum.new_goals,
                                        "new_stakes": retention_analysis.metrics.narrative_momentum.new_stakes,
                                    },
                                }
                                decision_dict = {
                                    "action": retention_analysis.decision.action,
                                    "reason": retention_analysis.decision.reason,
                                }
                                
                                retention_metrics_data.append({
                                    "segment_id": segment_id,
                                    "time_range": time_range_dict,
                                    "text": segment_text,
                                    "metrics": metrics_dict,
                                    "retention_value": retention_analysis.retention_value,
                                    "decision": decision_dict,
                                })
                            except Exception as e:
                                logger.warning(f"[ProcessVideo {video_id}] ⚠️ Failed to prepare retention metrics for segment {segment_id}: {e}")
                    
                    # Batch insert segments - use fresh session to avoid connection timeout
                    logger.info(f"[ProcessVideo {video_id} (user: {clerk_user_id_local})] 💾 Batch inserting {len(segments_data)} segments...")
                    async with async_session_maker() as save_db:
                        await crud.create_video_segments_batch(save_db, video_id, segments_data)
                        logger.info(f"[ProcessVideo {video_id} (user: {clerk_user_id_local})] ✅ Batch inserted {len(segments_data)} segments")
                        
                        # Batch insert retention metrics if any
                        if retention_metrics_data:
                            logger.info(f"[ProcessVideo {video_id} (user: {clerk_user_id_local})] 💾 Batch inserting {len(retention_metrics_data)} retention metrics...")
                            await crud.create_retention_metrics_batch(save_db, video_id, retention_metrics_data)
                            logger.info(f"[ProcessVideo {video_id} (user: {clerk_user_id_local})] ✅ Batch inserted {len(retention_metrics_data)} retention metrics")
                        
                        await crud.update_video_status(save_db, video_id, VideoStatus.SEGMENTS_ANALYZED)
                        logger.info(f"[ProcessVideo {video_id} (user: {clerk_user_id_local})] ✅ Saved {len(segments_data)} segments, status: SEGMENTS_ANALYZED")
                
                except Exception as e:
                    error_msg = f"Failed to save segments for video {video_id}: {str(e)}"
                    logger.error(f"[ProcessVideo {video_id}] ❌ {error_msg}", exc_info=True)
                    raise Exception(error_msg)
        else:
            logger.warning(f"[ProcessVideo {video_id} (user: {clerk_user_id_local})] ⚠️  No segments found")
            # Still mark as SEGMENTS_ANALYZED (empty list is valid) - use fresh session
            async with async_session_maker() as status_db:
                await crud.update_video_status(status_db, video_id, VideoStatus.SEGMENTS_ANALYZED)
                logger.info(f"[ProcessVideo {video_id} (user: {clerk_user_id_local})] ✅ Status updated to SEGMENTS_ANALYZED (empty)")
        
        # Update status to DONE - use fresh session to avoid connection timeout
        async with async_session_maker() as final_db:
            await crud.update_video_status(final_db, video_id, VideoStatus.DONE)
            
            # Log final segment counts
            segments_db = await crud.get_video_segments(final_db, video_id)
            fluff_count = sum(1 for s in segments_db if s.label == "FLUFF")
            
            logger.info(
                f"[ProcessVideo] ✅ Completed fluff detection for video: {video_id} (user_id: {clerk_user_id_local}) - "
                f"Total segments: {len(segments_db)} (FLUFF: {fluff_count})"
            )
            
            # Learning pipeline disabled
            # # Run learning pipeline after successful processing
            # try:
            #     from jobs.learning_job import run_learning_pipeline
            #     async with async_session_maker() as learning_db:
            #         result = await run_learning_pipeline(learning_db)
            #         if result.get("calibration_updated") or result.get("prompt_promoted") or result.get("new_prompt_created"):
            #             logger.info(f"Learning pipeline updated system: {result}")
            # except Exception as e:
            #     logger.warning(f"Learning pipeline failed (non-blocking): {str(e)}", exc_info=True)
    
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error processing video {video_id} (user_id: {clerk_user_id_local}): {error_msg}", exc_info=True)
        # Use separate session for error handling to ensure status update succeeds
        try:
            async with async_session_maker() as error_db:
                # Truncate error message to 1000 chars to avoid database issues
                truncated_error = error_msg[:1000] if len(error_msg) > 1000 else error_msg
                await crud.update_video_status(error_db, video_id, VideoStatus.FAILED, error_message=truncated_error)
        except Exception as db_error:
            logger.error(f"Failed to update status to FAILED: {str(db_error)}", exc_info=True)
        raise
    
    finally:
        # Clean up audio file after transcription (or on error)
        if audio_path:
            cleanup_audio_file(audio_path)

