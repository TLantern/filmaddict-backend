import asyncio
import logging
from typing import List, Optional
from uuid import UUID

from database import async_session_maker
from db import crud
from models import SegmentAnalysis
from utils.audio import extract_audio_async
from utils.filler_detection import detect_fillers
from utils.retention_scoring import compute_retention_scores_async
from utils.model_loader import get_embedding_model
from utils.storage import get_video_path
from utils.silence_detection import detect_silence_segments
from utils.video_cutter import process_video_cutting

logger = logging.getLogger(__name__)

# Segment grouping configuration
TARGET_SEGMENT_COUNT_MIN = 50
TARGET_SEGMENT_COUNT_MAX = 75
MAX_SEGMENT_COUNT = 100
RATING_SIMILARITY_THRESHOLD = 0.15
SCORE_SIMILARITY_THRESHOLD = 0.2
MIN_FLUFF_DURATION = 3.5  # seconds - minimum duration for a segment to qualify as FLUFF
HOOK_PROTECTION_WINDOW = 45.0  # seconds - first 45 seconds protected from FLUFF labeling (creator hook)


def _are_segments_similar(
    seg1: SegmentAnalysis,
    seg2: SegmentAnalysis,
    rating_threshold: float = RATING_SIMILARITY_THRESHOLD,
    score_threshold: float = SCORE_SIMILARITY_THRESHOLD,
) -> bool:
    """
    Check if two segments are similar enough to be grouped together.
    
    Criteria:
    - Same label
    - Similar ratings (within threshold)
    - Similar scores (repetition, filler within thresholds)
    - Adjacent in time (seg2 starts right after seg1 ends or very close)
    """
    # Must have same label
    if seg1.label != seg2.label:
        return False
    
    # Ratings must be similar
    if abs(seg1.rating - seg2.rating) > rating_threshold:
        return False
    
    # Scores must be similar
    if abs(seg1.repetition_score - seg2.repetition_score) > score_threshold:
        return False
    if abs(seg1.filler_density - seg2.filler_density) > score_threshold:
        return False
    
    # Must be temporally adjacent (seg2 starts within 2 seconds of seg1 end)
    time_gap = seg2.start_time - seg1.end_time
    if time_gap < 0 or time_gap > 2.0:
        return False
    
    return True


def _merge_segments(segments: List[SegmentAnalysis]) -> SegmentAnalysis:
    """
    Merge multiple segments into a single segment.
    
    - Combines text from all segments
    - Uses earliest start_time and latest end_time
    - Averages all scores
    - Uses most common label (or highest rating label if tie)
    - Combines reasons
    """
    if not segments:
        raise ValueError("Cannot merge empty segment list")
    
    if len(segments) == 1:
        return segments[0]
    
    # Combine text
    combined_text = " ".join(getattr(seg, '_segment_text', '') for seg in segments)
    
    # Time range
    start_time = min(seg.start_time for seg in segments)
    end_time = max(seg.end_time for seg in segments)
    
    # Average scores
    rating = sum(seg.rating for seg in segments) / len(segments)
    repetition_score = sum(seg.repetition_score for seg in segments) / len(segments)
    filler_density = sum(seg.filler_density for seg in segments) / len(segments)
    visual_change_score = sum(seg.visual_change_score for seg in segments) / len(segments)
    usefulness_score = sum(seg.usefulness_score for seg in segments) / len(segments)
    
    # Most common label (or highest rating if tie)
    label_counts = {}
    label_ratings = {}
    for seg in segments:
        label_counts[seg.label] = label_counts.get(seg.label, 0) + 1
        if seg.label not in label_ratings:
            label_ratings[seg.label] = []
        label_ratings[seg.label].append(seg.rating)
    
    max_count = max(label_counts.values())
    candidates = [label for label, count in label_counts.items() if count == max_count]
    
    if len(candidates) == 1:
        label = candidates[0]
    else:
        # Tie - use highest average rating
        label = max(candidates, key=lambda l: sum(label_ratings[l]) / len(label_ratings[l]))
    
    # Grade based on merged rating
    if rating >= 0.9:
        grade = "A"
    elif rating >= 0.8:
        grade = "B"
    elif rating >= 0.5:
        grade = "C"
    elif rating >= 0.2:
        grade = "D"
    else:
        grade = "F"
    
    # Combine reasons
    reasons = [seg.reason for seg in segments if seg.reason]
    if len(reasons) == 1:
        reason = reasons[0]
    else:
        reason = f"Merged {len(segments)} segments: " + " | ".join(set(reasons[:3]))  # Limit to avoid too long
    
    merged = SegmentAnalysis(
        start_time=start_time,
        end_time=end_time,
        label=label,
        rating=rating,
        grade=grade,
        reason=reason,
        repetition_score=repetition_score,
        filler_density=filler_density,
        visual_change_score=visual_change_score,
        usefulness_score=usefulness_score,
    )
    
    # Preserve segment text
    merged._segment_text = combined_text
    
    return merged


def _group_similar_segments(
    analyses: List[SegmentAnalysis],
    rating_threshold: float = RATING_SIMILARITY_THRESHOLD,
    score_threshold: float = SCORE_SIMILARITY_THRESHOLD,
) -> List[SegmentAnalysis]:
    """
    Group similar adjacent segments together.
    
    Iterates through segments and merges adjacent ones that are similar.
    """
    if not analyses:
        return []
    
    if len(analyses) == 1:
        return analyses
    
    grouped = []
    current_group = [analyses[0]]
    
    for i in range(1, len(analyses)):
        current_seg = analyses[i]
        last_seg = current_group[-1]
        
        if _are_segments_similar(last_seg, current_seg, rating_threshold, score_threshold):
            # Add to current group
            current_group.append(current_seg)
        else:
            # Merge current group and start new one
            grouped.append(_merge_segments(current_group))
            current_group = [current_seg]
    
    # Add final group
    if current_group:
        grouped.append(_merge_segments(current_group))
    
    return grouped


def _limit_segment_count(analyses: List[SegmentAnalysis]) -> List[SegmentAnalysis]:
    """
    Limit segment count to target range (50-75) with max 100.
    
    Strategy:
    - If > 100: Aggressively merge (widen thresholds)
    - If > 75: Merge adjacent segments with same label and similar ratings
    - If < 50: Keep as is (don't split)
    """
    count = len(analyses)
    
    if count <= MAX_SEGMENT_COUNT and count >= TARGET_SEGMENT_COUNT_MIN:
        # Within acceptable range
        return analyses
    
    if count > MAX_SEGMENT_COUNT:
        # Too many - aggressively merge
        logger.info(f"Segment count {count} exceeds max {MAX_SEGMENT_COUNT}, applying aggressive merging")
        # Widen thresholds for aggressive merging
        aggressive_rating_threshold = RATING_SIMILARITY_THRESHOLD * 2.0  # 0.30
        aggressive_score_threshold = SCORE_SIMILARITY_THRESHOLD * 2.0  # 0.40
        
        result = _group_similar_segments(analyses, aggressive_rating_threshold, aggressive_score_threshold)
        
        # If still too many, merge more aggressively (limit iterations to avoid excessive processing)
        max_iterations = 5
        iteration = 0
        while len(result) > MAX_SEGMENT_COUNT and iteration < max_iterations:
            aggressive_rating_threshold *= 1.5
            aggressive_score_threshold *= 1.5
            result = _group_similar_segments(result, aggressive_rating_threshold, aggressive_score_threshold)
            iteration += 1
            # Safety check to avoid infinite loop
            if aggressive_rating_threshold > 1.0:
                break
        
        return result
    
    if count > TARGET_SEGMENT_COUNT_MAX:
        # Between 75-100, merge moderately
        logger.info(f"Segment count {count} exceeds target max {TARGET_SEGMENT_COUNT_MAX}, applying moderate merging")
        moderate_rating_threshold = RATING_SIMILARITY_THRESHOLD * 1.5  # 0.225
        moderate_score_threshold = SCORE_SIMILARITY_THRESHOLD * 1.5  # 0.30
        
        result = _group_similar_segments(analyses, moderate_rating_threshold, moderate_score_threshold)
        
        # If still above target, merge more (limit iterations to avoid excessive processing)
        max_iterations = 3
        iteration = 0
        while len(result) > TARGET_SEGMENT_COUNT_MAX and iteration < max_iterations:
            moderate_rating_threshold *= 1.2
            moderate_score_threshold *= 1.2
            result = _group_similar_segments(result, moderate_rating_threshold, moderate_score_threshold)
            iteration += 1
            if moderate_rating_threshold > 0.5:
                break
        
        return result
    
    # count < 50 - keep as is (don't split segments)
    logger.info(f"Segment count {count} is below target min {TARGET_SEGMENT_COUNT_MIN}, keeping as is")
    return analyses


def _filter_short_fluff_segments(analyses: List[SegmentAnalysis]) -> List[SegmentAnalysis]:
    """
    Filter out FLUFF segments that are shorter than MIN_FLUFF_DURATION.
    
    Args:
        analyses: List of SegmentAnalysis objects
        
    Returns:
        List of SegmentAnalysis objects with short FLUFF segments removed
    """
    filtered = []
    removed_count = 0
    
    for analysis in analyses:
        if analysis.label == "FLUFF":
            duration = analysis.end_time - analysis.start_time
            if duration >= MIN_FLUFF_DURATION:
                filtered.append(analysis)
            else:
                removed_count += 1
                logger.debug(f"Removed FLUFF segment: {duration:.2f}s < {MIN_FLUFF_DURATION}s (start: {analysis.start_time:.2f}s, end: {analysis.end_time:.2f}s)")
        else:
            filtered.append(analysis)
    
    if removed_count > 0:
        logger.info(f"Filtered out {removed_count} FLUFF segments shorter than {MIN_FLUFF_DURATION}s")
    
    return filtered


def _filter_hook_protection(analyses: List[SegmentAnalysis]) -> List[SegmentAnalysis]:
    """
    Filter out FLUFF segments in the first 45 seconds (creator hook protection).
    
    The first 45 seconds of a video typically contain the creator's hook/introduction
    and should not be marked as fluff.
    
    Args:
        analyses: List of SegmentAnalysis objects
        
    Returns:
        List of SegmentAnalysis objects with FLUFF segments in first 45 seconds removed
    """
    filtered = []
    removed_count = 0
    
    for analysis in analyses:
        if analysis.label == "FLUFF" and analysis.start_time < HOOK_PROTECTION_WINDOW:
            removed_count += 1
            logger.debug(f"Removed FLUFF segment from first {HOOK_PROTECTION_WINDOW}s: {analysis.start_time:.2f}s - {analysis.end_time:.2f}s")
        else:
            filtered.append(analysis)
    
    if removed_count > 0:
        logger.info(f"Removed {removed_count} FLUFF segment(s) from first {HOOK_PROTECTION_WINDOW}s (hook protection)")
    
    return filtered


async def analyze_video_for_fluff(
    video_id: UUID,
    transcript_segments: Optional[List] = None,
    video_path: Optional[str] = None,
    skip_highlights: bool = False,
) -> List[SegmentAnalysis]:
    """
    Main pipeline orchestrator for fluff detection.
    
    Uses Colab GPU processing for 10x speedup when COLAB_API_URL is set.
    Falls back to local processing if Colab unavailable.
    
    Coordinates all 7 pipeline steps:
    1. Audio extraction
    2. Transcription (large-v3 with word timestamps) - openai whisper server (skipped if transcript provided)
    3. Highlight Detection - Local (rule-based analysis from transcript)
    4. Visual Change Scoring (V-JEPA2) - Colab GPU currently disabled
    4.5. Silence Removal - Cut silences > 1.3s from video and add timeline markers
    5. Filler Detection - Local
    6. Retention Scoring - Local
    7. LLM Final Judge - Local
    
    Args:
        video_id: UUID of the video to analyze
        transcript_segments: Optional list of TranscriptSegment objects (if provided, skips transcription)
        video_path: Optional video path (if not provided, will be fetched from database)
        
    Returns:
        List of SegmentAnalysis objects with labels and scores
    """
    logger.info(f"Starting fluff detection pipeline for video: {video_id}")
    
    # Check if transcript is already provided (from process_video.py)
    has_existing_transcript = transcript_segments is not None and len(transcript_segments) > 0
    
    try:
        # Get video path if not provided
        if video_path is None:
            async with async_session_maker() as db:
                video_path = await get_video_path(video_id, db, download_local=False)
                if not video_path:
                    raise ValueError(f"Video not found: {video_id}")
        
        # Use local processing by default (Colab disabled)
        import os
        colab_url = os.getenv("COLAB_API_URL")
        # Only use Colab if explicitly enabled via environment variable
        use_colab = os.getenv("USE_COLAB", "false").lower() == "true" and colab_url is not None
        
        if use_colab:
            try:
                logger.info(f"[FluffDetection {video_id}] 🚀 Using Colab GPU processing for 10x speedup")
                from utils.colab_processor import process_video_on_colab
                
                # Visual analysis disabled - always use default scores
                enable_visual = False
                logger.info(f"[FluffDetection {video_id}] 📍 Sending video to Colab API (visual analysis: disabled)...")
                
                # Process on Colab (all heavy lifting on GPU)
                words, sentences, segments, visual_scores_dict = await process_video_on_colab(
                    video_path, 
                    enable_visual=enable_visual
                )
                
                logger.info(f"[FluffDetection {video_id}] ✅ Colab processing complete: {len(words)} words, {len(sentences)} sentences")
                
                # Build words list from sentences (only if words list is empty or incomplete)
                if not words or len(words) == 0:
                    words = []
                    for sentence in sentences:
                        words.extend(sentence.words)
                
                # Use default visual scores (0.5) for all sentences
                visual_scores = [0.5] * len(sentences)
                
                # Save transcript from Colab results (database and Redis) - only if not already provided
                if not has_existing_transcript:
                    logger.info(f"[FluffDetection {video_id}] 📍 Saving transcript from Colab results to database and Redis...")
                    async with async_session_maker() as db:
                        from models import TranscriptSegment
                        from models import VideoStatus
                        
                        # Convert normalized sentences to transcript segments
                        transcript_segments_colab = [
                            TranscriptSegment(
                                start=sentence.start,
                                end=sentence.end,
                                text=sentence.text
                            )
                            for sentence in sentences
                        ]
                        
                        segments_data = [segment.model_dump() for segment in transcript_segments_colab]
                        await crud.create_transcript(db, video_id, segments_data)
                        await crud.update_video_status(db, video_id, VideoStatus.TRANSCRIBED)
                        logger.info(f"[FluffDetection {video_id}] ✅ Transcript saved to database: {len(transcript_segments_colab)} segments, status: TRANSCRIBED")
                    
                    # Also save to Redis cache
                    from utils.redis_cache import save_transcript_to_redis
                    save_transcript_to_redis(video_id, segments_data, ttl=86400 * 7)  # 7 days TTL
                else:
                    logger.info(f"[FluffDetection {video_id}] ⏭️  Skipping transcript save (already provided)")
                
                # Step 3: Find Highlights (using sentences from Colab)
                if not skip_highlights:
                    logger.info(f"[FluffDetection {video_id}] 📍 Step 3/7: Finding highlights from sentences...")
                    try:
                        from utils.highlights import find_highlights_from_sentences_async, aggregate_and_rank_highlights
                        
                        # Find highlights using sentences (sentence-by-sentence like fluff detection)
                        highlights = await find_highlights_from_sentences_async(sentences)
                        logger.info(f"[FluffDetection {video_id}] ✅ Found {len(highlights)} highlight candidates")
                        
                        # Aggregate, deduplicate, and rank highlights
                        async with async_session_maker() as db:
                            final_highlights = await aggregate_and_rank_highlights(
                                highlights,
                                max_results=10,
                                db_session=db
                            )
                            logger.info(f"[FluffDetection {video_id}] ✅ Aggregated to {len(final_highlights)} final highlights")
                            
                            # Save highlights to database (batch insert for performance)
                            if final_highlights:
                                logger.info(f"[FluffDetection {video_id}] 💾 Saving {len(final_highlights)} highlights to database...")
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
                                logger.info(f"[FluffDetection {video_id}] ✅ Saved {len(final_highlights)} highlights to database")
                            else:
                                logger.warning(f"[FluffDetection {video_id}] ⚠️  No highlights found to save")
                                
                    except Exception as e:
                        logger.error(f"[FluffDetection {video_id}] ❌ Failed to find highlights: {e}", exc_info=True)
                        logger.warning(f"[FluffDetection {video_id}] ⚠️  Continuing without highlights")
                else:
                    logger.info(f"[FluffDetection {video_id}] ⏭️  Skipping highlights generation (will be done in parallel)")
                
            except Exception as e:
                logger.error(
                    f"[FluffDetection {video_id}] ❌ Colab processing failed: {str(e)}",
                    exc_info=True
                )
                logger.warning(f"[FluffDetection {video_id}] ⚠️  Falling back to local processing (requires ML libraries)")
                use_colab = False
        
        words = []
        sentences = []
        
        # If transcript is already provided, convert it to Sentence/Word objects and skip transcription
        if has_existing_transcript:
            logger.info(f"[FluffDetection {video_id}] ✅ Using provided transcript, skipping transcription step")
            from models import Sentence, Word, TranscriptSegment
            
            # Convert TranscriptSegment objects to Sentence and Word objects
            for seg in transcript_segments:
                # Create words from transcript segment text (estimate timestamps)
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
                    words.append(word)
                    word_start += word_duration
                
                # Create sentence from segment
                sentence = Sentence(
                    start=seg.start,
                    end=seg.end,
                    text=seg.text,
                    words=segment_words
                )
                sentences.append(sentence)
            
            logger.info(f"[FluffDetection {video_id}] ✅ Converted {len(transcript_segments)} transcript segments to {len(words)} words, {len(sentences)} sentences")
            
            # Use default visual scores (no visual analysis)
            visual_scores = [0.5] * len(sentences)
            
        
        # Fallback to local processing (no ML models) - only if transcript not provided
        elif not use_colab:
            logger.info(f"[FluffDetection {video_id}] 🏠 Using local processing (no ML models)")
            
            # Step 1: Transcription
            logger.info(f"[FluffDetection {video_id}] 📍 Step 1/7: Extracting audio...")
            audio_path = await extract_audio_async(video_path)
            logger.info(f"[FluffDetection {video_id}] ✅ Audio extracted")
            
            try:
                logger.info(f"[FluffDetection {video_id}] 📍 Step 2/7: Transcribing audio (Whisper if available)...")
                
                # Try Whisper first to get words and sentences directly
                from models import Sentence, Word
                words = []
                sentences = []
                transcript_segments = []
                
                try:
                    from utils.transcription_whisper import transcribe_with_word_timestamps_async
                    words, sentences = await transcribe_with_word_timestamps_async(audio_path)
                    
                    logger.info(f"[FluffDetection {video_id}] ✅ Whisper transcription: {len(words)} words, {len(sentences)} sentences")
                    
                    # Build words list from sentences (only if not already built)
                    if not words or len(words) == 0:
                        words = []
                        for sentence in sentences:
                            words.extend(sentence.words)
                    
                    # Convert normalized sentences to TranscriptSegments for database storage
                    from models import TranscriptSegment
                    transcript_segments = [
                        TranscriptSegment(
                            start=sentence.start,
                            end=sentence.end,
                            text=sentence.text
                        )
                        for sentence in sentences
                    ]
                except (ImportError, Exception) as e:
                    # Fallback: use transcription.py (which will try Whisper)
                    logger.info(f"[FluffDetection {video_id}] Using transcription module (Whisper may be unavailable): {e}")
                    from utils.transcription import transcribe_audio_async
                    from models import SpeechSegment
                    timeline_items = await transcribe_audio_async(audio_path)
                    
                    # Filter to only speech segments (ignore silence segments)
                    speech_segments = [
                        item for item in timeline_items
                        if item.type == "speech" and isinstance(item, SpeechSegment)
                    ]
                    
                    # Create words and sentences from speech segments only
                    for seg in speech_segments:
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
                            words.append(word)
                            word_start += word_duration
                        
                        sentence = Sentence(
                            start=seg.start,
                            end=seg.end,
                            text=seg.text,
                            words=segment_words
                        )
                        sentences.append(sentence)
                    
                    logger.info(f"[FluffDetection {video_id}] ✅ Created {len(words)} words, {len(sentences)} sentences from {len(speech_segments)} speech segments (filtered from {len(timeline_items)} timeline items)")
                
                # Words list already built above, no need to rebuild
                
                # Convert normalized sentences to TranscriptSegments for database storage
                from models import TranscriptSegment
                transcript_segments_local = [
                    TranscriptSegment(
                        start=sentence.start,
                        end=sentence.end,
                        text=sentence.text
                    )
                    for sentence in sentences
                ]
                
                # Save transcript to database and Redis (ensure it's preserved) - only if not already provided
                if not has_existing_transcript:
                    logger.info(f"[FluffDetection {video_id}] 📍 Saving transcript to database and Redis...")
                    async with async_session_maker() as db:
                        from models import TranscriptSegment
                        from models import VideoStatus
                        
                        segments_data = [segment.model_dump() for segment in transcript_segments_local]
                        await crud.create_transcript(db, video_id, segments_data)
                        await crud.update_video_status(db, video_id, VideoStatus.TRANSCRIBED)
                        logger.info(f"[FluffDetection {video_id}] ✅ Transcript saved to database: {len(transcript_segments_local)} segments, status: TRANSCRIBED")
                    
                    # Also save to Redis cache
                    from utils.redis_cache import save_transcript_to_redis
                    save_transcript_to_redis(video_id, segments_data, ttl=86400 * 7)  # 7 days TTL
                else:
                    logger.info(f"[FluffDetection {video_id}] ⏭️  Skipping transcript save (already provided)")
                
                
                if not sentences:
                    logger.warning(f"[FluffDetection {video_id}] ⚠️  No sentences found")
                    return []
                
                # Step 3: Find Highlights (using transcript segments)
                if not skip_highlights:
                    logger.info(f"[FluffDetection {video_id}] 📍 Step 3/7: Finding highlights from sentences...")
                    try:
                        from utils.highlights import find_highlights_from_sentences_async, aggregate_and_rank_highlights
                        
                        # Find highlights using sentences (sentence-by-sentence like fluff detection)
                        highlights = await find_highlights_from_sentences_async(sentences)
                        logger.info(f"[FluffDetection {video_id}] ✅ Found {len(highlights)} highlight candidates")
                        
                        # Aggregate, deduplicate, and rank highlights
                        async with async_session_maker() as db:
                            final_highlights = await aggregate_and_rank_highlights(
                                highlights,
                                max_results=10,
                                db_session=db
                            )
                            logger.info(f"[FluffDetection {video_id}] ✅ Aggregated to {len(final_highlights)} final highlights")
                            
                            # Save highlights to database (batch insert for performance)
                            if final_highlights:
                                logger.info(f"[FluffDetection {video_id}] 💾 Saving {len(final_highlights)} highlights to database...")
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
                                logger.info(f"[FluffDetection {video_id}] ✅ Saved {len(final_highlights)} highlights to database")
                            else:
                                logger.warning(f"[FluffDetection {video_id}] ⚠️  No highlights found to save")
                                
                    except Exception as e:
                        logger.error(f"[FluffDetection {video_id}] ❌ Failed to find highlights: {e}", exc_info=True)
                        logger.warning(f"[FluffDetection {video_id}] ⚠️  Continuing without highlights")
                else:
                    logger.info(f"[FluffDetection {video_id}] ⏭️  Skipping highlights generation (will be done in parallel)")
                
                # Step 4: Visual Change Scoring (skip - no ML models)
                logger.info(f"[FluffDetection {video_id}] 📍 Step 4/7: Skipping visual analysis (no ML models, using default scores)")
                visual_scores = [0.5] * len(sentences)
            finally:
                # Clean up audio file (but keep transcript in database)
                # Only try to clean up if audio_path was created in local processing
                if not use_colab and 'audio_path' in locals():
                    try:
                        import os
                        if os.path.exists(audio_path):
                            os.remove(audio_path)
                    except:
                        pass
        
        # Step 3: Find Highlights (if not already done in local processing path)
        # This handles the case where transcript was provided (has_existing_transcript=True)
        if has_existing_transcript and sentences and not skip_highlights:
            logger.info(f"[FluffDetection {video_id}] 📍 Step 3/7: Finding highlights from sentences...")
            try:
                from utils.highlights import find_highlights_from_sentences_async, aggregate_and_rank_highlights
                
                # Find highlights using sentences (sentence-by-sentence like fluff detection)
                highlights = await find_highlights_from_sentences_async(sentences)
                logger.info(f"[FluffDetection {video_id}] ✅ Found {len(highlights)} highlight candidates")
                
                # Aggregate, deduplicate, and rank highlights
                async with async_session_maker() as db:
                    final_highlights = await aggregate_and_rank_highlights(
                        highlights,
                        max_results=10,
                        db_session=db
                    )
                    logger.info(f"[FluffDetection {video_id}] ✅ Aggregated to {len(final_highlights)} final highlights")
                    
                    # Save highlights to database (batch insert for performance)
                    if final_highlights:
                        logger.info(f"[FluffDetection {video_id}] 💾 Saving {len(final_highlights)} highlights to database...")
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
                        logger.info(f"[FluffDetection {video_id}] ✅ Saved {len(final_highlights)} highlights to database")
                    else:
                        logger.warning(f"[FluffDetection {video_id}] ⚠️  No highlights found to save")
                        
            except Exception as e:
                logger.error(f"[FluffDetection {video_id}] ❌ Failed to find highlights: {e}", exc_info=True)
                logger.warning(f"[FluffDetection {video_id}] ⚠️  Continuing without highlights")
        elif has_existing_transcript and skip_highlights:
            logger.info(f"[FluffDetection {video_id}] ⏭️  Skipping highlights generation (will be done in parallel)")
        
        # Step 4.5: Silence Removal - Cut silences > 1.3s and add timeline markers (start in parallel with filler/embeddings)
        logger.info(f"[FluffDetection {video_id}] 📍 Step 4.5/7: Starting silence detection in parallel with filler/embedding computation...")
        silence_cut_markers = []
        time_adjustments = []  # Track cumulative time removed for timestamp adjustment
        
        async def detect_silences_async():
            """Async function to detect silences."""
            try:
                # Detect silences in audio
                if 'audio_path' in locals() and audio_path:
                    return detect_silence_segments(audio_path)
                else:
                    # Extract audio if not already available
                    if not has_existing_transcript:
                        audio_path_local = await extract_audio_async(video_path)
                    else:
                        # If transcript provided, we need to extract audio for silence detection
                        audio_path_local = await extract_audio_async(video_path)
                    return detect_silence_segments(audio_path_local)
            except Exception as e:
                logger.warning(f"[FluffDetection {video_id}] ⚠️  Silence detection failed: {e}, continuing without cuts", exc_info=True)
                return []
        
        # Start silence detection in parallel with filler/embedding computation
        silence_task = detect_silences_async()
        
        # Steps 4.5, 5 & 5.5: Silence Detection, Filler Detection and Embedding Computation (all parallelized)
        logger.info(f"[FluffDetection {video_id}] 📍 Steps 4.5, 5 & 5.5/7: Detecting silences, filler words and computing embeddings in parallel...")
        
        async def compute_embeddings_async():
            """Async function to compute embeddings."""
            try:
                embedding_model = get_embedding_model()
                sentence_texts = [sentence.text for sentence in sentences]
                
                # Estimate time: ~0.1-0.2 seconds per sentence on CPU
                estimated_time = len(sentence_texts) * 0.15 / 60  # minutes
                logger.info(f"[FluffDetection {video_id}] 🔄 Encoding {len(sentence_texts)} sentences (estimated: {estimated_time:.1f} minutes on CPU)...")
                
                # Process in smaller chunks to avoid memory issues, but process multiple chunks in parallel
                chunk_size = 200  # Process 200 sentences at a time
                total_chunks = (len(sentence_texts) + chunk_size - 1) // chunk_size
                
                import time
                start_time = time.time()
                
                # Run encoding in executor to avoid blocking
                loop = asyncio.get_event_loop()
                
                def encode_chunk(chunk_texts):
                    """Helper function to encode a chunk of texts."""
                    return embedding_model.encode(chunk_texts, show_progress_bar=False, batch_size=64)
                
                # Process chunks in parallel batches (3-5 chunks at a time to balance memory and speed)
                concurrent_chunks = min(4, total_chunks)  # Process up to 4 chunks concurrently
                embeddings_list = []
                
                async def process_chunk_async(chunk_idx: int, chunk_texts: List[str]) -> List[List[float]]:
                    """Process a single chunk asynchronously."""
                    try:
                        chunk_embeddings = await loop.run_in_executor(None, encode_chunk, chunk_texts)
                        logger.info(f"[FluffDetection {video_id}] ✅ Chunk {chunk_idx + 1}/{total_chunks} complete ({len(chunk_embeddings)} embeddings)")
                        return [emb.tolist() for emb in chunk_embeddings]
                    except Exception as chunk_error:
                        logger.error(f"[FluffDetection {video_id}] ❌ Error processing chunk {chunk_idx + 1}: {chunk_error}", exc_info=True)
                        # Use zero embeddings for this chunk as fallback
                        embedding_dim = 1024
                        return [[0.0] * embedding_dim] * len(chunk_texts)
                
                # Process chunks in parallel batches
                for batch_start in range(0, total_chunks, concurrent_chunks):
                    batch_end = min(batch_start + concurrent_chunks, total_chunks)
                    batch_chunks = []
                    
                    for chunk_idx in range(batch_start, batch_end):
                        start_idx = chunk_idx * chunk_size
                        end_idx = min(start_idx + chunk_size, len(sentence_texts))
                        chunk_texts = sentence_texts[start_idx:end_idx]
                        batch_chunks.append((chunk_idx, chunk_texts))
                    
                    logger.info(f"[FluffDetection {video_id}] 🔄 Processing chunks {batch_start + 1}-{batch_end}/{total_chunks} in parallel...")
                    
                    # Process this batch of chunks concurrently
                    chunk_tasks = [process_chunk_async(chunk_idx, chunk_texts) for chunk_idx, chunk_texts in batch_chunks]
                    batch_results = await asyncio.gather(*chunk_tasks)
                    
                    # Collect results in order
                    for result in batch_results:
                        embeddings_list.extend(result)
                    
                    elapsed_total = time.time() - start_time
                    remaining_chunks = total_chunks - batch_end
                    if remaining_chunks > 0:
                        estimated_remaining = (elapsed_total / batch_end) * remaining_chunks / 60  # minutes
                        logger.info(f"[FluffDetection {video_id}] Progress: {batch_end}/{total_chunks} chunks complete (~{estimated_remaining:.1f} min remaining)")
                
                total_time = time.time() - start_time
                logger.info(f"[FluffDetection {video_id}] ✅ Embeddings computed: {len(embeddings_list)} embeddings total (took {total_time/60:.1f} minutes)")
                return embeddings_list
            except Exception as e:
                logger.error(f"[FluffDetection {video_id}] ❌ Failed to compute embeddings: {e}", exc_info=True)
                logger.warning(f"[FluffDetection {video_id}] ⚠️ Using zero embeddings as fallback (will result in lower quality retention scores)")
                # Fallback: use zero embeddings (will result in low novelty scores)
                embedding_dim = 1024  # BGE-large dimension
                return [[0.0] * embedding_dim] * len(sentences)
        
        # Run silence detection, filler detection and embedding computation in parallel
        loop = asyncio.get_event_loop()
        filler_task = loop.run_in_executor(None, detect_fillers, words, sentences)
        embeddings_task = compute_embeddings_async()
        
        filler_densities, embeddings_list, silence_segments_result = await asyncio.gather(
            filler_task, embeddings_task, silence_task, return_exceptions=True
        )
        
        # Handle silence detection result
        if isinstance(silence_segments_result, Exception):
            logger.warning(f"[FluffDetection {video_id}] ⚠️  Silence detection raised exception: {silence_segments_result}, continuing without cuts")
            silence_segments = []
        else:
            silence_segments = silence_segments_result
        
        logger.info(f"[FluffDetection {video_id}] ✅ Silence detection, filler detection and embedding computation complete (parallel execution)")
        
        # Process silence removal now that we have the results
        try:
            # Filter silences > 1.3s
            SILENCE_THRESHOLD = 1.3
            long_silences = [s for s in silence_segments if s.duration > SILENCE_THRESHOLD]
            
            if long_silences:
                logger.info(f"[FluffDetection {video_id}] ✅ Found {len(long_silences)} silences > {SILENCE_THRESHOLD}s")
                
                # Prepare segments to remove
                segments_to_remove = [(s.start, s.end) for s in long_silences]
                
                # Create markers for timeline (red lines where cuts were made)
                import uuid
                for silence in long_silences:
                    silence_cut_markers.append({
                        "id": str(uuid.uuid4()),
                        "time": silence.start,
                        "label": f"Silence cut ({silence.duration:.1f}s)"
                    })
                
                # Cut silences from video
                async with async_session_maker() as db:
                    new_storage_path = await process_video_cutting(
                        video_id=video_id,
                        segments_to_remove=segments_to_remove,
                        db=db
                    )
                    logger.info(f"[FluffDetection {video_id}] ✅ Video cut complete, new storage path: {new_storage_path}")
                    
                    # Update video_path to point to new cut video (for subsequent operations)
                    video_path = await get_video_path(video_id, db, download_local=False)
                    
                    # Save markers to timeline
                    from db import crud
                    await crud.create_or_update_timeline(
                        db=db,
                        video_id=video_id,
                        markers=silence_cut_markers
                    )
                    logger.info(f"[FluffDetection {video_id}] ✅ Saved {len(silence_cut_markers)} silence cut markers to timeline")
                
                # Adjust timestamps for sentences and words after cutting silences
                # Sort silences by start time
                sorted_silences = sorted(long_silences, key=lambda s: s.start)
                
                def get_cumulative_removed_time(timestamp: float) -> float:
                    """Calculate cumulative time removed before a given timestamp."""
                    total = 0.0
                    for silence in sorted_silences:
                        if silence.end <= timestamp:
                            total += silence.end - silence.start
                        elif silence.start < timestamp:
                            # Partial overlap - count only the part before timestamp
                            total += timestamp - silence.start
                    return total
                
                # Adjust all sentence and word timestamps
                sentences_to_keep = []
                for sentence in sentences:
                    # Check if sentence is entirely within a silence
                    is_in_silence = False
                    for silence in sorted_silences:
                        if sentence.start >= silence.start and sentence.end <= silence.end:
                            is_in_silence = True
                            break
                    
                    if is_in_silence:
                        continue  # Skip sentences entirely within silence
                    
                    # Adjust timestamps by subtracting cumulative removed time
                    original_start = sentence.start
                    original_end = sentence.end
                    
                    sentence.start = original_start - get_cumulative_removed_time(original_start)
                    sentence.end = original_end - get_cumulative_removed_time(original_end)
                    
                    # Adjust word timestamps
                    for word in sentence.words:
                        original_word_start = word.start
                        original_word_end = word.end
                        
                        word.start = original_word_start - get_cumulative_removed_time(original_word_start)
                        word.end = original_word_end - get_cumulative_removed_time(original_word_end)
                    
                    sentences_to_keep.append(sentence)
                
                sentences = sentences_to_keep
                # Rebuild words list from remaining sentences
                words = []
                for sentence in sentences:
                    words.extend(sentence.words)
                
                total_removed = sum(s.end - s.start for s in sorted_silences)
                logger.info(f"[FluffDetection {video_id}] ✅ Adjusted timestamps: removed {total_removed:.2f}s total, {len(sentences)} sentences remaining")
            else:
                logger.info(f"[FluffDetection {video_id}] ✅ No silences > {SILENCE_THRESHOLD}s found")
        except Exception as e:
            logger.warning(f"[FluffDetection {video_id}] ⚠️  Silence removal processing failed: {e}, continuing without cuts", exc_info=True)
        
        # Step 6: Retention Scoring (replaces old scoring system)
        logger.info(f"[FluffDetection {video_id}] 📍 Step 6/7: Computing retention scores for {len(sentences)} sentences...")
        retention_analyses = await compute_retention_scores_async(sentences, embeddings_list, filler_densities, video_id=str(video_id))
        logger.info(f"[FluffDetection {video_id}] ✅ Retention scoring complete: {len(retention_analyses)} segments")
        
        # Convert retention analyses to SegmentAnalysis format (only FLUFF segments)
        analyses = []
        for i, retention_analysis in enumerate(retention_analyses):
            # Only create segments for CUT decisions (FLUFF)
            if retention_analysis.decision.action != "CUT":
                continue
            
            # Use retention_value as rating
            rating = retention_analysis.retention_value
            
            # Assign grade based on retention value
            if rating >= 0.9:
                grade = "A"
            elif rating >= 0.8:
                grade = "B"
            elif rating >= 0.5:
                grade = "C"
            elif rating >= 0.2:
                grade = "D"
            else:
                grade = "F"
            
            # Create SegmentAnalysis from RetentionAnalysis (only FLUFF)
            analysis = SegmentAnalysis(
                start_time=retention_analysis.time_range.start,
                end_time=retention_analysis.time_range.end,
                label="FLUFF",
                rating=rating,
                grade=grade,
                reason=retention_analysis.decision.reason,
                repetition_score=0.0,  # Not used in retention scoring
                filler_density=filler_densities[i] if i < len(filler_densities) else 0.0,
                visual_change_score=0.5,  # Not used in retention scoring
                usefulness_score=rating,  # Use retention value as usefulness score
            )
            analysis._segment_id = retention_analysis.segment_id
            analysis._segment_text = retention_analysis.text
            analysis._retention_analysis = retention_analysis  # Store for database saving
            analyses.append(analysis)
        
        # Attach sentence index and text to analyses for database storage
        for i, analysis in enumerate(analyses):
            if i < len(sentences):
                sentence = sentences[i]
                analysis._segment_id = i + 1  # Use sentence index as segment_id
                analysis._segment_text = sentence.text
        
        
        # Step 6.5: Moment Scoring (compute top moments using window-based scoring) - OPTIONAL
        # Skip by default to save processing time (moment generation is disabled)
        enable_moment_scoring = os.getenv("ENABLE_MOMENT_SCORING", "false").lower() == "true"
        if enable_moment_scoring:
            logger.info(f"[FluffDetection {video_id}] 📍 Step 6.5/7: Computing moment scores...")
            try:
                from utils.moment_scoring import compute_moment_scores
                from models import TranscriptSegment
                
                # Create transcript segments from sentences if not already available
                if 'transcript_segments' not in locals() or not transcript_segments:
                    transcript_segments = [
                        TranscriptSegment(
                            start=sentence.start,
                            end=sentence.end,
                            text=sentence.text
                        )
                        for sentence in sentences
                    ]
                
                # Create empty repetition scores (repetition detection removed in step 5)
                repetition_scores = [0.0] * len(sentences)
                
                moment_result = compute_moment_scores(
                    video_id=video_id,
                    transcript_segments=transcript_segments,
                    sentences=sentences,
                    embeddings=embeddings_list,
                    filler_densities=filler_densities,
                    repetition_scores=repetition_scores,
                    silence_segments=[],  # Empty list since silences are already cut
                    top_k=10,
                )
                
                logger.info(f"[FluffDetection {video_id}] ✅ Moment scoring complete: {len(moment_result.top_moments)} top moments identified")
                # Log top moments
                for i, moment in enumerate(moment_result.top_moments[:5]):  # Log top 5
                    logger.info(
                        f"[FluffDetection {video_id}]   Top {i+1}: [{moment.window.start:.1f}s-{moment.window.end:.1f}s] "
                        f"score={moment.score:.3f} - {moment.justification}"
                    )
            except Exception as e:
                logger.warning(f"[FluffDetection {video_id}] ⚠️ Failed to compute moment scores: {e}", exc_info=True)
                # Continue without moment scores
        else:
            logger.info(f"[FluffDetection {video_id}] ⏭️  Skipping moment scoring (disabled by default, set ENABLE_MOMENT_SCORING=true to enable)")
        
        # Filter out FLUFF segments shorter than minimum duration
        logger.info(f"[FluffDetection {video_id}] 📍 Filtering FLUFF segments (minimum duration: {MIN_FLUFF_DURATION}s)...")
        analyses = _filter_short_fluff_segments(analyses)
        logger.info(f"[FluffDetection {video_id}] ✅ FLUFF filtering complete: {len(analyses)} segments remaining")
        
        # Protect hook (first 45 seconds) from FLUFF labeling
        logger.info(f"[FluffDetection {video_id}] 📍 Applying hook protection (first {HOOK_PROTECTION_WINDOW}s)...")
        analyses = _filter_hook_protection(analyses)
        logger.info(f"[FluffDetection {video_id}] ✅ Hook protection complete: {len(analyses)} segments remaining")
        
        # Keep only top 20 most important fluff segments (lowest rating = most fluff)
        MAX_FLUFF_SEGMENTS = 20
        if len(analyses) > MAX_FLUFF_SEGMENTS:
            logger.info(f"[FluffDetection {video_id}] 📍 Selecting top {MAX_FLUFF_SEGMENTS} most important fluff segments from {len(analyses)} candidates...")
            # Sort by rating ascending (lowest rating = most important fluff to remove)
            analyses.sort(key=lambda a: a.rating)
            analyses = analyses[:MAX_FLUFF_SEGMENTS]
            logger.info(f"[FluffDetection {video_id}] ✅ Selected top {len(analyses)} fluff segments")
        
        # Sort analyses by start_time for chronological order
        analyses.sort(key=lambda a: a.start_time)
        
        # Step 7: LLM Final Judge (optional, disabled by default for performance)
        enable_llm_judge = os.getenv("ENABLE_LLM_FINAL_JUDGE", "false").lower() == "true"
        if enable_llm_judge:
            logger.info(f"[FluffDetection {video_id}] 📍 Step 7/7: LLM final judge reviewing fluff detection quality...")
            try:
                from utils.llm_fallback import judge_fluff_detection_quality
                from models import TranscriptSegment
                
                # Create transcript segments from sentences if not already available
                if 'transcript_segments' not in locals() or not transcript_segments:
                    if sentences:
                        transcript_segments = [
                            TranscriptSegment(
                                start=sentence.start,
                                end=sentence.end,
                                text=sentence.text
                            )
                            for sentence in sentences
                        ]
                    else:
                        # Fallback: create from analyses
                        transcript_segments = [
                            TranscriptSegment(
                                start=analysis.start_time,
                                end=analysis.end_time,
                                text=getattr(analysis, '_segment_text', '')
                            )
                            for analysis in analyses
                            if hasattr(analysis, '_segment_text') and analysis._segment_text
                        ]
                
                # Get LLM judgment
                llm_judgment = await judge_fluff_detection_quality(transcript_segments, analyses)
                
                if llm_judgment.get("approved", True) and llm_judgment.get("confidence", 0) > 0.8:
                    logger.info(
                        f"[FluffDetection {video_id}] ✅ LLM final judge approved fluff detection "
                        f"(confidence: {llm_judgment.get('confidence', 0):.2f}): {llm_judgment.get('reason', '')}"
                    )
                else:
                    logger.info(
                        f"[FluffDetection {video_id}] ⚠️ LLM final judge not confident "
                        f"(confidence: {llm_judgment.get('confidence', 0):.2f}), using LLM-identified fluff segments"
                    )
                    
                    # If not approved, use LLM's top fluff segments as final decision
                    top_fluff_segments = llm_judgment.get("top_fluff_segments", [])
                    
                    if top_fluff_segments:
                        # Create a list of LLM-identified fluff time ranges
                        llm_fluff_ranges = []
                        for seg in top_fluff_segments:
                            start = float(seg.get("start_time", 0))
                            end = float(seg.get("end_time", 0))
                            if end > start:  # Valid time range
                                llm_fluff_ranges.append({
                                    "start": start,
                                    "end": end,
                                    "data": seg
                                })
                        
                        # Update analyses: mark segments that overlap with LLM fluff as FLUFF
                        updated_count = 0
                        tolerance = 3.0  # 3 second tolerance for matching
                        
                        for analysis in analyses:
                            # Check if this segment overlaps with any LLM-identified fluff
                            for llm_range in llm_fluff_ranges:
                                llm_start = llm_range["start"]
                                llm_end = llm_range["end"]
                                llm_seg = llm_range["data"]
                                
                                # Check for overlap (segments overlap if they share any time)
                                # Also check if start/end times are close (within tolerance)
                                overlaps = (
                                    (analysis.start_time <= llm_end and analysis.end_time >= llm_start) or
                                    abs(analysis.start_time - llm_start) < tolerance or
                                    abs(analysis.end_time - llm_end) < tolerance or
                                    abs(analysis.start_time - llm_end) < tolerance or
                                    abs(analysis.end_time - llm_start) < tolerance
                                )
                                
                                if overlaps:
                                    # This segment matches LLM fluff
                                    if analysis.label != "FLUFF":
                                        analysis.label = "FLUFF"
                                        analysis.rating = float(llm_seg.get("fluff_score", 0.3))
                                        analysis.reason = f"LLM final judge: {llm_seg.get('reason', 'Identified as fluff')}"
                                        # Update grade based on rating
                                        if analysis.rating >= 0.9:
                                            analysis.grade = "A"
                                        elif analysis.rating >= 0.8:
                                            analysis.grade = "B"
                                        elif analysis.rating >= 0.5:
                                            analysis.grade = "C"
                                        elif analysis.rating >= 0.2:
                                            analysis.grade = "D"
                                        else:
                                            analysis.grade = "F"
                                        updated_count += 1
                                    break
                        
                        logger.info(
                            f"[FluffDetection {video_id}] ✅ Updated {updated_count} segments based on LLM final judge "
                            f"(identified {len(top_fluff_segments)} top fluff segments)"
                        )
                    else:
                        logger.warning(
                            f"[FluffDetection {video_id}] ⚠️ LLM final judge not confident but no top fluff segments provided, "
                            f"keeping original fluff detection"
                        )
            except Exception as e:
                logger.warning(
                    f"[FluffDetection {video_id}] ⚠️ LLM final judge failed: {e}, "
                    f"continuing with original fluff detection",
                    exc_info=True
                )
                # Continue with original analyses if LLM judge fails
        else:
            logger.info(f"[FluffDetection {video_id}] ⏭️  Skipping LLM final judge (disabled by default, set ENABLE_LLM_FINAL_JUDGE=true to enable)")
        
        # Update segment IDs
        for i, analysis in enumerate(analyses):
            analysis._segment_id = i + 1
        
        # Log summary statistics
        fluff_count = sum(1 for a in analyses if a.label == "FLUFF")
        
        processing_mode = "Colab GPU" if use_colab else "Local"
        logger.info(
            f"[FluffDetection] Pipeline completed ({processing_mode}) for video {video_id}: "
            f"{len(analyses)} FLUFF segments detected"
        )
        return analyses
        
    except Exception as e:
        logger.error(f"Error in fluff detection pipeline for video {video_id}: {str(e)}", exc_info=True)
        raise
