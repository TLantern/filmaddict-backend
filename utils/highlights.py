import asyncio
import json
import logging
import os
import re
import statistics
from typing import List, Optional
from uuid import UUID

from openai import AsyncOpenAI, OpenAI
from models import Highlight, TranscriptChunk, TranscriptSegment
from utils.learning import apply_calibration

logger = logging.getLogger(__name__)

_openai_client: OpenAI = None
_async_openai_client: AsyncOpenAI = None

MAX_CONCURRENT_CHUNKS = 5  # Parallel API calls limit


def get_openai_client() -> OpenAI:
    """Get or create OpenAI client instance."""
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def get_async_openai_client() -> AsyncOpenAI:
    """Get or create async OpenAI client instance."""
    global _async_openai_client
    if _async_openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        _async_openai_client = AsyncOpenAI(api_key=api_key)
    return _async_openai_client

MIN_CHUNK_SECONDS = 30.0


def _ends_with_complete_sentence(text: str) -> bool:
    """
    Check if text ends with a complete sentence (sentence ending followed by space or end of string).
    
    Args:
        text: Text to check
        
    Returns:
        True if text ends with a complete sentence
    """
    text = text.strip()
    if not text:
        return False
    
    pattern = r"[.!?](?:\s+|$)"
    return bool(re.search(pattern, text))


def chunk_transcript(segments: List[TranscriptSegment], max_window_seconds: float = 90.0) -> List[TranscriptChunk]:
    """
    Group transcript segments into chunks of approximately 30-90 seconds while respecting sentence boundaries.
    
    Args:
        segments: List of transcript segments to chunk
        max_window_seconds: Maximum duration for a chunk in seconds (default: 60)
        
    Returns:
        List of TranscriptChunk objects with start, end, text, and segments
        
    Raises:
        ValueError: If max_window_seconds is less than MIN_CHUNK_SECONDS
    """
    if max_window_seconds < MIN_CHUNK_SECONDS:
        raise ValueError(f"max_window_seconds must be at least {MIN_CHUNK_SECONDS}")
    
    if not segments:
        return []
    
    chunks = []
    i = 0
    
    while i < len(segments):
        current_segments = [segments[i]]
        current_start = segments[i].start
        chunk_created = False
        i += 1
        
        while i < len(segments):
            segment = segments[i]
            potential_duration = segment.end - current_start
            
            if potential_duration >= max_window_seconds:
                duration_before = current_segments[-1].end - current_start
                
                if duration_before >= MIN_CHUNK_SECONDS:
                    combined_text = " ".join(s.text for s in current_segments)
                    
                    if _ends_with_complete_sentence(combined_text):
                        chunks.append(TranscriptChunk(
                            start=current_start,
                            end=current_segments[-1].end,
                            text=combined_text,
                            segments=current_segments.copy()
                        ))
                        chunk_created = True
                        break
                
                current_segments.append(segment)
                combined_text = " ".join(s.text for s in current_segments)
                
                if _ends_with_complete_sentence(combined_text):
                    chunks.append(TranscriptChunk(
                        start=current_start,
                        end=segment.end,
                        text=combined_text,
                        segments=current_segments.copy()
                    ))
                    chunk_created = True
                    i += 1
                    break
            
            current_segments.append(segment)
            i += 1
        
        if not chunk_created and current_segments:
            combined_text = " ".join(s.text for s in current_segments)
            chunks.append(TranscriptChunk(
                start=current_start,
                end=current_segments[-1].end,
                text=combined_text,
                segments=current_segments.copy()
            ))
    
    return chunks


async def _analyze_chunk(
    client: AsyncOpenAI,
    chunk: TranscriptChunk,
    system_prompt: str,
    user_template: str,
    semaphore: asyncio.Semaphore,
) -> List[Highlight]:
    """Analyze a single chunk for highlights (runs in parallel)."""
    async with semaphore:
        try:
            user_prompt = user_template.format(
                start=chunk.start,
                end=chunk.end,
                text=chunk.text
            )

            logger.info(f"Analyzing chunk {chunk.start:.2f}-{chunk.end:.2f}s for highlights")
            
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content.strip()
            highlights = []
            
            try:
                parsed = json.loads(content)
                highlights_data = parsed.get("highlights", parsed.get("results", [])) if isinstance(parsed, dict) else parsed
                
                if not isinstance(highlights_data, list):
                    highlights_data = []
                
                for h in highlights_data:
                    start = float(h.get("start", 0))
                    end = float(h.get("end", 0))
                    title = str(h.get("title", "")) or None
                    summary = str(h.get("summary", "")) or None
                    score = float(h.get("score", 0))
                    
                    start = max(start, chunk.start)
                    end = min(end, chunk.end)
                    if start >= end:
                        continue
                    
                    # Validate duration: must be 20-30s or 50-60s
                    duration = end - start
                    is_short_clip = 20 <= duration <= 30
                    is_long_clip = 50 <= duration <= 60
                    if not (is_short_clip or is_long_clip):
                        logger.debug(f"Skipping highlight with invalid duration {duration:.1f}s (must be 20-30s or 50-60s)")
                        continue
                    
                    score = max(1, min(10, score))
                    
                    # Only include highlights scoring 6.0 or higher
                    if score >= 6.0 and summary:
                        highlights.append(Highlight(start=start, end=end, title=title, summary=summary, score=score))
                
                logger.info(f"Found {len(highlights)} highlights in chunk {chunk.start:.2f}-{chunk.end:.2f}s")
                return highlights
                
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning(f"Failed to parse GPT response for chunk {chunk.start:.2f}-{chunk.end:.2f}s: {str(e)}")
                return []
                
        except Exception as e:
            logger.error(f"Error analyzing chunk {chunk.start:.2f}-{chunk.end:.2f}s: {str(e)}", exc_info=True)
            return []


async def find_highlights_async(
    chunks: List[TranscriptChunk],
    system_prompt: Optional[str] = None,
    user_prompt_template: Optional[str] = None,
) -> List[Highlight]:
    """
    Identify highlights in transcript chunks using GPT (PARALLEL processing).
    
    Processes all chunks concurrently for 3-5x faster results.
    """
    if not chunks:
        return []
    
    client = get_async_openai_client()
    
    default_system_prompt = (
        "You are an expert storytelling editor for short-form video. "
        "You identify moments that feel like mini-stories with a clear beginning, middle, and end. "
        "Start clips on powerful hooks—first-person statements ('I', 'We') or emotionally charged language. "
        "Favor clips that include enough setup and aftermath for viewers to understand the context "
        "even if they have not seen the rest of the video. "
        "Return only valid JSON objects with a 'highlights' key containing an array."
    )
    default_user_template = """Chunk time range: {start:.2f} - {end:.2f} seconds
Transcript text: {text}

Your job is to find moments that work as self-contained stories.

**CRITICAL DURATION REQUIREMENT:**
Clips MUST be either 20-30 seconds OR 50-60 seconds long. Do NOT return clips outside these ranges.
- Short clips: 20-30 seconds (quick impactful moments)
- Long clips: 50-60 seconds (deeper stories with full context)

Identify the most engaging, emotionally intense, or information-dense moments that would perform well as short-form content, **while preserving narrative context**. For each candidate:
- **START clips on hook moments**: Begin when the speaker says "I", "We", "You", or emotionally charged words (e.g., "never", "always", "worst", "best", "crazy", "unbelievable", "shocking").
- Prioritize starting near emotionally evoking language that grabs attention immediately.
- Avoid starting or ending in the middle of a sentence unless absolutely necessary.
- Prefer ranges where the speaker completes a thought or point (a mini beginning → middle → end).
- Extend or trim the range to fit within the required duration brackets (20-30s or 50-60s).

**IMPORTANT: Scoring Guidelines**
Use the full 1-10 range and be highly selective. Most moments should score 5-7. Only truly exceptional moments score 8+. Reserve 9-10 for the absolute best.

Score rubric:
- 9-10: Exceptional, viral-worthy moments with perfect narrative arc, high emotional impact, and universal appeal
- 7-8: Strong, engaging moments with clear story structure and good emotional resonance
- 5-6: Decent moments that work but lack exceptional qualities
- 3-4: Weak candidates that barely qualify as highlights
- 1-2: Poor quality, not suitable as highlights

**Only return highlights scoring 6.0 or higher.** Return 0–1 timestamp range per chunk (be highly selective - only the best moment).

Return your response as a JSON object with a "highlights" key containing an array. Each highlight must have: start (seconds), end (seconds), title (short catchy title), summary (2-3 sentences describing what happens), score (1-10).
Example format:
{{
  "highlights": [
    {{
      "start": 120.0,
      "end": 145.0,
      "title": "The Moment Everything Changed",
      "summary": "The speaker reveals a pivotal moment from their past. They explain how this experience fundamentally changed their perspective and set them on their current path.",
      "score": 8.5
    }}
  ]
}}"""
    
    system_prompt_to_use = system_prompt or default_system_prompt
    user_template_to_use = user_prompt_template or default_user_template
    
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_CHUNKS)
    
    tasks = [
        _analyze_chunk(client, chunk, system_prompt_to_use, user_template_to_use, semaphore)
        for chunk in chunks
    ]
    
    results = await asyncio.gather(*tasks)
    all_highlights = [h for chunk_highlights in results for h in chunk_highlights]
    
    logger.info(f"Total highlights found: {len(all_highlights)}")
    return all_highlights


def find_highlights(
    chunks: List[TranscriptChunk],
    system_prompt: Optional[str] = None,
    user_prompt_template: Optional[str] = None,
) -> List[Highlight]:
    """Sync wrapper for find_highlights_async."""
    return asyncio.get_event_loop().run_until_complete(
        find_highlights_async(chunks, system_prompt, user_prompt_template)
    )


def redistribute_scores(highlights: List[Highlight]) -> List[Highlight]:
    """
    Redistribute scores if they are too clustered to ensure better differentiation.
    
    If scores have low standard deviation (< 1.0), redistributes them using
    percentile-based mapping to spread across the 6-10 range more evenly.
    
    Args:
        highlights: List of Highlight objects with scores
        
    Returns:
        List of Highlight objects with redistributed scores
    """
    if len(highlights) < 2:
        return highlights
    
    scores = [h.score for h in highlights]
    
    # Calculate standard deviation
    try:
        std_dev = statistics.stdev(scores) if len(scores) > 1 else 0.0
    except statistics.StatisticsError:
        std_dev = 0.0
    
    # Only redistribute if scores are clustered (low std dev)
    if std_dev >= 1.0:
        return highlights
    
    # Redistribute using percentile-based mapping
    # Map percentiles to scores in 6.0-10.0 range
    sorted_highlights = sorted(highlights, key=lambda h: h.score, reverse=True)
    redistributed = []
    
    for i, highlight in enumerate(sorted_highlights):
        # Calculate percentile (0.0 = top, 1.0 = bottom)
        percentile = i / (len(sorted_highlights) - 1) if len(sorted_highlights) > 1 else 0.5
        
        # Map percentile to score range 6.0-10.0 (inverted: low percentile = high score)
        # Top highlights get 9-10, middle get 7-8, lower get 6-7
        if percentile <= 0.1:
            # Top 10%: 9.0-10.0 (percentile 0.0 -> 10.0, percentile 0.1 -> 9.0)
            new_score = 10.0 - (percentile / 0.1) * 1.0
        elif percentile <= 0.3:
            # Next 20%: 8.0-9.0 (percentile 0.1 -> 9.0, percentile 0.3 -> 8.0)
            new_score = 9.0 - ((percentile - 0.1) / 0.2) * 1.0
        elif percentile <= 0.6:
            # Next 30%: 7.0-8.0 (percentile 0.3 -> 8.0, percentile 0.6 -> 7.0)
            new_score = 8.0 - ((percentile - 0.3) / 0.3) * 1.0
        else:
            # Bottom 40%: 6.0-7.0 (percentile 0.6 -> 7.0, percentile 1.0 -> 6.0)
            new_score = 7.0 - ((percentile - 0.6) / 0.4) * 1.0
        
        new_score = max(6.0, min(10.0, new_score))
        
        redistributed.append(Highlight(
            start=highlight.start,
            end=highlight.end,
            title=highlight.title,
            summary=highlight.summary,
            score=new_score
        ))
    
    logger.info(f"Redistributed scores (std dev was {std_dev:.2f}, now spread across range)")
    return redistributed


async def aggregate_and_rank_highlights(
    highlights: List[Highlight],
    max_results: int = 10,
    db_session=None,
) -> List[Highlight]:
    """
    Aggregate, deduplicate, and rank highlights.
    
    Merges all highlight candidates from all chunks, deduplicates overlapping ranges
    (keeps highest score), sorts by score descending, and truncates to top results.
    Applies calibration offset to scores if calibration is enabled.
    
    Args:
        highlights: List of Highlight objects to process
        max_results: Maximum number of highlights to return (default: 10, range: 5-10)
        db_session: Optional database session for calibration lookup
        
    Returns:
        List of deduplicated and ranked Highlight objects
    """
    if not highlights:
        return []
    
    # Ensure max_results is within reasonable range
    max_results = max(5, min(10, max_results))
    
    # Apply calibration if enabled and db_session provided
    calibration_offset = 0.0
    if db_session:
        try:
            from db import crud
            config = await crud.get_calibration_config(db_session)
            if config:
                calibration_offset = config.score_offset
        except Exception as e:
            logger.warning(f"Failed to get calibration config: {str(e)}")
    
    # Apply calibration and create new highlight objects with calibrated scores
    calibrated_highlights = []
    for highlight in highlights:
        calibrated_score = apply_calibration(highlight.score, calibration_offset)
        calibrated_highlights.append(
            Highlight(
                start=highlight.start,
                end=highlight.end,
                title=highlight.title,
                summary=highlight.summary,
                score=calibrated_score
            )
        )
    
    # Redistribute scores if they are too clustered
    redistributed_highlights = redistribute_scores(calibrated_highlights)
    
    # Sort by score descending first (process highest scores first)
    sorted_highlights = sorted(redistributed_highlights, key=lambda h: h.score, reverse=True)
    
    # Deduplicate overlapping ranges - keep highest score
    deduplicated = []
    for highlight in sorted_highlights:
        overlap_found = False
        
        # Check against all existing highlights
        for i, existing in enumerate(deduplicated):
            # Check if ranges overlap (inclusive boundaries)
            if not (highlight.end < existing.start or highlight.start > existing.end):
                # Overlap detected
                if highlight.score > existing.score:
                    # Replace existing with new highlight (higher score)
                    deduplicated[i] = highlight
                # If scores are equal or existing is higher, keep existing
                overlap_found = True
                break
        
        if not overlap_found:
            deduplicated.append(highlight)
    
    # Re-sort after deduplication to ensure correct order
    deduplicated.sort(key=lambda h: h.score, reverse=True)
    
    # Truncate to top results
    result = deduplicated[:max_results]
    
    logger.info(f"Aggregated {len(highlights)} highlights to {len(result)} final highlights")
    return result

