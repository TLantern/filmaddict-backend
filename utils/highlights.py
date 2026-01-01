import asyncio
import json
import logging
import os
import re
from typing import List, Optional
from uuid import UUID

from models import Highlight, TranscriptChunk, TranscriptSegment, Sentence
from utils.learning import apply_calibration
from utils.llm_fallback import get_async_openai_client

logger = logging.getLogger(__name__)

MIN_CHUNK_SECONDS = 5.0  # Minimum highlight duration
MAX_CHUNK_SECONDS = 45.0  # Maximum highlight duration


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


def chunk_transcript(segments: List[TranscriptSegment], max_window_seconds: float = 45.0) -> List[TranscriptChunk]:
    """
    Group transcript segments into chunks of 5-45 seconds while respecting sentence boundaries.
    
    Args:
        segments: List of transcript segments to chunk
        max_window_seconds: Maximum duration for a chunk in seconds (default: 45)
        
    Returns:
        List of TranscriptChunk objects with start, end, text, and segments
        
    Raises:
        ValueError: If max_window_seconds is less than MIN_CHUNK_SECONDS
    """
    if max_window_seconds < MIN_CHUNK_SECONDS:
        raise ValueError(f"max_window_seconds must be at least {MIN_CHUNK_SECONDS}")
    
    if not segments:
        return []
    
    # Use MAX_CHUNK_SECONDS as the actual max
    max_window_seconds = min(max_window_seconds, MAX_CHUNK_SECONDS)
    
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
            
            # Create chunk when we reach the target/max duration
            if potential_duration >= max_window_seconds:
                duration_before = current_segments[-1].end - current_start
                
                # Check if we have minimum duration
                if duration_before >= MIN_CHUNK_SECONDS:
                    combined_text = " ".join(s.text for s in current_segments)
                    
                    # Prefer chunking at sentence boundaries, but force split if needed
                    if _ends_with_complete_sentence(combined_text) or duration_before >= MAX_CHUNK_SECONDS:
                        chunks.append(TranscriptChunk(
                            start=current_start,
                            end=current_segments[-1].end,
                            text=combined_text,
                            segments=current_segments.copy()
                        ))
                        chunk_created = True
                        break
                
                # Add current segment and check again
                current_segments.append(segment)
                combined_text = " ".join(s.text for s in current_segments)
                
                if _ends_with_complete_sentence(combined_text) or (segment.end - current_start) >= MAX_CHUNK_SECONDS:
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
            chunk_duration = current_segments[-1].end - current_start
            
            # Force split if chunk is too long
            if chunk_duration >= MAX_CHUNK_SECONDS:
                # Find the segment that crosses MAX_CHUNK_SECONDS
                split_idx = 0
                for idx, seg in enumerate(current_segments):
                    if seg.end - current_start >= MAX_CHUNK_SECONDS:
                        split_idx = max(1, idx)  # Ensure at least one segment
                        break
                
                # Create chunk up to split point (must meet minimum)
                split_segments = current_segments[:split_idx]
                split_duration = split_segments[-1].end - current_start
                if split_duration >= MIN_CHUNK_SECONDS:
                    combined_text = " ".join(s.text for s in split_segments)
                    chunks.append(TranscriptChunk(
                        start=current_start,
                        end=split_segments[-1].end,
                        text=combined_text,
                        segments=split_segments.copy()
                    ))
                    
                    # Continue with remaining segments
                    current_segments = current_segments[split_idx:]
                    if current_segments:
                        current_start = current_segments[0].start
                        # Don't increment i, continue processing remaining segments
                        continue
            
            # Final chunk creation - must meet minimum duration
            if current_segments:
                final_duration = current_segments[-1].end - current_start
                if final_duration >= MIN_CHUNK_SECONDS:
                    combined_text = " ".join(s.text for s in current_segments)
                    chunks.append(TranscriptChunk(
                        start=current_start,
                        end=current_segments[-1].end,
                        text=combined_text,
                        segments=current_segments.copy()
                    ))
    
    return chunks


def overlap_percentage(start1: float, end1: float, start2: float, end2: float) -> float:
    """Calculate overlap percentage between two time ranges."""
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    if overlap_end <= overlap_start:
        return 0.0
    overlap_duration = overlap_end - overlap_start
    range1_duration = end1 - start1
    return overlap_duration / range1_duration if range1_duration > 0 else 0.0


def _analyze_chunk_rule_based(chunk: TranscriptChunk) -> List[Highlight]:
    """Analyze a single chunk for highlights using simple rule-based logic."""
    highlights = []
    
    try:
        chunk_duration = chunk.end - chunk.start
        text = chunk.text.lower()
        
        # Simple scoring based on text features
        score = 6.0  # Base score
        
        # Boost score for emotional words
        emotional_words = ['never', 'always', 'worst', 'best', 'crazy', 'unbelievable', 
                          'shocking', 'amazing', 'incredible', 'terrible', 'awesome']
        emotional_count = sum(1 for word in emotional_words if word in text)
        score += min(2.0, emotional_count * 0.3)
        
        # Boost for first-person statements
        if any(text.startswith(prefix) for prefix in ['i ', 'we ', 'you ']):
            score += 0.5
        
        # Boost for question marks (engagement)
        if '?' in chunk.text:
            score += 0.3
        
        # Score based on duration (prefer longer segments)
        if 20 <= chunk_duration <= 35:
            score += 0.5  # Optimal range
        elif 5 <= chunk_duration < 20 or 35 < chunk_duration <= 45:
            score += 0.2  # Acceptable range
        else:
            # Outside 5-45 range, reject
            return []
        
        score = max(6.0, min(10.0, score))
        
        # Create highlight only if within valid duration range
        if score >= 6.0 and MIN_CHUNK_SECONDS <= chunk_duration <= MAX_CHUNK_SECONDS:
            # Use the full chunk as highlight (no duration restrictions)
            start = chunk.start
            end = chunk.end
            title = chunk.text[:50] + "..." if len(chunk.text) > 50 else chunk.text
            highlights.append(Highlight(start=start, end=end, title=title, summary=None, score=score))
        
        logger.info(f"Found {len(highlights)} highlights in chunk {chunk.start:.2f}-{chunk.end:.2f}s")
        return highlights
        
    except Exception as e:
        logger.error(f"Error analyzing chunk {chunk.start:.2f}-{chunk.end:.2f}s: {str(e)}", exc_info=True)
        return []


def _create_sentence_windows(sentences: List[Sentence], min_duration: float = 5.0, max_duration: float = 45.0) -> List[List[Sentence]]:
    """
    Create sentence windows that are 5-45 seconds long.
    
    Groups consecutive sentences into windows, similar to chunk_transcript but working with Sentence objects.
    """
    if not sentences:
        return []
    
    windows = []
    i = 0
    
    while i < len(sentences):
        window = [sentences[i]]
        window_start = sentences[i].start
        window_created = False
        i += 1
        
        # Add sentences until we reach max duration
        while i < len(sentences):
            sentence = sentences[i]
            potential_duration = sentence.end - window_start
            
            if potential_duration > max_duration:
                # Window would be too long, finish current window if it meets minimum
                if (window[-1].end - window_start) >= min_duration:
                    windows.append(window)
                    window_created = True
                break
            
            window.append(sentence)
            
            # If we've reached at least minimum duration and adding next would exceed max, create window now
            if potential_duration >= min_duration:
                if i + 1 >= len(sentences) or sentences[i + 1].end - window_start > max_duration:
                    windows.append(window)
                    window_created = True
                    i += 1
                    break
            
            i += 1
        
        # Handle case where we've processed all sentences but window hasn't been added yet
        if i >= len(sentences) and not window_created:
            window_duration = window[-1].end - window_start
            if window_duration >= min_duration:
                windows.append(window)
                window_created = True
        
        # If window wasn't created and we're still here, skip to next sentence
        if not window_created:
            # Current window is too short and we can't add more sentences
            # Skip to next sentence to avoid infinite loop
            if i >= len(sentences):
                break
    
    return windows


async def find_highlights_from_sentences_async(
    sentences: List[Sentence],
) -> List[Highlight]:
    """
    Identify highlights from sentences using LLM analysis.
    
    Works sentence-by-sentence like fluff detection:
    1. Creates sentence windows (5-45 seconds)
    2. Analyzes each window with LLM for highlight potential
    3. Groups high-scoring windows into final highlights
    
    Args:
        sentences: List of Sentence objects to analyze
        
    Returns:
        List of Highlight objects
    """
    if not sentences:
        return []
    
    try:
        logger.info(f"Creating sentence windows from {len(sentences)} sentences")
        
        # Create sentence windows (5-45 seconds)
        windows = _create_sentence_windows(sentences, MIN_CHUNK_SECONDS, MAX_CHUNK_SECONDS)
        logger.info(f"Created {len(windows)} sentence windows")
        
        if not windows:
            logger.warning("No valid sentence windows created")
            return []
        
        # Convert windows to text format for LLM
        window_data = []
        for window in windows:
            start = window[0].start
            end = window[-1].end
            text = " ".join(s.text for s in window)
            window_data.append({
                'start': start,
                'end': end,
                'text': text,
                'sentences': window
            })
        
        # Process windows in batches to avoid token limits while covering entire video
        # Split windows into batches of ~6000 chars each (leaving room for prompt)
        MAX_BATCH_CHARS = 6000
        all_highlights_data = []
        
        client = get_async_openai_client()
        
        # Split window_data into batches
        batches = []
        current_batch = []
        current_batch_size = 0
        
        for w in window_data:
            window_text = f"[{w['start']:.1f}s-{w['end']:.1f}s] {w['text']}\n"
            window_size = len(window_text)
            
            if current_batch_size + window_size > MAX_BATCH_CHARS and current_batch:
                batches.append(current_batch)
                current_batch = [w]
                current_batch_size = window_size
            else:
                current_batch.append(w)
                current_batch_size += window_size
        
        if current_batch:
            batches.append(current_batch)
        
        logger.info(f"Processing {len(window_data)} windows in {len(batches)} batches to cover entire video")
        
        # Process all batches in parallel
        async def process_batch(batch_idx: int, batch: List[dict]) -> List[dict]:
            """Process a single batch and return highlights data."""
            transcript_text = "\n".join([
                f"[{w['start']:.1f}s-{w['end']:.1f}s] {w['text']}"
                for w in batch
            ])
            
            prompt = f"""You are analyzing a video transcript to identify the best highlights.

Here are video segments with timestamps. Each segment is 5-45 seconds long:

{transcript_text}

Your task: Select the TOP 5-10 most engaging, interesting, or valuable segments as highlights.

Criteria for good highlights:
- Most engaging or entertaining moments
- Key insights or important information
- Emotional or impactful moments
- Memorable quotes or statements
- Moments that would make viewers want to watch the full video

CRITICAL REQUIREMENTS:
- You MUST return between 5 and 10 highlights (preferably 8-10 if there are enough good segments)
- If there are fewer than 5 good segments, return the best ones available (minimum 5 if possible)
- Each highlight must use the EXACT start and end timestamps from the segments above
- Each highlight duration must be 5-45 seconds
- Prioritize the most interesting segments
- Sort highlights by score (highest first)

Respond with JSON only:
{{
  "highlights": [
    {{
      "start": float (start timestamp in seconds, must match one of the segments above),
      "end": float (end timestamp in seconds, must match one of the segments above),
      "title": string (short catchy title, max 50 chars),
      "score": float (7.0-10.0, rating of how good this highlight is)
    }}
  ]
}}

IMPORTANT: Return between 5-10 highlights. Aim for 8-10 if there are enough good segments available."""

            try:
                response = await client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are an expert at identifying the most engaging moments in video content. Always respond with valid JSON only. You must return between 5-10 highlights in your response."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    max_tokens=3000,
                    temperature=0.3,
                )
                
                result = json.loads(response.choices[0].message.content)
                batch_highlights = result.get("highlights", [])
                logger.info(f"Batch {batch_idx + 1}/{len(batches)}: Found {len(batch_highlights)} highlights")
                return batch_highlights
            except Exception as e:
                logger.warning(f"Error processing batch {batch_idx + 1}: {str(e)}, continuing with other batches")
                return []
        
        # Process all batches concurrently
        batch_tasks = [process_batch(batch_idx, batch) for batch_idx, batch in enumerate(batches)]
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        # Collect results, handling exceptions
        for batch_idx, result in enumerate(batch_results):
            if isinstance(result, Exception):
                logger.warning(f"Batch {batch_idx + 1} raised exception: {result}")
                continue
            all_highlights_data.extend(result)
        
        highlights_data = all_highlights_data
        
        if not highlights_data:
            logger.warning("LLM returned no highlights, returning empty list")
            return []
        
        # Validate and create Highlight objects
        highlights = []
        valid_timestamps = {(w['start'], w['end']) for w in window_data}
        
        for h_data in highlights_data[:10]:
            try:
                start = float(h_data.get("start", 0))
                end = float(h_data.get("end", 0))
                duration = end - start
                
                # Validate duration
                if not (MIN_CHUNK_SECONDS <= duration <= MAX_CHUNK_SECONDS):
                    logger.warning(f"Skipping highlight [{start:.1f}s-{end:.1f}s] - duration {duration:.1f}s outside valid range")
                    continue
                
                # Validate timestamps match valid windows (allow small tolerance)
                timestamp_valid = any(
                    abs(start - w_start) < 0.1 and abs(end - w_end) < 0.1
                    for w_start, w_end in valid_timestamps
                )
                
                if not timestamp_valid:
                    logger.warning(f"Skipping highlight [{start:.1f}s-{end:.1f}s] - timestamps don't match valid windows")
                    continue
                
                title = str(h_data.get("title", ""))[:50]
                score = float(h_data.get("score", 7.0))
                score = max(6.0, min(10.0, score))
                
                highlights.append(Highlight(
                    start=start,
                    end=end,
                    title=title if title else None,
                    summary=None,
                    score=score
                ))
            except (ValueError, KeyError, TypeError) as e:
                logger.warning(f"Error parsing highlight data: {e}, skipping")
                continue
        
        # Sort by score descending
        highlights.sort(key=lambda h: h.score, reverse=True)
        
        # If we have fewer than 5 highlights, retry LLM with more lenient prompt
        if len(highlights) < 5:
            logger.warning(f"Only {len(highlights)} highlights found, retrying LLM with more lenient criteria")
            try:
                # Retry with batches using more lenient criteria - process in parallel
                async def process_retry_batch(batch_idx: int, batch: List[dict]) -> List[dict]:
                    """Process a retry batch with more lenient criteria."""
                    transcript_text = "\n".join([
                        f"[{w['start']:.1f}s-{w['end']:.1f}s] {w['text']}"
                        for w in batch
                    ])
                    
                    retry_prompt = f"""You are analyzing a video transcript to identify highlights.

Here are video segments with timestamps. Each segment is 5-45 seconds long:

{transcript_text}

Your task: Select AT LEAST 5-10 most engaging, interesting, or valuable segments as highlights. If you previously selected fewer, select MORE segments now.

CRITICAL: You MUST return at least 5 highlights. Select segments even if they are moderately interesting - we need at least 5 highlights.

Criteria (be more lenient):
- Engaging or entertaining moments
- Key insights or important information
- Emotional or impactful moments
- Memorable quotes or statements
- Moderately interesting segments are acceptable if needed to reach 5 highlights

CRITICAL REQUIREMENTS:
- You MUST return at least 5 highlights (preferably 8-10)
- Use the EXACT start and end timestamps from the segments above
- Each highlight duration must be 5-45 seconds
- Be more lenient - include segments even if they are only moderately interesting
- Sort highlights by score (highest first)

Respond with JSON only:
{{
  "highlights": [
    {{
      "start": float (start timestamp in seconds, must match one of the segments above),
      "end": float (end timestamp in seconds, must match one of the segments above),
      "title": string (short catchy title, max 50 chars),
      "score": float (6.0-10.0, rating of how good this highlight is - can use 6.0-7.0 for moderate highlights)
    }}
  ]
}}

IMPORTANT: Return AT LEAST 5 highlights. Include moderately interesting segments if needed."""

                    try:
                        retry_response = await client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {"role": "system", "content": "You are an expert at identifying engaging moments in video content. Always respond with valid JSON only. You MUST return at least 5 highlights - include moderately interesting segments if needed."},
                                {"role": "user", "content": retry_prompt}
                            ],
                            response_format={"type": "json_object"},
                            max_tokens=3000,
                            temperature=0.5,
                        )
                        
                        retry_result = json.loads(retry_response.choices[0].message.content)
                        batch_retry_highlights = retry_result.get("highlights", [])
                        return batch_retry_highlights
                    except Exception as e:
                        logger.warning(f"Error in retry batch {batch_idx + 1}: {str(e)}, continuing")
                        return []
                
                # Process all retry batches concurrently
                retry_tasks = [process_retry_batch(batch_idx, batch) for batch_idx, batch in enumerate(batches)]
                retry_results = await asyncio.gather(*retry_tasks, return_exceptions=True)
                
                retry_highlights_data = []
                for batch_idx, result in enumerate(retry_results):
                    if isinstance(result, Exception):
                        logger.warning(f"Retry batch {batch_idx + 1} raised exception: {result}")
                        continue
                    retry_highlights_data.extend(result)
                
                # Add retry highlights that don't overlap
                for h_data in retry_highlights_data[:20]:  # Check more candidates
                    try:
                        start = float(h_data.get("start", 0))
                        end = float(h_data.get("end", 0))
                        duration = end - start
                        
                        if not (MIN_CHUNK_SECONDS <= duration <= MAX_CHUNK_SECONDS):
                            continue
                        
                        timestamp_valid = any(
                            abs(start - w_start) < 0.1 and abs(end - w_end) < 0.1
                            for w_start, w_end in valid_timestamps
                        )
                        
                        if not timestamp_valid:
                            continue
                        
                        # Check for overlaps
                        overlaps = any(
                            not (end < h.start or start > h.end)
                            for h in highlights
                        )
                        
                        if not overlaps:
                            title = str(h_data.get("title", ""))[:50]
                            score = float(h_data.get("score", 7.0))
                            score = max(6.0, min(10.0, score))
                            
                            highlights.append(Highlight(
                                start=start,
                                end=end,
                                title=title if title else None,
                                summary=None,
                                score=score
                            ))
                            
                            if len(highlights) >= 10:
                                break
                    except (ValueError, KeyError, TypeError) as e:
                        logger.warning(f"Error parsing retry highlight data: {e}, skipping")
                        continue
                
                highlights.sort(key=lambda h: h.score, reverse=True)
                logger.info(f"After retry, have {len(highlights)} highlights")
            except Exception as e:
                logger.warning(f"Error in retry LLM call: {str(e)}, continuing with existing highlights")
        
        # Ensure we return 5-10 highlights when possible
        if len(highlights) > 10:
            highlights = highlights[:10]
            logger.info(f"LLM found {len(highlights)} highlights (limited to top 10)")
        elif len(highlights) < 5:
            logger.warning(f"Only {len(highlights)} highlights available after all attempts (target: 5-10)")
        else:
            logger.info(f"LLM found {len(highlights)} highlights (target: 5-10)")
        
        return highlights
        
    except Exception as e:
        logger.error(f"Error in LLM highlight detection from sentences: {str(e)}", exc_info=True)
        logger.warning("Returning empty highlights list due to LLM error")
        return []


async def find_highlights_async(
    chunks: List[TranscriptChunk],
    system_prompt: Optional[str] = None,
    user_prompt_template: Optional[str] = None,
) -> List[Highlight]:
    """
    Identify highlights in transcript chunks using LLM analysis.
    
    Uses LLM to select the top 5-10 highlights, each 5-45 seconds long, with timestamps.
    """
    if not chunks:
        return []
    
    try:
        logger.info(f"Analyzing {len(chunks)} chunks using LLM highlight detection")
        client = get_async_openai_client()
        
        # Filter chunks to only those within the valid duration range
        valid_chunks = [
            chunk for chunk in chunks 
            if MIN_CHUNK_SECONDS <= (chunk.end - chunk.start) <= MAX_CHUNK_SECONDS
        ]
        
        if not valid_chunks:
            logger.warning("No valid chunks in 5-45s range, returning empty highlights")
            return []
        
        # Process chunks in batches to avoid token limits while covering entire video
        # Split chunks into batches of ~6000 chars each (leaving room for prompt)
        MAX_BATCH_CHARS = 6000
        all_highlights_data = []
        
        # Split valid_chunks into batches
        batches = []
        current_batch = []
        current_batch_size = 0
        
        for chunk in valid_chunks:
            chunk_text = f"[{chunk.start:.1f}s-{chunk.end:.1f}s] {chunk.text}\n"
            chunk_size = len(chunk_text)
            
            if current_batch_size + chunk_size > MAX_BATCH_CHARS and current_batch:
                batches.append(current_batch)
                current_batch = [chunk]
                current_batch_size = chunk_size
            else:
                current_batch.append(chunk)
                current_batch_size += chunk_size
        
        if current_batch:
            batches.append(current_batch)
        
        logger.info(f"Processing {len(valid_chunks)} chunks in {len(batches)} batches to cover entire video")
        
        # Process all batches in parallel
        async def process_batch(batch_idx: int, batch: List[TranscriptChunk]) -> List[dict]:
            """Process a single batch and return highlights data."""
            transcript_text = "\n".join([
                f"[{chunk.start:.1f}s-{chunk.end:.1f}s] {chunk.text}"
                for chunk in batch
            ])
            
            prompt = f"""You are analyzing a video transcript to identify the best highlights.

Here are video segments with timestamps. Each segment is 5-45 seconds long:

{transcript_text}

Your task: Select the TOP 5-10 most engaging, interesting, or valuable segments as highlights.

Criteria for good highlights:
- Most engaging or entertaining moments
- Key insights or important information
- Emotional or impactful moments
- Memorable quotes or statements
- Moments that would make viewers want to watch the full video

CRITICAL REQUIREMENTS:
- You MUST return between 5 and 10 highlights (preferably 8-10 if there are enough good segments)
- If there are fewer than 5 good segments, return the best ones available (minimum 5 if possible)
- Each highlight must use the EXACT start and end timestamps from the segments above
- Each highlight duration must be 5-45 seconds
- Prioritize the most interesting segments
- Sort highlights by score (highest first)

Respond with JSON only:
{{
  "highlights": [
    {{
      "start": float (start timestamp in seconds, must match one of the segments above),
      "end": float (end timestamp in seconds, must match one of the segments above),
      "title": string (short catchy title, max 50 chars),
      "score": float (7.0-10.0, rating of how good this highlight is)
    }}
  ]
}}

IMPORTANT: Return between 5-10 highlights. Aim for 8-10 if there are enough good segments available."""

            try:
                response = await client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are an expert at identifying the most engaging moments in video content. Always respond with valid JSON only. You must return between 5-10 highlights in your response."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    max_tokens=3000,
                    temperature=0.3,
                )
                
                result = json.loads(response.choices[0].message.content)
                batch_highlights = result.get("highlights", [])
                logger.info(f"Batch {batch_idx + 1}/{len(batches)}: Found {len(batch_highlights)} highlights")
                return batch_highlights
            except Exception as e:
                logger.warning(f"Error processing batch {batch_idx + 1}: {str(e)}, continuing with other batches")
                return []
        
        # Process all batches concurrently
        batch_tasks = [process_batch(batch_idx, batch) for batch_idx, batch in enumerate(batches)]
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        # Collect results, handling exceptions
        for batch_idx, result in enumerate(batch_results):
            if isinstance(result, Exception):
                logger.warning(f"Batch {batch_idx + 1} raised exception: {result}")
                continue
            all_highlights_data.extend(result)
        
        highlights_data = all_highlights_data
        
        if not highlights_data:
            logger.warning("LLM returned no highlights, returning empty list")
            return []
        
        # Validate and create Highlight objects
        highlights = []
        valid_timestamps = {(chunk.start, chunk.end) for chunk in valid_chunks}
        
        for h_data in highlights_data[:10]:  # Limit to max 10
            try:
                start = float(h_data.get("start", 0))
                end = float(h_data.get("end", 0))
                duration = end - start
                
                # Validate duration
                if not (MIN_CHUNK_SECONDS <= duration <= MAX_CHUNK_SECONDS):
                    logger.warning(f"Skipping highlight [{start:.1f}s-{end:.1f}s] - duration {duration:.1f}s outside valid range")
                    continue
                
                # Validate timestamps match valid chunks (allow small tolerance for floating point)
                timestamp_valid = any(
                    abs(start - chunk_start) < 0.1 and abs(end - chunk_end) < 0.1
                    for chunk_start, chunk_end in valid_timestamps
                )
                
                if not timestamp_valid:
                    logger.warning(f"Skipping highlight [{start:.1f}s-{end:.1f}s] - timestamps don't match valid chunks")
                    continue
                
                title = str(h_data.get("title", ""))[:50]
                score = float(h_data.get("score", 7.0))
                score = max(6.0, min(10.0, score))  # Clamp to 6-10 range
                
                highlights.append(Highlight(
                    start=start,
                    end=end,
                    title=title if title else None,
                    summary=None,
                    score=score
                ))
            except (ValueError, KeyError, TypeError) as e:
                logger.warning(f"Error parsing highlight data: {e}, skipping")
                continue
        
        # Sort by score descending
        highlights.sort(key=lambda h: h.score, reverse=True)
        
        # If we have fewer than 5 highlights, retry LLM with more lenient prompt
        if len(highlights) < 5:
            logger.warning(f"Only {len(highlights)} highlights found, retrying LLM with more lenient criteria")
            try:
                # Retry with batches using more lenient criteria - process in parallel
                async def process_retry_batch(batch_idx: int, batch: List[TranscriptChunk]) -> List[dict]:
                    """Process a retry batch with more lenient criteria."""
                    transcript_text = "\n".join([
                        f"[{chunk.start:.1f}s-{chunk.end:.1f}s] {chunk.text}"
                        for chunk in batch
                    ])
                    
                    retry_prompt = f"""You are analyzing a video transcript to identify highlights.

Here are video segments with timestamps. Each segment is 5-45 seconds long:

{transcript_text}

Your task: Select AT LEAST 5-10 most engaging, interesting, or valuable segments as highlights. If you previously selected fewer, select MORE segments now.

CRITICAL: You MUST return at least 5 highlights. Select segments even if they are moderately interesting - we need at least 5 highlights.

Criteria (be more lenient):
- Engaging or entertaining moments
- Key insights or important information
- Emotional or impactful moments
- Memorable quotes or statements
- Moderately interesting segments are acceptable if needed to reach 5 highlights

CRITICAL REQUIREMENTS:
- You MUST return at least 5 highlights (preferably 8-10)
- Use the EXACT start and end timestamps from the segments above
- Each highlight duration must be 5-45 seconds
- Be more lenient - include segments even if they are only moderately interesting
- Sort highlights by score (highest first)

Respond with JSON only:
{{
  "highlights": [
    {{
      "start": float (start timestamp in seconds, must match one of the segments above),
      "end": float (end timestamp in seconds, must match one of the segments above),
      "title": string (short catchy title, max 50 chars),
      "score": float (6.0-10.0, rating of how good this highlight is - can use 6.0-7.0 for moderate highlights)
    }}
  ]
}}

IMPORTANT: Return AT LEAST 5 highlights. Include moderately interesting segments if needed."""

                    try:
                        retry_response = await client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {"role": "system", "content": "You are an expert at identifying engaging moments in video content. Always respond with valid JSON only. You MUST return at least 5 highlights - include moderately interesting segments if needed."},
                                {"role": "user", "content": retry_prompt}
                            ],
                            response_format={"type": "json_object"},
                            max_tokens=3000,
                            temperature=0.5,
                        )
                        
                        retry_result = json.loads(retry_response.choices[0].message.content)
                        batch_retry_highlights = retry_result.get("highlights", [])
                        return batch_retry_highlights
                    except Exception as e:
                        logger.warning(f"Error in retry batch {batch_idx + 1}: {str(e)}, continuing")
                        return []
                
                # Process all retry batches concurrently
                retry_tasks = [process_retry_batch(batch_idx, batch) for batch_idx, batch in enumerate(batches)]
                retry_results = await asyncio.gather(*retry_tasks, return_exceptions=True)
                
                retry_highlights_data = []
                for batch_idx, result in enumerate(retry_results):
                    if isinstance(result, Exception):
                        logger.warning(f"Retry batch {batch_idx + 1} raised exception: {result}")
                        continue
                    retry_highlights_data.extend(result)
                
                # Add retry highlights that don't overlap
                for h_data in retry_highlights_data[:20]:  # Check more candidates
                    try:
                        start = float(h_data.get("start", 0))
                        end = float(h_data.get("end", 0))
                        duration = end - start
                        
                        if not (MIN_CHUNK_SECONDS <= duration <= MAX_CHUNK_SECONDS):
                            continue
                        
                        timestamp_valid = any(
                            abs(start - chunk_start) < 0.1 and abs(end - chunk_end) < 0.1
                            for chunk_start, chunk_end in valid_timestamps
                        )
                        
                        if not timestamp_valid:
                            continue
                        
                        # Check for overlaps
                        overlaps = any(
                            not (end < h.start or start > h.end)
                            for h in highlights
                        )
                        
                        if not overlaps:
                            title = str(h_data.get("title", ""))[:50]
                            score = float(h_data.get("score", 7.0))
                            score = max(6.0, min(10.0, score))
                            
                            highlights.append(Highlight(
                                start=start,
                                end=end,
                                title=title if title else None,
                                summary=None,
                                score=score
                            ))
                            
                            if len(highlights) >= 10:
                                break
                    except (ValueError, KeyError, TypeError) as e:
                        logger.warning(f"Error parsing retry highlight data: {e}, skipping")
                        continue
                
                highlights.sort(key=lambda h: h.score, reverse=True)
                logger.info(f"After retry, have {len(highlights)} highlights")
            except Exception as e:
                logger.warning(f"Error in retry LLM call: {str(e)}, continuing with existing highlights")
        
        # Ensure we return 5-10 highlights when possible
        if len(highlights) > 10:
            highlights = highlights[:10]
            logger.info(f"LLM found {len(highlights)} highlights (limited to top 10)")
        elif len(highlights) < 5:
            logger.warning(f"Only {len(highlights)} highlights available after all attempts (target: 5-10)")
        else:
            logger.info(f"LLM found {len(highlights)} highlights (target: 5-10)")
        
        return highlights
        
    except Exception as e:
        logger.error(f"Error in LLM highlight detection: {str(e)}", exc_info=True)
        logger.warning("Returning empty highlights list due to LLM error")
        return []


async def _find_highlights_rule_based(chunks: List[TranscriptChunk]) -> List[Highlight]:
    """
    Fallback rule-based highlight detection.
    
    Uses local rule-based analysis for fast processing without API calls.
    Ensures at least 5 highlights if possible by being more lenient with scoring.
    """
    if not chunks:
        return []
    
    logger.info(f"Analyzing {len(chunks)} chunks using rule-based highlight detection")
    
    # Process chunks in parallel
    loop = asyncio.get_event_loop()
    tasks = [
        loop.run_in_executor(None, _analyze_chunk_rule_based, chunk)
        for chunk in chunks
    ]
    
    results = await asyncio.gather(*tasks)
    all_highlights = [h for chunk_highlights in results for h in chunk_highlights]
    
    # If we have fewer than 5 highlights, try with lower score threshold
    if len(all_highlights) < 5:
        logger.info(f"Only {len(all_highlights)} highlights found, trying with more lenient criteria")
        # Re-analyze chunks with lower score threshold
        for chunk in chunks:
            if len(all_highlights) >= 10:
                break
            
            chunk_duration = chunk.end - chunk.start
            if not (MIN_CHUNK_SECONDS <= chunk_duration <= MAX_CHUNK_SECONDS):
                continue
            
            # Check if this chunk already has a highlight
            chunk_has_highlight = any(
                abs(chunk.start - h.start) < 0.1 and abs(chunk.end - h.end) < 0.1
                for h in all_highlights
            )
            
            if not chunk_has_highlight:
                # Create a highlight with lower score (minimum 6.0)
                text = chunk.text.lower()
                score = 6.0  # Minimum score
                
                # Still apply some scoring boosts but be more lenient
                emotional_words = ['never', 'always', 'worst', 'best', 'crazy', 'unbelievable', 
                                  'shocking', 'amazing', 'incredible', 'terrible', 'awesome']
                emotional_count = sum(1 for word in emotional_words if word in text)
                score += min(1.0, emotional_count * 0.2)  # Smaller boost
                
                if any(text.startswith(prefix) for prefix in ['i ', 'we ', 'you ']):
                    score += 0.3
                
                if '?' in chunk.text:
                    score += 0.2
                
                score = max(6.0, min(10.0, score))
                
                title = chunk.text[:50] + "..." if len(chunk.text) > 50 else chunk.text
                all_highlights.append(Highlight(
                    start=chunk.start,
                    end=chunk.end,
                    title=title,
                    summary=None,
                    score=score
                ))
    
    # Sort by score
    all_highlights.sort(key=lambda h: h.score, reverse=True)
    
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
    
    # Quick check: if only 2 items, use simple range check
    n = len(highlights)
    if n < 2:
        return highlights
    
    # Fast variance calculation using single pass
    scores = [h.score for h in highlights]
    mean = sum(scores) / n
    sum_sq_diff = sum((x - mean) * (x - mean) for x in scores)
    std_dev = (sum_sq_diff / (n - 1)) ** 0.5 if n > 1 else 0.0
    
    # Only redistribute if scores are clustered (low std dev)
    if std_dev >= 1.0:
        return highlights
    
    # Redistribute using percentile-based mapping
    sorted_highlights = sorted(highlights, key=lambda h: h.score, reverse=True)
    denom = n - 1 if n > 1 else 1
    
    # Pre-calculate all new scores first
    new_scores = []
    for i in range(n):
        percentile = i / denom if n > 1 else 0.5
        if percentile <= 0.1:
            new_score = 10.0 - (percentile / 0.1) * 1.0
        elif percentile <= 0.3:
            new_score = 9.0 - ((percentile - 0.1) / 0.2) * 1.0
        elif percentile <= 0.6:
            new_score = 8.0 - ((percentile - 0.3) / 0.3) * 1.0
        else:
            new_score = 7.0 - ((percentile - 0.6) / 0.4) * 1.0
        new_scores.append(max(6.0, min(10.0, new_score)))
    
    # Batch create objects using model_copy
    redistributed = [
        highlight.model_copy(update={"score": new_scores[i]})
        for i, highlight in enumerate(sorted_highlights)
    ]
    
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
        max_results: Maximum number of highlights to return (default: 10, minimum: 5)
        db_session: Optional database session for calibration lookup
        
    Returns:
        List of deduplicated and ranked Highlight objects (minimum 5 highlights, or all available if fewer)
    """
    if not highlights:
        return []
    
    # Ensure max_results is at least 5 (minimum requirement)
    max_results = max(5, max_results)
    
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
    
    # Apply calibration and redistribute in one pass to avoid double object creation
    if calibration_offset != 0.0:
        # Apply calibration first, then redistribute
        calibrated_scores = [apply_calibration(h.score, calibration_offset) for h in highlights]
        calibrated_highlights = [
            highlight.model_copy(update={"score": score})
            for highlight, score in zip(highlights, calibrated_scores)
        ]
    else:
        calibrated_highlights = highlights
    
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
    
    # Filter highlights to ensure they're within 5-45 second range
    filtered = []
    for h in deduplicated:
        duration = h.end - h.start
        # Only accept highlights between 5-45 seconds
        if MIN_CHUNK_SECONDS <= duration <= MAX_CHUNK_SECONDS:
            filtered.append(h)
        else:
            logger.warning(f"Rejecting highlight [{h.start:.1f}s-{h.end:.1f}s] - duration {duration:.1f}s outside valid range ({MIN_CHUNK_SECONDS}-{MAX_CHUNK_SECONDS}s)")
    
    # Ensure we return 5-max_results highlights
    min_highlights = 5
    max_highlights = min(20, max_results)  # Cap at 20 to allow more highlights for filters
    
    if len(filtered) < min_highlights:
        logger.warning(f"Only {len(filtered)} highlights available after filtering, but minimum is {min_highlights}. Trying to find more from all highlights.")
        # Try to find more highlights that might have been filtered out
        # Lower score threshold temporarily if needed
        all_candidates = sorted(redistributed_highlights, key=lambda h: h.score, reverse=True)
        for h in all_candidates:
            if len(filtered) >= min_highlights:
                break
            duration = h.end - h.start
            # Check if it's in filtered already
            already_in = any(
                abs(h.start - f.start) < 0.1 and abs(h.end - f.end) < 0.1
                for f in filtered
            )
            if not already_in and MIN_CHUNK_SECONDS <= duration <= MAX_CHUNK_SECONDS:
                # Check for significant overlap (be lenient to reach minimum)
                significant_overlap = any(
                    overlap_percentage(h.start, h.end, f.start, f.end) > 0.8  # More lenient threshold (0.8 instead of 0.5)
                    for f in filtered
                )
                if not significant_overlap:
                    filtered.append(h)
                    logger.info(f"Added additional highlight [{h.start:.1f}s-{h.end:.1f}s] to reach minimum")
        
        # If still fewer than 5, be even more lenient with overlap threshold
        if len(filtered) < min_highlights:
            logger.warning(f"Only {len(filtered)} highlights after first pass, trying with more lenient overlap rules to reach {min_highlights}")
            for h in all_candidates:
                if len(filtered) >= min_highlights:
                    break
                duration = h.end - h.start
                already_in = any(
                    abs(h.start - f.start) < 0.1 and abs(h.end - f.end) < 0.1
                    for f in filtered
                )
                if not already_in and MIN_CHUNK_SECONDS <= duration <= MAX_CHUNK_SECONDS:
                    # Very lenient overlap check (0.9) to prioritize reaching minimum
                    significant_overlap = any(
                        overlap_percentage(h.start, h.end, f.start, f.end) > 0.9
                        for f in filtered
                    )
                    if not significant_overlap:
                        filtered.append(h)
                        logger.info(f"Added additional highlight [{h.start:.1f}s-{h.end:.1f}s] with lenient overlap rules")
        
        if len(filtered) < min_highlights:
            logger.warning(f"Still only {len(filtered)} highlights available after trying to find more (minimum: {min_highlights}). Returning all available.")
        result = filtered
    elif len(filtered) > max_highlights:
        # Take top max_highlights
        result = filtered[:max_highlights]
    else:
        # Between min and max, return all
        result = filtered
    
    logger.info(f"Aggregated {len(highlights)} highlights to {len(result)} final highlights (target: {min_highlights}-{max_highlights})")
    return result

