import json
import logging
import os
from typing import List, Optional

from openai import AsyncOpenAI

from models import SemanticSegment

logger = logging.getLogger(__name__)

_async_openai_client: Optional[AsyncOpenAI] = None


def get_async_openai_client() -> AsyncOpenAI:
    """Get or create async OpenAI client instance."""
    global _async_openai_client
    if _async_openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        _async_openai_client = AsyncOpenAI(api_key=api_key)
    return _async_openai_client


async def judge_segment_hybrid(
    segment: SemanticSegment,
    prior_context: List[SemanticSegment],
    visual_change_score: float,
    repetition_score: float,
    filler_density: float,
) -> dict:
    """
    Hybrid LLM + visual analysis judgment for segment classification.
    
    Uses LLM to analyze semantic content and combines with visual analysis
    to determine if segment is FLUFF.
    
    Args:
        segment: SemanticSegment to judge
        prior_context: List of prior SemanticSegment objects for context
        visual_change_score: Visual change score (0.0-1.0)
        repetition_score: Repetition score (0.0-1.0)
        filler_density: Filler word density (0.0-1.0)
        
    Returns:
        dict with keys: label ("FLUFF"), confidence (0.0-1.0), reason (str), or None if not FLUFF
    """
    enable_llm = os.getenv("ENABLE_LLM_FALLBACK", "true").lower() == "true"
    
    if not enable_llm:
        # Fallback to rule-based if LLM disabled (low thresholds for fast processing)
        if repetition_score > 0.3 and visual_change_score < 0.1:
            return {"label": "FLUFF", "confidence": 0.8, "reason": "High repetition with minimal visual change"}
        else:
            return None
    
    try:
        client = get_async_openai_client()
        
        # Build context text
        context_text = "\n".join([f"[{s.start_time:.1f}s-{s.end_time:.1f}s] {s.text}" for s in prior_context[-5:]])
        if not context_text:
            context_text = "No prior context"
        
        # Format scores for LLM
        visual_desc = "high" if visual_change_score > 0.6 else "moderate" if visual_change_score > 0.3 else "low"
        repetition_desc = "high" if repetition_score > 0.7 else "moderate" if repetition_score > 0.4 else "low"
        filler_desc = "high" if filler_density > 0.3 else "moderate" if filler_density > 0.15 else "low"
        
        prompt = f"""You are analyzing a video segment to classify it as FLUFF or not.

Prior context (last 5 segments):
{context_text}

Current segment [{segment.start_time:.1f}s-{segment.end_time:.1f}s]:
{segment.text}

Technical analysis:
- Visual change: {visual_desc} ({visual_change_score:.2f})
- Repetition: {repetition_desc} ({repetition_score:.2f})
- Filler words: {filler_desc} ({filler_density:.2f})

Classify this segment considering BOTH semantic content AND visual analysis:
- FLUFF: No new information, repeats prior content, minimal visual interest
- Not FLUFF: Introduces new information or has significant visual/emotional value

Respond with JSON only:
{{"label": "FLUFF" or null, "confidence": 0.0-1.0, "reason": "brief explanation"}}

If it's FLUFF, return {{"label": "FLUFF", ...}}. If it's not FLUFF, return {{"label": null, ...}}."""

        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert at analyzing video content. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            max_tokens=150,
            temperature=0.3,
        )
        
        result = json.loads(response.choices[0].message.content)
        
        label = result.get("label")
        if label:
            label = label.upper()
        
        confidence = float(result.get("confidence", 0.7))
        reason = result.get("reason", "LLM analysis")
        
        # Only return FLUFF, return None otherwise
        if label == "FLUFF":
            logger.info(f"LLM hybrid judgment for segment {segment.segment_id}: FLUFF (confidence: {confidence:.2f})")
            return {"label": "FLUFF", "confidence": confidence, "reason": reason}
        else:
            return None
            
    except Exception as e:
        logger.error(f"Error in LLM hybrid judgment: {str(e)}", exc_info=True)
        # Fallback to rule-based on error (low thresholds for fast processing)
        if repetition_score > 0.3 and visual_change_score < 0.1:
            return {"label": "FLUFF", "confidence": 0.6, "reason": "Fallback: High repetition with minimal visual change"}
        else:
            return None


async def judge_segment_text_only(
    segment: SemanticSegment,
    prior_context: List[SemanticSegment],
    repetition_score: float,
    filler_density: float,
) -> dict:
    """
    Text-only LLM analysis judgment for segment classification (no visual analysis).
    
    Uses LLM to analyze semantic content based on text, repetition, and filler words
    to determine if segment is FLUFF.
    
    Args:
        segment: SemanticSegment to judge
        prior_context: List of prior SemanticSegment objects for context
        repetition_score: Repetition score (0.0-1.0)
        filler_density: Filler word density (0.0-1.0)
        
    Returns:
        dict with keys: label ("FLUFF"), confidence (0.0-1.0), reason (str), or None if not FLUFF
    """
    enable_llm = os.getenv("ENABLE_LLM_FALLBACK", "true").lower() == "true"
    
    if not enable_llm:
        # Fallback to rule-based if LLM disabled
        if repetition_score > 0.3:
            return {"label": "FLUFF", "confidence": 0.8, "reason": "High repetition detected"}
        else:
            return None
    
    try:
        client = get_async_openai_client()
        
        # Build context text
        context_text = "\n".join([f"[{s.start_time:.1f}s-{s.end_time:.1f}s] {s.text}" for s in prior_context[-5:]])
        if not context_text:
            context_text = "No prior context"
        
        # Format scores for LLM
        repetition_desc = "high" if repetition_score > 0.7 else "moderate" if repetition_score > 0.4 else "low"
        filler_desc = "high" if filler_density > 0.3 else "moderate" if filler_density > 0.15 else "low"
        
        prompt = f"""You are analyzing a video segment transcript to classify it as FLUFF or not.

Prior context (last 5 segments):
{context_text}

Current segment [{segment.start_time:.1f}s-{segment.end_time:.1f}s]:
{segment.text}

Technical analysis:
- Repetition: {repetition_desc} ({repetition_score:.2f})
- Filler words: {filler_desc} ({filler_density:.2f})

Classify this segment based on semantic content:
- FLUFF: No new information, repeats prior content, lacks substance
- Not FLUFF: Introduces new information, has educational or informative value

Respond with JSON only:
{{"label": "FLUFF" or null, "confidence": 0.0-1.0, "reason": "brief explanation"}}

If it's FLUFF, return {{"label": "FLUFF", ...}}. If it's not FLUFF, return {{"label": null, ...}}."""

        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert at analyzing video transcript content. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            max_tokens=150,
            temperature=0.3,
        )
        
        result = json.loads(response.choices[0].message.content)
        
        label = result.get("label")
        if label:
            label = label.upper()
        
        confidence = float(result.get("confidence", 0.7))
        reason = result.get("reason", "LLM analysis")
        
        # Only return FLUFF, return None otherwise
        if label == "FLUFF":
            logger.info(f"LLM text-only judgment for segment {segment.segment_id}: FLUFF (confidence: {confidence:.2f})")
            return {"label": "FLUFF", "confidence": confidence, "reason": reason}
        else:
            return None
            
    except Exception as e:
        logger.error(f"Error in LLM text-only judgment: {str(e)}", exc_info=True)
        # Fallback to rule-based on error
        if repetition_score > 0.3:
            return {"label": "FLUFF", "confidence": 0.6, "reason": "Fallback: High repetition detected"}
        else:
            return None


async def judge_sentence_text_only(
    sentence,
    prior_context: List,
    repetition_score: float,
    filler_density: float,
) -> dict:
    """
    Text-only LLM analysis judgment for sentence classification (no visual analysis).
    
    Uses LLM to analyze semantic content based on text, repetition, and filler words
    to determine if sentence is FLUFF.
    
    Args:
        sentence: Sentence object with start, end, and text
        prior_context: List of prior Sentence objects for context
        repetition_score: Repetition score (0.0-1.0)
        filler_density: Filler word density (0.0-1.0)
        
    Returns:
        dict with keys: label ("FLUFF"), confidence (0.0-1.0), reason (str), or None if not FLUFF
    """
    from models import Sentence
    
    enable_llm = os.getenv("ENABLE_LLM_FALLBACK", "true").lower() == "true"
    
    if not enable_llm:
        # Fallback to rule-based if LLM disabled
        if repetition_score > 0.3:
            return {"label": "FLUFF", "confidence": 0.8, "reason": "High repetition detected"}
        else:
            return None
    
    try:
        client = get_async_openai_client()
        
        # Build context text from prior sentences
        context_text = "\n".join([f"[{s.start:.1f}s-{s.end:.1f}s] {s.text}" for s in prior_context[-5:]])
        if not context_text:
            context_text = "No prior context"
        
        # Format scores for LLM
        repetition_desc = "high" if repetition_score > 0.7 else "moderate" if repetition_score > 0.4 else "low"
        filler_desc = "high" if filler_density > 0.3 else "moderate" if filler_density > 0.15 else "low"
        
        prompt = f"""You are analyzing a video sentence transcript to classify it as FLUFF or not.

Prior context (last 5 sentences):
{context_text}

Current sentence [{sentence.start:.1f}s-{sentence.end:.1f}s]:
{sentence.text}

Technical analysis:
- Repetition: {repetition_desc} ({repetition_score:.2f})
- Filler words: {filler_desc} ({filler_density:.2f})

Classify this sentence based on semantic content:
- FLUFF: No new information, repeats prior content, lacks substance
- Not FLUFF: Introduces new information, has educational or informative value

Respond with JSON only:
{{"label": "FLUFF" or null, "confidence": 0.0-1.0, "reason": "brief explanation"}}

If it's FLUFF, return {{"label": "FLUFF", ...}}. If it's not FLUFF, return {{"label": null, ...}}."""

        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert at analyzing video transcript content. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            max_tokens=150,
            temperature=0.3,
        )
        
        result = json.loads(response.choices[0].message.content)
        
        label = result.get("label")
        if label:
            label = label.upper()
        
        confidence = float(result.get("confidence", 0.7))
        reason = result.get("reason", "LLM analysis")
        
        # Only return FLUFF, return None otherwise
        if label == "FLUFF":
            logger.info(f"LLM text-only judgment for sentence [{sentence.start:.1f}s-{sentence.end:.1f}s]: FLUFF (confidence: {confidence:.2f})")
            return {"label": "FLUFF", "confidence": confidence, "reason": reason}
        else:
            return None
            
    except Exception as e:
        logger.error(f"Error in LLM text-only judgment: {str(e)}", exc_info=True)
        # Fallback to rule-based on error
        if repetition_score > 0.3:
            return {"label": "FLUFF", "confidence": 0.6, "reason": "Fallback: High repetition detected"}
        else:
            return None


async def judge_borderline_segment(
    segment: SemanticSegment,
    prior_context: List[SemanticSegment],
) -> str:
    """
    Optional LLM fallback for borderline segments (0.4 < usefulness < 0.6).
    
    Prompt: "Does this segment introduce new semantic information relative to prior context?"
    Returns: KEEP, TRIM, or CUT
    
    Args:
        segment: SemanticSegment to judge
        prior_context: List of prior SemanticSegment objects for context
        
    Returns:
        "KEEP", "TRIM", or "CUT"
    """
    enable_fallback = os.getenv("ENABLE_LLM_FALLBACK", "true").lower() == "true"
    
    if not enable_fallback:
        return "KEEP"  # Default to keeping if fallback disabled
    
    try:
        client = get_async_openai_client()
        
        # Build context text
        context_text = "\n".join([f"[{s.start_time:.1f}s-{s.end_time:.1f}s] {s.text}" for s in prior_context[-5:]])
        
        prompt = f"""You are analyzing a video segment to determine if it introduces new semantic information.

Prior context (last 5 segments):
{context_text}

Current segment [{segment.start_time:.1f}s-{segment.end_time:.1f}s]:
{segment.text}

Does this segment introduce new semantic information relative to the prior context?

Respond with exactly one word:
- KEEP: Segment introduces new information and should be kept
- TRIM: Segment has some value but could be shortened
- CUT: Segment repeats prior information and should be removed"""

        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert at analyzing video content for semantic redundancy."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=10,
            temperature=0.3,
        )
        
        judgment = response.choices[0].message.content.strip().upper()
        
        if judgment in ["KEEP", "TRIM", "CUT"]:
            logger.info(f"LLM judgment for segment {segment.segment_id}: {judgment}")
            return judgment
        else:
            logger.warning(f"Unexpected LLM response: {judgment}, defaulting to KEEP")
            return "KEEP"
            
    except Exception as e:
        logger.error(f"Error in LLM fallback: {str(e)}", exc_info=True)
        return "KEEP"  # Default to keeping on error


async def judge_fluff_detection_quality(
    transcript_segments: List,
    fluff_analyses: List,
) -> dict:
    """
    Final LLM judge to review fluff detection quality.
    
    If confidence > 80%, passes through the fluff detection results.
    If confidence < 80%, LLM identifies the top 5-10 segments with most fluff.
    
    Args:
        transcript_segments: List of transcript segments with start, end, text
        fluff_analyses: List of SegmentAnalysis objects with fluff detection results
        
    Returns:
        dict with keys:
        - approved: bool (True if confidence > 80%)
        - confidence: float (0.0-1.0)
        - reason: str
        - top_fluff_segments: List[dict] (if not approved, contains top 5-10 fluff segments)
            Each dict has: start_time, end_time, text, fluff_score
    """
    enable_llm = os.getenv("ENABLE_LLM_FALLBACK", "true").lower() == "true"
    
    if not enable_llm:
        logger.info("LLM final judge disabled, passing through fluff detection")
        return {
            "approved": True,
            "confidence": 0.9,
            "reason": "LLM judge disabled, auto-approved",
            "top_fluff_segments": []
        }
    
    if not transcript_segments or not fluff_analyses:
        logger.warning("Empty transcript or analyses, skipping LLM final judge")
        return {
            "approved": True,
            "confidence": 0.8,
            "reason": "Empty input, auto-approved",
            "top_fluff_segments": []
        }
    
    try:
        client = get_async_openai_client()
        
        # Build full transcript with timestamps
        transcript_text = "\n".join([
            f"[{seg.start:.1f}s-{seg.end:.1f}s] {seg.text}"
            for seg in transcript_segments
        ])
        
        # Build fluff detection summary
        fluff_segments = [a for a in fluff_analyses if a.label == "FLUFF"]
        
        fluff_summary = "\n".join([
            f"[{a.start_time:.1f}s-{a.end_time:.1f}s] FLUFF (rating: {a.rating:.2f}, reason: {a.reason})"
            for a in sorted(fluff_segments, key=lambda x: x.start_time)[:20]  # Show first 20 for context
        ])
        
        if not fluff_summary:
            fluff_summary = "No FLUFF segments detected"
        
        # Limit transcript to avoid token limits
        transcript_preview = transcript_text[:8000] if len(transcript_text) > 8000 else transcript_text
        
        prompt = f"""You are a final quality judge reviewing fluff detection results for a video transcript.

Full transcript:
{transcript_preview}

Fluff detection results (first 20 FLUFF segments):
{fluff_summary}

Summary:
- Total segments analyzed: {len(fluff_analyses)}
- FLUFF segments detected: {len(fluff_segments)}

Your task:
1. Review the fluff detection quality. Are the FLUFF segments correctly identified as fluff (repetitive, uninformative, filler content)?
2. If you're confident (>80%) that the fluff detection is accurate, approve it.
3. If you're not confident (<80%), identify EXACTLY 5-10 segments (no more, no less) that contain the MOST fluff in the transcript. Sort them by fluff_score (highest first).

Respond with JSON only:
{{
  "approved": true/false,
  "confidence": 0.0-1.0,
  "reason": "brief explanation of your judgment",
  "top_fluff_segments": [
    {{"start_time": float, "end_time": float, "text": "segment text", "fluff_score": 0.0-1.0, "reason": "why this is fluff"}}
  ]
}}

If approved=true, top_fluff_segments can be empty.
If approved=false, provide EXACTLY 5-10 segments (no more than 10, prioritize by fluff_score) with the most fluff."""

        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert at analyzing video content quality and identifying fluff. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            max_tokens=2000,
            temperature=0.3,
        )
        
        result = json.loads(response.choices[0].message.content)
        
        approved = result.get("approved", True)
        confidence = float(result.get("confidence", 0.8))
        reason = result.get("reason", "LLM final judgment")
        top_fluff_segments = result.get("top_fluff_segments", [])
        
        # Validate top_fluff_segments structure
        validated_segments = []
        for seg in top_fluff_segments:
            if isinstance(seg, dict) and "start_time" in seg and "end_time" in seg:
                validated_segments.append({
                    "start_time": float(seg.get("start_time", 0)),
                    "end_time": float(seg.get("end_time", 0)),
                    "text": str(seg.get("text", "")),
                    "fluff_score": float(seg.get("fluff_score", 0.5)),
                    "reason": str(seg.get("reason", "LLM identified as fluff"))
                })
        
        # Limit to 5-10 segments (sort by fluff_score descending and take top 10)
        if len(validated_segments) > 10:
            validated_segments.sort(key=lambda x: x.get("fluff_score", 0.5), reverse=True)
            validated_segments = validated_segments[:10]
            logger.warning(
                f"LLM returned {len(top_fluff_segments)} fluff segments, limiting to top 10 by fluff_score"
            )
        elif len(validated_segments) > 0:
            # Sort by fluff_score even if within limit for consistency
            validated_segments.sort(key=lambda x: x.get("fluff_score", 0.5), reverse=True)
        
        logger.info(
            f"LLM final judge: approved={approved}, confidence={confidence:.2f}, "
            f"top_fluff_segments={len(validated_segments)}"
        )
        
        return {
            "approved": approved,
            "confidence": confidence,
            "reason": reason,
            "top_fluff_segments": validated_segments
        }
            
    except Exception as e:
        logger.error(f"Error in LLM final judge: {str(e)}", exc_info=True)
        # Fallback: approve if we have fluff segments, otherwise pass through
        return {
            "approved": True,
            "confidence": 0.7,
            "reason": f"Fallback: LLM error occurred ({str(e)})",
            "top_fluff_segments": []
        }
