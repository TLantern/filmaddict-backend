"""
Explanation builder for unified verdict explanations.

Generates human-readable explanations for FLUFF and HIGHLIGHT verdicts
using a consistent schema with evidence and action hints.
"""

import json
import logging
from typing import List, Tuple, Optional

from models import VerdictExplanation, RetentionMetrics
from utils.moment_scoring import MomentScoreComponents
from utils.llm_fallback import get_async_openai_client

logger = logging.getLogger(__name__)

# Threshold for determining confidence levels
CONFIDENCE_LOW_THRESHOLD = 0.2
CONFIDENCE_MEDIUM_THRESHOLD = 0.4


def build_fluff_explanation(
    metrics: RetentionMetrics,
    retention_value: float,
    threshold: float = 0.4,
) -> VerdictExplanation:
    """
    Build FLUFF explanation from retention metrics.
    
    Args:
        metrics: RetentionMetrics object
        retention_value: Retention value (0.0-1.0)
        threshold: Threshold for FLUFF decision (default: 0.4)
        
    Returns:
        VerdictExplanation with verdict="FLUFF"
    """
    # Map metrics to human-readable phrases
    # For emotional_delta, use absolute value since it can be negative
    # Low absolute value means flat emotional delivery
    emotional_delta_abs = abs(metrics.emotional_delta.value)
    
    metric_mappings = [
        (metrics.semantic_novelty.value, "Repeats earlier content"),
        (metrics.information_density.value, "Low information per second"),
        (emotional_delta_abs, "Flat emotional delivery"),
        (metrics.narrative_momentum.value, "No forward movement"),
    ]
    
    # Sort by value (ascending) to find bottom signals
    metric_mappings.sort(key=lambda x: x[0])
    
    # Select bottom 2-3 metrics (the worst performers)
    num_metrics = min(3, len(metric_mappings))
    selected_metrics = metric_mappings[:num_metrics]
    
    # Extract evidence phrases
    evidence = [phrase for _, phrase in selected_metrics]
    
    # Ensure we have at least one evidence item
    if not evidence:
        evidence = ["Low retention score"]
    
    # Determine confidence based on how far below threshold
    distance_below_threshold = threshold - retention_value
    if distance_below_threshold >= 0.3:
        confidence = "high"
    elif distance_below_threshold >= 0.15:
        confidence = "medium"
    else:
        confidence = "low"
    
    # Action hint
    action_hint = "Remove"
    
    return VerdictExplanation(
        verdict="FLUFF",
        confidence=confidence,
        evidence=evidence,
        action_hint=action_hint,
    )


def build_fluff_explanation_from_segment(
    segment_label: str,
    repetition_score: float,
    filler_density: float,
    visual_change_score: float,
    usefulness_score: float,
) -> VerdictExplanation:
    """
    Build FLUFF explanation from segment scores when retention metrics are unavailable.
    
    Args:
        segment_label: Segment label (should be "FLUFF")
        repetition_score: Repetition score (0.0-1.0)
        filler_density: Filler word density (0.0-1.0)
        visual_change_score: Visual change score (0.0-1.0)
        usefulness_score: Usefulness score (0.0-1.0)
        
    Returns:
        VerdictExplanation with verdict="FLUFF"
    """
    evidence = []
    
    # Build evidence from available scores
    if repetition_score > 0.7:
        evidence.append("High repetition")
    if filler_density > 0.3:
        evidence.append("Many filler words")
    if visual_change_score < 0.3:
        evidence.append("Little visual change")
    if usefulness_score < 0.3:
        evidence.append("Low usefulness")
    
    # If no specific evidence, use generic
    if not evidence:
        evidence.append("Low quality segment")
    
    # Limit to 3 items
    evidence = evidence[:3]
    
    # Determine confidence based on scores
    avg_low_score = (repetition_score + filler_density + (1 - visual_change_score) + (1 - usefulness_score)) / 4
    if avg_low_score > 0.7:
        confidence = "high"
    elif avg_low_score > 0.5:
        confidence = "medium"
    else:
        confidence = "low"
    
    return VerdictExplanation(
        verdict="FLUFF",
        confidence=confidence,
        evidence=evidence,
        action_hint="Remove",
    )


def build_highlight_explanation(
    components: MomentScoreComponents,
) -> VerdictExplanation:
    """
    Build HIGHLIGHT explanation from moment score components.
    
    Args:
        components: MomentScoreComponents object
        
    Returns:
        VerdictExplanation with verdict="HIGHLIGHT"
    """
    # Map components to human-readable phrases
    # For positive signals: higher is better
    # For negative signals (repetition, filler): lower is better
    signal_mappings: List[Tuple[float, str]] = []
    
    # Positive signals
    if components.novelty > 0.7:
        signal_mappings.append((components.novelty, "Introduces a new idea or turn"))
    if components.emotional_delta > 0.6:
        signal_mappings.append((components.emotional_delta, "Clear emotional shift"))
    if components.information_density > 0.7:
        signal_mappings.append((components.information_density, "Highly quotable"))
    
    # Negative signals (inverted - lower values are better)
    if components.repetition < 0.3:
        signal_mappings.append((1.0 - components.repetition, "Clean, non-redundant segment"))
    if components.filler_density < 0.2:
        signal_mappings.append((1.0 - components.filler_density, "Tight delivery"))
    if components.silence_penalty < 0.1:
        signal_mappings.append((1.0 - components.silence_penalty, "No dead air"))
    
    # Sort by strength (descending) to find strongest signals
    signal_mappings.sort(key=lambda x: x[0], reverse=True)
    
    # Select top 2-3 strongest signals
    num_signals = min(3, len(signal_mappings))
    selected_signals = signal_mappings[:num_signals]
    
    # Extract evidence phrases
    evidence = [phrase for _, phrase in selected_signals]
    
    # If no strong signals, use default evidence
    if not evidence:
        evidence = ["Moderate scoring across metrics"]
    
    # Determine confidence based on component strength
    # Use average of top signals
    if selected_signals:
        avg_strength = sum(score for score, _ in selected_signals) / len(selected_signals)
        if avg_strength >= 0.8:
            confidence = "high"
        elif avg_strength >= 0.6:
            confidence = "medium"
        else:
            confidence = "low"
    else:
        confidence = "medium"
    
    # Action hint
    action_hint = "Clip (30–60s)"
    
    return VerdictExplanation(
        verdict="HIGHLIGHT",
        confidence=confidence,
        evidence=evidence,
        action_hint=action_hint,
    )


async def build_highlight_explanation_from_text_async(
    text: str,
    score: float,
) -> VerdictExplanation:
    """
    Build HIGHLIGHT explanation using LLM to generate evidence bullet points from transcript text.
    
    Args:
        text: Transcript text for the highlight segment
        score: Highlight score (6.0-10.0)
        
    Returns:
        VerdictExplanation with verdict="HIGHLIGHT" and LLM-generated evidence
    """
    try:
        client = get_async_openai_client()
        
        prompt = f"""Analyze this video transcript segment and provide 2-3 bullet point reasons why this segment makes a good highlight.

Transcript: "{text}"

Provide concise, specific reasons why this segment would be engaging or valuable as a highlight. Focus on what makes it stand out - emotional impact, key insights, memorable moments, engaging delivery, etc.

Respond with JSON only:
{{
  "reasons": [
    "First reason (brief and specific)",
    "Second reason (brief and specific)",
    "Third reason (optional, brief and specific)"
  ]
}}

Return 2-3 reasons as bullet points. Each reason should be one short sentence."""
        
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert at identifying engaging moments in video content. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            max_tokens=200,
            temperature=0.3,
        )
        
        result = json.loads(response.choices[0].message.content)
        reasons = result.get("reasons", [])
        
        # Ensure we have at least one reason
        if not reasons:
            reasons = ["Engaging content"]
        
        # Limit to 3 items
        evidence = reasons[:3]
        
        # Determine confidence based on score
        if score >= 9.0:
            confidence = "high"
        elif score >= 8.0:
            confidence = "high"
        elif score >= 7.0:
            confidence = "medium"
        else:
            confidence = "medium"
        
        return VerdictExplanation(
            verdict="HIGHLIGHT",
            confidence=confidence,
            evidence=evidence,
            action_hint="Clip (30–60s)",
        )
    except Exception as e:
        logger.warning(f"Failed to generate LLM evidence for highlight: {e}, using fallback")
        # Fallback to simple rule-based evidence
        evidence = []
        if score >= 9.0:
            evidence.append("Exceptional engagement potential")
            confidence = "high"
        elif score >= 8.0:
            evidence.append("Strong engagement potential")
            confidence = "high"
        elif score >= 7.0:
            evidence.append("Good engagement potential")
            confidence = "medium"
        else:
            evidence.append("Moderate engagement potential")
            confidence = "medium"
        
        if not evidence:
            evidence = ["Scored as highlight"]
        
        return VerdictExplanation(
            verdict="HIGHLIGHT",
            confidence=confidence,
            evidence=evidence,
            action_hint="Clip (30–60s)",
        )


def build_highlight_explanation_from_score(
    score: float,
    title: Optional[str] = None,
    summary: Optional[str] = None,
) -> VerdictExplanation:
    """
    DEPRECATED: Use build_highlight_explanation_from_text_async instead.
    
    Build HIGHLIGHT explanation from highlight score and basic data.
    
    This is a simpler version that doesn't require expensive moment score components.
    Used when full moment scoring data is not available.
    
    Args:
        score: Highlight score (6.0-10.0)
        title: Optional title text
        summary: Optional summary text
        
    Returns:
        VerdictExplanation with verdict="HIGHLIGHT"
    """
    evidence = []
    
    # Determine evidence based on score range
    if score >= 9.0:
        evidence.append("Exceptional engagement potential")
        confidence = "high"
    elif score >= 8.0:
        evidence.append("Strong engagement potential")
        confidence = "high"
    elif score >= 7.0:
        evidence.append("Good engagement potential")
        confidence = "medium"
    else:
        evidence.append("Moderate engagement potential")
        confidence = "medium"
    
    # Add text-based evidence if available
    if title or summary:
        text = (title or "") + " " + (summary or "")
        text_lower = text.lower()
        
        # Check for emotional indicators
        emotional_words = ['never', 'always', 'worst', 'best', 'crazy', 'unbelievable', 
                          'shocking', 'amazing', 'incredible', 'terrible', 'awesome']
        if any(word in text_lower for word in emotional_words):
            evidence.append("Contains emotional language")
        
        # Check for questions (engagement)
        if '?' in text:
            evidence.append("Includes engaging questions")
        
        # Check for first-person (personal connection)
        if any(text_lower.startswith(prefix) for prefix in ['i ', 'we ', 'you ']):
            evidence.append("Personal or direct address")
    
    # Limit to 3 evidence items
    evidence = evidence[:3]
    
    # Ensure at least one evidence item
    if not evidence:
        evidence = ["Scored as highlight"]
    
    # Action hint
    action_hint = "Clip (30–60s)"
    
    return VerdictExplanation(
        verdict="HIGHLIGHT",
        confidence=confidence,
        evidence=evidence,
        action_hint=action_hint,
    )

