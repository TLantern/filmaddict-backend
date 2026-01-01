import logging
from typing import List, Optional, Dict

import numpy as np

from models import Sentence, SegmentAnalysis

logger = logging.getLogger(__name__)


def score_segments(
    sentences: List[Sentence],
    repetition_scores: List[float],
    filler_densities: List[float],
    visual_scores: List[float],
    llm_judgments: Optional[List[Dict]] = None,
) -> List[SegmentAnalysis]:
    """
    Compute usefulness scores and assign labels to sentences using rule-based analysis.
    Optionally incorporates LLM judgments when provided.
    
    Formula: usefulness_score =
        0.40 * semantic_novelty
      + 0.25 * information_density
      + 0.15 * emotional_intensity
      + 0.10 * audience_relevance
      + 0.10 * visual_engagement
    
    Where:
    - semantic_novelty = 1 - repetition_score (inverse of repetition)
    - information_density = 1 - filler_density (inverse of filler)
    - emotional_intensity = derived from sentence characteristics
    - audience_relevance = derived from sentence characteristics
    - visual_engagement = visual_change_score
    
    Args:
        sentences: List of Sentence objects
        repetition_scores: List of repetition scores (0.0-1.0)
        filler_densities: List of filler densities (0.0-1.0)
        visual_scores: List of visual change scores (0.0-1.0)
        llm_judgments: Optional list of LLM judgment dicts with keys: label, confidence, reason
        
    Returns:
        List of SegmentAnalysis objects with scores and labels
    """
    if not sentences:
        return []
    
    if len(sentences) != len(repetition_scores) or len(sentences) != len(filler_densities) or len(sentences) != len(visual_scores):
        raise ValueError("All input lists must have the same length")
    
    if llm_judgments and len(llm_judgments) != len(sentences):
        logger.warning(f"LLM judgments length ({len(llm_judgments)}) doesn't match sentences length ({len(sentences)}), ignoring LLM judgments")
        llm_judgments = None
    
    logger.info(f"Scoring {len(sentences)} sentences using rule-based analysis")
    
    # First pass: calculate all ratings
    ratings = []
    sentence_data = []
    
    for i, sentence in enumerate(sentences):
        repetition = repetition_scores[i]
        filler = filler_densities[i]
        visual = visual_scores[i]
        
        # Compute component scores for usefulness formula
        semantic_novelty = 1 - repetition
        information_density = 1 - filler
        visual_engagement = visual
        
        # Derive emotional_intensity from sentence characteristics
        # Higher for sentences with more content and lower repetition/filler
        word_count = len(sentence.text.split()) if sentence.text else 0
        # Normalize: longer sentences with less filler = higher emotional intensity
        emotional_intensity = min(1.0, (word_count / 50.0) * (1 - filler) * (1 - repetition * 0.5))
        
        # Derive audience_relevance from sentence characteristics
        # Higher for sentences that introduce new content (low repetition) and have substance
        audience_relevance = semantic_novelty * information_density
        
        # Compute raw usefulness score using weighted formula
        usefulness_score = (
            0.40 * semantic_novelty
            + 0.25 * information_density
            + 0.15 * emotional_intensity
            + 0.10 * audience_relevance
            + 0.10 * visual_engagement
        )
        
        # Stage 1: Calculate fluff penalty with aggressive nonlinear exponent
        # This makes the penalty actually bite for low-quality sentences
        fluff_penalty = (0.6 * repetition + 0.4 * filler) ** 1.7
        
        # Raw rating: subtract fluff penalty from usefulness score
        raw_rating = usefulness_score - fluff_penalty
        
        ratings.append(raw_rating)
        sentence_data.append({
            'sentence': sentence,
            'repetition': repetition,
            'filler': filler,
            'visual': visual,
            'usefulness_score': usefulness_score,
            'raw_rating': raw_rating,
        })
    
    # Stage 2: Distribution Stretching
    # Compute mean and standard deviation of raw ratings
    raw_ratings = np.array(ratings)
    mu = np.mean(raw_ratings)
    sigma = np.std(raw_ratings)
    
    # Apply z-score normalization and sigmoid squash
    # This forces separation between segments
    z_scores = (raw_ratings - mu) / (sigma + 1e-6)
    stretched_ratings = 1 / (1 + np.exp(-1.3 * z_scores))
    
    # Update sentence_data with stretched ratings
    for i, data in enumerate(sentence_data):
        data['rating'] = stretched_ratings[i]
    
    # Apply momentum (temporal context) adjustments
    for i in range(len(sentence_data)):
        if i > 0:  # First sentence has no previous sentence
            prev_rating = sentence_data[i - 1]['rating']
            if prev_rating > 0.6:
                sentence_data[i]['rating'] += 0.05
            elif prev_rating < 0.3:
                sentence_data[i]['rating'] -= 0.05
            # Clamp rating after momentum adjustment
            sentence_data[i]['rating'] = max(0.0, min(1.0, sentence_data[i]['rating']))
    
    # Recalculate ratings list with momentum adjustments
    ratings = [data['rating'] for data in sentence_data]
    
    # Step 1: Collect distribution - sort ratings
    R = sorted(ratings)
    
    # Step 2: Calculate dynamic cutoffs for labels
    # - FLUFF: bottom 20% (<= 20th percentile)
    if R:
        p75 = np.percentile(R, 75)
        p20 = np.percentile(R, 20)
    else:
        p75 = 0.75
        p20 = 0.20
    
    # Calculate percentiles for grading
    if R:
        p90 = np.percentile(R, 90)
        p80 = np.percentile(R, 80)
        p50 = np.percentile(R, 50)
        p20 = np.percentile(R, 20)
    else:
        p90 = 0.9
        p80 = 0.8
        p50 = 0.5
        p20 = 0.2
    
    # Classify and grade sentences
    analyses = []
    for i, data in enumerate(sentence_data):
        rating = data['rating']
        sentence = data['sentence']
        repetition = data['repetition']
        filler = data['filler']
        visual = data['visual']
        usefulness_score = data['usefulness_score']
        
        # Only create FLUFF segments (skip others)
        llm_judgment = llm_judgments[i] if llm_judgments and i < len(llm_judgments) and llm_judgments[i] else None
        
        # Determine if this should be labeled as FLUFF
        is_fluff = False
        reason = ""
        
        if llm_judgment and llm_judgment.get('label', '').upper() == "FLUFF" and llm_judgment.get('confidence', 0) >= 0.7:
            # High-confidence LLM judgment indicates FLUFF
            is_fluff = True
            reason = f"LLM analysis: {llm_judgment.get('reason', 'No reason provided')} (confidence: {llm_judgment.get('confidence', 0):.2f})"
        else:
            # Only label as FLUFF if rating is low (bottom 20th percentile)
            if rating <= p20:
                is_fluff = True
                reason = f"Low quality sentence (rating: {rating:.2f}, percentile: <=20th)"
        
        # Skip non-FLUFF segments
        if not is_fluff:
            continue
        
        # Grade assignment based on rating percentiles
        if rating >= p90:
            grade = "A"
        elif rating >= p80:
            grade = "B"
        elif rating >= p50:
            grade = "C"
        elif rating >= p20:
            grade = "D"
        else:
            grade = "F"
        
        analysis = SegmentAnalysis(
            start_time=sentence.start,
            end_time=sentence.end,
            label="FLUFF",
            rating=rating,
            grade=grade,
            reason=reason,
            repetition_score=repetition,
            filler_density=filler,
            visual_change_score=visual,
            usefulness_score=usefulness_score,
        )
        # Store sentence text for later retrieval
        analysis._segment_text = sentence.text
        analyses.append(analysis)
    
    logger.info(f"Scored {len(analyses)} FLUFF segments")
    
    return analyses
