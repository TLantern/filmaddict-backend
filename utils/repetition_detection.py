import logging
from typing import List

from models import Sentence

logger = logging.getLogger(__name__)


def _simple_text_similarity(text1: str, text2: str) -> float:
    """Simple text similarity based on word overlap (no ML models)."""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    if not words1 or not words2:
        return 0.0
    
    intersection = words1 & words2
    union = words1 | words2
    return len(intersection) / len(union) if union else 0.0


def detect_repetition(sentences: List[Sentence]) -> List[float]:
    """
    Detect repetition by comparing sentence texts to all prior sentences (no ML models).
    
    Rules:
    - > 0.5 → near-duplicate (repetition_score = 1.0)
    - 0.3-0.5 → paraphrase (repetition_score = similarity)
    - < 0.3 → unique (repetition_score = similarity)
    
    Args:
        sentences: List of Sentence objects
        
    Returns:
        List of repetition scores (0.0-1.0) for each sentence
    """
    if not sentences:
        return []
    
    logger.info(f"Detecting repetition in {len(sentences)} sentences (rule-based)")
    
    repetition_scores = []
    
    # First pass: compute all repetition scores (except first sentence)
    raw_scores = []
    for i, current_sentence in enumerate(sentences):
        if i == 0:
            raw_scores.append(None)  # Will compute separately
            continue
        
        max_similarity = 0.0
        
        # Compare to all prior sentences
        for prior_sentence in sentences[:i]:
            similarity = _simple_text_similarity(current_sentence.text, prior_sentence.text)
            max_similarity = max(max_similarity, similarity)
        
        # Apply rules
        if max_similarity > 0.5:
            repetition_score = 1.0
        elif max_similarity >= 0.3:
            repetition_score = max_similarity
        else:
            repetition_score = max_similarity
        
        raw_scores.append(repetition_score)
    
    # Compute global average similarity for first sentence normalization
    if len(sentences) > 1:
        first_similarities = []
        for other_sentence in sentences[1:]:
            similarity = _simple_text_similarity(sentences[0].text, other_sentence.text)
            first_similarities.append(similarity)
        
        # Use average similarity as baseline for first sentence
        first_baseline = sum(first_similarities) / len(first_similarities) if first_similarities else 0.0
        # Apply same rules as other sentences
        if first_baseline > 0.5:
            first_repetition_score = 1.0
        elif first_baseline >= 0.3:
            first_repetition_score = first_baseline
        else:
            first_repetition_score = first_baseline
        
        raw_scores[0] = first_repetition_score
    else:
        raw_scores[0] = 0.0
    
    # Build final scores list
    for i, current_sentence in enumerate(sentences):
        repetition_score = raw_scores[i]
        repetition_scores.append(repetition_score)
    
    logger.info(f"Computed repetition scores: {[f'{s:.2f}' for s in repetition_scores]}")
    return repetition_scores
