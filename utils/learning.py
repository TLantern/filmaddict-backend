import logging
import os
import hashlib
from typing import List, Dict, Optional, Tuple
from uuid import UUID
from datetime import datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession
from db import crud
from db.models import HighlightFeedback, PromptVersion
from models import FeedbackType

logger = logging.getLogger(__name__)

MIN_FEEDBACK_THRESHOLD = int(os.getenv("MIN_FEEDBACK_THRESHOLD", "50"))
MIN_FEEDBACK_FOR_NEW_PROMPT = int(os.getenv("MIN_FEEDBACK_FOR_NEW_PROMPT", "1"))
CALIBRATION_ENABLED = os.getenv("CALIBRATION_ENABLED", "true").lower() == "true"


def calculate_calibration_offset(
    predicted_scores: List[float],
    actual_ratings: List[float],
) -> float:
    """
    Calculate offset to calibrate predicted scores based on user feedback.
    
    Args:
        predicted_scores: List of predicted scores from the model (1-10 scale)
        actual_ratings: List of actual user ratings (0-100 scale)
        
    Returns:
        Adjustment factor to apply to future predictions (1-10 scale)
    """
    if len(predicted_scores) != len(actual_ratings) or len(predicted_scores) == 0:
        return 0.0
    
    # Convert 0-100 scale ratings to 1-10 scale for comparison
    scaled_ratings = [rating / 10.0 for rating in actual_ratings]
    
    avg_predicted = sum(predicted_scores) / len(predicted_scores)
    avg_actual = sum(scaled_ratings) / len(scaled_ratings)
    
    offset = avg_actual - avg_predicted
    logger.info(f"Calculated calibration offset: {offset:.2f} (predicted avg: {avg_predicted:.2f}, actual avg: {avg_actual:.2f})")
    return offset


def apply_calibration(score: float, offset: float) -> float:
    """
    Apply calibration offset to a score.
    
    Args:
        score: Original score
        offset: Calibration offset
        
    Returns:
        Calibrated score (clamped to 1-10 range)
    """
    if not CALIBRATION_ENABLED:
        return score
    
    calibrated = score + offset
    return max(1.0, min(10.0, calibrated))


async def update_calibration_online(
    db: AsyncSession,
    highlight_id: UUID,
    rating: float,
    predicted_score: float,
) -> Optional[float]:
    """
    Update calibration config online from a single RATING feedback.
    
    Args:
        db: Database session
        highlight_id: Highlight ID (for logging)
        rating: User rating (0-100 scale)
        predicted_score: Model's predicted score (1-10 scale)
        
    Returns:
        New calibration offset if sufficient data, None otherwise
    """
    scaled_rating = rating / 10.0  # Convert 0-100 to 0-10 scale
    
    config = await crud.update_calibration_config_online(
        db,
        predicted_score=predicted_score,
        actual_rating=scaled_rating,
    )
    
    if config and config.feedback_count >= MIN_FEEDBACK_THRESHOLD:
        logger.info(
            f"Updated calibration online: feedback_count={config.feedback_count}, "
            f"offset={config.score_offset:.2f}"
        )
        return config.score_offset
    
    return None


async def update_calibration_from_feedback(
    db: AsyncSession,
    days: int = 7,
) -> Optional[float]:
    """
    Update calibration config from recent feedback data.
    
    Args:
        db: Database session
        days: Number of days to look back for feedback
        
    Returns:
        New calibration offset if sufficient data, None otherwise
    """
    feedback_list = await crud.get_recent_feedback(db, days=days)
    
    if len(feedback_list) < MIN_FEEDBACK_THRESHOLD:
        return None
    
    predicted_scores = []
    actual_ratings = []
    
    for feedback in feedback_list:
        if feedback.confidence_score is not None and feedback.highlight:
            predicted_scores.append(feedback.highlight.score)
            actual_ratings.append(feedback.confidence_score)
    
    if len(predicted_scores) < MIN_FEEDBACK_THRESHOLD:
        return None
    
    offset = calculate_calibration_offset(predicted_scores, actual_ratings)
    
    await crud.create_or_update_calibration_config(
        db,
        score_offset=offset,
        sample_size=len(predicted_scores),
    )
    
    logger.info(f"Updated calibration config with offset: {offset:.2f} (sample size: {len(predicted_scores)})")
    return offset


def extract_feedback_patterns(feedback_data: List[HighlightFeedback]) -> Dict[str, any]:
    """
    Analyze feedback to find success/failure patterns based on editor quality signals.
    
    Only uses editor feedback (POSITIVE, NEGATIVE, RATING) to classify highlights.
    View-based metrics like SKIP are excluded from pattern extraction.
    
    Args:
        feedback_data: List of feedback records
        
    Returns:
        Dictionary with patterns extracted from feedback including text feedback analysis
    """
    if not feedback_data:
        return {}
    
    successful_highlights = []
    failed_highlights = []
    successful_text_feedback = []
    failed_text_feedback = []
    
    for feedback in feedback_data:
        if not feedback.highlight:
            continue
        
        highlight = feedback.highlight
        
        # Ratings are now 0-100 scale, so threshold is 70 instead of 7
        # Only use editor quality feedback: POSITIVE, NEGATIVE, and RATING
        # SKIP is a view-based metric and not used for learning patterns
        if feedback.feedback_type == FeedbackType.POSITIVE.value or (
            feedback.feedback_type == FeedbackType.CONFIDENCE_SCORE.value and feedback.confidence_score and feedback.confidence_score >= 70
        ):
            successful_highlights.append(highlight)
            if feedback.text_feedback:
                successful_text_feedback.append(feedback.text_feedback.lower())
        elif feedback.feedback_type == FeedbackType.NEGATIVE.value or (
            feedback.feedback_type == FeedbackType.CONFIDENCE_SCORE.value and feedback.confidence_score and feedback.confidence_score < 70
        ):
            failed_highlights.append(highlight)
            if feedback.text_feedback:
                failed_text_feedback.append(feedback.text_feedback.lower())
    
    patterns = {
        "successful_durations": [],
        "failed_durations": [],
        "successful_scores": [],
        "failed_scores": [],
        "successful_reasons": [],
        "failed_reasons": [],
        "successful_text_feedback": successful_text_feedback,
        "failed_text_feedback": failed_text_feedback,
    }
    
    for highlight in successful_highlights:
        duration = highlight.end - highlight.start
        patterns["successful_durations"].append(duration)
        patterns["successful_scores"].append(highlight.score)
        if highlight.summary:
            patterns["successful_reasons"].append(highlight.summary.lower())
    
    for highlight in failed_highlights:
        duration = highlight.end - highlight.start
        patterns["failed_durations"].append(duration)
        patterns["failed_scores"].append(highlight.score)
        if highlight.summary:
            patterns["failed_reasons"].append(highlight.summary.lower())
    
    return patterns


def generate_improved_prompt(
    feedback_data: List[HighlightFeedback],
    current_prompt: str,
) -> str:
    """
    Generate improved prompt based on feedback patterns.
    
    Args:
        feedback_data: List of feedback records
        current_prompt: Current prompt template
        
    Returns:
        Enhanced prompt template
    """
    patterns = extract_feedback_patterns(feedback_data)
    
    if not patterns.get("successful_durations") and not patterns.get("failed_durations"):
        return current_prompt
    
    avg_successful_duration = (
        sum(patterns["successful_durations"]) / len(patterns["successful_durations"])
        if patterns["successful_durations"]
        else None
    )
    avg_failed_duration = (
        sum(patterns["failed_durations"]) / len(patterns["failed_durations"])
        if patterns["failed_durations"]
        else None
    )
    
    avg_successful_score = (
        sum(patterns["successful_scores"]) / len(patterns["successful_scores"])
        if patterns["successful_scores"]
        else None
    )
    
    enhancement = []
    
    if avg_successful_duration:
        enhancement.append(f"Preferred duration: {avg_successful_duration:.1f} seconds")
    
    if avg_successful_score:
        enhancement.append(f"Target score range: {avg_successful_score:.1f}+")
    
    if patterns["successful_reasons"]:
        common_words = {}
        for reason in patterns["successful_reasons"]:
            words = reason.split()
            for word in words:
                if len(word) > 4:
                    common_words[word] = common_words.get(word, 0) + 1
        
        top_words = sorted(common_words.items(), key=lambda x: x[1], reverse=True)[:5]
        if top_words:
            enhancement.append(f"Successful patterns include: {', '.join([w[0] for w in top_words])}")
    
    # Analyze text feedback for common themes
    if patterns.get("successful_text_feedback"):
        text_words = {}
        for text in patterns["successful_text_feedback"]:
            words = text.split()
            for word in words:
                if len(word) > 4:
                    text_words[word] = text_words.get(word, 0) + 1
        
        top_text_words = sorted(text_words.items(), key=lambda x: x[1], reverse=True)[:5]
        if top_text_words:
            enhancement.append(f"User feedback highlights: {', '.join([w[0] for w in top_text_words])}")
    
    if patterns.get("failed_text_feedback"):
        failed_words = {}
        for text in patterns["failed_text_feedback"]:
            words = text.split()
            for word in words:
                if len(word) > 4:
                    failed_words[word] = failed_words.get(word, 0) + 1
        
        top_failed_words = sorted(failed_words.items(), key=lambda x: x[1], reverse=True)[:3]
        if top_failed_words:
            enhancement.append(f"Avoid patterns with: {', '.join([w[0] for w in top_failed_words])}")
    
    if enhancement:
        enhanced_prompt = f"""{current_prompt}

Based on user feedback analysis:
- {chr(10).join('- ' + e for e in enhancement)}
"""
        return enhanced_prompt
    
    return current_prompt


async def evaluate_prompt_performance(
    db: AsyncSession,
    version_id: UUID,
    days: int = 7,
) -> Dict[str, float]:
    """
    Calculate performance metrics for a prompt version based on editor feedback.
    
    Reads rolling metrics directly from PromptVersion columns (updated online).
    The editor is treated as a labeler, not an audience. Only explicit quality signals
    are used: ratings, positive/negative flags, saves.
    
    CTR and view-based metrics are intentionally excluded because the editor is the one
    viewing clips, not end users.
    
    Args:
        db: Database session
        version_id: Prompt version ID
        days: Number of days to analyze (kept for API compatibility, but metrics come from rolling columns)
        
    Returns:
        Dictionary with performance metrics:
        - avg_rating: Average rating (0-100 scale) from RATING feedback events
        - sample_size: Number of RATING feedback events (total_rated)
        - save_rate: Saves per rated highlight (total_saves / total_rated)
        - positive_rate: Positive feedback count / total_rated
        - negative_rate: Negative feedback count / total_rated
    """
    prompt_version = await crud.get_prompt_version_by_id(db, version_id)
    
    if not prompt_version:
        return {
            "avg_rating": 0.0,
            "save_rate": 0.0,
            "positive_rate": 0.0,
            "negative_rate": 0.0,
            "sample_size": 0,
        }
    
    # Return rolling metrics directly from PromptVersion columns
    metrics = {
        "avg_rating": prompt_version.avg_confidence_score,  # 0-100 scale
        "save_rate": prompt_version.save_rate,
        "positive_rate": prompt_version.positive_rate,
        "negative_rate": prompt_version.negative_rate,
        "sample_size": prompt_version.total_rated,
    }
    
    return metrics


def select_prompt_version(video_id: UUID, prompt_versions: List[PromptVersion]) -> Optional[PromptVersion]:
    """
    Select prompt version using consistent hashing.
    
    Args:
        video_id: Video UUID
        prompt_versions: List of active prompt versions
        
    Returns:
        Selected prompt version
    """
    if not prompt_versions:
        return None
    
    if len(prompt_versions) == 1:
        return prompt_versions[0]
    
    video_id_str = str(video_id)
    hash_value = int(hashlib.md5(video_id_str.encode()).hexdigest(), 16)
    selected_index = hash_value % len(prompt_versions)
    
    return prompt_versions[selected_index]


def compare_prompt_versions(
    metrics_a: Dict[str, float],
    metrics_b: Dict[str, float],
) -> Dict[str, any]:
    """
    Compare performance of two prompt versions based on editor feedback.
    
    Prompt promotion is based purely on editor feedback (ratings, saves, positive/negative flags),
    not viewer CTR or other view-based metrics.
    
    Args:
        metrics_a: Metrics for version A (active version)
        metrics_b: Metrics for version B (candidate version)
        
    Returns:
        Comparison results with statistical significance:
        - improvement: Dictionary with rating_improvement and optionally save_rate_improvement
        - significant: True if improvement is statistically significant (both versions have sufficient sample size)
        - winner: "A" or "B" based on avg_rating comparison
    """
    rating_improvement = metrics_b["avg_rating"] - metrics_a["avg_rating"]
    
    improvement = {
        "rating_improvement": rating_improvement,
    }
    
    # Optionally include save_rate improvement
    if "save_rate" in metrics_a and "save_rate" in metrics_b:
        improvement["save_rate_improvement"] = metrics_b["save_rate"] - metrics_a["save_rate"]
    
    # Significance check: both versions must have sufficient sample size
    # and rating improvement must exceed threshold
    significant = (
        abs(rating_improvement) > 0.5
        and metrics_a["sample_size"] >= MIN_FEEDBACK_THRESHOLD
        and metrics_b["sample_size"] >= MIN_FEEDBACK_THRESHOLD
    )
    
    return {
        "improvement": improvement,
        "significant": significant,
        "winner": "B" if metrics_b["avg_rating"] > metrics_a["avg_rating"] else "A",
    }


def extract_successful_features(feedback_data: List[HighlightFeedback]) -> Dict[str, any]:
    """
    Learn preferred duration, score thresholds, content patterns.
    
    Args:
        feedback_data: List of feedback records
        
    Returns:
        Dictionary with learned features
    """
    patterns = extract_feedback_patterns(feedback_data)
    
    features = {}
    
    if patterns["successful_durations"]:
        durations = patterns["successful_durations"]
        features["preferred_duration_min"] = min(durations)
        features["preferred_duration_max"] = max(durations)
        features["preferred_duration_avg"] = sum(durations) / len(durations)
    
    if patterns["successful_scores"]:
        scores = patterns["successful_scores"]
        features["minimum_score_threshold"] = min(scores)
        features["optimal_score_range"] = (min(scores), max(scores))
    
    if patterns["successful_reasons"]:
        reason_words = {}
        for reason in patterns["successful_reasons"]:
            for word in reason.split():
                if len(word) > 4:
                    reason_words[word] = reason_words.get(word, 0) + 1
        
        top_keywords = sorted(reason_words.items(), key=lambda x: x[1], reverse=True)[:10]
        features["successful_keywords"] = [word for word, _ in top_keywords]
    
    return features


async def update_prompt_version_metrics_online(
    db: AsyncSession,
    prompt_version_id: UUID,
    feedback_type: Optional[str] = None,
    rating: Optional[float] = None,
    is_save: bool = False,
) -> None:
    """
    Update prompt version rolling metrics online from a single feedback event.
    
    Args:
        db: Database session
        prompt_version_id: Prompt version ID
        feedback_type: Feedback type (RATING, POSITIVE, NEGATIVE) or None
        rating: Rating value (0-100) if feedback_type is RATING
        is_save: True if this is a save event
    """
    await crud.update_prompt_version_metrics_online(
        db,
        prompt_version_id=prompt_version_id,
        feedback_type=feedback_type,
        rating=rating,
        is_save=is_save,
    )


def calculate_highlight_quality_score(
    highlight,
    learned_features: Dict[str, any],
) -> float:
    """
    Calculate composite quality score based on learned features.
    
    Args:
        highlight: Highlight object
        learned_features: Learned features from feedback
        
    Returns:
        Quality-adjusted score
    """
    base_score = highlight.score
    duration = highlight.end - highlight.start
    
    adjustments = []
    
    if "preferred_duration_min" in learned_features:
        preferred_min = learned_features["preferred_duration_min"]
        preferred_max = learned_features["preferred_duration_max"]
        
        if preferred_min <= duration <= preferred_max:
            adjustments.append(0.5)
        elif duration < preferred_min * 0.5 or duration > preferred_max * 1.5:
            adjustments.append(-0.5)
    
    if "minimum_score_threshold" in learned_features:
        threshold = learned_features["minimum_score_threshold"]
        if base_score < threshold:
            adjustments.append(-1.0)
    
    adjustment = sum(adjustments)
    quality_score = base_score + adjustment
    
    return max(1.0, min(10.0, quality_score))

