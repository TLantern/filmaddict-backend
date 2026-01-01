import logging
import os
from typing import List, Dict, Optional, Tuple, Mapping
import numpy as np
from uuid import UUID

from models import TranscriptSegment, Sentence, SilenceSegment

logger = logging.getLogger(__name__)

# Configuration
MOMENT_WINDOW_MIN = float(os.getenv("MOMENT_WINDOW_MIN", "30.0"))
MOMENT_WINDOW_MAX = float(os.getenv("MOMENT_WINDOW_MAX", "60.0"))
MOMENT_TOP_K = int(os.getenv("MOMENT_TOP_K", "10"))
MOMENT_NOVELTY_HISTORY_WINDOW = int(os.getenv("MOMENT_NOVELTY_HISTORY_WINDOW", "5"))


class MomentWindow:
    """Represents a time window for moment scoring."""
    def __init__(
        self,
        start: float,
        end: float,
        text: str,
        segment_indices: List[int],
        sentences: List[Sentence],
    ):
        self.start = start
        self.end = end
        self.text = text
        self.segment_indices = segment_indices
        self.sentences = sentences
        self.duration = end - start


class MomentScoreComponents:
    """Components of the moment score formula."""
    def __init__(
        self,
        novelty: float,
        repetition: float,
        emotional_delta: float,
        information_density: float,
        filler_density: float,
        silence_penalty: float,
    ):
        self.novelty = novelty
        self.repetition = repetition
        self.emotional_delta = emotional_delta
        self.information_density = information_density
        self.filler_density = filler_density
        self.silence_penalty = silence_penalty


class MomentScore:
    """Complete moment score with metadata."""
    def __init__(
        self,
        score: float,
        window: MomentWindow,
        components: MomentScoreComponents,
        justification: str,
    ):
        self.score = score
        self.window = window
        self.components = components
        self.justification = justification


class MomentScoreResult:
    """Result containing top moments."""
    def __init__(
        self,
        video_id: UUID,
        top_moments: List[MomentScore],
    ):
        self.video_id = video_id
        self.top_moments = top_moments


def create_sliding_windows(
    transcript_segments: List[TranscriptSegment],
    sentences: List[Sentence],
    min_window: float = MOMENT_WINDOW_MIN,
    max_window: float = MOMENT_WINDOW_MAX,
) -> List[MomentWindow]:
    """
    Create sliding windows (30-60s) from transcript segments.
    
    Windows respect sentence boundaries and don't cut sentences mid-way.
    Uses adaptive window sizing within the min-max range.
    
    Args:
        transcript_segments: List of transcript segments
        sentences: List of Sentence objects (must align with segments)
        min_window: Minimum window size in seconds (default: 30.0)
        max_window: Maximum window size in seconds (default: 60.0)
        
    Returns:
        List of MomentWindow objects
    """
    if not transcript_segments or not sentences:
        return []
    
    # If video is shorter than min_window, create a single window
    if transcript_segments:
        video_duration = max(seg.end for seg in transcript_segments)
        if video_duration <= min_window:
            # Create single window for entire video
            text = " ".join(seg.text for seg in transcript_segments)
            segment_indices = list(range(len(transcript_segments)))
            window_sentences = [s for s in sentences if any(
                seg.start <= s.start < seg.end or seg.start < s.end <= seg.end
                for seg in transcript_segments
            )]
            return [MomentWindow(
                start=transcript_segments[0].start,
                end=video_duration,
                text=text,
                segment_indices=segment_indices,
                sentences=window_sentences,
            )]
    
    windows = []
    current_start = transcript_segments[0].start if transcript_segments else 0.0
    current_segment_idx = 0
    step_size = min_window / 2  # 50% overlap for sliding windows
    
    while current_segment_idx < len(transcript_segments):
        # Collect segments until we reach target window size
        window_segments = []
        window_segment_indices = []
        window_end = current_start + min_window
        
        # First, try to reach min_window
        while current_segment_idx < len(transcript_segments):
            seg = transcript_segments[current_segment_idx]
            if seg.end <= window_end:
                window_segments.append(seg)
                window_segment_indices.append(current_segment_idx)
                current_segment_idx += 1
            elif seg.start < window_end:
                # Segment overlaps with window, include it
                window_segments.append(seg)
                window_segment_indices.append(current_segment_idx)
                window_end = seg.end
                current_segment_idx += 1
                break
            else:
                break
        
        # If we haven't reached min_window, extend to max_window
        while len(window_segments) > 0 and window_end - current_start < min_window:
            if current_segment_idx < len(transcript_segments):
                seg = transcript_segments[current_segment_idx]
                if seg.start < window_end + (max_window - (window_end - current_start)):
                    window_segments.append(seg)
                    window_segment_indices.append(current_segment_idx)
                    window_end = max(window_end, seg.end)
                    current_segment_idx += 1
                else:
                    break
            else:
                break
        
        # Cap window at max_window
        if window_end - current_start > max_window:
            window_end = current_start + max_window
            # Trim segments that extend beyond max_window
            window_segments = [s for s in window_segments if s.start < window_end]
            if window_segments:
                window_end = max(s.end for s in window_segments if s.end <= window_end)
        
        if not window_segments:
            break
        
        # Get sentences that fall within this window
        window_sentences = [
            s for s in sentences
            if any(
                (seg.start <= s.start < seg.end or seg.start < s.end <= seg.end or
                 (s.start <= seg.start and s.end >= seg.end))
                for seg in window_segments
            )
        ]
        
        # Combine text
        window_text = " ".join(seg.text for seg in window_segments)
        
        # Create window
        window = MomentWindow(
            start=current_start,
            end=window_end,
            text=window_text,
            segment_indices=window_segment_indices,
            sentences=window_sentences,
        )
        windows.append(window)
        
        # Move to next window (sliding)
        current_start += step_size
        
        # Advance to next segment that hasn't been included
        if current_segment_idx < len(transcript_segments):
            # Find first segment that starts after current_start
            while current_segment_idx < len(transcript_segments):
                if transcript_segments[current_segment_idx].start >= current_start:
                    break
                current_segment_idx += 1
        
        # If we've processed all segments, break
        if current_segment_idx >= len(transcript_segments):
            break
    
    logger.info(f"Created {len(windows)} sliding windows (min={min_window}s, max={max_window}s)")
    return windows


def compute_window_novelty(
    window: MomentWindow,
    window_embedding: List[float],
    recent_windows: List[MomentWindow],
    recent_embeddings: List[List[float]],
) -> float:
    """
    Compute novelty by comparing window embedding to recent history.
    
    Args:
        window: Current window
        window_embedding: Embedding vector for the current window
        recent_windows: List of recent windows (history)
        recent_embeddings: List of embeddings for recent windows
        
    Returns:
        Novelty score (0.0-1.0), where 1.0 is maximum novelty
    """
    if not recent_windows or not recent_embeddings:
        return 1.0  # Maximum novelty if no history
    
    if len(recent_windows) != len(recent_embeddings):
        raise ValueError("recent_windows and recent_embeddings must have same length")
    
    current_emb = np.array(window_embedding)
    
    # Compute cosine similarity to all recent windows
    max_similarity = 0.0
    for hist_emb in recent_embeddings:
        hist_emb_array = np.array(hist_emb)
        dot_product = np.dot(current_emb, hist_emb_array)
        norm_current = np.linalg.norm(current_emb)
        norm_hist = np.linalg.norm(hist_emb_array)
        if norm_current > 0 and norm_hist > 0:
            similarity = dot_product / (norm_current * norm_hist)
            max_similarity = max(max_similarity, similarity)
    
    # Novelty is inverse of similarity
    novelty = 1.0 - max_similarity
    return max(0.0, min(1.0, novelty))  # Clamp to [0, 1]


def compute_window_repetition(
    window: MomentWindow,
    sentence_repetition_scores: List[float],
    sentence_to_index_map: Mapping[Tuple[float, float, int], int],
) -> float:
    """
    Compute repetition score for a window by aggregating sentence-level scores.
    
    Args:
        window: Current window
        sentence_repetition_scores: List of repetition scores for all sentences
        sentence_to_index_map: Mapping from Sentence objects to their indices
        
    Returns:
        Repetition score (0.0-1.0)
    """
    if not window.sentences:
        return 0.0
    
    scores = []
    for sentence in window.sentences:
        key = (sentence.start, sentence.end, hash(sentence.text))
        if key in sentence_to_index_map:
            idx = sentence_to_index_map[key]
            if idx < len(sentence_repetition_scores):
                scores.append(sentence_repetition_scores[idx])
    
    if not scores:
        return 0.0
    
    # Use average repetition score
    return sum(scores) / len(scores)


def compute_window_emotional_delta(
    window_sentiment: float,
    window_intensity: float,
    prior_window_sentiment: Optional[float],
    prior_window_intensity: Optional[float],
) -> float:
    """
    Compute emotional delta by comparing current window to prior window.
    
    Args:
        window_sentiment: Current window sentiment (-1 to 1)
        window_intensity: Current window intensity (0 to 1)
        prior_window_sentiment: Prior window sentiment (None if first window)
        prior_window_intensity: Prior window intensity (None if first window)
        
    Returns:
        Emotional delta (0.0-1.0, normalized)
    """
    if prior_window_sentiment is None or prior_window_intensity is None:
        # First window: use intensity as baseline
        return window_intensity
    
    # Compute deltas
    sentiment_delta = window_sentiment - prior_window_sentiment
    intensity_delta = window_intensity - prior_window_intensity
    
    # Combined delta (weighted)
    emotional_delta = (sentiment_delta * 0.6 + intensity_delta * 0.4)
    
    # Normalize from [-1, 1] to [0, 1]
    normalized_delta = (emotional_delta + 1.0) / 2.0
    return max(0.0, min(1.0, normalized_delta))


def compute_window_information_density(
    window: MomentWindow,
    all_window_texts: List[str],
) -> float:
    """
    Compute information density using TF-IDF on window-level text.
    
    Args:
        window: Current window
        all_window_texts: List of all window texts for TF-IDF computation
        
    Returns:
        Information density score (0.0-1.0)
    """
    if not window.text or not all_window_texts:
        return 0.0
    
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        logger.warning("scikit-learn not available, using fallback information density")
        return _compute_information_density_fallback(window)
    
    try:
        # Filler words and stopwords (same as retention_scoring.py)
        FILLER_WORDS = {
            "um", "uh", "like", "you know", "basically", "so", "well",
            "actually", "literally", "kind of", "sort of", "i mean",
            "right", "okay", "ok", "yeah", "yep", "hmm", "er", "ah",
        }
        STOPWORDS = {
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
            "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
            "to", "was", "were", "will", "with", "this", "but", "they",
            "have", "had", "what", "said", "each", "which", "their", "time",
        }
        
        def tokenize_and_filter(text: str) -> List[str]:
            import re
            tokens = re.findall(r'\b\w+\b', text.lower())
            filtered = [t for t in tokens if t not in FILLER_WORDS and t not in STOPWORDS]
            return filtered
        
        # Compute TF-IDF
        vectorizer = TfidfVectorizer(
            tokenizer=tokenize_and_filter,
            token_pattern=None,
            lowercase=True,
            min_df=1,
            max_df=0.95,
        )
        tfidf_matrix = vectorizer.fit_transform(all_window_texts)
        
        # Find index of current window
        window_idx = all_window_texts.index(window.text) if window.text in all_window_texts else 0
        if window_idx >= tfidf_matrix.shape[0]:
            window_idx = 0
        
        # Get TF-IDF weights for this window
        row = tfidf_matrix[window_idx]
        tfidf_weights = row.toarray()[0]
        tfidf_sum = float(np.sum(tfidf_weights))
        
        # Normalize by duration (typical good window: 2.0 TF-IDF sum per second)
        if window.duration > 0:
            density_value = tfidf_sum / window.duration
            normalized_density = min(1.0, density_value / 2.0)
        else:
            normalized_density = 0.0
        
        return normalized_density
        
    except Exception as e:
        logger.warning(f"TF-IDF calculation failed: {e}, using fallback")
        return _compute_information_density_fallback(window)


def _compute_information_density_fallback(window: MomentWindow) -> float:
    """Fallback information density calculation without scikit-learn."""
    import re
    FILLER_WORDS = {
        "um", "uh", "like", "you know", "basically", "so", "well",
        "actually", "literally", "kind of", "sort of", "i mean",
        "right", "okay", "ok", "yeah", "yep", "hmm", "er", "ah",
    }
    STOPWORDS = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
        "to", "was", "were", "will", "with", "this", "but", "they",
    }
    
    tokens = re.findall(r'\b\w+\b', window.text.lower())
    meaningful_tokens = [t for t in tokens if t not in FILLER_WORDS and t not in STOPWORDS]
    meaningful_count = len(meaningful_tokens)
    
    if window.duration > 0:
        density_value = meaningful_count / window.duration
        normalized_density = min(1.0, density_value / 10.0)  # Assume 10 tokens/second is good
    else:
        normalized_density = 0.0
    
    return normalized_density


def compute_window_filler_density(
    window: MomentWindow,
    sentence_filler_densities: List[float],
    sentence_to_index_map: Mapping[Tuple[float, float, int], int],
) -> float:
    """
    Compute filler density for a window.
    
    Args:
        window: Current window
        sentence_filler_densities: List of filler densities for all sentences
        sentence_to_index_map: Mapping from Sentence objects to their indices
        
    Returns:
        Filler density (fillers per second, normalized 0.0-1.0)
    """
    if not window.sentences or window.duration <= 0:
        return 0.0
    
    # Aggregate filler counts (assuming filler_density is already normalized per sentence)
    total_filler_density = 0.0
    count = 0
    for sentence in window.sentences:
        key = (sentence.start, sentence.end, hash(sentence.text))
        if key in sentence_to_index_map:
            idx = sentence_to_index_map[key]
            if idx < len(sentence_filler_densities):
                total_filler_density += sentence_filler_densities[idx]
                count += 1
    
    if count == 0:
        return 0.0
    
    # Average filler density (already normalized 0-1)
    avg_filler_density = total_filler_density / count
    return avg_filler_density


def compute_window_silence_penalty(
    window: MomentWindow,
    silence_segments: List[SilenceSegment],
) -> float:
    """
    Compute silence penalty for a window.
    
    Args:
        window: Current window
        silence_segments: List of silence segments in the video
        
    Returns:
        Silence penalty (0.0-1.0), where 1.0 is maximum penalty
    """
    if not silence_segments:
        return 0.0
    
    # Find overlapping silence segments
    total_silence_duration = 0.0
    for silence in silence_segments:
        # Check if silence overlaps with window
        overlap_start = max(window.start, silence.start)
        overlap_end = min(window.end, silence.end)
        if overlap_start < overlap_end:
            total_silence_duration += overlap_end - overlap_start
    
    # Normalize by window duration (penalty is ratio of silence)
    if window.duration > 0:
        silence_penalty = min(1.0, total_silence_duration / window.duration)
    else:
        silence_penalty = 0.0
    
    return silence_penalty


def _get_window_embedding(
    window: MomentWindow,
    sentence_embeddings: List[List[float]],
    sentence_to_index_map: Mapping[Tuple[float, float, int], int],
) -> List[float]:
    """Compute window embedding as average of sentence embeddings in the window."""
    if not window.sentences:
        # Return zero embedding if no sentences
        if sentence_embeddings:
            return [0.0] * len(sentence_embeddings[0])
        return []
    
    embeddings = []
    for sentence in window.sentences:
        key = (sentence.start, sentence.end, hash(sentence.text))
        if key in sentence_to_index_map:
            idx = sentence_to_index_map[key]
            if idx < len(sentence_embeddings):
                embeddings.append(sentence_embeddings[idx])
    
    if not embeddings:
        if sentence_embeddings:
            return [0.0] * len(sentence_embeddings[0])
        return []
    
    # Average embeddings
    avg_embedding = np.mean([np.array(emb) for emb in embeddings], axis=0)
    return avg_embedding.tolist()


def _compute_window_sentiment(
    window: MomentWindow,
    all_sentences: List[Sentence],
    sentence_sentiments: List[float],
    sentence_intensities: List[float],
    sentence_to_index_map: Mapping[Tuple[float, float, int], int],
) -> Tuple[float, float]:
    """
    Compute average sentiment and intensity for a window.
    
    Returns:
        Tuple of (sentiment, intensity)
    """
    if not window.sentences:
        return (0.0, 0.5)
    
    sentiments = []
    intensities = []
    
    for sentence in window.sentences:
        key = (sentence.start, sentence.end, hash(sentence.text))
        if key in sentence_to_index_map:
            idx = sentence_to_index_map[key]
            if idx < len(sentence_sentiments) and idx < len(sentence_intensities):
                sentiments.append(sentence_sentiments[idx])
                intensities.append(sentence_intensities[idx])
    
    if not sentiments:
        return (0.0, 0.5)
    
    avg_sentiment = sum(sentiments) / len(sentiments)
    avg_intensity = sum(intensities) / len(intensities)
    
    return (avg_sentiment, avg_intensity)


def generate_moment_justification(
    window: MomentWindow,
    components: MomentScoreComponents,
) -> str:
    """
    Generate a brief justification explaining why this moment scored well.
    
    Args:
        window: The window being scored
        components: Score components
        
    Returns:
        Brief justification text
    """
    reasons = []
    
    if components.novelty > 0.7:
        reasons.append("high novelty")
    if components.repetition < 0.3:
        reasons.append("low repetition")
    if components.emotional_delta > 0.6:
        reasons.append("strong emotional shift")
    if components.information_density > 0.7:
        reasons.append("high information density")
    if components.filler_density < 0.2:
        reasons.append("minimal fillers")
    if components.silence_penalty < 0.1:
        reasons.append("little silence")
    
    if reasons:
        return f"Scored highly due to: {', '.join(reasons)}."
    else:
        return "Moderate score across all metrics."


def compute_moment_scores(
    video_id: UUID,
    transcript_segments: List[TranscriptSegment],
    sentences: List[Sentence],
    embeddings: List[List[float]],
    filler_densities: List[float],
    repetition_scores: List[float],
    silence_segments: List[SilenceSegment],
    top_k: int = MOMENT_TOP_K,
) -> MomentScoreResult:
    """
    Main function to compute moment scores for video windows.
    
    Applies formula:
    MomentScore = (Novelty × (1 - Repetition) × EmotionalDelta × InformationDensity) / (1 + FillerDensity + SilencePenalty)
    
    Args:
        video_id: Video ID
        transcript_segments: List of transcript segments
        sentences: List of Sentence objects
        embeddings: List of embeddings for each sentence
        filler_densities: List of filler densities for each sentence
        repetition_scores: List of repetition scores for each sentence
        silence_segments: List of silence segments
        top_k: Number of top moments to return (default: 10)
        
    Returns:
        MomentScoreResult with top moments
    """
    if not transcript_segments or not sentences:
        logger.warning(f"No transcript segments or sentences for video {video_id}")
        return MomentScoreResult(video_id=video_id, top_moments=[])
    
    if len(sentences) != len(embeddings) or len(sentences) != len(filler_densities) or len(sentences) != len(repetition_scores):
        raise ValueError("sentences, embeddings, filler_densities, and repetition_scores must have the same length")
    
    logger.info(f"Computing moment scores for video {video_id} ({len(transcript_segments)} segments, {len(sentences)} sentences)")
    
    # Create sentence-to-index mapping using hashable key (start, end, text hash)
    sentence_to_index_map = {(sentence.start, sentence.end, hash(sentence.text)): i for i, sentence in enumerate(sentences)}
    
    # Compute sentiment and intensity for all sentences (if not already computed)
    # For now, we'll compute them using the emotional delta logic from retention_scoring
    try:
        from utils.retention_scoring import compute_emotional_delta
        emotional_delta_metrics = compute_emotional_delta(sentences)
        sentence_sentiments = []
        sentence_intensities = []
        for metrics in emotional_delta_metrics:
            # Use current sentiment and intensity
            sentence_sentiments.append(metrics.sentiment_curr)
            sentence_intensities.append(metrics.intensity_curr)
    except Exception as e:
        logger.warning(f"Failed to compute emotional metrics: {e}, using defaults")
        sentence_sentiments = [0.0] * len(sentences)
        sentence_intensities = [0.5] * len(sentences)
    
    # Create sliding windows
    logger.info(f"Creating sliding windows...")
    windows = create_sliding_windows(transcript_segments, sentences)
    
    if not windows:
        logger.warning(f"No windows created for video {video_id}")
        return MomentScoreResult(video_id=video_id, top_moments=[])
    
    # Compute embeddings for all windows
    logger.info(f"Computing window embeddings...")
    window_embeddings = []
    for window in windows:
        window_emb = _get_window_embedding(window, embeddings, sentence_to_index_map)
        window_embeddings.append(window_emb)
    
    # Get all window texts for TF-IDF
    all_window_texts = [w.text for w in windows]
    
    # Compute scores for each window
    logger.info(f"Computing scores for {len(windows)} windows...")
    moment_scores = []
    
    for i, window in enumerate(windows):
        # Get recent windows for novelty computation
        history_start_idx = max(0, i - MOMENT_NOVELTY_HISTORY_WINDOW)
        recent_windows = windows[history_start_idx:i]
        recent_embeddings = window_embeddings[history_start_idx:i]
        
        # Compute novelty
        novelty = compute_window_novelty(
            window,
            window_embeddings[i],
            recent_windows,
            recent_embeddings,
        )
        
        # Compute repetition
        repetition = compute_window_repetition(
            window,
            repetition_scores,
            sentence_to_index_map,
        )
        
        # Compute emotional delta
        window_sentiment, window_intensity = _compute_window_sentiment(
            window,
            sentences,
            sentence_sentiments,
            sentence_intensities,
            sentence_to_index_map,
        )
        prior_sentiment = None
        prior_intensity = None
        if i > 0:
            prior_sentiment, prior_intensity = _compute_window_sentiment(
                windows[i-1],
                sentences,
                sentence_sentiments,
                sentence_intensities,
                sentence_to_index_map,
            )
        emotional_delta = compute_window_emotional_delta(
            window_sentiment,
            window_intensity,
            prior_sentiment,
            prior_intensity,
        )
        
        # Compute information density
        information_density = compute_window_information_density(
            window,
            all_window_texts,
        )
        
        # Compute filler density
        filler_density = compute_window_filler_density(
            window,
            filler_densities,
            sentence_to_index_map,
        )
        
        # Compute silence penalty
        silence_penalty = compute_window_silence_penalty(
            window,
            silence_segments,
        )
        
        # Compute final score using formula
        # MomentScore = (Novelty × (1 - Repetition) × EmotionalDelta × InformationDensity) / (1 + FillerDensity + SilencePenalty)
        numerator = novelty * (1.0 - repetition) * emotional_delta * information_density
        denominator = 1.0 + filler_density + silence_penalty
        score = numerator / denominator if denominator > 0 else 0.0
        
        # Create components object
        components = MomentScoreComponents(
            novelty=novelty,
            repetition=repetition,
            emotional_delta=emotional_delta,
            information_density=information_density,
            filler_density=filler_density,
            silence_penalty=silence_penalty,
        )
        
        # Generate justification
        justification = generate_moment_justification(window, components)
        
        # Create moment score
        moment_score = MomentScore(
            score=score,
            window=window,
            components=components,
            justification=justification,
        )
        moment_scores.append(moment_score)
    
    # Rank by score (highest first)
    moment_scores.sort(key=lambda ms: ms.score, reverse=True)
    
    # Take top K
    top_moments = moment_scores[:top_k]
    
    logger.info(f"Computed {len(moment_scores)} moment scores, returning top {len(top_moments)}")
    for i, moment in enumerate(top_moments):
        logger.info(f"  Top {i+1}: [{moment.window.start:.1f}s-{moment.window.end:.1f}s] score={moment.score:.3f} - {moment.justification}")
    
    return MomentScoreResult(video_id=video_id, top_moments=top_moments)

