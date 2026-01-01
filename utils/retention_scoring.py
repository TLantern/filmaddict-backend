import asyncio
import logging
import os
from typing import List, Dict, Optional, Tuple
import numpy as np

from models import Sentence
from models import (
    RetentionAnalysis,
    TimeRange,
    SemanticNoveltyMetrics,
    InformationDensityMetrics,
    EmotionalDeltaMetrics,
    NarrativeMomentumMetrics,
    RetentionMetrics,
    RetentionDecision,
)

logger = logging.getLogger(__name__)

# Configuration
RETENTION_THRESHOLD = float(os.getenv("RETENTION_THRESHOLD", "0.4"))
SEMANTIC_NOVELTY_WINDOW = int(os.getenv("SEMANTIC_NOVELTY_WINDOW", "10"))

# Filler words (from filler_detection.py)
FILLER_WORDS = {
    "um", "uh", "like", "you know", "basically", "so", "well",
    "actually", "literally", "kind of", "sort of", "i mean",
    "right", "okay", "ok", "yeah", "yep", "hmm", "er", "ah",
}

# Stopwords (common English stopwords)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
    "to", "was", "were", "will", "with", "the", "this", "but", "they",
    "have", "had", "what", "said", "each", "which", "their", "time",
    "if", "up", "out", "many", "then", "them", "these", "so", "some",
    "her", "would", "make", "like", "into", "him", "has", "two", "more",
    "very", "after", "words", "long", "than", "first", "been", "call",
    "who", "oil", "sit", "now", "find", "down", "day", "did", "get",
    "come", "made", "may", "part",
}


def compute_semantic_novelty(
    sentences: List[Sentence],
    embeddings: List[List[float]],
    window_size: int = SEMANTIC_NOVELTY_WINDOW,
) -> List[SemanticNoveltyMetrics]:
    """
    Compute semantic novelty by comparing embeddings to recent history.
    
    Args:
        sentences: List of Sentence objects
        embeddings: List of embedding vectors for each sentence
        window_size: Number of previous segments to compare against
        
    Returns:
        List of SemanticNoveltyMetrics objects
    """
    if not sentences or not embeddings:
        return []
    
    if len(sentences) != len(embeddings):
        raise ValueError("Sentences and embeddings must have the same length")
    
    logger.info(f"Computing semantic novelty for {len(sentences)} segments (window_size={window_size})")
    
    metrics_list = []
    total = len(sentences)
    
    for i in range(len(sentences)):
        if (i + 1) % 100 == 0 or i == 0:
            logger.debug(f"[Semantic Novelty] Progress: {i+1}/{total} segments ({100*(i+1)//total}%)")
        current_embedding = np.array(embeddings[i])
        
        # Get recent history (previous segments within window)
        start_idx = max(0, i - window_size)
        history_embeddings = [np.array(emb) for emb in embeddings[start_idx:i]]
        
        if not history_embeddings:
            # First segment or no history - maximum novelty
            max_similarity = 0.0
            novelty_value = 1.0
            window_size_actual = 1  # Ensure window_size >= 1
        else:
            window_size_actual = len(history_embeddings)
            # Compute cosine similarity to all history segments
            similarities = []
            for hist_emb in history_embeddings:
                # Cosine similarity
                dot_product = np.dot(current_embedding, hist_emb)
                norm_current = np.linalg.norm(current_embedding)
                norm_hist = np.linalg.norm(hist_emb)
                if norm_current > 0 and norm_hist > 0:
                    similarity = dot_product / (norm_current * norm_hist)
                    similarities.append(similarity)
            
            max_similarity = max(similarities) if similarities else 0.0
            # Novelty is inverse of similarity (higher similarity = lower novelty)
            novelty_value = 1.0 - max_similarity
        
        # Clamp values to [0, 1] range to handle floating point precision issues
        novelty_value = max(0.0, min(1.0, novelty_value))
        max_similarity = max(0.0, min(1.0, max_similarity))
        
        metrics = SemanticNoveltyMetrics(
            value=float(novelty_value),
            max_similarity_to_history=float(max_similarity),
            window_size=window_size_actual,
        )
        metrics_list.append(metrics)
    
    logger.info(f"Computed semantic novelty: {[f'{m.value:.2f}' for m in metrics_list]}")
    return metrics_list


def compute_information_density(
    sentences: List[Sentence],
    filler_densities: List[float],
) -> List[InformationDensityMetrics]:
    """
    Compute information density using TF-IDF weighted tokens, downweighting filler words and stopwords.
    
    Args:
        sentences: List of Sentence objects
        filler_densities: List of filler density scores (0.0-1.0)
        
    Returns:
        List of InformationDensityMetrics objects
    """
    if not sentences:
        return []
    
    logger.info(f"Computing information density for {len(sentences)} segments")
    total = len(sentences)
    
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        logger.warning("scikit-learn not available, using fallback information density calculation")
        return _compute_information_density_fallback(sentences, filler_densities)
    
    logger.info(f"[Information Density] Computing TF-IDF for {total} segments...")
    # Prepare texts
    texts = [sentence.text for sentence in sentences]
    
    # Custom tokenizer that filters filler words and stopwords
    def tokenize_and_filter(text: str) -> List[str]:
        import re
        tokens = re.findall(r'\b\w+\b', text.lower())
        # Filter out filler words and stopwords
        filtered = [t for t in tokens if t not in FILLER_WORDS and t not in STOPWORDS]
        return filtered
    
    # Compute TF-IDF
    try:
        vectorizer = TfidfVectorizer(
            tokenizer=tokenize_and_filter,
            token_pattern=None,  # Use custom tokenizer
            lowercase=True,
            min_df=1,  # Include all terms
            max_df=0.95,  # Exclude terms that appear in >95% of documents
        )
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # Get feature names and weights
        feature_names = vectorizer.get_feature_names_out()
        
        metrics_list = []
        for i, sentence in enumerate(sentences):
            if (i + 1) % 50 == 0 or i == 0:
                logger.info(f"[Information Density] Progress: {i+1}/{total} segments ({100*(i+1)//total}%)")
            
            # Get TF-IDF weights for this sentence
            row = tfidf_matrix[i]
            tfidf_weights = row.toarray()[0]
            
            # Sum of TF-IDF weights (information density)
            tfidf_sum = float(np.sum(tfidf_weights))
            
            # Count meaningful tokens (non-filler, non-stopword)
            tokens = tokenize_and_filter(sentence.text)
            meaningful_count = len(tokens)
            
            # Normalize by duration
            duration = sentence.end - sentence.start
            if duration > 0:
                density_value = tfidf_sum / duration
            else:
                density_value = 0.0
            
            # Normalize to 0-1 range (heuristic: divide by max expected value)
            # Typical good segment: 5-10 meaningful tokens, 0.5-2.0 TF-IDF sum per second
            normalized_density = min(1.0, density_value / 2.0)
            
            metrics = InformationDensityMetrics(
                value=float(normalized_density),
                meaningful_token_count=meaningful_count,
                tfidf_weight_sum=float(tfidf_sum),
                duration_seconds=float(duration),
            )
            metrics_list.append(metrics)
        
    except Exception as e:
        logger.warning(f"TF-IDF calculation failed: {e}, using fallback")
        return _compute_information_density_fallback(sentences, filler_densities)
    
    logger.info(f"Computed information density: {[f'{m.value:.2f}' for m in metrics_list]}")
    return metrics_list


def _compute_information_density_fallback(
    sentences: List[Sentence],
    filler_densities: List[float],
) -> List[InformationDensityMetrics]:
    """Fallback information density calculation without scikit-learn."""
    metrics_list = []
    
    for i, sentence in enumerate(sentences):
        import re
        tokens = re.findall(r'\b\w+\b', sentence.text.lower())
        meaningful_tokens = [t for t in tokens if t not in FILLER_WORDS and t not in STOPWORDS]
        meaningful_count = len(meaningful_tokens)
        
        duration = sentence.end - sentence.start
        if duration > 0:
            # Simple density: meaningful tokens per second
            density_value = meaningful_count / duration
            # Normalize (assume 5-10 tokens/second is good)
            normalized_density = min(1.0, density_value / 10.0)
        else:
            normalized_density = 0.0
        
        metrics = InformationDensityMetrics(
            value=float(normalized_density),
            meaningful_token_count=meaningful_count,
            tfidf_weight_sum=float(meaningful_count),  # Approximate
            duration_seconds=float(duration),
        )
        metrics_list.append(metrics)
    
    return metrics_list


def compute_emotional_delta(
    sentences: List[Sentence],
) -> List[EmotionalDeltaMetrics]:
    """
    Compute emotional delta using sentiment analysis, tracking sentiment and intensity changes.
    
    Args:
        sentences: List of Sentence objects
        
    Returns:
        List of EmotionalDeltaMetrics objects
    """
    if not sentences:
        return []
    
    logger.info(f"Computing emotional delta for {len(sentences)} segments")
    total = len(sentences)
    
    # Try to use transformers sentiment analysis
    sentiments = []
    intensities = []
    
    try:
        from transformers import pipeline
        
        logger.info(f"[Emotional Delta] Loading sentiment analysis model...")
        # Load sentiment analysis pipeline
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            device=-1,  # Use CPU by default
        )
        
        logger.info(f"[Emotional Delta] Analyzing sentiment for {total} segments...")
        texts = [sentence.text for sentence in sentences]
        results = sentiment_pipeline(texts)
        logger.info(f"[Emotional Delta] ✅ Sentiment analysis complete for {len(results)} segments")
        
        for result in results:
            label = result['label'].upper()
            score = result['score']
            
            # Map labels to sentiment scores (-1 to 1)
            if 'POSITIVE' in label:
                sentiment = score  # 0 to 1
            elif 'NEGATIVE' in label:
                sentiment = -score  # -1 to 0
            else:  # NEUTRAL
                sentiment = 0.0
            
            # Intensity is the absolute value of the score
            intensity = score
            
            sentiments.append(sentiment)
            intensities.append(intensity)
            
    except Exception as e:
        logger.warning(f"Sentiment analysis failed: {e}, using fallback")
        # Fallback: simple heuristic based on emotional words
        sentiments, intensities = _compute_sentiment_fallback(sentences)
    
    # Compute deltas
    logger.info(f"[Emotional Delta] Computing deltas for {total} segments...")
    metrics_list = []
    for i, sentence in enumerate(sentences):
        if (i + 1) % 50 == 0 or i == 0:
            logger.info(f"[Emotional Delta] Progress: {i+1}/{total} segments ({100*(i+1)//total}%)")
        if i == 0:
            # First segment: no previous segment
            sentiment_prev = 0.0
            intensity_prev = 0.5
        else:
            sentiment_prev = sentiments[i - 1]
            intensity_prev = intensities[i - 1]
        
        sentiment_curr = sentiments[i]
        intensity_curr = intensities[i]
        
        # Emotional delta: combination of sentiment change and intensity change
        sentiment_delta = sentiment_curr - sentiment_prev
        intensity_delta = intensity_curr - intensity_prev
        
        # Combined delta (normalized to -1 to 1, then to 0-1)
        emotional_delta = (sentiment_delta * 0.6 + intensity_delta * 0.4)
        # Normalize to 0-1 range
        normalized_delta = (emotional_delta + 1.0) / 2.0
        
        metrics = EmotionalDeltaMetrics(
            value=float(emotional_delta),  # Keep raw delta for reference
            sentiment_prev=float(sentiment_prev),
            sentiment_curr=float(sentiment_curr),
            intensity_prev=float(intensity_prev),
            intensity_curr=float(intensity_curr),
        )
        metrics_list.append(metrics)
    
    logger.info(f"Computed emotional delta: {[f'{m.value:.2f}' for m in metrics_list]}")
    return metrics_list


def _compute_sentiment_fallback(sentences: List[Sentence]) -> Tuple[List[float], List[float]]:
    """Fallback sentiment calculation using simple heuristics."""
    positive_words = {
        "good", "great", "excellent", "amazing", "wonderful", "fantastic",
        "love", "like", "happy", "joy", "excited", "best", "perfect",
        "awesome", "brilliant", "outstanding", "superb", "incredible",
    }
    negative_words = {
        "bad", "terrible", "awful", "horrible", "worst", "hate", "dislike",
        "sad", "angry", "frustrated", "disappointed", "failed", "problem",
        "issue", "error", "mistake", "wrong", "broken",
    }
    
    sentiments = []
    intensities = []
    
    for sentence in sentences:
        text_lower = sentence.text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        total_emotional_words = positive_count + negative_count
        if total_emotional_words > 0:
            sentiment = (positive_count - negative_count) / total_emotional_words
            intensity = min(1.0, total_emotional_words / 5.0)  # Normalize
        else:
            sentiment = 0.0
            intensity = 0.3  # Baseline intensity
        
        sentiments.append(sentiment)
        intensities.append(intensity)
    
    return sentiments, intensities


def compute_narrative_momentum(
    sentences: List[Sentence],
) -> List[NarrativeMomentumMetrics]:
    """
    Compute narrative momentum by extracting new entities, events, goals, and stakes.
    
    Args:
        sentences: List of Sentence objects
        
    Returns:
        List of NarrativeMomentumMetrics objects
    """
    if not sentences:
        return []
    
    logger.info(f"Computing narrative momentum for {len(sentences)} segments")
    total = len(sentences)
    
    # Try to use spaCy for NER
    try:
        import spacy
        logger.info(f"[Narrative Momentum] Loading spaCy model...")
        nlp = spacy.load("en_core_web_sm")
        use_spacy = True
        logger.info(f"[Narrative Momentum] ✅ spaCy model loaded")
    except (ImportError, OSError):
        logger.warning("spaCy not available, using fallback NER")
        use_spacy = False
        nlp = None
    
    # Track entities, events, goals, stakes across segments
    all_entities = set()
    all_events = set()
    all_goals = set()
    all_stakes = set()
    
    # Keywords for events, goals, stakes
    event_keywords = {
        "happened", "occurred", "started", "ended", "began", "finished",
        "changed", "moved", "went", "came", "arrived", "left", "created",
        "destroyed", "built", "broke", "fixed", "solved", "failed",
    }
    goal_keywords = {
        "want", "need", "need", "goal", "objective", "aim", "target",
        "plan", "intend", "trying", "attempt", "strive", "purpose",
    }
    stakes_keywords = {
        "risk", "danger", "threat", "stake", "consequence", "impact",
        "critical", "important", "crucial", "vital", "essential",
        "urgent", "emergency", "problem", "issue", "challenge",
    }
    
    logger.info(f"[Narrative Momentum] Extracting entities/events/goals/stakes for {total} segments...")
    metrics_list = []
    
    for i, sentence in enumerate(sentences):
        if (i + 1) % 50 == 0 or i == 0:
            logger.info(f"[Narrative Momentum] Progress: {i+1}/{total} segments ({100*(i+1)//total}%)")
        
        text = sentence.text
        
        # Extract entities
        if use_spacy and nlp:
            doc = nlp(text)
            entities = {ent.text.lower() for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE", "PRODUCT", "EVENT"]}
        else:
            # Fallback: extract capitalized words (likely proper nouns)
            import re
            capitalized = re.findall(r'\b[A-Z][a-z]+\b', text)
            entities = {e.lower() for e in capitalized}
        
        # Extract events, goals, stakes using keywords
        text_lower = text.lower()
        events = {kw for kw in event_keywords if kw in text_lower}
        goals = {kw for kw in goal_keywords if kw in text_lower}
        stakes = {kw for kw in stakes_keywords if kw in text_lower}
        
        # Count new items (not seen in previous segments)
        new_entities = len(entities - all_entities)
        new_events = len(events - all_events)
        new_goals = len(goals - all_goals)
        new_stakes = len(stakes - all_stakes)
        
        # Update tracking sets
        all_entities.update(entities)
        all_events.update(events)
        all_goals.update(goals)
        all_stakes.update(stakes)
        
        # Compute momentum score (normalized)
        # Each new item contributes, with diminishing returns
        momentum_value = min(1.0, (
            new_entities * 0.3 +
            new_events * 0.3 +
            new_goals * 0.2 +
            new_stakes * 0.2
        ))
        
        metrics = NarrativeMomentumMetrics(
            value=float(momentum_value),
            new_entities=new_entities,
            new_events=new_events,
            new_goals=new_goals,
            new_stakes=new_stakes,
        )
        metrics_list.append(metrics)
    
    logger.info(f"Computed narrative momentum: {[f'{m.value:.2f}' for m in metrics_list]}")
    return metrics_list


def compute_retention_value(
    semantic_novelty: List[SemanticNoveltyMetrics],
    information_density: List[InformationDensityMetrics],
    emotional_delta: List[EmotionalDeltaMetrics],
    narrative_momentum: List[NarrativeMomentumMetrics],
) -> List[float]:
    """
    Compute retention value using weighted formula:
    RetentionValue = 0.40*N + 0.25*I + 0.20*E + 0.15*M
    
    Args:
        semantic_novelty: List of semantic novelty metrics
        information_density: List of information density metrics
        emotional_delta: List of emotional delta metrics
        narrative_momentum: List of narrative momentum metrics
        
    Returns:
        List of retention values (0.0-1.0)
    """
    if not semantic_novelty or not information_density or not emotional_delta or not narrative_momentum:
        return []
    
    if len(semantic_novelty) != len(information_density) or \
       len(information_density) != len(emotional_delta) or \
       len(emotional_delta) != len(narrative_momentum):
        raise ValueError("All metric lists must have the same length")
    
    retention_values = []
    
    for i in range(len(semantic_novelty)):
        N = semantic_novelty[i].value
        I = information_density[i].value
        
        # Normalize emotional delta to 0-1 range
        E_raw = emotional_delta[i].value
        E = (E_raw + 1.0) / 2.0  # Map from [-1, 1] to [0, 1]
        
        M = narrative_momentum[i].value
        
        # Weighted formula
        retention_value = 0.40 * N + 0.25 * I + 0.20 * E + 0.15 * M
        
        # Clamp to [0, 1]
        retention_value = max(0.0, min(1.0, retention_value))
        
        retention_values.append(retention_value)
    
    logger.info(f"Computed retention values: {[f'{v:.2f}' for v in retention_values]}")
    return retention_values


def make_retention_decision(
    retention_value: float,
    metrics: RetentionMetrics,
    threshold: float = RETENTION_THRESHOLD,
) -> RetentionDecision:
    """
    Make retention decision based on threshold.
    
    Args:
        retention_value: Retention value (0.0-1.0)
        metrics: RetentionMetrics object
        threshold: Threshold for CUT decision (default: 0.4)
        
    Returns:
        RetentionDecision object
    """
    if retention_value < threshold:
        action = "CUT"
        reason = "Retention value below threshold"
    else:
        action = "KEEP"
        # Generate reason based on which metrics contributed most
        reasons = []
        if metrics.semantic_novelty.value > 0.7:
            reasons.append("high semantic novelty")
        if metrics.information_density.value > 0.7:
            reasons.append("strong information density")
        if metrics.emotional_delta.value > 0.5:
            reasons.append("positive emotional progression")
        if metrics.narrative_momentum.value > 0.6:
            reasons.append("strong narrative progression")
        
        if reasons:
            reason = "High " + " and ".join(reasons) + "."
        else:
            reason = "Retention value above threshold."
    
    return RetentionDecision(action=action, reason=reason)


async def compute_semantic_novelty_async(
    sentences: List[Sentence],
    embeddings: List[List[float]],
    window_size: int = SEMANTIC_NOVELTY_WINDOW,
) -> List[SemanticNoveltyMetrics]:
    """Async wrapper for compute_semantic_novelty."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, compute_semantic_novelty, sentences, embeddings, window_size)


async def compute_information_density_async(
    sentences: List[Sentence],
    filler_densities: List[float],
) -> List[InformationDensityMetrics]:
    """Async wrapper for compute_information_density."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, compute_information_density, sentences, filler_densities)


async def compute_emotional_delta_async(
    sentences: List[Sentence],
) -> List[EmotionalDeltaMetrics]:
    """Async wrapper for compute_emotional_delta."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, compute_emotional_delta, sentences)


async def compute_narrative_momentum_async(
    sentences: List[Sentence],
) -> List[NarrativeMomentumMetrics]:
    """Async wrapper for compute_narrative_momentum."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, compute_narrative_momentum, sentences)


async def compute_retention_scores_async(
    sentences: List[Sentence],
    embeddings: List[List[float]],
    filler_densities: List[float],
    video_id: str = "",
) -> List[RetentionAnalysis]:
    """
    Async function to compute retention scores for all segments using parallel execution.
    
    Args:
        sentences: List of Sentence objects
        embeddings: List of embedding vectors for each sentence
        filler_densities: List of filler density scores (0.0-1.0)
        
    Returns:
        List of RetentionAnalysis objects
    """
    if not sentences:
        return []
    
    if len(sentences) != len(embeddings) or len(sentences) != len(filler_densities):
        raise ValueError("Sentences, embeddings, and filler_densities must have the same length")
    
    logger.info(f"Computing retention scores for {len(sentences)} segments (parallel execution)")
    total = len(sentences)
    
    # Compute metrics in parallel
    # Emotional Delta and Narrative Momentum can start immediately (only need sentences)
    # Semantic Novelty needs embeddings, Information Density needs filler_densities
    logger.info(f"[Retention Scoring] Starting parallel metric computation...")
    
    # Start independent metrics immediately
    emotional_delta_task = compute_emotional_delta_async(sentences)
    narrative_momentum_task = compute_narrative_momentum_async(sentences)
    
    # Start dependent metrics (they should already be ready)
    semantic_novelty_task = compute_semantic_novelty_async(sentences, embeddings)
    information_density_task = compute_information_density_async(sentences, filler_densities)
    
    # Wait for all metrics to complete in parallel
    semantic_novelty_metrics, information_density_metrics, emotional_delta_metrics, narrative_momentum_metrics = await asyncio.gather(
        semantic_novelty_task,
        information_density_task,
        emotional_delta_task,
        narrative_momentum_task
    )
    
    logger.info(f"[Retention Scoring] ✅ All metrics computed in parallel")
    
    # Compute retention values
    logger.info(f"[Retention Scoring] Computing final retention values...")
    retention_values = compute_retention_value(
        semantic_novelty_metrics,
        information_density_metrics,
        emotional_delta_metrics,
        narrative_momentum_metrics,
    )
    logger.info(f"[Retention Scoring] ✅ Retention values computed")
    
    # Create RetentionAnalysis objects
    logger.info(f"[Retention Scoring] Creating retention analysis objects...")
    analyses = []
    for i, sentence in enumerate(sentences):
        if (i + 1) % 20 == 0 or i == 0:
            logger.info(f"[Retention Scoring] Progress: {i+1}/{total} analyses created ({100*(i+1)//total}%)")
        metrics = RetentionMetrics(
            semantic_novelty=semantic_novelty_metrics[i],
            information_density=information_density_metrics[i],
            emotional_delta=emotional_delta_metrics[i],
            narrative_momentum=narrative_momentum_metrics[i],
        )
        
        decision = make_retention_decision(retention_values[i], metrics)
        
        time_range = TimeRange(
            start=sentence.start,
            end=sentence.end,
            duration=sentence.end - sentence.start,
        )
        
        analysis = RetentionAnalysis(
            video_id=video_id,
            segment_id=i + 1,
            time_range=time_range,
            text=sentence.text,
            metrics=metrics,
            retention_value=retention_values[i],
            decision=decision,
        )
        analyses.append(analysis)
    
    logger.info(f"Computed {len(analyses)} retention analyses")
    return analyses


def compute_retention_scores(
    sentences: List[Sentence],
    embeddings: List[List[float]],
    filler_densities: List[float],
    video_id: str = "",
) -> List[RetentionAnalysis]:
    """
    Synchronous wrapper for compute_retention_scores_async.
    For use in synchronous contexts. In async contexts, use compute_retention_scores_async directly.
    
    Args:
        sentences: List of Sentence objects
        embeddings: List of embedding vectors for each sentence
        filler_densities: List of filler density scores (0.0-1.0)
        
    Returns:
        List of RetentionAnalysis objects
    """
    return asyncio.run(compute_retention_scores_async(sentences, embeddings, filler_densities, video_id))

