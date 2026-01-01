import logging
import re
from typing import List

from models import Sentence, SemanticSegment

logger = logging.getLogger(__name__)

# Sentence start heuristics configuration
SENTENCE_START_MARKERS = {
    "so", "then", "now", "but", "and", "because", "however", "therefore", "also",
    "meanwhile", "afterward", "finally", "first", "second", "next", "today",
    "yesterday", "tomorrow", "when", "while", "before", "after", "during", "once",
    "suddenly", "eventually", "immediately", "soon", "later", "earlier", "i", "we",
    "you", "they", "this", "that", "these", "those", "he", "she", "it", "since",
    "thus", "hence"
}

MIN_TOKENS_BEFORE_SPLIT = 5
MAX_TOKENS_WITHOUT_PUNCTUATION = 40


def _starts_with_sentence_marker(text: str) -> bool:
    """Check if text starts with a sentence start marker (case-insensitive)."""
    if not text.strip():
        return False
    first_word = text.strip().split()[0].lower().rstrip('.,!?;:')
    return first_word in SENTENCE_START_MARKERS


def _count_tokens(text: str) -> int:
    """Simple token count (split by whitespace)."""
    return len(text.split())


def segment_by_similarity(
    sentences: List[Sentence],
    window_size: int = 3,
    similarity_threshold: float = 0.3,
) -> List[SemanticSegment]:
    """
    Create segments using sentence-start heuristics.
    
    Starts a new segment when a sentence begins with discourse markers, pronouns,
    temporal transitions, or causal connectors. Combines sentences that don't
    start with these markers into the same segment.
    
    Args:
        sentences: List of Sentence objects with timestamps
        window_size: Ignored (kept for compatibility)
        similarity_threshold: Ignored (kept for compatibility)
        
    Returns:
        List of SemanticSegment objects grouped by sentence-start heuristics
    """
    if not sentences:
        return []
    
    logger.info(f"Creating segments from {len(sentences)} sentences using sentence-start heuristics")
    
    segments = []
    current_segment_sentences = []
    segment_id = 1
    
    for i, sentence in enumerate(sentences):
        text = sentence.text.strip()
        if not text:
            continue
        
        # Check if we should start a new segment
        should_start_new = False
        
        # Always start first segment
        if i == 0:
            should_start_new = True
        # Start new segment if sentence begins with a marker
        elif _starts_with_sentence_marker(text):
            # Only start new segment if current segment has minimum tokens
            current_text = " ".join(s.text for s in current_segment_sentences)
            if _count_tokens(current_text) >= MIN_TOKENS_BEFORE_SPLIT:
                should_start_new = True
        
        if should_start_new and current_segment_sentences:
            # Save current segment
            first_sentence = current_segment_sentences[0]
            last_sentence = current_segment_sentences[-1]
            segment_text = " ".join(s.text for s in current_segment_sentences)
            
            segment = SemanticSegment(
                segment_id=segment_id,
                start_time=first_sentence.start,
                end_time=last_sentence.end,
                text=segment_text,
                embedding=None,
            )
            segments.append(segment)
            segment_id += 1
            current_segment_sentences = []
        
        current_segment_sentences.append(sentence)
    
    # Add final segment
    if current_segment_sentences:
        first_sentence = current_segment_sentences[0]
        last_sentence = current_segment_sentences[-1]
        segment_text = " ".join(s.text for s in current_segment_sentences)
        
        segment = SemanticSegment(
            segment_id=segment_id,
            start_time=first_sentence.start,
            end_time=last_sentence.end,
            text=segment_text,
            embedding=None,
        )
        segments.append(segment)
    
    logger.info(f"Created {len(segments)} semantic segments using sentence-start heuristics")
    return segments
