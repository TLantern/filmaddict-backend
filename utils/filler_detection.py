import logging
import re
from typing import Dict, List

from models import Word, Sentence

logger = logging.getLogger(__name__)

FILLER_WORDS = {
    "um", "uh", "like", "you know", "basically", "so", "well",
    "actually", "literally", "kind of", "sort of", "i mean",
    "right", "okay", "ok", "yeah", "yep", "hmm", "er", "ah",
}


def detect_fillers(
    words: List[Word],
    sentences: List[Sentence],
) -> List[float]:
    """
    Detect filler words, repeated phrases, and sentence restarts per sentence.
    
    Args:
        words: List of Word objects with timestamps
        sentences: List of Sentence objects
        
    Returns:
        List of filler_density scores (0.0-1.0) for each sentence
    """
    if not sentences:
        return []
    
    logger.info(f"Detecting fillers in {len(sentences)} sentences")
    
    filler_densities = []
    
    for sentence in sentences:
        # Get words in this sentence
        sentence_words = [
            w for w in words
            if sentence.start <= w.start < sentence.end
        ]
        
        if not sentence_words:
            filler_densities.append(0.0)
            continue
        
        total_words = len(sentence_words)
        filler_count = 0
        restart_count = 0
        
        # Detect filler words
        for word in sentence_words:
            word_lower = word.word.lower().strip(".,!?;:")
            if word_lower in FILLER_WORDS:
                filler_count += 1
        
        # Detect repeated phrases (same 3+ word sequence within the sentence)
        word_sequence = [w.word.lower() for w in sentence_words]
        for i in range(len(word_sequence) - 2):
            phrase = " ".join(word_sequence[i:i+3])
            # Check if phrase appears again later in sentence
            for j in range(i + 3, len(word_sequence) - 2):
                if " ".join(word_sequence[j:j+3]) == phrase:
                    # Check time difference
                    time_diff = abs(sentence_words[j].start - sentence_words[i].start)
                    if time_diff < 10.0:
                        restart_count += 1
                        break
        
        # Calculate filler density
        filler_density = min(1.0, (filler_count + restart_count) / max(1, total_words))
        filler_densities.append(filler_density)
    
    logger.info(f"Computed filler densities: {[f'{d:.2f}' for d in filler_densities]}")
    return filler_densities
