import logging
from typing import List

from models import Sentence, Word

logger = logging.getLogger(__name__)

# Segment length constraints
TARGET_SEGMENT_LENGTH = 3.0  # seconds
MIN_SEGMENT_LENGTH = 1.5  # seconds
MAX_SEGMENT_LENGTH = 6.0  # seconds


def _merge_sentences(sentences: List[Sentence], start_idx: int, end_idx: int) -> Sentence:
    """Merge multiple sentences into one."""
    merged_sentences = sentences[start_idx:end_idx + 1]
    
    # Combine all words
    merged_words = []
    for sent in merged_sentences:
        merged_words.extend(sent.words)
    
    # Combine text
    merged_text = " ".join(sent.text for sent in merged_sentences)
    
    # Use earliest start and latest end
    start_time = merged_sentences[0].start
    end_time = merged_sentences[-1].end
    
    return Sentence(
        start=start_time,
        end=end_time,
        text=merged_text,
        words=merged_words
    )


def _split_sentence(sentence: Sentence) -> List[Sentence]:
    """Split a sentence that's too long into multiple segments targeting 3 seconds."""
    duration = sentence.end - sentence.start
    if duration <= MAX_SEGMENT_LENGTH:
        return [sentence]
    
    # Calculate how many segments we need
    num_segments = int(duration / TARGET_SEGMENT_LENGTH) + (1 if duration % TARGET_SEGMENT_LENGTH > 0 else 0)
    segment_duration = duration / num_segments
    
    split_sentences = []
    words = sentence.words
    
    if not words:
        # No words, split by time only
        for i in range(num_segments):
            seg_start = sentence.start + (i * segment_duration)
            seg_end = sentence.start + ((i + 1) * segment_duration) if i < num_segments - 1 else sentence.end
            split_sentences.append(Sentence(
                start=seg_start,
                end=seg_end,
                text=sentence.text,  # Same text for all segments
                words=[]
            ))
        return split_sentences
    
    # Split words proportionally
    words_per_segment = len(words) / num_segments
    
    for i in range(num_segments):
        start_word_idx = int(i * words_per_segment)
        end_word_idx = int((i + 1) * words_per_segment) if i < num_segments - 1 else len(words)
        
        segment_words = words[start_word_idx:end_word_idx]
        if not segment_words:
            continue
        
        seg_start = segment_words[0].start
        seg_end = segment_words[-1].end
        
        # Extract text for this segment
        segment_text = " ".join(w.word for w in segment_words)
        
        split_sentences.append(Sentence(
            start=seg_start,
            end=seg_end,
            text=segment_text,
            words=segment_words
        ))
    
    return split_sentences if split_sentences else [sentence]


def normalize_segment_lengths(sentences: List[Sentence]) -> List[Sentence]:
    """
    Normalize segment lengths to target 3 seconds.
    
    Rules:
    - If segment < 1.5s → merge with adjacent segment(s)
    - If segment > 6s → split into multiple segments
    
    Args:
        sentences: List of Sentence objects
        
    Returns:
        List of normalized Sentence objects
    """
    if not sentences:
        return []
    
    logger.info(f"Normalizing segment lengths: {len(sentences)} sentences (target: {TARGET_SEGMENT_LENGTH}s, min: {MIN_SEGMENT_LENGTH}s, max: {MAX_SEGMENT_LENGTH}s)")
    
    # First pass: Split segments that are too long
    split_sentences = []
    for sentence in sentences:
        duration = sentence.end - sentence.start
        if duration > MAX_SEGMENT_LENGTH:
            split = _split_sentence(sentence)
            split_sentences.extend(split)
            logger.debug(f"Split sentence {duration:.2f}s into {len(split)} segments")
        else:
            split_sentences.append(sentence)
    
    if not split_sentences:
        return []
    
    # Second pass: Merge segments that are too short
    normalized = []
    i = 0
    
    while i < len(split_sentences):
        current = split_sentences[i]
        duration = current.end - current.start
        
        if duration < MIN_SEGMENT_LENGTH:
            # Try to merge with next segment(s)
            merged = current
            merge_count = 1
            
            # Keep merging until we reach min length or run out of segments
            while duration < MIN_SEGMENT_LENGTH and i + merge_count < len(split_sentences):
                next_sentence = split_sentences[i + merge_count]
                merged = _merge_sentences(split_sentences, i, i + merge_count)
                duration = merged.end - merged.start
                merge_count += 1
                
                # Stop if merged segment would be too long
                if duration >= MAX_SEGMENT_LENGTH:
                    break
            
            normalized.append(merged)
            i += merge_count
            if merge_count > 1:
                logger.debug(f"Merged {merge_count} short segments into {duration:.2f}s segment")
        else:
            normalized.append(current)
            i += 1
    
    # Final validation: ensure no segments violate constraints
    final_normalized = []
    i = 0
    while i < len(normalized):
        sentence = normalized[i]
        duration = sentence.end - sentence.start
        
        if duration < MIN_SEGMENT_LENGTH:
            # Still too short - merge with previous if available, otherwise keep as is
            if final_normalized:
                prev_idx = len(final_normalized) - 1
                # Create a temporary list with previous and current for merging
                temp_list = [final_normalized[-1], sentence]
                merged = _merge_sentences(temp_list, 0, 1)
                final_normalized[-1] = merged
                logger.debug(f"Final merge: combined segments to {merged.end - merged.start:.2f}s")
            else:
                final_normalized.append(sentence)
            i += 1
        elif duration > MAX_SEGMENT_LENGTH:
            # Still too long - split again
            split = _split_sentence(sentence)
            final_normalized.extend(split)
            logger.debug(f"Final split: split {duration:.2f}s segment into {len(split)} segments")
            i += 1
        else:
            final_normalized.append(sentence)
            i += 1
    
    logger.info(f"Segment normalization complete: {len(sentences)} → {len(final_normalized)} segments")
    
    # Log statistics
    if final_normalized:
        durations = [s.end - s.start for s in final_normalized]
        avg_duration = sum(durations) / len(durations)
        min_duration = min(durations)
        max_duration = max(durations)
        logger.info(f"Segment duration stats: avg={avg_duration:.2f}s, min={min_duration:.2f}s, max={max_duration:.2f}s")
    
    return final_normalized

