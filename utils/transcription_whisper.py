import asyncio
import json
import logging
import os
import time
import threading
from pathlib import Path
from typing import List, Tuple, Optional, Set

from models import Word, Sentence

logger = logging.getLogger(__name__)

# Cache for sentence starters (loaded once, reused)
_sentence_starters: Optional[Set[str]] = None


def _load_sentence_starters() -> Set[str]:
    """Load sentence starters from JSON file (cached)."""
    global _sentence_starters
    if _sentence_starters is not None:
        return _sentence_starters
    
    # Get path to sentence_start.json (same directory as this file)
    current_dir = Path(__file__).parent
    json_path = current_dir / "sentence_start.json"
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            starters_list = data.get("sentence_starters", [])
            _sentence_starters = {starter.lower() for starter in starters_list}
            logger.info(f"Loaded {len(_sentence_starters)} sentence starters from {json_path}")
            return _sentence_starters
    except Exception as e:
        logger.error(f"Failed to load sentence starters from {json_path}: {e}")
        # Fallback to empty set - will result in all text being one sentence
        _sentence_starters = set()
        return _sentence_starters


def _is_sentence_starter(word_text: str) -> bool:
    """Check if a word is a sentence starter (case-insensitive, punctuation stripped)."""
    if not word_text:
        return False
    
    # Strip punctuation and convert to lowercase
    word_clean = word_text.strip(".,!?;:\"'").lower()
    if not word_clean:
        return False
    
    sentence_starters = _load_sentence_starters()
    return word_clean in sentence_starters

# OpenAI Whisper API (hosted on OpenAI servers - fast and reliable)
try:
    from openai import OpenAI
    _openai_available = True
except ImportError:
    _openai_available = False

_openai_client: Optional[OpenAI] = None


def _log_progress_while_processing(file_size_mb: float, stop_event: threading.Event):
    """
    Log periodic progress updates while API is processing.
    
    Args:
        file_size_mb: File size in MB for estimation
        stop_event: Event to stop logging when processing completes
    """
    start_time = time.time()
    update_interval = 5.0  # Log every 5 seconds
    
    # Estimate processing time: ~2-3 seconds per MB is typical
    estimated_time = file_size_mb * 2.5
    
    while not stop_event.is_set():
        elapsed = time.time() - start_time
        
        if elapsed < estimated_time:
            progress_pct = min(90, (elapsed / estimated_time) * 100)  # Cap at 90% until done
            logger.info(f"⏳ Processing... {progress_pct:.0f}% (elapsed: {elapsed:.1f}s, estimated: {estimated_time:.1f}s)")
        else:
            logger.info(f"⏳ Processing... taking longer than expected (elapsed: {elapsed:.1f}s)")
        
        # Wait for next update or stop signal
        if stop_event.wait(update_interval):
            break


def _get_openai_client() -> OpenAI:
    """Get or create OpenAI client instance."""
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required for transcription")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def transcribe_with_word_timestamps(audio_path: str) -> Tuple[List[Word], List[Sentence]]:
    """
    Transcribe audio file using OpenAI Whisper API (hosted on their servers).
    
    Uses OpenAI's Whisper API which provides fast, accurate transcription with
    word-level timestamps. Much faster than local CPU processing.
    
    Do NOT clean filler words - preserve um, uh, like, etc.
    
    Args:
        audio_path: Full path to the audio file
        
    Returns:
        Tuple of (words, sentences) with timestamps
        
    Raises:
        ImportError: If OpenAI library is not installed
        FileNotFoundError: If audio file doesn't exist
        Exception: If transcription fails
    """
    if not _openai_available:
        raise ImportError(
            "OpenAI library is not installed. Install with:\n"
            "pip install openai\n"
            "And set OPENAI_API_KEY environment variable."
        )
    
    # Check if it's a local file path or HTTP URL
    is_http_url = audio_path.startswith("http://") or audio_path.startswith("https://")
    
    # Only check file existence for local paths, not HTTP URLs
    if not is_http_url and not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    try:
        # Get file size for progress estimation
        file_size_mb = os.path.getsize(audio_path) / (1024 * 1024) if os.path.exists(audio_path) else 0
        logger.info(f"📤 Starting transcription with OpenAI Whisper API: {audio_path} ({file_size_mb:.2f} MB)")
        
        client = _get_openai_client()
        
        # Start progress logging in background thread
        stop_progress = threading.Event()
        progress_thread = threading.Thread(
            target=_log_progress_while_processing,
            args=(file_size_mb, stop_progress),
            daemon=True
        )
        progress_thread.start()
        
        # Open audio file and transcribe using OpenAI Whisper API
        # The API handles all audio processing, chunking, and returns word-level timestamps
        start_time = time.time()
        try:
            with open(audio_path, "rb") as audio_file:
                logger.info("📤 Sending audio to OpenAI Whisper API...")
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json",  # Get word-level timestamps
                    timestamp_granularities=["word"]  # Request word-level timestamps
                )
        finally:
            # Stop progress logging
            stop_progress.set()
            progress_thread.join(timeout=1.0)
        
        elapsed_time = time.time() - start_time
        logger.info(f"✅ Received transcription from OpenAI API in {elapsed_time:.1f}s")
        
        # Log raw response structure for debugging
        logger.debug(f"OpenAI API response type: {type(transcript)}")
        if hasattr(transcript, '__dict__'):
            logger.debug(f"OpenAI API response attributes: {list(transcript.__dict__.keys())}")
        
        # Convert OpenAI response to our Word and Sentence models
        words = []
        sentences = []
        current_sentence_words = []
        current_sentence_start = None
        current_sentence_text = ""
        
        # OpenAI returns words with timestamps
        # Handle both dict and object formats
        transcript_words = getattr(transcript, 'words', None) or []
        transcript_duration = getattr(transcript, 'duration', 0) or 0
        transcript_text = getattr(transcript, 'text', '') or ''
        
        # Validate we got a response
        if not transcript_words and not transcript_text:
            logger.error("❌ OpenAI API returned empty response - no words or text")
            raise Exception("OpenAI Whisper API returned empty response")
        
        logger.info(f"📊 OpenAI API response: {len(transcript_words)} words, duration: {transcript_duration:.1f}s, text length: {len(transcript_text)} chars")
        
        # If we have text but no words, log a warning
        if transcript_text and not transcript_words:
            logger.warning("⚠️ OpenAI API returned text but no word-level timestamps - will estimate timestamps")
        
        if transcript_words:
            logger.info(f"📝 Processing {len(transcript_words)} words from OpenAI API (duration: {transcript_duration:.1f}s)")
            for word_data in transcript_words:
                # Handle both dict and object formats
                if isinstance(word_data, dict):
                    word_start = word_data.get('start', 0)
                    word_end = word_data.get('end', word_start + 0.1)
                    word_text = word_data.get('word', '').strip()
                else:
                    word_start = getattr(word_data, 'start', 0)
                    word_end = getattr(word_data, 'end', word_start + 0.1)
                    word_text = getattr(word_data, 'word', '').strip()
                
                if not word_text:
                    continue
                
                word = Word(
                    start=word_start,
                    end=word_end,
                    word=word_text.strip(".,!?;:"),
                    confidence=0.9,  # OpenAI doesn't provide confidence, use default
                )
                words.append(word)
                
                # Check if this word is a sentence starter
                is_starter = _is_sentence_starter(word_text)
                
                # If we have a sentence in progress and this is a starter, end previous sentence
                if current_sentence_start is not None and is_starter and current_sentence_words:
                    # End previous sentence at the end of the last word we added to it
                    prev_word_end = current_sentence_words[-1].end
                    sentence = Sentence(
                        start=current_sentence_start,
                        end=prev_word_end,
                        text=current_sentence_text.strip(),
                        words=current_sentence_words.copy(),
                    )
                    sentences.append(sentence)
                    
                    # Start new sentence with this starter word
                    current_sentence_words = []
                    current_sentence_start = word_start
                    current_sentence_text = ""
                
                # If no sentence in progress, start one
                if current_sentence_start is None:
                    current_sentence_start = word_start
                
                current_sentence_words.append(word)
                current_sentence_text += word_text + " "
            
            # Add final sentence if exists
            if current_sentence_words and current_sentence_start is not None:
                final_end = words[-1].end if words else current_sentence_start + 1.0
                sentence = Sentence(
                    start=current_sentence_start,
                    end=final_end,
                    text=current_sentence_text.strip(),
                    words=current_sentence_words.copy(),
                )
                sentences.append(sentence)
        else:
            # Fallback: if word-level timestamps not available, use text and estimate timestamps
            logger.warning("Word-level timestamps not available, estimating from text")
            transcript_text = getattr(transcript, 'text', '') or ''
            transcript_duration = getattr(transcript, 'duration', 0) or 0
            
            if transcript_text:
                # Split text into words and estimate timestamps
                text_words = transcript_text.split()
                word_duration = transcript_duration / len(text_words) if text_words and transcript_duration > 0 else 0.5
                
                for i, word_text in enumerate(text_words):
                    word_start = i * word_duration
                    word_end = (i + 1) * word_duration
                    
                    word = Word(
                        start=word_start,
                        end=word_end,
                        word=word_text.strip(".,!?;:"),
                        confidence=0.9,
                    )
                    words.append(word)
                    
                    # Check if this word is a sentence starter
                    is_starter = _is_sentence_starter(word_text)
                    
                    # If we have a sentence in progress and this is a starter, end previous sentence
                    if current_sentence_start is not None and is_starter and current_sentence_words:
                        # End previous sentence at the end of the last word we added to it
                        prev_word_end = current_sentence_words[-1].end
                        sentence = Sentence(
                            start=current_sentence_start,
                            end=prev_word_end,
                            text=current_sentence_text.strip(),
                            words=current_sentence_words.copy(),
                        )
                        sentences.append(sentence)
                        
                        # Start new sentence with this starter word
                        current_sentence_words = []
                        current_sentence_start = word_start
                        current_sentence_text = ""
                    
                    # If no sentence in progress, start one
                    if current_sentence_start is None:
                        current_sentence_start = word_start
                    
                    current_sentence_words.append(word)
                    current_sentence_text += word_text + " "
                
                # Add final sentence if exists
                if current_sentence_words and current_sentence_start is not None:
                    final_end = words[-1].end if words else current_sentence_start + 1.0
                    sentence = Sentence(
                        start=current_sentence_start,
                        end=final_end,
                        text=current_sentence_text.strip(),
                        words=current_sentence_words.copy(),
                    )
                    sentences.append(sentence)
        
        # Validate we have data before returning
        if not words:
            logger.warning("⚠️ No words extracted from OpenAI transcription - this may indicate an issue")
            if transcript_text:
                logger.info(f"Fallback: Using transcript text ({len(transcript_text)} chars) to create words")
            else:
                raise Exception("OpenAI API returned empty transcription - no words or text found")
        
        logger.info(f"✅ Transcription complete: {len(words)} words, {len(sentences)} sentences")
        if transcript_duration > 0:
            processing_speed = transcript_duration / elapsed_time if elapsed_time > 0 else 0
            logger.info(f"⚡ Processing speed: {processing_speed:.1f}x real-time (processed {transcript_duration:.1f}s in {elapsed_time:.1f}s)")
        
        # Log sample of first few words to verify data quality
        if words:
            sample_words = words[:5]
            logger.debug(f"📝 Sample words: {[(w.word, f'{w.start:.2f}-{w.end:.2f}s') for w in sample_words]}")
        
        return words, sentences
        
    except Exception as e:
        logger.error(f"Error transcribing audio with OpenAI API: {str(e)}", exc_info=True)
        raise Exception(f"Failed to transcribe audio: {str(e)}")


async def transcribe_with_word_timestamps_async(audio_path: str) -> Tuple[List[Word], List[Sentence]]:
    """
    Async wrapper for transcribe_with_word_timestamps.
    
    Args:
        audio_path: Full path to the audio file
        
    Returns:
        Tuple of (words, sentences) with timestamps
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, transcribe_with_word_timestamps, audio_path)
