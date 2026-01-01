import asyncio
import logging
import os
from typing import List

from models import TimelineItem, TranscriptSegment

logger = logging.getLogger(__name__)


def transcribe_audio(audio_path: str) -> List[TimelineItem]:
    """
    Transcribe audio using Whisper and detect silence segments.
    
    This function performs transcription using OpenAI Whisper API, then performs
    post-processing to detect and insert silence segments between speech segments.
    
    Args:
        audio_path: Full path to the audio file
        
    Returns:
        List of TimelineItem objects (union of SpeechSegment and SilenceSegment) 
        sorted by start time, representing the unified timeline
        
    Raises:
        FileNotFoundError: If audio file doesn't exist
        ImportError: If Whisper is not available
        Exception: If transcription fails
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    try:
        from utils.transcription_whisper import transcribe_with_word_timestamps
        logger.info(f"📥 Using OpenAI Whisper API transcription for: {audio_path}")
        words, sentences = transcribe_with_word_timestamps(audio_path)
        
        # Validate we received data from OpenAI
        if not sentences:
            logger.error("❌ OpenAI Whisper API returned no sentences")
            raise Exception("OpenAI Whisper API returned empty transcription - no sentences found")
        
        if not words:
            logger.error("❌ OpenAI Whisper API returned no words")
            raise Exception("OpenAI Whisper API returned empty transcription - no words found")
        
        logger.info(f"✅ Received {len(words)} words and {len(sentences)} sentences from OpenAI Whisper API")
        
        # Convert sentences to TranscriptSegments (existing transcription logic)
        speech_segments = []
        for sentence in sentences:
            if not sentence.text or not sentence.text.strip():
                logger.warning(f"Skipping empty sentence at {sentence.start:.2f}s-{sentence.end:.2f}s")
                continue
            speech_segments.append(TranscriptSegment(
                start=sentence.start,
                end=sentence.end,
                text=sentence.text,
            ))
        
        if not speech_segments:
            logger.error("❌ No valid transcript segments created from OpenAI transcription")
            raise Exception("Failed to create transcript segments from OpenAI transcription")
        
        logger.info(f"✅ Created {len(speech_segments)} transcript segments from OpenAI Whisper API")
        logger.debug(f"📝 Sample segment: {speech_segments[0].text[:50]}..." if speech_segments else "No segments")
        
        # Post-processing: Detect silence and merge with speech segments
        try:
            from utils.silence_detection import detect_silence_segments, merge_transcript_with_silence
            
            logger.info(f"🔇 Post-processing: Detecting silence segments...")
            silence_segments = detect_silence_segments(audio_path)
            
            timeline = merge_transcript_with_silence(speech_segments, silence_segments)
            logger.info(f"✅ Unified timeline created: {len(timeline)} items ({len(speech_segments)} speech + {len(silence_segments)} silence)")
            
            return timeline
            
        except ImportError as silence_error:
            # If pydub is not available, log warning and return speech segments only
            logger.warning(
                f"⚠️ Silence detection not available (pydub not installed): {silence_error}. "
                "Returning speech segments only. Install pydub for silence detection."
            )
            # Convert to TimelineItem format (SpeechSegment)
            from models import SpeechSegment
            timeline: List[TimelineItem] = [
                SpeechSegment(start=seg.start, end=seg.end, text=seg.text)
                for seg in speech_segments
            ]
            return timeline
        except Exception as silence_error:
            # If silence detection fails, log error and return speech segments only
            logger.warning(
                f"⚠️ Silence detection failed: {silence_error}. "
                "Returning speech segments only."
            )
            from models import SpeechSegment
            timeline: List[TimelineItem] = [
                SpeechSegment(start=seg.start, end=seg.end, text=seg.text)
                for seg in speech_segments
            ]
            return timeline
        
    except ImportError as e:
        error_msg = (
            "Whisper transcription is not available. "
            "Install required dependencies: pip install torch torchaudio soundfile\n"
            "Or use Colab processing by setting COLAB_API_URL environment variable."
        )
        logger.error(error_msg)
        raise ImportError(error_msg) from e
    except Exception as e:
        error_msg = f"Whisper transcription failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise Exception(error_msg) from e


async def transcribe_audio_async(audio_path: str) -> List[TimelineItem]:
    """
    Async wrapper for transcribe_audio function.
    
    Args:
        audio_path: Full path to the audio file
        
    Returns:
        List of TimelineItem objects (union of SpeechSegment and SilenceSegment)
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, transcribe_audio, audio_path)


def cleanup_audio_file(audio_path: str) -> None:
    """
    Delete audio file after transcription.
    
    Args:
        audio_path: Full path to the audio file to delete
    """
    try:
        if os.path.exists(audio_path):
            os.remove(audio_path)
            logger.info(f"Cleaned up audio file: {audio_path}")
    except Exception as e:
        logger.warning(f"Failed to cleanup audio file {audio_path}: {str(e)}")

