import asyncio
import logging
import os
import re
from typing import List

from openai import OpenAI
from models import TranscriptSegment

logger = logging.getLogger(__name__)

_openai_client: OpenAI = None


def get_openai_client() -> OpenAI:
    """Get or create OpenAI client instance."""
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def normalize_text(text: str) -> str:
    """
    Normalize transcript text by stripping whitespace and fixing spacing.
    
    Args:
        text: Raw transcript text
        
    Returns:
        Normalized text
    """
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def transcribe_audio(audio_path: str) -> List[TranscriptSegment]:
    """
    Transcribe audio file using OpenAI Whisper API.
    
    Args:
        audio_path: Full path to the audio file
        
    Returns:
        List of TranscriptSegment objects with start, end, and text
        
    Raises:
        FileNotFoundError: If audio file doesn't exist
        Exception: If transcription fails
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    client = get_openai_client()
    
    try:
        logger.info(f"Transcribing audio: {audio_path}")
        
        with open(audio_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="verbose_json",
                timestamp_granularities=["segment"],
            )
        
        segments = []
        if hasattr(transcript, "segments") and transcript.segments:
            for segment in transcript.segments:
                normalized_text = normalize_text(segment.text)
                if normalized_text:
                    segments.append(
                        TranscriptSegment(
                            start=segment.start,
                            end=segment.end,
                            text=normalized_text,
                        )
                    )
        else:
            normalized_text = normalize_text(transcript.text)
            if normalized_text:
                segments.append(
                    TranscriptSegment(
                        start=0.0,
                        end=transcript.duration if hasattr(transcript, "duration") else 0.0,
                        text=normalized_text,
                    )
                )
        
        logger.info(f"Successfully transcribed audio: {len(segments)} segments")
        return segments
        
    except Exception as e:
        logger.error(f"Error transcribing audio: {str(e)}", exc_info=True)
        raise Exception(f"Failed to transcribe audio: {str(e)}")


async def transcribe_audio_async(audio_path: str) -> List[TranscriptSegment]:
    """
    Async wrapper for transcribe_audio function.
    
    Args:
        audio_path: Full path to the audio file
        
    Returns:
        List of TranscriptSegment objects
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

