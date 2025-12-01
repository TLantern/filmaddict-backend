import asyncio
import logging
import os
import uuid

import ffmpeg

logger = logging.getLogger(__name__)

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
AUDIO_DIR = os.path.join(UPLOAD_DIR, "audio")


def extract_audio(video_path: str) -> str:
    """
    Extract audio track from video file using FFmpeg.
    
    Outputs MP3 format (16kHz, mono, 64kbps) optimized for Whisper API and file size.
    Uses compressed format to stay under OpenAI's 25MB limit.
    Stores audio in uploads/audio/ directory with UUID-based filename.
    
    Args:
        video_path: Full path to the video file
        
    Returns:
        Full path to the extracted audio file
        
    Raises:
        FileNotFoundError: If video file doesn't exist
        Exception: If FFmpeg extraction fails
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    os.makedirs(AUDIO_DIR, exist_ok=True)
    
    audio_filename = f"{uuid.uuid4()}.mp3"
    audio_path = os.path.join(AUDIO_DIR, audio_filename)
    
    try:
        logger.info(f"Extracting audio from video: {video_path}")
        
        stream = ffmpeg.input(video_path)
        stream = ffmpeg.output(
            stream,
            audio_path,
            acodec="libmp3lame",
            ac=1,
            ar=16000,
            audio_bitrate="64k",
        )
        ffmpeg.run(stream, overwrite_output=True, quiet=True)
        
        if not os.path.exists(audio_path):
            raise Exception(f"Audio extraction failed: output file not created")
        
        logger.info(f"Successfully extracted audio to: {audio_path}")
        return audio_path
        
    except ffmpeg.Error as e:
        error_message = e.stderr.decode() if e.stderr else str(e)
        logger.error(f"FFmpeg error extracting audio: {error_message}")
        raise Exception(f"Failed to extract audio: {error_message}")
    except Exception as e:
        logger.error(f"Error extracting audio: {str(e)}", exc_info=True)
        raise


async def extract_audio_async(video_path: str) -> str:
    """
    Async wrapper for extract_audio function.
    
    Args:
        video_path: Full path to the video file
        
    Returns:
        Full path to the extracted audio file
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, extract_audio, video_path)

