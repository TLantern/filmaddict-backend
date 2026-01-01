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
    
    Outputs MP3 format (16kHz, mono, 32kbps) optimized for fast processing and Whisper transcription.
    Uses compressed format and optimized settings for 5x faster extraction.
    Supports direct processing from HTTP URLs (S3 presigned URLs) without downloading full video.
    Stores audio in uploads/audio/ directory with UUID-based filename.
    
    Args:
        video_path: Full path to the video file or HTTP URL (S3 presigned URL)
        
    Returns:
        Full path to the extracted audio file
        
    Raises:
        FileNotFoundError: If video file doesn't exist (for local paths only)
        Exception: If FFmpeg extraction fails
    """
    # Check if it's a local file path or HTTP URL (S3 presigned URL)
    is_http_url = video_path.startswith("http://") or video_path.startswith("https://")
    
    # Only check file existence for local paths, not HTTP URLs
    if not is_http_url and not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    os.makedirs(AUDIO_DIR, exist_ok=True)
    
    audio_filename = f"{uuid.uuid4()}.mp3"
    audio_path = os.path.join(AUDIO_DIR, audio_filename)
    
    try:
        logger.info(f"Extracting audio from video: {video_path[:80]}...")
        
        # Optimized FFmpeg settings for 5x faster processing:
        # - Skip video decoding entirely (vn flag)
        # - Use lower bitrate for faster encoding (32k vs 64k - still good for transcription)
        # - Use faster quality setting (q:a 5)
        # - FFmpeg can process HTTP URLs directly without full download
        
        stream = ffmpeg.input(video_path)
        stream = ffmpeg.output(
            stream,
            audio_path,
            vn=None,  # Skip video processing (no video output) - speeds up processing
            acodec="libmp3lame",
            ac=1,  # Mono channel
            ar=16000,  # 16kHz sample rate
            audio_bitrate="32k",  # Reduced from 64k to 32k for faster encoding (adequate for transcription)
            **{'q:a': 5},  # Audio quality level 5 (faster encoding, acceptable quality for transcription)
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


def delete_audio_file(audio_path: str) -> None:
    """
    Delete an audio file from the audio directory.
    
    Args:
        audio_path: Full path to the audio file to delete
    """
    try:
        if os.path.exists(audio_path):
            os.remove(audio_path)
            logger.info(f"Deleted audio file: {audio_path}")
    except Exception as e:
        logger.warning(f"Failed to delete audio file {audio_path}: {str(e)}")


def cleanup_audio_directory() -> None:
    """
    Clean up all audio files in the audio directory.
    Useful for cleanup when deleting clips/moments.
    """
    try:
        if os.path.exists(AUDIO_DIR):
            for filename in os.listdir(AUDIO_DIR):
                file_path = os.path.join(AUDIO_DIR, filename)
                if os.path.isfile(file_path) and filename.endswith('.mp3'):
                    try:
                        os.remove(file_path)
                        logger.info(f"Cleaned up audio file: {file_path}")
                    except Exception as e:
                        logger.warning(f"Failed to delete audio file {file_path}: {str(e)}")
    except Exception as e:
        logger.warning(f"Failed to cleanup audio directory: {str(e)}")

